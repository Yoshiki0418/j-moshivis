"""One Terra v11 utterance through the user's ChatGPT-authenticated Codex CLI."""

import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

from .production_budget import MODEL
from .visual_dialogue_prompts_v11 import ASSISTANT, USER


class CodexInvocationError(RuntimeError):
    """The request could not be confirmed as successful. Never retry it blindly."""


class CodexTransientError(CodexInvocationError):
    """A transport interruption that may clear after a cooldown."""


class CodexIncompleteAtLimit(CodexInvocationError):
    """Quota reported after output started; retain the unconfirmed reservation."""


class CodexUsageLimit(CodexInvocationError):
    """The CLI explicitly rejected the turn due to a usage limit."""


class CodexRateLimit(CodexUsageLimit):
    """A temporary rejection before any output; retry_after is in seconds."""

    def __init__(self, message="Codex rate limit reached", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


_DISABLED = (
    "shell_tool", "shell_snapshot", "unified_exec", "code_mode_host", "apps", "plugins", "hooks",
    "multi_agent", "multi_agent_v2", "browser_use", "browser_use_external", "computer_use",
    "image_generation", "view_image", "daemon_auto_start", "unbounded_connection_retries",
    "skill_mcp_dependency_install", "skill_search", "sleep_tool",
)
_MIME = {"data:image/jpeg;base64": ".jpg", "data:image/png;base64": ".png", "data:image/webp;base64": ".webp"}
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens")
_QUOTA = re.compile(r"(?:hit|reached|exceeded) (?:your |the )?usage limit|usage[ _-]limit[ _-](?:reached|exceeded)|"
                    r"(?:weekly|weekly usage)[ _-]limit[ _-](?:reached|exceeded)|(?:hit|reached|exceeded) (?:your |the )?weekly (?:usage )?limit|"
                    r"insufficient_quota|quota[ _-](?:exceeded|exhausted)|out of (?:usage|credits)", re.I)
_RATE = re.compile(r"(?:hit|reached|exceeded) (?:your |the )?rate limit|rate[ _-]limit[ _-](?:reached|exceeded)|"
                   r"rate[ _-]limited|too many requests|\b429\b", re.I)
_RECONNECT = re.compile(r"^reconnecting(?:\.{3}|…)?\s+\d+/\d+\b", re.I)
_CODE_MODE_DISABLED = ("Code Mode is unavailable because code-mode host is disabled. "
                       "Code mode will fail closed; enable `features.code_mode_host` "
                       "and install `codex-code-mode-host`.")
_CLI_TOOL_ITEMS = frozenset({"command_execution", "file_change", "mcp_tool_call", "dynamic_tool_call",
                             "collab_agent_tool_call", "web_search", "image_view", "image_generation",
                             "function_call_output", "sub_agent_activity", "sleep"})


def _failure(errors, stderr, returncode, response_started):
    # Persist only known diagnostic categories, never arbitrary provider text or account identifiers.
    diagnostic = "\n".join([*errors, stderr.decode(errors="replace")])
    if _QUOTA.search(diagnostic):
        if response_started:
            return CodexIncompleteAtLimit("Codex usage limit reached during a turn")
        return CodexUsageLimit("Codex usage limit reached")
    if not response_started:
        if _RATE.search(diagnostic):
            delay = re.search(r"(?:retry[ _-]after(?:_seconds)?[\s\"':=]+|try again in\s+)(\d+(?:\.\d+)?)\s*(?:seconds?|s)?", diagnostic, re.I)
            return CodexRateLimit(retry_after=float(delay[1]) if delay else None)
    for pattern, reason in (
        (r"\b(?:401|403)\b|unauthorized|forbidden|not logged in|refresh token", "authentication or access denied"),
        (r"model.{0,70}(?:not found|not supported|unsupported|not available)", "model unavailable"),
        (r"unexpected argument|unknown feature|invalid config|failed to parse|configuration error", "invalid CLI configuration"),
        (r"permission denied|read-only file system", "filesystem access denied"),
        (r"stream|reconnect|connection|dns|timed out|interrupted", "connection interrupted"),
        (r"\b5\d\d\b", "provider error"),
    ):
        if re.search(pattern, diagnostic, re.I):
            if reason in ("connection interrupted", "provider error") or (
                    reason == "authentication or access denied" and re.search(r"\b403\b|forbidden", diagnostic, re.I)):
                return CodexTransientError(f"CLI {reason} (exit {returncode})")
            return CodexInvocationError(f"CLI {reason} (exit {returncode})")
    stage = "after partial output" if response_started else "before a confirmed response"
    return CodexInvocationError(f"CLI failure {stage} (exit {returncode})")


def _environment():
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL", "OPENAI_ORG_ID",
                "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
        env.pop(key, None)
    return env


def _schema(assistant):
    properties = {"text": {"type": "string"}}
    if assistant:
        properties["tasks"] = {"type": "array", "items": {"type": "string", "enum": ["ocr", "counting"]}}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _inputs(request, image_path):
    instructions = request.get("instructions")
    if request.get("model") != MODEL or instructions not in (USER, ASSISTANT):
        raise CodexInvocationError("Expected the Terra v11 role request")
    try:
        (message,) = request["input"]
        text, image = message["content"]
        prefix, encoded = image["image_url"].split(",", 1)
        extension = _MIME[prefix]
        if (message["role"] != "user" or text["type"] != "input_text" or not isinstance(text["text"], str)
                or image["type"] != "input_image" or image["detail"] != "high"):
            raise ValueError
        source = Path(image_path).read_bytes()
        if base64.b64encode(source).decode("ascii") != encoded:
            raise CodexInvocationError("The source image does not match the requested bytes")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CodexInvocationError("Invalid dialogue context or source image") from exc
    return instructions, text["text"], source, extension


def _events(raw):
    errors, completed, reported_models = [], [], set()
    pending_notices = []
    recovered_notices = 0
    response_started = False
    turn_started = False
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError
        except ValueError:
            raise CodexInvocationError("Invalid CLI JSON event") from None
        kind = event.get("type")
        if kind == "turn.started":
            turn_started = True
        if event.get("model") is not None:
            if event["model"] != MODEL:
                raise CodexInvocationError("CLI reported an unexpected model")
            reported_models.add(event["model"])
        if kind == "turn.failed":
            errors.append(json.dumps(event))
        diagnostic = event if kind == "error" else None
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item")
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "error":
                if not turn_started and kind == "item.completed" and item.get("message") == _CODE_MODE_DISABLED:
                    continue
                diagnostic = item
            elif item_type in ("reasoning", "agent_message"):
                response_started = True
            elif item_type != "user_message":
                category = item_type if isinstance(item_type, str) and item_type in _CLI_TOOL_ITEMS else "unknown item"
                raise CodexInvocationError(f"Unexpected CLI activity ({category})")
        if diagnostic is not None:
            message = str(diagnostic.get("message", ""))
            if (_RECONNECT.match(message)
                    or message.startswith("Falling back from WebSockets to HTTPS transport.")):
                pending_notices.append(json.dumps(diagnostic))
            else:
                errors.append(json.dumps(diagnostic))
        if kind == "turn.completed":
            recovered_notices += len(pending_notices)
            pending_notices.clear()
            completed.append(event)
    return errors + pending_notices, completed, next(iter(reported_models), None), response_started, recovered_notices


class CodexTransport:
    def __init__(self, executable=None, timeout=240):
        self.executable = shutil.which(executable or "codex")
        if not self.executable or not 0 < timeout <= 3600:
            raise CodexInvocationError("Codex executable is missing or timeout is invalid")
        self.executable = str(Path(self.executable).resolve())
        self.timeout = timeout
        self.identity = None

    def preflight(self):
        """Verify saved authentication without reading or transmitting login credentials."""
        env = _environment()
        try:
            def local(args):
                return subprocess.run([self.executable, *args], cwd=tempfile.gettempdir(), env=env,
                                      capture_output=True, text=True, timeout=min(15, self.timeout), check=False)
            auth = local(["login", "status"])
            if auth.returncode or "logged in using chatgpt" not in (auth.stdout + auth.stderr).lower():
                raise CodexInvocationError("A ChatGPT CLI login is required")
            version = local(["--version"])
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CodexInvocationError("Could not check Codex CLI authentication or version") from exc
        reported = re.search(r"^codex-cli ([\w.+-]+)$", version.stdout, re.M)
        if version.returncode or reported is None:
            raise CodexInvocationError("Could not identify the Codex CLI version")
        self.identity = dict(model=MODEL, auth_method="chatgpt", transport="codex_cli", cli_version=reported.group(1))
        return dict(self.identity)

    def generate(self, request, image_path):
        identity = self.identity or self.preflight()
        instructions, context, source, suffix = _inputs(request, image_path)
        assistant = instructions == ASSISTANT
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="jmv-terra-cli-") as directory:
            work = Path(directory)
            image = work / ("image" + suffix)
            image.write_bytes(source)
            schema = work / "schema.json"
            schema.write_text(json.dumps(_schema(assistant)), encoding="utf-8")
            output = work / "response.json"
            cmd = [self.executable, "--no-daemon", "exec", "--ignore-user-config", "--ephemeral",
                   "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--json",
                   "--model", MODEL, "--cd", str(work), "--output-schema", str(schema),
                   "--output-last-message", str(output), "--image", str(image)]
            for setting in ('forced_login_method="chatgpt"', 'model_provider="openai"', 'model_reasoning_effort="low"',
                            'approval_policy="never"', 'web_search="disabled"', 'project_doc_max_bytes=0',
                            'history.persistence="none"', 'mcp_servers={}',
                            "developer_instructions=" + json.dumps(instructions, ensure_ascii=False)):
                cmd.extend(["-c", setting])
            for feature in _DISABLED:
                cmd.extend(["--disable", feature])
            cmd.append("-")
            try:
                proc = subprocess.Popen(cmd, cwd=work, env=_environment(), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    stdout, stderr = proc.communicate(context.encode("utf-8"), timeout=self.timeout)
                finally:
                    if proc.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
            except subprocess.TimeoutExpired:
                raise CodexTransientError("Codex invocation timed out") from None
            except OSError as exc:
                raise CodexInvocationError("Could not execute Codex") from exc

            errors, completed, reported, response_started, recovered = _events(stdout)
            if errors or proc.returncode:
                raise _failure(errors, stderr, proc.returncode, bool(completed) or response_started)
            if len(completed) != 1 or not output.is_file() or output.stat().st_size > 1024 * 1024:
                raise CodexInvocationError("Missing CLI completion or final response")
            raw = output.read_bytes()
        try:
            reply = json.loads(raw)
            if (not isinstance(reply, dict) or set(reply) != set(_schema(assistant)["required"])
                    or not isinstance(reply["text"], str) or not reply["text"].strip()):
                raise ValueError
            if assistant and (not isinstance(reply["tasks"], list) or
                              any(task not in ("ocr", "counting") for task in reply["tasks"])):
                raise ValueError
            usage = completed[0].get("usage", {})
            if not isinstance(usage, dict) or any(type(usage[k]) is not int or usage[k] < 0
                                                   for k in _USAGE_KEYS if k in usage):
                raise ValueError
        except (ValueError, UnicodeError) as exc:
            raise CodexInvocationError("Invalid final utterance or usage record") from exc
        receipt = dict(identity, reported_model=reported, usage={k: usage[k] for k in _USAGE_KEYS if k in usage},
                       response_sha256=hashlib.sha256(raw).hexdigest(), elapsed_seconds=round(time.monotonic()-start, 3))
        if recovered:
            receipt["recovered_transport_notices"] = recovered
        return reply, receipt
