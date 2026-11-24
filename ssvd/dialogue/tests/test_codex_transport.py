"""Exercise the CLI protocol with a local fake executable and no authentication or model traffic."""

import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue.codex_transport import CodexInvocationError, CodexRateLimit, CodexTransport, CodexUsageLimit, _events
from ssvd.dialogue.production_budget import MODEL
from ssvd.dialogue.regeneration_roles import next_step, VISUAL_V11
from ssvd.dialogue.visual_dialogue_prompts_v11 import ASSISTANT, USER


# A fake executable inspects the real argv / image / stdin that a worker would deliver.
_FAKE = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
folder = Path(__file__).resolve().parent
mode = (folder / "mode").read_text()
if args == ["login", "status"]:
    print("Logged in using " + ("an API key" if mode=="api_key" else "ChatGPT"), file=sys.stderr)
    sys.exit(0)
if args == ["--version"]:
    print("codex-cli 0.171.0-alpha.13")
    sys.exit(0)
if mode in ("quota", "partial_then_quota", "temporary"):
    if mode == "partial_then_quota":
        print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"partial"}}))
    if mode == "temporary":
        print("429 Too Many Requests; retry-after: 12", file=sys.stderr)
        print(json.dumps({"type":"turn.failed"}))
    else:
        print(json.dumps({"type":"turn.failed","error":{"message":"You have hit your usage limit."}}))
    sys.exit(1)
if mode == "unknown":
    print(json.dumps({"type":"turn.failed","error":{"message":"Request interrupted while reading response"}}))
    sys.exit(1)
if mode == "invalid_events":
    print("truncated event")
    sys.exit(1)
image = Path(args[args.index("--image")+1])
out = Path(args[args.index("--output-last-message")+1])
schema = json.loads(Path(args[args.index("--output-schema")+1]).read_text())
config = dict(i.split("=",1) for i in args if "=" in i)
snapshot = dict(args=args,image=image.read_bytes().hex(),role=json.loads(config["developer_instructions"]),
                context=json.load(sys.stdin),schema=schema,workdir=str(Path.cwd()),
                has_key=any(name in os.environ for name in ("OPENAI_API_KEY","CODEX_API_KEY","CODEX_ACCESS_TOKEN")))
(folder / "seen").write_text(json.dumps(snapshot))
reply={"text":"静かな雰囲気だね。"}
if "tasks" in schema["required"]: reply["tasks"]=["counting"]
if mode=="extra": reply["old_dialogue"]="unused"
out.write_text(json.dumps(reply))
if mode=="wrong_model": print(json.dumps({"type":"turn.started","model":"other-model"}))
if mode=="tool": print(json.dumps({"type":"item.started","item":{"type":"command_execution"}}))
if mode.startswith("native_fallback"):
    print(json.dumps({"type":"item.completed","item":{"type":"error","message":
                     "Falling back from WebSockets to HTTPS transport. unexpected status 403 Forbidden"}}))
    if mode=="native_fallback_failure":
        print(json.dumps({"type":"turn.failed","error":{"message":"HTTP 403 Forbidden"}}))
        sys.exit(1)
    if mode=="native_fallback_bad_json": out.write_text("invalid json")
if mode=="reconnect": print(json.dumps({"type":"error","message":"Reconnecting… 1/5"}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":2,"arbitrary_debug":"private"}}))
'''


class CodexTransportTests(unittest.TestCase):
    def setUp(self):
        files = tempfile.TemporaryDirectory()
        self.addCleanup(files.cleanup)
        self.root = Path(files.name)
        self.executable = self.root / "fake-codex"
        self.executable.write_text(_FAKE)
        self.executable.chmod(0o700)
        self.image = self.root / "never-send-this-uid.jpg"
        self.image.write_bytes(b"same source bytes")
        self.mode = self.root / "mode"
        self.mode.write_text("ok")
        self.seen = self.root / "seen"
        self.cli = CodexTransport(str(self.executable))

    def request(self, prior=()):
        url = "data:image/jpeg;base64," + base64.b64encode(self.image.read_bytes()).decode()
        return next_step("fixture", "ocr", url, list(prior), pairs=4, model=MODEL, version=VISUAL_V11)["request"]

    def test_either_speaker_uses_pinned_role_checked_image_full_history_and_chatgpt_only(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused", "CODEX_API_KEY":"unused", "CODEX_ACCESS_TOKEN":"unused"}):
            identity = self.cli.preflight()
            self.assertEqual(identity["auth_method"], "chatgpt")
            self.assertEqual(identity["cli_version"], "0.171.0-alpha.13")
            for instructions, history in ((USER, []), (ASSISTANT, [{"text":"賑やかだね。"}])):
                request = self.request(history)
                reply, receipt = self.cli.generate(request, self.image)
                seen = json.loads(self.seen.read_text())
                cmd = seen["args"]
                self.assertEqual(cmd[cmd.index("--model")+1], MODEL)
                self.assertIn('forced_login_method="chatgpt"', cmd)
                self.assertIn("--ephemeral", cmd)
                self.assertNotIn(str(self.image), cmd)
                self.assertFalse(seen["has_key"])
                self.assertEqual(bytes.fromhex(seen["image"]), self.image.read_bytes())
                self.assertEqual(seen["role"], instructions)
                self.assertEqual(seen["context"], json.loads(request["input"][0]["content"][0]["text"]))
                self.assertFalse(Path(seen["workdir"]).exists())
                self.assertEqual("tasks" in reply, instructions == ASSISTANT)
                self.assertEqual(receipt["model"], MODEL)
                self.assertIsNone(receipt["reported_model"])
                self.assertEqual(receipt["usage"], {"input_tokens":4,"output_tokens":2})
                self.assertNotIn("静か", json.dumps(receipt, ensure_ascii=False))

    def test_source_changed_between_preview_and_submission_does_not_invoke_model(self):
        request = self.request()
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(CodexInvocationError, "image does not match"):
            self.cli.generate(request, self.image)
        self.assertFalse(self.seen.exists())

    def test_key_only_login_does_not_invoke_model(self):
        self.mode.write_text("api_key")
        with self.assertRaisesRegex(CodexInvocationError, "ChatGPT CLI login"):
            self.cli.generate(self.request(), self.image)
        self.assertFalse(self.seen.exists())

    def test_explicit_limit_and_ambiguous_failures_are_distinct(self):
        for mode in ("quota", "partial_then_quota", "temporary", "unknown", "invalid_events", "extra", "wrong_model", "tool"):
            with self.subTest(mode=mode):
                self.mode.write_text(mode)
                with self.assertRaises(CodexInvocationError) as error:
                    self.cli.generate(self.request(), self.image)
                self.assertEqual(isinstance(error.exception, CodexUsageLimit), mode in ("quota", "temporary"))
                self.assertEqual(isinstance(error.exception, CodexRateLimit), mode == "temporary")
                if mode == "temporary":
                    self.assertEqual(error.exception.retry_after, 12)
                if mode == "unknown":
                    self.assertIn("connection interrupted", str(error.exception))

    def test_transient_reconnect_with_successful_completion_is_accepted(self):
        self.mode.write_text("reconnect")
        reply, _ = self.cli.generate(self.request(), self.image)
        self.assertIsInstance(reply["text"], str)

    def test_native_https_recovery_still_requires_a_valid_final_utterance(self):
        self.mode.write_text("native_fallback")
        reply, receipt = self.cli.generate(self.request(), self.image)
        self.assertEqual(receipt["recovered_transport_notices"], 1)
        self.assertTrue(reply["text"])
        for mode in ("native_fallback_failure", "native_fallback_bad_json"):
            with self.subTest(mode=mode):
                self.mode.write_text(mode)
                with self.assertRaises(CodexInvocationError):
                    self.cli.generate(self.request(), self.image)

    def test_real_cli_disabled_code_mode_advisory_before_turn_is_informational(self):
        message = ("Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; "
                   "enable `features.code_mode_host` and install `codex-code-mode-host`.")
        advisory = {"type": "item.completed", "item": {"type": "error", "message": message}}
        start = {"type": "turn.started"}
        user = {"type": "item.completed", "item": {"type": "user_message"}}
        reply = {"type": "item.completed", "item": {"type": "agent_message"}}
        end = {"type": "turn.completed"}

        def events(*items):
            return _events("\n".join(json.dumps(item) for item in items).encode())

        self.assertEqual(events(advisory, start, user, reply, end), ([], [end], None, True, 0))
        notice = {"type": "item.completed", "item": {"type": "error", "message":
                  "Falling back from WebSockets to HTTPS transport. unexpected status 403 Forbidden"}}
        reconnect = {"type": "item.completed", "item": {"type": "error", "message": "Reconnecting... 1/5"}}
        self.assertEqual(events(start, reconnect, notice, reply, end), ([], [end], None, True, 2))
        self.assertTrue(events(start, notice)[0])
        self.assertTrue(events(start, notice, {"type": "turn.failed"}, end)[0])
        self.assertTrue(events(start, reply, end, notice)[0])
        self.assertTrue(events(start, advisory, end)[0])
        self.assertTrue(events({"type": "item.completed", "item": {"type": "error", "message": "model unavailable"}}, start, end)[0])
        with self.assertRaisesRegex(CodexInvocationError, "command_execution"):
            events(advisory, start, {"type": "item.started", "item": {"type": "command_execution"}})


class SupervisedFailureTests(unittest.TestCase):
    def test_quota_before_and_after_output_and_transient_errors(self):
        from ssvd.dialogue.codex_transport import _failure, CodexIncompleteAtLimit, CodexTransientError
        for started, kind in ((False, CodexUsageLimit), (True, CodexIncompleteAtLimit)):
            self.assertIs(type(_failure(["weekly limit reached"], b"", 1, started)), kind)
        self.assertIsInstance(_failure(["403 forbidden"], b"", 1, False), CodexTransientError)
        self.assertNotIsInstance(_failure(["401 unauthorized"], b"", 1, False), CodexTransientError)
