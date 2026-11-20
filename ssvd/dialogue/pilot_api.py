"""Synchronous, standard-tier Responses calls for the Luna/Terra comparison."""

import hashlib
import json
import re
from http.client import HTTPException
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

from ssvd.dialogue.regeneration_roles import UTTERANCE_OUTPUT_LIMIT

MODELS = ("gpt-6-luna", "gpt-6.1-terra")
_ENDPOINT = "https://api.openai.com/v1/responses"
_COUNT_FIELDS = ("model", "input", "instructions", "text", "reasoning")
# Above this threshold both models change pricing for the *entire* request.
_SHORT_CONTEXT_LIMIT = 272_000


class _Rates(NamedTuple):
    ordinary: int
    cached: int
    written: int
    output: int


# Nano-USD per token, Standard global API, short context, default 30-min cache TTL.
# https://developers.openai.com/api/docs/models/gpt-6-luna
# https://developers.openai.com/api/docs/models/gpt-6.1-terra
_RATES = {
    "gpt-6-luna": _Rates(100, 10, 125, 500),
    "gpt-6.1-terra": _Rates(1000, 50, 1250, 5000),
}


class PilotRequestRejected(RuntimeError):
    """A definite HTTP rejection before generation. Safe to release this reservation."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _tokens(value):
    if type(value) is not int or value < 0:
        raise RuntimeError("Invalid OpenAI token count")
    return value


def _fingerprint(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()


def _reply(response):
    """Preserve the usage receipt even for refusals, truncated or malformed outputs."""
    if response.get("status") != "completed" or not isinstance(response.get("output"), list):
        return {}
    texts = []
    for item in response["output"]:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            return {}
        for part in item["content"]:
            if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                return {}
            texts.append(part["text"])
    try:
        reply = json.loads("".join(texts))
    except (ValueError, RecursionError):
        return {}
    return reply if isinstance(reply, dict) else {}


class OpenAIPilotAPI:
    def __init__(self, api_key: str):
        if not api_key or not api_key.isascii() or any(ord(c) <= 32 or ord(c) >= 127 for c in api_key):
            raise ValueError("A valid OPENAI_API_KEY is required")
        self._api_key = api_key
        self._opener = build_opener(_NoRedirect())
        self._quoted = None

    def prepare(self, body: dict) -> dict:
        """Accept only the one-message requests emitted by regeneration_roles."""
        model = body.get("model")
        if model not in MODELS:
            raise ValueError("Unsupported pilot model")
        messages = body.get("input")
        if (not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict)
                or messages[0].get("role") != "user"):
            raise ValueError("Expected one user message")
        parts = messages[0].get("content")
        if not isinstance(parts, list) or len(parts) != 2 or not all(isinstance(p, dict) for p in parts):
            raise ValueError("Expected text and one source image for every pilot step")
        if parts[0].get("type") != "input_text" or not isinstance(parts[0].get("text"), str):
            raise ValueError("Invalid pilot text")
        content = [{"type": "input_text", "text": parts[0]["text"]}]
        image = parts[1]
        if (image.get("type") != "input_image" or not isinstance(image.get("image_url"), str)
                or not image["image_url"].startswith("data:image/") or image.get("detail") != "high"):
            raise ValueError("Expected a high-detail inline image")
        content.append({"type": "input_image", "image_url": image["image_url"], "detail": "high"})
        if not isinstance(body.get("instructions"), str):
            raise ValueError("Expected role instructions")
        limit = body.get("max_output_tokens")
        if type(limit) is not int or limit != UTTERANCE_OUTPUT_LIMIT:
            raise ValueError("Expected the pilot output limit for this step")
        # Construct a fresh, allowlisted request. In particular no tools, cache markers,
        # previous responses, regional endpoints or project-selected premium tier.
        return {
            "model": model, "input": [{"role": "user", "content": content}],
            "instructions": body["instructions"], "text": {"format": {"type": "json_object"}},
            "reasoning": {"effort": "low"}, "store": False, "service_tier": "default",
            "prompt_cache_options": {"mode": "explicit"},
            "max_output_tokens": limit,
        }

    def _post(self, url, body, *, request_id=None):
        headers = {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"}
        if request_id is not None:
            if str(UUID(request_id)) != request_id:
                raise ValueError("Expected a canonical request UUID")
            headers["X-Client-Request-Id"] = request_id
        request = Request(url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                          headers=headers)
        try:
            with self._opener.open(request, timeout=120) as response:
                result = json.load(response)
        except HTTPError as error:
            status = error.code
            error.close()
            # Only unconditional authentication, route, size, format or quota rejections.
            # Generic 400, timeout, conflict and server errors retain the reservation.
            if status in (401, 403, 404, 413, 415, 422, 429):
                raise PilotRequestRejected(f"OpenAI rejected the request (HTTP {status})") from None
            raise RuntimeError(f"OpenAI request outcome unknown (HTTP {status})") from None
        except (URLError, OSError, HTTPException, ValueError, RecursionError):
            raise RuntimeError("OpenAI transport or response decoding failed; outcome unknown") from None
        if not isinstance(result, dict):
            raise RuntimeError("Invalid OpenAI response envelope")
        return result

    def reserve_usd(self, body: dict) -> float:
        """Quote all input, including actual image tokens, plus the full output cap.

        The official counter takes input, text and reasoning options. The output
        limit includes both visible and reasoning tokens. The count endpoint is free.
        https://developers.openai.com/api/docs/guides/token-counting
        https://help.openai.com/en/articles/4936856-understanding-and-counting-tokens
        """
        self._quoted = None
        body = self.prepare(body)
        count = self._post(_ENDPOINT + "/input_tokens", {key: body[key] for key in _COUNT_FIELDS})
        if count.get("object") != "response.input_tokens":
            raise RuntimeError("Invalid OpenAI token-count response")
        input_tokens = _tokens(count.get("input_tokens"))
        if not 0 < input_tokens <= _SHORT_CONTEXT_LIMIT:
            raise ValueError("Request outside the pilot's short-context pricing")
        rates = _RATES[body["model"]]
        upper = input_tokens * rates.written + body["max_output_tokens"] * rates.output
        self._quoted = _fingerprint(body), input_tokens
        return upper / 1_000_000_000

    def generate(self, body: dict, *, request_id: str | None = None) -> tuple[dict, dict]:
        """Send a quoted request once. Preserve the reservation on unknown outcomes."""
        body = self.prepare(body)
        if self._quoted is None or self._quoted[0] != _fingerprint(body):
            raise ValueError("First obtain a new reservation for this request")
        _, counted_input = self._quoted
        self._quoted = None
        response = self._post(_ENDPOINT, body, request_id=request_id)
        response_id = response.get("id")
        model = body["model"]
        if (not isinstance(response_id, str) or not response_id.startswith("resp_")
                or not re.fullmatch(re.escape(model) + r"(-\d{4}-\d{2}-\d{2})?", response.get("model") or "")
                or response.get("service_tier", "default") != "default"
                or response.get("status") not in ("completed", "incomplete", "failed", "cancelled")):
            raise RuntimeError("Unexpected OpenAI response or non-final status; verify billing")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("OpenAI returned no usage; verify billing")
        input_tokens = _tokens(usage.get("input_tokens"))
        output_tokens = _tokens(usage.get("output_tokens"))
        details = usage.get("input_tokens_details") or {}
        if not isinstance(details, dict):
            raise RuntimeError("Invalid OpenAI cache usage")
        # Explicit mode with no breakpoints normally has zero cache reads/writes.
        # https://developers.openai.com/api/docs/guides/prompt-caching
        cached = _tokens(details.get("cached_tokens", 0))
        written = _tokens(details.get("cache_write_tokens", 0))
        ordinary = input_tokens - cached - written
        if ordinary < 0 or input_tokens > counted_input or output_tokens > body["max_output_tokens"]:
            raise RuntimeError("OpenAI usage outside reserved pricing; verify billing")
        rates = _RATES[model]
        cost = ordinary * rates.ordinary + cached * rates.cached + written * rates.written + output_tokens * rates.output
        return _reply(response), {
            "response_id": response_id, "model": response["model"], "status": response.get("status"),
            "usage": usage, "cost_usd": cost / 1_000_000_000,
        }
