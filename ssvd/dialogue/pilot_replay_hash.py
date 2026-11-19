"""Verify historical request bytes when a stored observation lost its JSON key order."""

import copy
import hashlib
from itertools import permutations
import json
import re

from ssvd.dialogue.regeneration_roles import _ASSISTANT


def request_digest(body):
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def matches_saved_request(body, digest):
    """Only try alternative key orders of a single active task (at most 6! possibilities).

    The original pilot saved decoded replies with sorted keys but interpolated the
    in-memory model order into this one literal string of a later assistant prompt.
    Any changed value, utterance, task sequence, or outer request must still fail.
    """
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return False
    if request_digest(body) == digest:
        return True
    if body.get("instructions") != _ASSISTANT:
        return False
    messages = body.get("input")
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        return False
    content = messages[0].get("content")
    if (messages[0].get("role") != "user" or not isinstance(content, list) or len(content) != 1
            or not isinstance(content[0], dict) or content[0].get("type") != "input_text"):
        return False
    text = content[0].get("text")
    lines = text.split("\n") if isinstance(text, str) else []
    if len(lines) != 3 or lines[2] != "今回の発話を JSON で返してください。":
        return False
    try:
        context = json.loads(lines[1])
        active = context["画像の観察情報"]["この返答の根拠"]
    except (ValueError, TypeError, KeyError):
        return False
    if not isinstance(active, dict) or json.dumps(context, ensure_ascii=False) != lines[1]:
        return False
    if active.get("kind") not in ("counting", "ocr"):
        return False
    allowed = {"kind", "mode", "target", "reason"}
    allowed.update({"count"} if active["kind"] == "counting" else {"written", "reading"})
    if not {"kind", "mode", "target"} <= active.keys() <= allowed:
        return False

    candidate = copy.deepcopy(body)
    prompt = candidate["input"][0]["content"][0]
    for order in permutations(active):
        context["画像の観察情報"]["この返答の根拠"] = {k: active[k] for k in order}
        lines[1] = json.dumps(context, ensure_ascii=False)
        prompt["text"] = "\n".join(lines)
        if request_digest(candidate) == digest:
            return True
    return False
