"""Versioned audio-dialogue protocol identifiers for image evaluation.

``vision_v1`` is intentionally kept on the historical bare-question path.
The assistant-prelude protocol injects audio without teacher-forcing the
synchronized assistant text stream and is therefore diagnostic-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BARE_QUESTION_PROTOCOL = "bare_question_v1"
ASSISTANT_PRELUDE_PROTOCOL = "assistant_prelude_audio_only_v2"
AUDIO_DIALOGUE_PROTOCOLS = (
    BARE_QUESTION_PROTOCOL,
    ASSISTANT_PRELUDE_PROTOCOL,
)


def audio_protocol_contract_from_lock(lock_path: Path) -> dict[str, str | None]:
    """Read the optional audio protocol fields without changing frozen suite code.

    Older locks, including ``vision_v1``, predate an explicit field and map to
    the historical bare-question protocol. Paper suite v2 locks reject the
    assistant-prelude variant; these optional fields remain readable for
    diagnostic/runtime metadata only.
    """

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    scope = lock.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("suite lock scope must be an object")
    protocol = str(scope.get("audio_dialogue_protocol", BARE_QUESTION_PROTOCOL))
    assistant = scope.get("assistant_prelude")
    digest = None
    if assistant is not None:
        if not isinstance(assistant, dict):
            raise ValueError("scope.assistant_prelude must be an object")
        value = assistant.get("wav_sha256")
        digest = None if value is None else str(value)
    return {
        "frozen_audio_dialogue_protocol": protocol,
        "frozen_assistant_prelude_wav_sha256": digest,
    }


def suite_freezes_audio_protocol(
    suite_contract: dict[str, Any],
    *,
    protocol: str,
    assistant_prelude_sha256: str | None,
) -> bool:
    """Return whether a suite lock fully pins the requested audio protocol."""

    if suite_contract.get("frozen_audio_dialogue_protocol") != protocol:
        return False
    if protocol == BARE_QUESTION_PROTOCOL:
        return assistant_prelude_sha256 is None
    return bool(assistant_prelude_sha256) and (
        suite_contract.get("frozen_assistant_prelude_wav_sha256")
        == assistant_prelude_sha256
    )
