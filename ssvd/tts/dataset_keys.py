"""Stable identifiers and source-group split helpers for speech datasets.

This module intentionally has no audio or ML dependencies. Dataset assembly
and its unit tests can therefore share the exact same keying logic without
loading a TTS engine, PyTorch, SciPy, or soundfile.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def speech_sample_uid(record: Mapping[str, Any]) -> str:
    """Return the storage ID for one source image and dialogue task.

    Normal-dialogue IDs retain their legacy value. Task-specific variants use
    a suffix so counting/OCR audio cannot overwrite the normal dialogue (or one
    another) for the same source image.
    """

    source_uid = str(record["uid"])
    task_mode = str(record.get("task_mode", "normal"))
    storage_uid = source_uid if task_mode == "normal" else f"{source_uid}__task_{task_mode}"
    revision = record.get("human_revision")
    if isinstance(revision, Mapping) and revision.get("review_id"):
        storage_uid += f"__revision_{revision['review_id'][:16]}"
    return storage_uid


def split_group_key(metadata: Mapping[str, Any]) -> str:
    """Return the source-level key shared by every task variant of an image."""

    return str(
        metadata.get("split_group_key")
        or metadata.get("source_uid")
        or metadata["uid"]
    )


def is_eval_group(key: str, ratio: float, seed: str) -> bool:
    """Assign a complete source group to eval deterministically."""

    if not 0.0 <= ratio < 1.0:
        raise ValueError("eval_ratio must satisfy 0 <= ratio < 1")
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < ratio
