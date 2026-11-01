"""Versioned extraction of answer text from frame-aligned model output."""

from __future__ import annotations

from typing import Any


SPECIAL_TEXT_IDS = {0, 1, 2, 3, 4, 5, 6}
EXTRACTION_VERSION = "question-end-overlap-until-eot.v1"


def extract_answer_events(
    events: list[dict[str, Any]],
    *,
    question_end_frame: int,
    overlap_frames: int,
    special_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select the first response after the question while preserving raw data.

    A small overlap window admits natural full-duplex turn overlap.  Special
    SentencePiece/Moshi padding IDs are ignored; after content starts, EOS (2)
    closes the answer.  ID 0 is *not* an utterance terminator: the training
    interleaver inserts it between sparse text tokens, so stopping at 0 would
    truncate most answers after their first piece.  Token ID 9 is a normal
    SentencePiece whitespace marker and must remain.
    """

    special = SPECIAL_TEXT_IDS if special_ids is None else special_ids
    start_frame = max(0, question_end_frame - overlap_frames)
    content: list[dict[str, Any]] = []
    started = False
    for event in events:
        if int(event["frame"]) < start_frame:
            continue
        token_id = int(event["token_id"])
        if token_id == 2 and started:
            break
        if token_id in special:
            continue
        started = True
        content.append(event)
    return content
