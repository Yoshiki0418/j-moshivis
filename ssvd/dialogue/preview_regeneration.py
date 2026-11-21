"""Prepare one regeneration step from a pending Japanese Daily image, without sending it."""

import argparse
import base64
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue.regeneration_roles import (
    MARKER_VERSIONS, PAIR_COUNTS, PROMPT_VERSION, VISUAL_V4, VISUAL_V5, VISUAL_V6, VISUAL_V7, VISUAL_V8,
    VISUAL_V9, VISUAL_V11, next_step,
)
from ssvd.dialogue import regeneration_roles_v3
from ssvd.review_store import (
    DEFAULT_REVIEW_EVENTS, REVIEW_REQUIRED_STATUS, ReviewStore,
    content_fingerprint, review_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = "japanese_daily_v1"
OBSERVATION_VERSION = "grounded_roles_v3"
PREVIEW_VERSIONS = (VISUAL_V11, PROMPT_VERSION, VISUAL_V9, VISUAL_V8, VISUAL_V7, VISUAL_V6, VISUAL_V5, VISUAL_V4,
                    OBSERVATION_VERSION)


def _load_source(root, uid):
    store = ReviewStore(root / DEFAULT_REVIEW_EVENTS)
    if (COLLECTION, uid) in human_touched_uids(store):
        raise ValueError("This image already has a human review decision")

    catalog = root / "data/.dataset_viewer/catalog.sqlite3"
    with closing(sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("""
            SELECT i.image_path, i.auxiliary_kind, i.review_id, i.dialogue_offset,
                   i.dialogue_length, s.path AS source_path
            FROM items AS i JOIN sources AS s ON s.source_id = i.dialogue_source
            WHERE i.collection = ? AND i.uid = ? AND i.review_required = 1
                AND i.review_status = 'pending' AND i.has_image AND i.has_dialogue
                AND i.auxiliary_kind IN ('counting', 'ocr')
            """, (COLLECTION, uid)).fetchone()
    if row is None:
        raise ValueError("No pending OCR/counting source for this image")
    with (root / row["source_path"]).open("rb") as stream:
        stream.seek(row["dialogue_offset"])
        source = json.loads(stream.read(row["dialogue_length"]))
    kind = row["auxiliary_kind"]
    image_path = root / row["image_path"]
    if (source["uid"] != uid or "qwen" not in source["generator"].lower()
            or source.get("review_required") is not True
            or source.get("verification_status") != REVIEW_REQUIRED_STATUS
            or source.get("auxiliary_task", {}).get("kind") != kind
            or (root / source["image"]).resolve() != image_path.resolve()):
        raise ValueError("The source and indexed task do not match")
    image_bytes = image_path.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest != source.get("image_sha256") or content_fingerprint(source, image_sha256=digest) != row["review_id"]:
        raise ValueError("The source image or dialogue has changed")
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(image_path.suffix.lower())
    if not mime:
        raise ValueError("Unsupported image format for preview")
    data_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
    return source, kind, data_url, row["review_id"]


def _review_task(task, pair, *, marker_only=False):
    """Attach the review label to its exchange; only older formats carry answer details."""
    kind = task["kind"]
    question_index = 2 * (pair - 1)
    result = dict(kind=kind, grounding_source="vlm_visual",
                  question_turn_index=question_index, answer_turn_index=question_index + 1)
    if marker_only:
        return result
    mode = task["mode"]
    result["answer_mode"] = mode
    if mode != "exact" and isinstance(task.get("reason"), str) and task["reason"].strip():
        result["reason"] = task["reason"]
    if kind == "counting":
        result["counted_target"] = task["target"]
        if mode != "unknown":
            field = {"exact": "answer_count", "approximate": "estimated_count",
                     "lower_bound": "minimum_count"}[mode]
            result[field] = task["count"]
    else:
        result["ocr_target"] = task["target"]
        if mode != "unknown":
            field = "recognized_text" if mode == "exact" else "visible_fragment"
            result[field] = task["written"]
            result["spoken_reading"] = task["reading"]
    return result


def preview(root, uid, *, model="gpt-6-luna", pairs=None, transcript=None, recovery=False,
            prompt_version=PROMPT_VERSION):
    """Replay decoded replies bound to the same image, review revision and settings."""
    if pairs is not None and (type(pairs) is not int or pairs not in PAIR_COUNTS):
        raise ValueError("Japanese Daily candidates use 4 to 8 question-answer pairs")
    if prompt_version not in PREVIEW_VERSIONS:
        raise ValueError("Unsupported prompt version")
    source, kind, image, base = _load_source(root, uid)
    identity = dict(collection=COLLECTION, uid=uid, base_fingerprint=base,
                    image_sha256=source["image_sha256"], kind=kind, model=model,
                    pairs=pairs, prompt_version=prompt_version)
    replies = []
    if transcript is not None:
        if not isinstance(transcript, dict) or transcript.get("identity") != identity:
            raise ValueError("Transcript belongs to a different source or configuration")
        replies = transcript.get("replies")
        if not isinstance(replies, list):
            raise ValueError("Expected a list of decoded JSON replies")
    if prompt_version == OBSERVATION_VERSION:
        step = regeneration_roles_v3.next_step(uid, kind, image, replies, model=model, pairs=pairs, recovery=recovery)
    else:
        step = next_step(uid, kind, image, replies, model=model, pairs=pairs, version=prompt_version)
    if step["status"] != "pending_review":
        return {"identity": identity, **step}

    if prompt_version == OBSERVATION_VERSION:
        evidence = step["grounding"]
        tasks = [_review_task(task, step["task_pairs"][task["kind"]]) for task in evidence["tasks"]]
    else:
        tasks = [_review_task(exchange["task"], exchange["pair"], marker_only=prompt_version in MARKER_VERSIONS)
                 for exchange in step["exchanges"]]
    auxiliary = tasks[0] if tasks else {"kind": "none", "grounding_source": "none"}
    # The review fingerprint covers this entire dictionary, including a second exchange.
    if len(tasks) > 1:
        auxiliary["additional_tasks"] = tasks[1:]
    speakers = {"user": "ユーザー", "assistant": "アシスタント"}
    candidate = dict(uid=uid, image=source["image"], image_sha256=identity["image_sha256"],
                     task_mode="normal", generator=model, prompt_version=prompt_version,
                     dialogue=[{"speaker": speakers[t["role"]], "text": t["text"]} for t in step["dialogue"]],
                     auxiliary_task=auxiliary, verification_status=REVIEW_REQUIRED_STATUS)
    if "review_warnings" in step:
        candidate["review_warnings"] = step["review_warnings"]
    if "split_group_key" in source:
        candidate["split_group_key"] = source["split_group_key"]
    candidate.update(review_metadata(candidate))
    return {"identity": identity, "status": "pending_review", "candidate": candidate,
            "pair_count": step["pair_count"],
            **{key: step[key] for key in ("grounding", "task_pairs") if key in step}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--pairs", type=int, choices=PAIR_COUNTS)
    parser.add_argument("--transcript", type=Path, help="JSON with identity and ordered decoded replies")
    parser.add_argument("--prompt-version", choices=PREVIEW_VERSIONS, default=PROMPT_VERSION)
    parser.add_argument("--recovery", action="store_true", help="Apply the follow-up pilot replay checks")
    args = parser.parse_args()
    try:
        transcript = json.loads(args.transcript.read_text(encoding="utf-8")) if args.transcript else None
        result = preview(ROOT, args.uid, model=args.model, pairs=args.pairs,
                         transcript=transcript, recovery=args.recovery, prompt_version=args.prompt_version)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
