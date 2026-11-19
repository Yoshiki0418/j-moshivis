"""Prepare one regeneration step from a pending Japanese Daily image, without sending it."""

import argparse
import base64
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue.regeneration_roles import next_step
from ssvd.review_store import (
    DEFAULT_REVIEW_EVENTS, REVIEW_REQUIRED_STATUS, ReviewStore,
    content_fingerprint, review_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = "japanese_daily_v1"
PROMPT_VERSION = "grounded_roles_v1"


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


def preview(root, uid, *, model="gpt-6-luna", pairs=None, transcript=None):
    """Replay decoded replies bound to the same image, review revision and settings."""
    if pairs is not None and (type(pairs) is not int or pairs not in (4, 5, 6)):
        raise ValueError("Japanese Daily candidates use 4 to 6 question-answer pairs")
    source, kind, image, base = _load_source(root, uid)
    identity = dict(collection=COLLECTION, uid=uid, base_fingerprint=base,
                    image_sha256=source["image_sha256"], kind=kind, model=model,
                    pairs=pairs, prompt_version=PROMPT_VERSION)
    replies = []
    if transcript is not None:
        if not isinstance(transcript, dict) or transcript.get("identity") != identity:
            raise ValueError("Transcript belongs to a different source or configuration")
        replies = transcript.get("replies")
        if not isinstance(replies, list):
            raise ValueError("Expected a list of decoded JSON replies")
    step = next_step(uid, kind, image, replies, model=model, pairs=pairs)
    if step["status"] != "pending_review":
        return {"identity": identity, **step}

    question_index = 2 * (step["task_pair"] - 1)
    evidence = step["grounding"]
    auxiliary = dict(kind=kind, grounding_source="vlm_visual",
                     question_turn_index=question_index, answer_turn_index=question_index + 1)
    if kind == "ocr":
        auxiliary.update(recognized_text=evidence["written"], spoken_reading=evidence["reading"])
    else:
        auxiliary.update(counted_target=evidence["target"], answer_count=evidence["count"])
    speakers = {"user": "ユーザー", "assistant": "アシスタント"}
    candidate = dict(uid=uid, image=source["image"], image_sha256=identity["image_sha256"],
                     task_mode="normal", generator=model, prompt_version=PROMPT_VERSION,
                     dialogue=[{"speaker": speakers[t["role"]], "text": t["text"]} for t in step["dialogue"]],
                     auxiliary_task=auxiliary, verification_status=REVIEW_REQUIRED_STATUS)
    if "split_group_key" in source:
        candidate["split_group_key"] = source["split_group_key"]
    candidate.update(review_metadata(candidate))
    return {"identity": identity, "status": "pending_review", "candidate": candidate,
            "grounding": evidence, "task_pair": step["task_pair"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--pairs", type=int, choices=(4, 5, 6))
    parser.add_argument("--transcript", type=Path, help="JSON with identity and ordered decoded replies")
    args = parser.parse_args()
    try:
        transcript = json.loads(args.transcript.read_text(encoding="utf-8")) if args.transcript else None
        result = preview(ROOT, args.uid, model=args.model, pairs=args.pairs, transcript=transcript)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
