# 構築済みディレクトリを走査し、学習用 JSONL (train/eval) を作成する
#   {"path": .../stereo_dialogue.wav, "duration": sec, "text": 全発話連結, "image": .../image.jpg}
# フィルタ: 必要ファイルが揃い、asr_check の mean_cer が閾値以下のもの
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dataset_keys import is_eval_group, split_group_key  # noqa: E402

SSVD_DIR = Path(__file__).resolve().parents[1]
if str(SSVD_DIR) not in sys.path:
    sys.path.insert(0, str(SSVD_DIR))

from review_store import (  # noqa: E402
    DEFAULT_REVIEW_EVENTS,
    ReviewStore,
    content_fingerprint,
    image_file_sha256,
    requires_review,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _review_record_from_speech_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact content fingerprinted before speech synthesis."""

    speaker_names = {"USER": "ユーザー", "ASSISTANT": "アシスタント"}
    dialogue = []
    for segment in meta.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        speaker = speaker_names.get(segment.get("speaker"))
        text = segment.get("text")
        if speaker is not None and isinstance(text, str):
            dialogue.append({"speaker": speaker, "text": text})
    return {
        "uid": meta.get("source_uid", meta.get("uid")),
        "image": meta.get("image"),
        "image_sha256": meta.get("image_sha256"),
        "task_mode": meta.get("source_task_mode", meta.get("task_mode", "normal")),
        "dialogue": dialogue,
        "auxiliary_task": meta.get("auxiliary_task"),
        "verification_status": meta.get("verification_status"),
        "generator": meta.get("generator"),
    }


def _review_gate(
    meta: Mapping[str, Any],
    latest_decisions: Mapping[str, Mapping[str, Any]],
    decision_uids: set[str],
    actual_image_asset_sha256: str | None = None,
    latest_revisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Return eligibility, exclusion reason, and immutable approval evidence."""

    required = meta.get("review_required") is True or requires_review(meta)
    review_id = meta.get("review_id")
    revision = meta.get("human_revision") or {}
    base = revision.get("base_fingerprint") or meta.get("source_fingerprint") or review_id
    current_revision = (latest_revisions or {}).get(base)
    source_uid = str(meta.get("source_uid", meta.get("uid", "")))
    if not review_id and source_uid in decision_uids:
        # Pre-existing ordinary audio may lack fingerprints. It must not win
        # over a correction for the same source/task, or an explicit rejection.
        candidates = [event for event in (latest_revisions or {}).values()
            if str(event.get("uid")) == source_uid
            and str(event.get("revised_record", {}).get("task_mode") or "normal") == str(meta.get("task_mode") or "normal")]
        if candidates:
            return False, "review_superseded", None
        for event in reversed(list(latest_decisions.values())):
            if (str(event.get("uid")) == source_uid
                    and str(event.get("task_mode") or "normal") == str(meta.get("task_mode") or "normal")):
                if event.get("decision") == "rejected":
                    return False, "review_rejected", None
                break
    if current_revision and current_revision.get("fingerprint") != review_id:
        return False, "review_superseded", None
    event = latest_decisions.get(review_id)
    if not required and not revision:
        if event and event.get("decision") == "rejected":
            return False, "review_rejected", None
        # Optional approval never turns an ordinary sample into mandatory.
        evidence = dict(event) if event and event.get("decision") == "approved" else None
        if evidence and content_fingerprint(_review_record_from_speech_meta(meta)) != review_id:
            evidence = None  # An optional label still must refer to these exact texts.
        if evidence:
            evidence.pop("revised_record", None)
            evidence["status"] = "approved"
        return True, None, evidence
    if (required and meta.get("review_required") is not True) or not isinstance(review_id, str):
        return False, "review_pending", None
    reconstructed = _review_record_from_speech_meta(meta)
    if content_fingerprint(reconstructed) != review_id:
        return False, "review_stale", None
    expected_image_asset_sha256 = meta.get("image_asset_sha256")
    if (
        not isinstance(expected_image_asset_sha256, str)
        or len(expected_image_asset_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_image_asset_sha256
        )
        or not isinstance(actual_image_asset_sha256, str)
    ):
        return False, "review_asset_missing", None
    if actual_image_asset_sha256 != expected_image_asset_sha256:
        return False, "review_asset_stale", None
    event = latest_decisions.get(review_id)
    if event is None:
        source_uid = str(meta.get("source_uid", meta.get("uid", "")))
        reason = "review_stale" if source_uid in decision_uids else "review_pending"
        return False, reason, None
    decision = event.get("decision") or event.get("status")
    if decision == "rejected":
        return False, "review_rejected", None
    if decision != "approved" and required:
        return False, "review_pending", None
    if decision != "approved":
        return True, None, None
    evidence = dict(event)
    evidence.pop("revised_record", None)  # Keep provenance, not a duplicate full dialogue.
    evidence["status"] = "approved"
    evidence["review_id"] = review_id
    return True, None, evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/speech/data_stereo")
    ap.add_argument("--out_train", default="data/speech/train.jsonl")
    ap.add_argument("--out_eval", default="data/speech/eval.jsonl")
    ap.add_argument("--max_cer", type=float, default=0.3)
    ap.add_argument("--eval_ratio", type=float, default=0.02)
    ap.add_argument(
        "--split_seed",
        default="jmoshivis-group-split-v1",
        help="Stable seed used for source-group train/eval assignment",
    )
    ap.add_argument("--require_asr", action="store_true", help="asr_check.json が無いものを除外")
    ap.add_argument(
        "--review-ledger",
        default=str(REPO_ROOT / DEFAULT_REVIEW_EVENTS),
        help="append-only human review event JSONL (exact review_id approval required)",
    )
    args = ap.parse_args()

    review_store = ReviewStore(args.review_ledger)
    latest_decisions = review_store.latest_decisions()
    latest_revisions = review_store.latest_revisions()
    decision_uids = {
        str(event.get("uid"))
        for event in latest_decisions.values()
        if event.get("uid") is not None
    }
    rows, stats = [], {
        "ok": 0,
        "missing": 0,
        "high_cer": 0,
        "no_asr": 0,
        "review_approved": 0,
        "review_pending": 0,
        "review_rejected": 0,
        "review_stale": 0,
        "review_superseded": 0,
        "review_asset_missing": 0,
        "review_asset_stale": 0,
    }
    for d in sorted(Path(args.datadir).iterdir()):
        wav, align, meta, img = d / "stereo_dialogue.wav", d / "assistant_dialogue.refined.json", d / "dialogue.json", d / "image.jpg"
        if not (wav.exists() and align.exists() and meta.exists() and img.exists()):
            stats["missing"] += 1; continue
        asr = d / "asr_check.json"
        if asr.exists():
            if json.loads(asr.read_text(encoding="utf-8"))["mean_cer"] > args.max_cer:
                stats["high_cer"] += 1; continue
        elif args.require_asr:
            stats["no_asr"] += 1; continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        actual_image_asset_sha256 = None
        if m.get("review_required") is True or requires_review(m) or m.get("human_revision"):
            try:
                actual_image_asset_sha256 = image_file_sha256(img)
            except OSError:
                # The review gate records a fail-closed exclusion below.
                pass
        review_ok, review_reason, human_review = _review_gate(
            m,
            latest_decisions,
            decision_uids,
            actual_image_asset_sha256,
            latest_revisions,
        )
        if not review_ok:
            stats[review_reason] += 1
            continue
        if human_review is not None:
            stats["review_approved"] += 1
        group_key = split_group_key(m)
        fixed_split = m.get("split")
        split = fixed_split if fixed_split in {"train", "eval"} else (
            "eval" if is_eval_group(group_key, args.eval_ratio, args.split_seed) else "train"
        )
        row = {"path": str(wav.resolve()), "duration": round(sf.info(wav).duration, 3),
               "text": " ".join(s["text"] for s in m["segments"]), "image": str(img.resolve()),
               "uid": m["uid"], "source_uid": m.get("source_uid", m["uid"]),
               "task_mode": m.get("task_mode", "normal"),
               "normal_strategy": m.get("normal_strategy"),
               "human_revision": m.get("human_revision"),
               "auxiliary_task": m.get("auxiliary_task"),
               "verification_status": m.get("verification_status"),
               "review_required": bool(m.get("review_required", False)),
               "human_certified": human_review is not None,
               "review_id": m.get("review_id"),
               "image_sha256": m.get("image_sha256"),
               "image_asset_sha256": m.get("image_asset_sha256"),
               "split_group_key": group_key, "grounding": m.get("grounding"),
               "split": split, "generator": m.get("generator"),
               "user_voice": m.get("user_voice")}
        if human_review is not None:
            row["human_review"] = human_review
        rows.append(row)
        stats["ok"] += 1
    rows.sort(key=lambda row: (row["split_group_key"], row["uid"]))
    for path, part in [
        (args.out_train, [row for row in rows if row["split"] == "train"]),
        (args.out_eval, [row for row in rows if row["split"] == "eval"]),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(part)} rows ({sum(r['duration'] for r in part)/3600:.1f} h)")
    print(stats)


if __name__ == "__main__":
    main()
