"""Freeze a paired Japanese Daily pilot from pending, image-verified Qwen dialogues."""

from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue.preview_regeneration import COLLECTION, _load_source
from ssvd.dialogue.regeneration_roles import next_step
from ssvd.images.collect_japanese_daily_v1 import KnownImageIndex, image_fingerprints
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore
from jmoshivis.evaluation.schema import read_jsonl, sha256_file


SEED = "japanese-daily-model-pilot-issue12-v1"
# These are sampling hints from the old record, never instructions for the generator.
QUOTAS = {
    "ocr": {"latin_hint": 4, "difficult_text_hint": 3, "mixed_scene_hint": 2, "general": 5},
    "counting": {"overlap_hint": 2, "mixed_scene_hint": 2, "general": 2},
}
TEXT_DIFFICULTY = re.compile(
    r"(?:小さ|細か|かすれ|ぼやけ|遠く|遠目).{0,8}(?:字|文|表記|看板)|"
    r"(?:文字|字|表記|看板).{0,8}(?:小さ|細か|かすれ|ぼやけ|読みにく|読みづら|読み取れ)"
)
OVERLAP = re.compile(r"重な|重なり|隠れ|かくれ|手前.{0,12}奥")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]{2}")


def _checked_file(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Frozen evaluation file differs from its lock: {path}")


def _fixed_evaluation(root):
    """Bind the held-out v3 subset to its locked index and add external eval pixels."""
    locks = root / "j-moshivis/evaluation"
    vision = json.loads((locks / "suite_v1.lock.json").read_text())
    stair = json.loads((locks / "stair_caption_ja_val_clean_v1.lock.json").read_text())
    reference = vision["training_overlap_reference"]
    eval_path = root / "data/speech_v3/eval.jsonl"
    fp_path = root / "data/evaluation/v3_image_index.jsonl"
    _checked_file(eval_path, reference["v3_eval_manifest_sha256"])
    _checked_file(fp_path, reference["fingerprint_index_sha256"])

    heldout = {row["uid"] for _, row in read_jsonl(eval_path)}
    known = KnownImageIndex(near_threshold=4, review_threshold=8)
    indexed = set()
    for _, row in read_jsonl(fp_path):
        uid = row["uid"]
        if uid in heldout:
            if row.get("error") or not all(row.get(k) for k in ("file_sha256", "pixel_sha256", "phash", "dhash")):
                raise ValueError("Incomplete fingerprint for held-out v3 image")
            known.add(uid, row, row["path"])
            indexed.add(uid)
    if not heldout or indexed != heldout:
        raise ValueError("Missing held-out v3 images in the frozen index")

    metadata = {
        "v3_eval": {"images": len(heldout), "manifest_sha256": reference["v3_eval_manifest_sha256"],
                    "index_sha256": reference["fingerprint_index_sha256"]},
    }
    for name, base, spec in (("vision_v1_raw", root, vision["manifests"]["raw"]),
                             ("stair_caption", locks, stair["manifest"])):
        manifest = (base / spec["path"]).resolve()
        _checked_file(manifest, spec["sha256"])
        images = {}
        for _, row in read_jsonl(manifest):
            path = (manifest.parent / row["image_path"]).resolve()
            digest = row["source"]["image_sha256"]
            if images.setdefault(path, digest) != digest:
                raise ValueError(f"Inconsistent image digest in {name}")
        if not images:
            raise ValueError(f"Empty evaluation image inventory: {name}")
        for path, digest in images.items():
            fp = image_fingerprints(path)
            if fp["file_sha256"] != digest:
                raise ValueError(f"Evaluation image differs from its manifest: {path}")
            known.add(f"{name}:{digest}", fp, str(path))
        metadata[name] = {"images": len(images), "manifest_sha256": spec["sha256"]}
    return heldout, known, metadata


def _candidates(root, touched):
    with closing(sqlite3.connect((root / "data/.dataset_viewer/catalog.sqlite3").as_uri()
                                 + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT i.uid, i.source_uid, i.auxiliary_kind, i.summary_json, i.dialogue_offset,
                   i.dialogue_length, s.path AS source_path
            FROM items i JOIN sources s ON i.dialogue_source = s.source_id
            WHERE i.collection = ? AND i.review_required = 1 AND i.review_status = 'pending'
              AND i.has_image AND i.has_dialogue AND i.auxiliary_kind IN ('counting','ocr')
        """, (COLLECTION,))
        result = []
        for row in rows:
            if (COLLECTION, row["uid"]) in touched:
                continue
            summary = json.loads(row["summary_json"])
            if (summary.get("image_integrity_status") == "verified" and not summary.get("human_revision")
                    and "qwen" in str(summary.get("generator", "")).lower()):
                result.append(row)
    return sorted(result, key=lambda r: hashlib.sha256(f"{SEED}:{r['uid']}".encode()).digest())


def _sampling_hints(root, row):
    with (root / row["source_path"]).open("rb") as handle:
        handle.seek(row["dialogue_offset"])
        source = json.loads(handle.read(row["dialogue_length"]))
    if source["uid"] != row["uid"]:
        raise ValueError("Indexed offset no longer identifies the source dialogue")
    turns = " ".join(turn["text"] for turn in source["dialogue"])
    task = source["auxiliary_task"]
    buckets = set(json.loads(row["summary_json"]).get("collection_buckets", []))
    hints = []
    if row["auxiliary_kind"] == "ocr":
        if LATIN.search(str(task.get("recognized_text", ""))):
            hints.append("latin_hint")
        if TEXT_DIFFICULTY.search(turns):
            hints.append("difficult_text_hint")
        if "count_friendly_everyday" in buckets:
            hints.append("mixed_scene_hint")
    else:
        if OVERLAP.search(turns):
            hints.append("overlap_hint")
        if "japanese_text_rich" in buckets:
            hints.append("mixed_scene_hint")
    return [*hints, "general"]


def prepare_manifest(root: Path) -> dict:
    root = root.resolve()
    events = root / DEFAULT_REVIEW_EVENTS
    if not events.is_file():
        raise FileNotFoundError(f"Missing human decision history: {events}")
    store = ReviewStore(events)
    heldout, known, reference = _fixed_evaluation(root)
    rows = _candidates(root, human_touched_uids(store))
    quota = {kind: counts.copy() for kind, counts in QUOTAS.items()}
    samples = []
    skipped = Counter()
    for row in rows:
        uid, kind = row["uid"], row["auxiliary_kind"]
        needed = quota[kind]
        if not any(needed.values()):
            continue
        if uid in heldout or row["source_uid"] in heldout:
            skipped["heldout_identity"] += 1
            continue
        hints = _sampling_hints(root, row)
        stratum = next((hint for hint in hints if needed[hint]), None)
        if stratum is None:
            continue
        source, current_kind, image, base = _load_source(root, uid)
        if current_kind != kind:
            raise ValueError("Task changed during pilot selection; prepare a fresh manifest")
        if source.get("split_group_key") in heldout:
            skipped["heldout_identity"] += 1
            continue
        image_path = (root / source["image"]).resolve()
        relative = image_path.relative_to(root)
        fp = image_fingerprints(image_path)
        if fp["file_sha256"] != source["image_sha256"]:
            raise ValueError("Candidate image changed during pilot selection")
        overlap = known.match(fp)["status"]
        if overlap != "accepted":
            skipped[overlap] += 1
            continue
        known.add(uid, fp, str(relative))  # Detect duplicates within the pilot as well.
        pairs = next_step(uid, kind, image, [])["pair_count"]
        samples.append(dict(uid=uid, kind=kind, base_fingerprint=base,
                            image_sha256=fp["file_sha256"], image_path=str(relative),
                            pairs=pairs, sampling_hint=stratum))
        needed[stratum] -= 1
        if all(not any(counts.values()) for counts in quota.values()):
            break
    if any(any(counts.values()) for counts in quota.values()):
        raise ValueError(f"Insufficient eligible pilot candidates for remaining quotas: {quota}")
    touched = human_touched_uids(store)
    if any((COLLECTION, sample["uid"]) in touched for sample in samples):
        raise ValueError("Human review changed during pilot selection; prepare a fresh manifest")
    return {
        "schema_version": 1, "collection": COLLECTION, "seed": SEED, "samples": samples,
        "candidate_pool_before_overlap": dict(Counter(row["auxiliary_kind"] for row in rows)),
        "fixed_evaluation": reference,
        "overlap_exclusions_encountered": dict(skipped),
        "overlap_thresholds": {"near": 4, "review": 8},
    }
