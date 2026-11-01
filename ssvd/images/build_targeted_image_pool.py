#!/usr/bin/env python3
"""Build image-only v4 candidate pools for counting and Japanese OCR.

This command deliberately does *not* create questions, answers, dialogues, or
audio.  Existing source annotations are retained only as reference evidence
for a later Sonnet/local-model dialogue generation stage.  Dialogue generator,
turn count, and train/eval split all remain undecided in the emitted records.

Only newly collected sources are accepted:

* counting: Visual Genome images referenced by TallyQA train (COCO is skipped)
* OCR: JSSODa train only

Every image is decoded and fingerprinted.  Exact and perceptual overlaps with
the frozen v3 image index are routed to a review manifest rather than the clean
candidate pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "j-moshivis"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from jmoshivis.evaluation.overlap.fingerprints import (  # noqa: E402
    fingerprint_provenance,
    hamming,
    image_fingerprints,
)


SCHEMA_VERSION = "jmoshivis.targeted-image-pool.v1"
VG_IMAGE_DATA_URL = (
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
    "image_data.json.zip"
)
VG_IMAGE_DATA_ZIP_SHA256 = "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a"
VG_IMAGE_DATA_JSON_SHA256 = "93d0053d7e8f451497646fa513e8fb9b1a889df6e757010b88e51e2e21803487"
TALLYQA_REVISION = "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f"


class HammingMultiIndex:
    """Exact candidate index for a bounded Hamming-distance search.

    A 64-bit value is divided into ``threshold + 1`` disjoint blocks. Any
    value within ``threshold`` bit flips must share at least one complete block
    (pigeonhole principle), so the bucket union has no false negatives. Exact
    Hamming distance is then evaluated only for that much smaller candidate
    set.
    """

    def __init__(self, threshold: int) -> None:
        if not 0 <= threshold < 64:
            raise ValueError("Hamming threshold must be between 0 and 63")
        self.threshold = threshold
        chunks = threshold + 1
        base, remainder = divmod(64, chunks)
        widths = [base + (index < remainder) for index in range(chunks)]
        offset = 64
        self.parts: list[tuple[int, int]] = []
        for width in widths:
            offset -= width
            self.parts.append((offset, (1 << width) - 1))
        self.buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self.hash_by_uid: dict[str, str] = {}
        self.item_by_uid: dict[str, dict[str, str]] = {}

    def add(self, uid: str, value: str, item: dict[str, str]) -> None:
        self.hash_by_uid[uid] = value
        self.item_by_uid[uid] = item
        integer = int(value, 16)
        for index, (offset, mask) in enumerate(self.parts):
            self.buckets[(index, (integer >> offset) & mask)].append(uid)

    def candidate_uids(self, value: str) -> set[str]:
        integer = int(value, 16)
        result: set[str] = set()
        for index, (offset, mask) in enumerate(self.parts):
            result.update(self.buckets.get((index, (integer >> offset) & mask), ()))
        return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", buffering=8 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_manual_overlap_decisions(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        raise ValueError(f"invalid manual overlap decision file: {path}")
    result: dict[str, dict[str, Any]] = {}
    for decision in value["decisions"]:
        if not isinstance(decision, dict) or not decision.get("candidate_uid"):
            raise ValueError(f"invalid manual overlap decision entry: {decision!r}")
        uid = str(decision["candidate_uid"])
        if uid in result:
            raise ValueError(f"duplicate manual overlap decision: {uid}")
        result[uid] = decision
    return result


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    # The manifests are project-owned and already pinned by content hashes.
    # Avoid resolving every row through the shared filesystem: realpath() on
    # tens of thousands of paths is needlessly slow and does not change these
    # non-symlink inputs.
    return Path(os.path.abspath(path))


def stored_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(absolute)


def deferred_dialogue_state() -> dict[str, Any]:
    return {
        "status": "deferred_until_image_collection_is_frozen",
        "generator": None,
        "turn_count": None,
        "allowed_generator_families": ["claude_sonnet_5", "local_vlm"],
        "contains_generated_dialogue": False,
    }


def load_vg_image_data(path: Path) -> dict[int, dict[str, Any]]:
    archive_hash = sha256_file(path)
    if archive_hash != VG_IMAGE_DATA_ZIP_SHA256:
        raise ValueError(
            f"unexpected Visual Genome image_data archive hash: {archive_hash}"
        )
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "image_data.json"]
        if len(names) != 1:
            raise ValueError(f"expected one image_data.json in {path}, found {names}")
        payload = archive.read(names[0])
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != VG_IMAGE_DATA_JSON_SHA256:
        raise ValueError(f"unexpected Visual Genome image_data JSON hash: {payload_hash}")
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise ValueError("Visual Genome image_data.json must contain a list")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in result:
            raise ValueError(f"duplicate Visual Genome image id: {image_id}")
        result[image_id] = row
    return result


def load_counting_candidates(
    path: Path, vg_image_data: Mapping[int, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aggregate evidence by original photo, excluding every COCO-origin image."""
    groups: dict[str, dict[str, Any]] = {}
    visual_genome_image_ids: set[int] = set()
    coco_origin_image_ids: set[int] = set()
    for row in read_jsonl(path):
        image = row.get("image")
        if not isinstance(image, dict) or image.get("source_dataset") != "Visual Genome":
            continue
        image_id = int(image["source_image_id"])
        visual_genome_image_ids.add(image_id)
        metadata = vg_image_data.get(image_id)
        if metadata is None:
            raise ValueError(f"Visual Genome image id missing from image_data.json: {image_id}")
        if metadata.get("coco_id") is not None:
            coco_origin_image_ids.add(image_id)
            continue
        flickr_id = metadata.get("flickr_id")
        source_photo_uid = (
            f"flickr:{int(flickr_id)}"
            if flickr_id is not None
            else f"visual_genome:{image_id}"
        )
        group = groups.setdefault(
            source_photo_uid,
            {"variants": {}, "reference_annotations": []},
        )
        group["variants"][image_id] = {
            "path": stored_path(project_path(str(image["path"]))),
            "declared_file_sha256": str(image["sha256"]),
            "source_split": image.get("source_split"),
            "base_image_uid": row.get("base_image_uid"),
            "license": image.get("license"),
            "official_metadata": {
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "url": metadata.get("url"),
                "coco_id": None,
                "flickr_id": flickr_id,
            },
        }
        group["reference_annotations"].append(
            {
                "source_question_id": row.get("source", {}).get("question_id"),
                "question_en": row.get("question_en"),
                "answer_count": row.get("answer_count"),
                "data_source": row.get("source", {}).get("data_source"),
                "is_simple": row.get("question_complexity", {}).get("is_simple"),
                "tallyqa_revision": TALLYQA_REVISION,
            }
        )

    images: list[dict[str, Any]] = []
    for source_photo_uid, group in groups.items():
        canonical_id = min(group["variants"])
        canonical = group["variants"][canonical_id]
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "candidate_uid": f"counting:{source_photo_uid}",
            "capability": "counting",
            "path": canonical["path"],
            "source_photo_uid": source_photo_uid,
            "split_group_key": source_photo_uid,
            "source": {
                "dataset": "Visual Genome",
                "source_image_id": str(canonical_id),
                "merged_source_image_ids": [str(value) for value in sorted(group["variants"])],
                "source_split": canonical["source_split"],
                "base_image_uid": canonical["base_image_uid"],
                "license": canonical["license"],
                "official_image_metadata": canonical["official_metadata"],
                "official_image_metadata_url": VG_IMAGE_DATA_URL,
                "annotation_dataset": "TallyQA",
                "annotation_split": "train",
            },
            "declared_file_sha256": canonical["declared_file_sha256"],
            "reference_annotations": group["reference_annotations"],
            "reference_annotation_role": (
                "source evidence for later dialogue generation; not dialogue text"
            ),
            "split": None,
            "training_ready": False,
            "dialogue_generation": deferred_dialogue_state(),
        }
        candidate["reference_annotations"].sort(
            key=lambda item: (item.get("source_question_id") is None, item.get("source_question_id"))
        )
        images.append(candidate)
    stats = {
        "tallyqa_visual_genome_images": len(visual_genome_image_ids),
        "official_coco_origin_images_excluded": len(coco_origin_image_ids),
        "non_coco_visual_genome_image_ids": (
            len(visual_genome_image_ids) - len(coco_origin_image_ids)
        ),
        "source_photo_groups": len(images),
        "same_flickr_photo_variants_merged": sum(
            max(0, len(group["variants"]) - 1) for group in groups.values()
        ),
    }
    return sorted(images, key=lambda row: row["candidate_uid"]), stats


def _ocr_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    if "ground_truth" in row:
        text = row["ground_truth"]
    elif "text" in row:
        text = row["text"]
    else:
        raise ValueError(f"OCR row lacks exact transcription: {row.get('uid')}")
    return {
        "kind": "exact_transcription",
        "text": text,
        "orientation": row.get("orientation"),
        "num_columns": row.get("num_columns"),
    }


def load_ocr_candidates(named_paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for priority, (name, path) in enumerate(named_paths):
        for row in read_jsonl(path):
            image_path = project_path(str(row["path"]))
            declared_hash = row.get("image_sha256")
            split_group_key = f"{name}:image:{row['uid']}"
            candidate = {
                "schema_version": SCHEMA_VERSION,
                "candidate_uid": f"ocr:{name}:{row['uid']}",
                "capability": "japanese_ocr",
                "path": stored_path(image_path),
                "source": {
                    "dataset": row.get("source", name),
                    "source_id": row.get("source_id"),
                    "source_revision": row.get("source_revision"),
                    "source_url": row.get("source_url"),
                    "license": row.get("license"),
                    "license_url": row.get("license_url"),
                    "source_manifest": stored_path(path),
                },
                "declared_file_sha256": declared_hash,
                "split_group_key": split_group_key,
                "reference_annotations": [_ocr_reference(row)],
                "reference_annotation_role": (
                    "exact OCR ground truth for validation and later dialogue generation; "
                    "not dialogue text"
                ),
                "mixture_priority": "auxiliary_synthetic_long_page",
                "source_order": priority,
                "split": None,
                "training_ready": False,
                "dialogue_generation": deferred_dialogue_state(),
            }
            candidates.append(candidate)
    return sorted(candidates, key=lambda row: (row["source_order"], row["candidate_uid"]))


def fingerprint_worker(path_value: str) -> tuple[str, dict[str, Any]]:
    path = project_path(path_value)
    try:
        return path_value, image_fingerprints(path)
    except Exception as exc:  # Keep the complete audit instead of losing all progress.
        return path_value, {"error": f"{type(exc).__name__}: {exc}"}


def fingerprint_candidates(
    candidates: list[dict[str, Any]], *, workers: int, cache_path: Path
) -> dict[str, dict[str, Any]]:
    paths = sorted({str(row["path"]) for row in candidates})
    result: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        for row in read_jsonl(cache_path):
            cached_path = row.get("path")
            cached_fingerprint = row.get("fingerprints")
            if isinstance(cached_path, str) and isinstance(cached_fingerprint, dict):
                result[cached_path] = cached_fingerprint
    pending = [path for path in paths if path not in result]
    if result:
        print(f"loaded {len(result):,} cached fingerprints", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (path, fingerprint) in enumerate(
            executor.map(fingerprint_worker, pending, chunksize=32), 1
        ):
            result[path] = fingerprint
            if index % 2_000 == 0 or index == len(pending):
                print(f"fingerprinted {index:,}/{len(pending):,} pending images", flush=True)
    atomic_write_jsonl(
        cache_path,
        ({"path": path, "fingerprints": result[path]} for path in paths),
    )
    return result


def load_v3_index(path: Path, *, review_threshold: int) -> dict[str, Any]:
    file_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    pixel_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    phash_index = HammingMultiIndex(review_threshold)
    dhash_index = HammingMultiIndex(review_threshold)
    rows = 0
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        rows += 1
        item = {"uid": str(row["uid"]), "path": str(row["path"])}
        file_map[str(row["file_sha256"])].append(item)
        pixel_map[str(row["pixel_sha256"])].append(item)
        phash_value = str(row["phash"])
        dhash_value = str(row["dhash"])
        phash_index.add(item["uid"], phash_value, item)
        dhash_index.add(item["uid"], dhash_value, item)
    return {
        "file_map": file_map,
        "pixel_map": pixel_map,
        "phash_index": phash_index,
        "dhash_index": dhash_index,
        "rows": rows,
    }


def overlap_result(
    fp: Mapping[str, Any],
    v3: Mapping[str, Any],
    *,
    near_threshold: int,
    review_threshold: int,
) -> dict[str, Any]:
    if fp.get("error"):
        return {"status": "fingerprint_error", "error": fp["error"], "matches": []}
    exact: list[dict[str, Any]] = []
    for kind in ("file_sha256", "pixel_sha256"):
        lookup = v3[f"{kind.removesuffix('_sha256')}_map"]
        for item in lookup.get(str(fp[kind]), ())[:5]:
            exact.append({"kind": kind, **item})
    if exact:
        return {"status": "excluded_exact_v3_overlap", "matches": exact}

    phash_value = str(fp["phash"])
    dhash_value = str(fp["dhash"])
    phash_index: HammingMultiIndex = v3["phash_index"]
    dhash_index: HammingMultiIndex = v3["dhash_index"]
    candidate_uids = (
        phash_index.candidate_uids(phash_value)
        & dhash_index.candidate_uids(dhash_value)
    )
    perceptual = [
        {
            "uid": uid,
            "path": phash_index.item_by_uid[uid]["path"],
            "phash_distance": hamming(phash_value, phash_index.hash_by_uid[uid]),
            "dhash_distance": hamming(dhash_value, dhash_index.hash_by_uid[uid]),
        }
        for uid in candidate_uids
    ]
    perceptual = [
        item
        for item in perceptual
        if item["phash_distance"] <= review_threshold
        and item["dhash_distance"] <= review_threshold
    ]
    perceptual.sort(
        key=lambda item: (
            max(item["phash_distance"], item["dhash_distance"]),
            item["phash_distance"] + item["dhash_distance"],
        )
    )
    strong = [
        item
        for item in perceptual
        if min(item["phash_distance"], item["dhash_distance"]) <= near_threshold
    ]
    if strong:
        return {"status": "excluded_near_v3_overlap", "matches": strong[:5]}
    if perceptual:
        return {"status": "review_near_v3_overlap", "matches": perceptual[:5]}
    return {"status": "accepted_new_image", "matches": []}


def verify_declared_hash(candidate: Mapping[str, Any], fp: Mapping[str, Any]) -> None:
    declared = candidate.get("declared_file_sha256")
    if declared and not fp.get("error") and declared != fp.get("file_sha256"):
        raise ValueError(
            f"source image changed for {candidate['candidate_uid']}: "
            f"declared {declared}, observed {fp.get('file_sha256')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tallyqa-manifest",
        type=Path,
        default=REPO_ROOT / "data/images/counting/tallyqa/metadata_train.jsonl",
    )
    parser.add_argument(
        "--vg-image-data-zip",
        type=Path,
        default=REPO_ROOT / "data/images/counting/visual_genome/image_data.json.zip",
    )
    parser.add_argument(
        "--v3-image-index",
        type=Path,
        default=REPO_ROOT / "data/evaluation/v3_image_index.jsonl",
    )
    parser.add_argument(
        "--jssoda-train-manifest",
        type=Path,
        default=REPO_ROOT / "data/images/ocr/jssoda/metadata_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/images/targeted_v4/image_pool_v1",
    )
    parser.add_argument(
        "--manual-overlap-decisions",
        type=Path,
        default=REPO_ROOT / "data/images/targeted_v4/manual_overlap_decisions.json",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--review-threshold", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [
        args.tallyqa_manifest,
        args.vg_image_data_zip,
        args.v3_image_index,
        args.jssoda_train_manifest,
        args.manual_overlap_decisions,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 <= args.near_threshold <= args.review_threshold:
        raise ValueError("expected 0 <= near-threshold <= review-threshold")
    manual_overlap_decisions = load_manual_overlap_decisions(
        args.manual_overlap_decisions
    )

    print("loading image-only candidates", flush=True)
    vg_image_data = load_vg_image_data(args.vg_image_data_zip)
    counting, counting_selection = load_counting_candidates(
        args.tallyqa_manifest, vg_image_data
    )
    ocr = load_ocr_candidates(
        [
            ("jssoda_train", args.jssoda_train_manifest),
        ]
    )
    candidates = counting + ocr
    print(
        f"counting={len(counting):,}, OCR={len(ocr):,}, total={len(candidates):,}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = fingerprint_candidates(
        candidates,
        workers=args.workers,
        cache_path=args.output_dir / "fingerprints.jsonl",
    )
    print("loading frozen v3 image index", flush=True)
    v3 = load_v3_index(args.v3_image_index, review_threshold=args.review_threshold)

    accepted: dict[str, list[dict[str, Any]]] = {"counting": [], "japanese_ocr": []}
    review: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    accepted_source_counts: Counter[str] = Counter()
    seen_new_pixels: dict[str, str] = {}
    applied_manual_decisions: set[str] = set()
    for index, candidate in enumerate(candidates, 1):
        fp = fingerprints[str(candidate["path"])]
        verify_declared_hash(candidate, fp)
        audit = overlap_result(
            fp,
            v3,
            near_threshold=args.near_threshold,
            review_threshold=args.review_threshold,
        )
        manual_decision = manual_overlap_decisions.get(str(candidate["candidate_uid"]))
        if manual_decision is not None:
            applied_manual_decisions.add(str(candidate["candidate_uid"]))
            required_status = manual_decision.get("automatic_status_required")
            if audit["status"] != required_status:
                raise ValueError(
                    f"manual decision for {candidate['candidate_uid']} expected "
                    f"{required_status}, observed {audit['status']}"
                )
            if manual_decision.get("decision") != "accept_false_positive":
                raise ValueError(
                    f"unsupported manual decision for {candidate['candidate_uid']}: "
                    f"{manual_decision.get('decision')}"
                )
            matched_uids = {str(item.get("uid")) for item in audit.get("matches", [])}
            expected_match = str(manual_decision.get("compared_v3_uid"))
            if expected_match not in matched_uids:
                raise ValueError(
                    f"manual decision match changed for {candidate['candidate_uid']}: "
                    f"expected {expected_match}, observed {sorted(matched_uids)}"
                )
            audit = {
                "status": "accepted_manual_false_positive",
                "automatic_audit": audit,
                "manual_decision": manual_decision,
            }
        accepted_status = audit["status"] in {
            "accepted_new_image",
            "accepted_manual_false_positive",
        }
        if (
            candidate["capability"] == "counting"
            and not fp.get("error")
            and accepted_status
        ):
            width, height = int(fp["width"]), int(fp["height"])
            aspect_ratio = max(width / height, height / width)
            if min(width, height) < 300 or aspect_ratio > 2.0:
                audit = {
                    "status": "excluded_counting_image_quality",
                    "matches": [],
                    "quality": {
                        "width": width,
                        "height": height,
                        "short_side_minimum": 300,
                        "observed_aspect_ratio": aspect_ratio,
                        "maximum_aspect_ratio": 2.0,
                    },
                }
                accepted_status = False
        if not fp.get("error") and accepted_status:
            prior = seen_new_pixels.get(str(fp["pixel_sha256"]))
            if prior is not None:
                audit = {
                    "status": "excluded_duplicate_within_new_pool",
                    "matches": [{"candidate_uid": prior}],
                }
                accepted_status = False
            else:
                seen_new_pixels[str(fp["pixel_sha256"])] = str(candidate["candidate_uid"])
        enriched = dict(candidate)
        enriched.pop("source_order", None)
        enriched["fingerprints"] = fp
        enriched["v3_overlap_audit"] = {
            **audit,
            "near_threshold": args.near_threshold,
            "review_threshold": args.review_threshold,
        }
        status_counts[audit["status"]] += 1
        source_name = str(candidate["source"]["dataset"])
        source_counts[source_name] += 1
        if accepted_status:
            accepted[str(candidate["capability"])].append(enriched)
            accepted_source_counts[source_name] += 1
        else:
            review.append(enriched)
        if index % 10_000 == 0 or index == len(candidates):
            print(f"audited {index:,}/{len(candidates):,} candidates", flush=True)

    unapplied_decisions = set(manual_overlap_decisions) - applied_manual_decisions
    if unapplied_decisions:
        raise ValueError(
            f"manual overlap decisions did not match candidates: {sorted(unapplied_decisions)}"
        )

    counting_output = args.output_dir / "counting_images.jsonl"
    ocr_output = args.output_dir / "ocr_images.jsonl"
    review_output = args.output_dir / "review_excluded_images.jsonl"
    atomic_write_jsonl(counting_output, accepted["counting"])
    atomic_write_jsonl(ocr_output, accepted["japanese_ocr"])
    atomic_write_jsonl(review_output, review)

    output_hashes = {
        path.name: sha256_file(path)
        for path in (counting_output, ocr_output, review_output)
    }
    accepted_counting_facts = [
        fact
        for image in accepted["counting"]
        for fact in image["reference_annotations"]
    ]
    accepted_counting_data_sources = Counter(
        str(fact.get("data_source")) for fact in accepted_counting_facts
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "image_collection_only",
        "generated_dialogues": 0,
        "turn_count": None,
        "train_eval_split_assigned": False,
        "candidate_policy": {
            "counting": (
                "newly downloaded Visual Genome photos only; official coco_id origins "
                "excluded, Flickr-origin duplicates merged, short side >=300 px, aspect <=2.0"
            ),
            "ocr": (
                "JSSODa train pages only; retained as auxiliary long-document OCR. "
                "A separate modern short scene-text bank is still pending"
            ),
            "dialogue_generation": "deferred to Claude Sonnet 5 or a local VLM",
            "v3_exact_overlap": "excluded",
            "v3_perceptual_overlap": "excluded or routed to manual review",
            "manual_false_positive_review": stored_path(
                args.manual_overlap_decisions
            ),
        },
        "input_candidates": len(candidates),
        "input_counting_images": len(counting),
        "input_ocr_images": len(ocr),
        "accepted_counting_images": len(accepted["counting"]),
        "accepted_counting_reference_facts": len(accepted_counting_facts),
        "accepted_counting_data_source_facts": dict(
            sorted(accepted_counting_data_sources.items())
        ),
        "accepted_counting_images_with_answer_ge_2": sum(
            any(
                isinstance(fact.get("answer_count"), int)
                and fact["answer_count"] >= 2
                for fact in image["reference_annotations"]
            )
            for image in accepted["counting"]
        ),
        "accepted_ocr_images": len(accepted["japanese_ocr"]),
        "counting_source_selection": counting_selection,
        "review_or_excluded_images": len(review),
        "status_counts": dict(sorted(status_counts.items())),
        "input_source_counts": dict(sorted(source_counts.items())),
        "accepted_source_counts": dict(sorted(accepted_source_counts.items())),
        "v3_indexed_images": v3["rows"],
        "overlap_thresholds": {
            "near": args.near_threshold,
            "review": args.review_threshold,
        },
        "fingerprint_provenance": fingerprint_provenance(),
        "inputs": [
            {"path": stored_path(path), "sha256": sha256_file(path)} for path in inputs
        ],
        "outputs": output_hashes,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
