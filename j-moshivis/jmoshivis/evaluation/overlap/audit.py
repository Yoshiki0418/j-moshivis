"""Audit benchmark images against v3 and emit an immutable clean manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..schema import EvalExample, read_jsonl, read_manifest, sha256_file, write_manifest
from .build_index import collect_train_index_inputs, verify_image_index_metadata
from .fingerprints import fingerprint_provenance, hamming, image_fingerprints


def coco_uid(coco_id: int) -> str:
    stem = f"{int(coco_id):012d}"
    return hashlib.sha256(f"coco_{stem}".encode("utf-8")).hexdigest()


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[str, dict[int, Any]] | None = None
        self._values: set[str] = set()

    def add(self, value: str) -> None:
        if value in self._values:
            return
        self._values.add(value)
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = hamming(value, node[0])
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def query(self, value: str, threshold: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        matches: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = hamming(value, node_value)
            if distance <= threshold:
                matches.append((distance, node_value))
            low, high = distance - threshold, distance + threshold
            stack.extend(child for edge, child in children.items() if low <= edge <= high)
        return sorted(matches)


def _load_train_uids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for _, row in read_jsonl(path):
            if row.get("uid"):
                result.add(str(row["uid"]))
    return result


def _load_index(path: Path | None) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    BKTree,
    BKTree,
    set[str],
    set[str],
]:
    file_map: dict[str, list[dict[str, str]]] = {}
    pixel_map: dict[str, list[dict[str, str]]] = {}
    dhash_map: dict[str, list[dict[str, str]]] = {}
    phash_map: dict[str, list[dict[str, str]]] = {}
    phash_tree, dhash_tree = BKTree(), BKTree()
    successful_uids: set[str] = set()
    error_uids: set[str] = set()
    if path is None:
        return (
            file_map, pixel_map, dhash_map, phash_map, phash_tree, dhash_tree,
            successful_uids, error_uids,
        )
    for _, row in read_jsonl(path):
        if row.get("error"):
            error_uids.add(str(row.get("uid")))
            continue
        successful_uids.add(str(row["uid"]))
        item = {"uid": str(row["uid"]), "path": str(row["path"])}
        for key, target in (
            ("file_sha256", file_map), ("pixel_sha256", pixel_map),
            ("dhash", dhash_map), ("phash", phash_map),
        ):
            target.setdefault(str(row[key]), []).append(item)
        phash_tree.add(str(row["phash"]))
        dhash_tree.add(str(row["dhash"]))
    error_uids -= successful_uids
    return (
        file_map, pixel_map, dhash_map, phash_map, phash_tree, dhash_tree,
        successful_uids, error_uids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--image-index", type=Path)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--review-threshold", type=int, default=8)
    parser.add_argument("--exclude-near", action="store_true")
    parser.add_argument(
        "--allow-incomplete-index",
        action="store_true",
        help="Diagnostic only: continue with a partial/error index, never marking unmatched rows clean.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = read_manifest(args.manifest, check_files=True)
    train_uids = _load_train_uids(args.train_manifest)
    if args.image_index is not None:
        train_index_images, input_manifests = collect_train_index_inputs(args.train_manifest)
        verify_image_index_metadata(
            args.image_index,
            input_manifests,
            len(train_index_images),
        )
    print(f"eval questions={len(examples)}, train UIDs={len(train_uids)}")
    if args.dry_run:
        print(f"image index: {args.image_index or 'not supplied (source-ID audit only)'}")
        return 0
    (
        file_map, pixel_map, dhash_map, phash_map, phash_tree, dhash_tree,
        indexed_uids, index_error_uids,
    ) = _load_index(args.image_index)
    missing_index_uids = train_uids - indexed_uids if args.image_index else set()
    incomplete_index = bool(args.image_index and (missing_index_uids or index_error_uids))
    if incomplete_index and not args.allow_incomplete_index:
        raise RuntimeError(
            "image fingerprint index is incomplete: "
            f"missing={len(missing_index_uids)}, errors={len(index_error_uids)}. "
            "Resume build_index until complete; --allow-incomplete-index is diagnostic only."
        )

    # Fingerprint each unique evaluation image only once.
    fingerprints: dict[Path, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    clean: list[EvalExample] = []
    reason_counts: dict[str, int] = {}
    for example in examples:
        image_path = example.resolve_image(args.manifest).resolve()
        fp = fingerprints.setdefault(image_path, image_fingerprints(image_path))
        reasons: list[str] = []
        matches: dict[str, Any] = {}
        source_coco = example.source.get("coco_id")
        if source_coco is not None:
            candidate_uid = coco_uid(int(source_coco))
            if candidate_uid in train_uids:
                reasons.append("source_coco_uid")
                matches["source_uid"] = candidate_uid
        if fp["file_sha256"] in file_map:
            reasons.append("file_sha256")
            matches["file_sha256"] = file_map[str(fp["file_sha256"])][:20]
        if fp["pixel_sha256"] in pixel_map:
            reasons.append("pixel_sha256")
            matches["pixel_sha256"] = pixel_map[str(fp["pixel_sha256"])][:20]
        if args.image_index:
            p_near = phash_tree.query(str(fp["phash"]), args.review_threshold)
            d_near = dhash_tree.query(str(fp["dhash"]), args.review_threshold)
            p_by_uid = {
                item["uid"]: {"distance": distance, "hash": value, **item}
                for distance, value in p_near
                for item in phash_map[value]
            }
            d_by_uid = {
                item["uid"]: {"distance": distance, "hash": value, **item}
                for distance, value in d_near
                for item in dhash_map[value]
            }
            perceptual = [
                {
                    "uid": uid,
                    "path": p_by_uid[uid]["path"],
                    "phash_distance": p_by_uid[uid]["distance"],
                    "dhash_distance": d_by_uid[uid]["distance"],
                }
                for uid in p_by_uid.keys() & d_by_uid.keys()
            ]
            perceptual.sort(key=lambda row: (max(row["phash_distance"], row["dhash_distance"]),
                                              row["phash_distance"] + row["dhash_distance"]))
            strong = [
                row for row in perceptual
                if min(row["phash_distance"], row["dhash_distance"]) <= args.near_threshold
            ]
            if strong:
                reasons.append("perceptual_near_duplicate")
                matches["perceptual_candidates"] = strong[:20]
            elif perceptual:
                reasons.append("perceptual_review_candidate")
                matches["perceptual_candidates"] = perceptual[:20]

        exact_reasons = {"source_coco_uid", "file_sha256", "pixel_sha256"}
        excluded = bool(exact_reasons.intersection(reasons)) or (
            args.exclude_near and any(reason.startswith("perceptual_") for reason in reasons)
        )
        has_perceptual_reason = any(reason.startswith("perceptual_") for reason in reasons)
        if excluded:
            status = "excluded"
        elif has_perceptual_reason:
            status = "review_near"
        elif args.image_index is None:
            status = "source_only_not_fingerprint_audited"
        elif incomplete_index:
            status = "incomplete_index_not_fully_audited"
        else:
            status = "clean"
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        audit = {
            "sample_id": example.sample_id,
            "image_path": str(image_path),
            "status": status,
            "reasons": reasons,
            "matches": matches,
            "fingerprints": fp,
        }
        audits.append(audit)
        if not excluded:
            metadata = copy.deepcopy(example.metadata)
            metadata["overlap_status"] = status
            metadata["overlap_audit"] = {
                "status": status,
                "reasons": reasons,
                "near_threshold": args.near_threshold,
                "review_threshold": args.review_threshold,
            }
            output_dir = args.clean_manifest.resolve().parent
            audio = example.resolve_audio(args.manifest)
            clean.append(
                replace(
                    example,
                    image_path=os.path.relpath(image_path, output_dir),
                    audio_path=None if audio is None else os.path.relpath(audio.resolve(), output_dir),
                    metadata=metadata,
                )
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_count, digest = write_manifest(args.clean_manifest, clean)
    strict_clean_count = sum(row["status"] == "clean" for row in audits)
    summary = {
        "schema_version": "jmoshivis.overlap-audit.v1",
        "eval_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "train_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.train_manifest
        ],
        "image_index": None if args.image_index is None else {
            "path": str(args.image_index.resolve()),
            "sha256": sha256_file(args.image_index),
        },
        "fingerprint": fingerprint_provenance(),
        "input_questions": len(examples),
        "output_questions": output_count,
        "strict_clean_questions": strict_clean_count,
        "clean_questions": strict_clean_count,
        "excluded_questions": sum(row["status"] == "excluded" for row in audits),
        "review_near_questions": sum(row["status"] == "review_near" for row in audits),
        "unique_eval_images": len(fingerprints),
        "reason_counts": reason_counts,
        "near_threshold": args.near_threshold,
        "review_threshold": args.review_threshold,
        "near_duplicates_excluded": args.exclude_near,
        "training_uids": len(train_uids),
        "successfully_indexed_training_uids": len(train_uids & indexed_uids),
        "missing_index_uids": len(missing_index_uids),
        "index_error_uids": len(index_error_uids),
        "clean_manifest_sha256": digest,
    }
    args.report.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
