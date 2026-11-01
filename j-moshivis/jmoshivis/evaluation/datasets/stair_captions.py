"""Build a Japanese caption evaluation manifest from official STAIR annotations.

Only local COCO images absent from the supplied v3 manifests are selected.
The separate overlap auditor must still check file, pixel, and perceptual
fingerprints before the subset is called clean.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from ..overlap.audit import coco_uid
from ..schema import EvalExample, read_jsonl, sha256_file, write_manifest


SOURCE_URL = "https://github.com/STAIR-Lab-CIT/STAIR-captions"
SOURCE_COMMIT = "6ac656e65d80cbe2c914d93626a1d5454e35a5d6"
ARCHIVE_MEMBER = "stair_captions_v1.2_val.json"
CAPTION_QUESTION = "この画像の内容を一文で説明してください。"


def build_manifest(
    archive: Path,
    coco_root: Path,
    train_manifests: list[Path],
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation manifest: {output}")
    archive_hash = sha256_file(archive)
    with tarfile.open(archive, "r:gz") as package:
        member = package.extractfile(ARCHIVE_MEMBER)
        if member is None:
            raise ValueError(f"missing {ARCHIVE_MEMBER} in {archive}")
        source = json.load(member)

    train_uids = {
        str(row["uid"])
        for manifest in train_manifests
        for _, row in read_jsonl(manifest)
        if row.get("uid")
    }
    captions: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for annotation in source["annotations"]:
        value = str(annotation["caption"]).strip()
        if value:
            captions[int(annotation["image_id"])].append((int(annotation["id"]), value))
    licenses = {int(item["id"]): item for item in source["licenses"]}
    stats: Counter[str] = Counter()
    examples: list[EvalExample] = []
    for image in sorted(source["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        references = sorted(captions[image_id])
        if len(references) != 5:
            stats["not_five_references"] += 1
            continue
        if coco_uid(image_id) in train_uids:
            stats["training_coco_id"] += 1
            continue
        name = f"{image_id:012d}.jpg"
        paths = [coco_root / split / name for split in ("val2017", "train2017")]
        path = next((item for item in paths if item.is_file()), None)
        if path is None:
            stats["image_not_local"] += 1
            continue
        license_info = licenses.get(int(image["license"]), {})
        examples.append(
            EvalExample(
                sample_id=f"stair-v1.2-val-{image_id:012d}",
                benchmark="stair-captions-ja-val2014-overlap-clean-candidate",
                split="validation",
                domain="general_world",
                task="caption",
                image_path=os.path.relpath(path.resolve(), output.resolve().parent),
                question=CAPTION_QUESTION,
                prompt_text=CAPTION_QUESTION,
                answers=[caption for _, caption in references],
                question_type="caption",
                source={
                    "dataset": "STAIR Captions v1.2",
                    "url": SOURCE_URL,
                    "revision": SOURCE_COMMIT,
                    "annotation_archive_sha256": archive_hash,
                    "annotation_member": ARCHIVE_MEMBER,
                    "coco_id": image_id,
                    "image_sha256": sha256_file(path),
                    "image_license": license_info,
                    "annotation_license": "CC BY 4.0",
                },
                metadata={
                    "moshivis_metric": "cider_ja",
                    "caption_language": "ja",
                    "reference_annotation_ids": [id for id, _ in references],
                    "selection": "source COCO ID excluded; fingerprint audit pending",
                },
            )
        )
        stats["selected_source_id_clean_candidates"] += 1
    if not examples:
        raise ValueError("no STAIR val images remain after source-ID exclusion")
    count, digest = write_manifest(output, examples)
    summary = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": archive_hash,
        "source_split": "STAIR v1.2 val (COCO val2014)",
        "source_images": len(source["images"]),
        "source_annotations": len(source["annotations"]),
        "counts": dict(stats),
        "manifest": str(output.resolve()),
        "manifest_sha256": digest,
        "manifest_questions": count,
        "overlap_status": "candidate_only; run overlap.audit with v3 image index",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.archive, args.coco_root, args.train_manifest, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
