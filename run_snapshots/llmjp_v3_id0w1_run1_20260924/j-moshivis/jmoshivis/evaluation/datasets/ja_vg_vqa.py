"""Prepare JA-VG-VQA-500 as a general-world Japanese VQA manifest.

The source has 500 image records and several QA pairs per image.  ``official``
selects the first QA exactly as SakanaAI's published evaluator does;
``balanced`` chooses one category-balanced QA per image for analysis; ``full``
flattens every QA.  COCO-linked records are kept in the raw manifest by
default and must be removed by ``audit_overlap``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..schema import EvalExample, sha256_file, write_manifest


DATASET_ID = "SakanaAI/JA-VG-VQA-500"
DEFAULT_REVISION = "aa13b70cd939faa1783187a752be2c0ca0d8303b"
SOURCE_URL = "https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500"


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def question_category(question: str) -> str:
    compact = question.replace(" ", "")
    if any(token in compact for token in ("何色", "どんな色", "色は")):
        return "color"
    if any(token in compact for token in ("いくつ", "何個", "何人", "何匹", "何台", "何本", "何枚")):
        return "count"
    if any(token in compact for token in ("どこ", "どちら", "場所")):
        return "where"
    if "誰" in compact:
        return "who"
    if any(token in compact for token in ("なぜ", "どうして")):
        return "why"
    if "いつ" in compact:
        return "when"
    if any(token in compact for token in ("どのよう", "どんな", "どうやって", "どうしてい")):
        return "how"
    if compact.startswith(("これは", "この", "彼", "彼女", "人は", "動物は")) and compact.endswith(("か?", "か？")):
        return "yes_no_or_attribute"
    return "what_other"


def _stable_key(seed: int, *values: Any) -> str:
    joined = ":".join(str(value) for value in values)
    return hashlib.sha256(f"{seed}:{joined}".encode("utf-8")).hexdigest()


def select_core(rows: list[dict[str, Any]], seed: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose one QA per image while greedily balancing question categories."""

    counts: dict[str, int] = {}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: _stable_key(seed, item["image_id"])):
        qas = [qa for qa in row.get("qas", []) if qa.get("question") and qa.get("answer")]
        if not qas:
            continue
        qas.sort(
            key=lambda qa: (
                counts.get(question_category(str(qa["question"])), 0),
                _stable_key(seed, row["image_id"], qa.get("qa_id", qa["question"])),
            )
        )
        chosen = qas[0]
        category = question_category(str(chosen["question"]))
        counts[category] = counts.get(category, 0) + 1
        selected.append((row, chosen))
    return selected


def _encoded_image_bytes(image: Any) -> bytes:
    """Return the source-encoded image without a decode/re-encode round trip."""

    if isinstance(image, dict) and image.get("bytes") is not None:
        return bytes(image["bytes"])
    if isinstance(image, dict) and image.get("path"):
        return Path(image["path"]).read_bytes()
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    raise TypeError(
        "JA-VG image must be loaded with datasets.Image(decode=False); "
        f"got {type(image).__name__}"
    )


def _save_image(image: Any, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _encoded_image_bytes(image)
    if destination.suffix.lower() in {".jpg", ".jpeg"} and not payload.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"source image is not JPEG but destination is {destination}")
    expected = hashlib.sha256(payload).hexdigest()
    if destination.exists() and sha256_file(destination) == expected:
        return expected
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--profile", choices=("official", "balanced", "full"), default="official")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--arrow-file",
        type=Path,
        help="Optional cached split .arrow file; avoids network/cache locks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-coco-linked", action="store_true")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            f"would load {args.arrow_file or DATASET_ID + '@' + args.revision} split={args.split}, "
            f"profile={args.profile}, exclude_coco_linked={args.exclude_coco_linked}"
        )
        return 0
    try:
        from datasets import Dataset, Image as DatasetImage, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The dataset adapter needs Hugging Face datasets. Run it with .venv-data/bin/python."
        ) from exc

    dataset = (
        Dataset.from_file(str(args.arrow_file))
        if args.arrow_file is not None
        else load_dataset(DATASET_ID, split=args.split, revision=args.revision)
    )
    # The default datasets.Image feature returns a PIL object.  Saving that
    # object would silently recompress every JPEG and weaken exact overlap
    # detection against the v3 training corpus.  Preserve the encoded bytes.
    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    source_artifact_sha256 = sha256_file(args.arrow_file) if args.arrow_file else None
    rows = [dict(row) for row in dataset]
    if args.limit_images is not None:
        rows = rows[: args.limit_images]
    raw_images = len(rows)
    coco_linked = sum(_present(row.get("coco_id")) for row in rows)
    if args.exclude_coco_linked:
        rows = [row for row in rows if not _present(row.get("coco_id"))]

    if args.profile == "official":
        pairs = [
            (row, row["qas"][0])
            for row in rows
            if row.get("qas") and row["qas"][0].get("question") and row["qas"][0].get("answer")
        ]
    elif args.profile == "balanced":
        pairs = select_core(rows, args.seed)
    else:
        pairs = [
            (row, qa)
            for row in rows
            for qa in row.get("qas", [])
            if qa.get("question") and qa.get("answer")
        ]
    pairs.sort(key=lambda pair: (int(pair[0]["image_id"]), int(pair[1].get("qa_id", 0))))

    images_dir = args.output_dir / "images"
    manifest_path = args.output_dir / f"manifest_{args.split}_{args.profile}.jsonl"
    examples: list[EvalExample] = []
    saved: dict[int, str] = {}
    for row, qa in pairs:
        image_id = int(row["image_id"])
        qa_id = int(qa.get("qa_id", 0))
        image_path = images_dir / f"{image_id}.jpg"
        if image_id not in saved:
            saved[image_id] = _save_image(row["image"], image_path)
        relative_image = image_path.relative_to(manifest_path.parent)
        coco_id = int(row["coco_id"]) if _present(row.get("coco_id")) else None
        flickr_id = int(row["flickr_id"]) if _present(row.get("flickr_id")) else None
        examples.append(
            EvalExample(
                sample_id=f"ja-vg-vqa-500-{args.split}-{image_id}-{qa_id}",
                benchmark="ja-vg-vqa-500",
                split=args.split,
                domain="general_world",
                task="vqa",
                image_path=str(relative_image),
                question=str(qa["question"]).strip(),
                answers=[str(qa["answer"]).strip()],
                question_type=question_category(str(qa["question"])),
                source={
                    "dataset_id": DATASET_ID,
                    "dataset_revision": args.revision,
                    "dataset_url": SOURCE_URL,
                    "license": "CC-BY-4.0",
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "image_url": row.get("url"),
                    "coco_id": coco_id,
                    "flickr_id": flickr_id,
                    "image_sha256": saved[image_id],
                },
                metadata={
                    "profile": args.profile,
                    "selection_seed": args.seed if args.profile == "balanced" else None,
                    "overlap_status": "source_overlap_candidate" if coco_id is not None else "not_audited",
                },
            )
        )

    count, manifest_hash = write_manifest(manifest_path, examples)
    categories: dict[str, int] = {}
    for example in examples:
        key = example.question_type or "unknown"
        categories[key] = categories.get(key, 0) + 1
    summary = {
        "schema_version": "jmoshivis.image-eval.dataset-summary.v1",
        "dataset_id": DATASET_ID,
        "dataset_revision": args.revision,
        "loaded_from_arrow": str(args.arrow_file.resolve()) if args.arrow_file else None,
        "source_artifact_sha256": source_artifact_sha256,
        "split": args.split,
        "profile": args.profile,
        "raw_image_records": raw_images,
        "coco_linked_image_records": coco_linked,
        "excluded_coco_linked": args.exclude_coco_linked,
        "output_images": len(saved),
        "output_questions": count,
        "question_categories": categories,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
    }
    (args.output_dir / f"summary_{args.split}_{args.profile}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("IMPORTANT: run overlap audit before using this set for paper results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
