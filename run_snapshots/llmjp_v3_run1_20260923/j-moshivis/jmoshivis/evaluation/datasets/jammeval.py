"""Prepare the four single-image JAMMEval tasks as evaluation manifests.

The adapter intentionally keeps the image payload embedded in the source
Parquet file byte-for-byte.  Images are stored by SHA-256, so questions that
share an image also share one local file without JPEG/PNG recompression.

The public JAMMEval collection contains seven configurations.  J-MoshiVis
starts with the four single-image, natural-image configurations:

* CC-OCR-JA-Refined (short-answer OCR)
* CVQA-JA-Refined (multiple choice)
* Heron-Bench-Refined (short answer)
* JA-VLM-Bench-Refined (short answer)

CVQA's published text protocol labels choices A--D.  Spoken Japanese can make
Latin letters ambiguous, so the audio manifest uses the labels 1--4 and keeps
the official A--D mapping in ``metadata`` for paper-compatible re-scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..schema import EvalExample, ManifestError, write_manifest


DATASET_ID = "llm-jp/JAMMEval"
DEFAULT_REVISION = "fc602195d5c9142c579e9a0e8434a9d7c54ab1f0"
SOURCE_URL = "https://huggingface.co/datasets/llm-jp/JAMMEval"
OFFICIAL_GITLAB_URL = "https://gitlab.llm-jp.nii.ac.jp/datasets/jammeval"

SHORT_ANSWER_PROMPT_SUFFIX = "\n上記の質問に対して、正確かつ簡潔に答えてください。"
OFFICIAL_MC_PROMPT_SUFFIX = (
    "\nステップバイステップで考えてから答えてください。\n"
    "最後の行は 'Answer: $LETTER' の形式で、選択肢のアルファベットで回答してください。"
)
AUDIO_MC_PROMPT_SUFFIX = (
    "\n上記の質問に対して、正しい選択肢の番号を1から4の数字で一つだけ答えてください。"
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    config: str
    slug: str
    task: str
    question_type: str
    source_dataset_url: str
    license_text: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "CC-OCR-JA-Refined": DatasetSpec(
        config="CC-OCR-JA-Refined",
        slug="jammeval-cc-ocr-ja-refined",
        task="vqa",
        question_type="ocr",
        source_dataset_url="https://huggingface.co/datasets/wulipc/CC-OCR",
        license_text="MIT",
    ),
    "CVQA-JA-Refined": DatasetSpec(
        config="CVQA-JA-Refined",
        slug="jammeval-cvqa-ja-refined",
        task="multiple_choice",
        question_type="japanese_cultural_knowledge",
        source_dataset_url="https://huggingface.co/datasets/afaji/cvqa",
        license_text="Per-example; see the License and Image Source fields",
    ),
    "Heron-Bench-Refined": DatasetSpec(
        config="Heron-Bench-Refined",
        slug="jammeval-heron-bench-refined",
        task="vqa",
        question_type="japanese_cultural_knowledge",
        source_dataset_url="https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench",
        license_text="Public domain / CC BY 1.0 / CC BY 2.0 (per image)",
    ),
    "JA-VLM-Bench-Refined": DatasetSpec(
        config="JA-VLM-Bench-Refined",
        slug="jammeval-ja-vlm-bench-refined",
        task="vqa",
        question_type="japanese_cultural_knowledge",
        source_dataset_url="https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild",
        license_text="Unsplash License",
    ),
}


def _required_text(row: Mapping[str, Any], key: str, row_number: int) -> str:
    value = row.get(key)
    if value is None or not str(value).strip():
        raise ManifestError(f"row {row_number}: missing or empty {key!r}")
    return str(value).strip()


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _image_bytes(value: Any, row_number: int) -> bytes:
    """Return original encoded bytes; decoded PIL images are rejected."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif isinstance(value, Mapping):
        raw = value.get("bytes")
        if raw is not None:
            payload = bytes(raw)
        else:
            source_path = value.get("path")
            if not source_path:
                raise ManifestError(f"row {row_number}: image has neither bytes nor path")
            path = Path(str(source_path))
            if not path.is_file():
                raise ManifestError(f"row {row_number}: image path does not exist: {path}")
            payload = path.read_bytes()
    elif isinstance(value, Image.Image):
        raise ManifestError(
            f"row {row_number}: image was decoded before conversion; load it with "
            "datasets.Image(decode=False) to preserve the original bytes"
        )
    else:
        raise ManifestError(f"row {row_number}: unsupported image value {type(value).__name__}")
    if not payload:
        raise ManifestError(f"row {row_number}: image payload is empty")
    return payload


def _image_extension(payload: bytes, row_number: int) -> str:
    """Detect the encoded format without changing the image payload."""

    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise ManifestError(f"row {row_number}: invalid encoded image: {exc}") from exc
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }
    try:
        return extensions[image_format]
    except KeyError as exc:
        raise ManifestError(
            f"row {row_number}: unsupported encoded image format {image_format!r}"
        ) from exc


def _store_image(value: Any, images_dir: Path, row_number: int) -> tuple[Path, str]:
    payload = _image_bytes(value, row_number)
    digest = hashlib.sha256(payload).hexdigest()
    extension = _image_extension(payload, row_number)
    destination = images_dir / digest[:2] / f"{digest}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise RuntimeError(f"content-address collision or corrupt cached image: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return destination, digest


def _source_common(
    spec: DatasetSpec,
    *,
    revision: str,
    image_sha256: str,
    original_id: str,
) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_config": spec.config,
        "dataset_revision": revision,
        "dataset_url": SOURCE_URL,
        "official_gitlab_url": OFFICIAL_GITLAB_URL,
        "source_dataset_url": spec.source_dataset_url,
        "original_id": original_id,
        "license": spec.license_text,
        "image_sha256": image_sha256,
    }


def _build_short_answer(
    spec: DatasetSpec,
    row: Mapping[str, Any],
    *,
    row_number: int,
    split: str,
    image_path: str,
    image_sha256: str,
    revision: str,
) -> EvalExample:
    original_id = _required_text(row, "original_id", row_number)
    question = _required_text(row, "question", row_number)
    answer = _required_text(row, "answer", row_number)
    source = _source_common(
        spec,
        revision=revision,
        image_sha256=image_sha256,
        original_id=original_id,
    )
    question_type = spec.question_type
    metadata: dict[str, Any] = {
        "official_answer_format": "short_answer",
        "official_prompt_suffix": SHORT_ANSWER_PROMPT_SUFFIX,
        "overlap_status": "not_audited",
    }

    if spec.config == "CC-OCR-JA-Refined":
        source["ocr"] = _optional_text(row, "ocr")
        source["image_name"] = _optional_text(row, "image_name")
    elif spec.config == "Heron-Bench-Refined":
        source["context"] = _optional_text(row, "context")
        image_category = _optional_text(row, "image_category")
        source["image_category"] = image_category
        if image_category:
            question_type = image_category
    elif spec.config == "JA-VLM-Bench-Refined":
        source["image_url"] = _optional_text(row, "image_url")
        source["page_url"] = _optional_text(row, "page_url")

    return EvalExample(
        sample_id=f"{spec.slug}-{split}-{original_id}",
        benchmark=spec.slug,
        split=split,
        domain="japan",
        task="vqa",
        image_path=image_path,
        question=question,
        answers=[answer],
        prompt_text=question + SHORT_ANSWER_PROMPT_SUFFIX,
        question_type=question_type,
        source=source,
        metadata=metadata,
    )


def _build_cvqa(
    spec: DatasetSpec,
    row: Mapping[str, Any],
    *,
    row_number: int,
    split: str,
    image_path: str,
    image_sha256: str,
    revision: str,
) -> EvalExample:
    original_id = _required_text(row, "original_id", row_number)
    question = _required_text(row, "question", row_number)
    raw_choices = row.get("options")
    if not isinstance(raw_choices, (list, tuple)) or len(raw_choices) != 4:
        raise ManifestError(f"row {row_number}: CVQA requires exactly four options")
    choices = [str(choice).strip() for choice in raw_choices]
    if any(not choice for choice in choices):
        raise ManifestError(f"row {row_number}: CVQA options must not be empty")
    try:
        answer_index = int(row["answer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"row {row_number}: invalid CVQA answer index") from exc
    if not 0 <= answer_index < len(choices):
        raise ManifestError(
            f"row {row_number}: CVQA answer index {answer_index} is outside 0..{len(choices) - 1}"
        )

    audio_labels = [str(index) for index in range(1, len(choices) + 1)]
    official_labels = [chr(ord("A") + index) for index in range(len(choices))]
    numbered_choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(audio_labels, choices)
    )
    source = _source_common(
        spec,
        revision=revision,
        image_sha256=image_sha256,
        original_id=original_id,
    )
    source.update(
        {
            "license": _optional_text(row, "License") or spec.license_text,
            "image_source": _optional_text(row, "Image Source"),
            "image_type": _optional_text(row, "Image Type"),
        }
    )
    category = _optional_text(row, "Category")
    metadata = {
        "official_answer_format": "multiple_choice_letter",
        "official_choice_labels": official_labels,
        "official_choice_map": dict(zip(official_labels, choices)),
        "official_correct_choice": official_labels[answer_index],
        "official_prompt_suffix": OFFICIAL_MC_PROMPT_SUFFIX,
        "audio_choice_labels": audio_labels,
        "answer_index": answer_index,
        "category": category,
        "overlap_status": "not_audited",
    }
    return EvalExample(
        sample_id=f"{spec.slug}-{split}-{original_id}",
        benchmark=spec.slug,
        split=split,
        domain="japan",
        task="multiple_choice",
        image_path=image_path,
        question=question,
        answers=[choices[answer_index]],
        prompt_text=f"{question}\n{numbered_choices}{AUDIO_MC_PROMPT_SUFFIX}",
        choices=choices,
        choice_labels=audio_labels,
        correct_choice=audio_labels[answer_index],
        question_type=category or spec.question_type,
        source=source,
        metadata=metadata,
    )


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: str,
    split: str,
    output_dir: str | Path,
    revision: str = DEFAULT_REVISION,
    limit: int | None = None,
    loaded_from: str | None = None,
) -> dict[str, Any]:
    """Convert decoded-off dataset rows and write a manifest plus summary."""

    try:
        spec = DATASET_SPECS[config]
    except KeyError as exc:
        raise ValueError(f"unsupported JAMMEval config: {config!r}") from exc
    destination = Path(output_dir)
    manifest_path = destination / f"manifest_{split}.jsonl"
    images_dir = destination / "images"
    examples: list[EvalExample] = []
    unique_images: set[str] = set()

    for row_number, row in enumerate(rows, 1):
        if limit is not None and len(examples) >= limit:
            break
        if not isinstance(row, Mapping):
            raise ManifestError(f"row {row_number}: expected a mapping")
        image_destination, image_sha256 = _store_image(row.get("image"), images_dir, row_number)
        unique_images.add(image_sha256)
        image_path = str(image_destination.relative_to(manifest_path.parent))
        kwargs = {
            "row_number": row_number,
            "split": split,
            "image_path": image_path,
            "image_sha256": image_sha256,
            "revision": revision,
        }
        if spec.config == "CVQA-JA-Refined":
            example = _build_cvqa(spec, row, **kwargs)
        else:
            example = _build_short_answer(spec, row, **kwargs)
        examples.append(example)

    count, manifest_hash = write_manifest(manifest_path, examples)
    summary = {
        "schema_version": "jmoshivis.image-eval.dataset-summary.v1",
        "dataset_id": DATASET_ID,
        "dataset_config": config,
        "dataset_revision": revision,
        "dataset_url": SOURCE_URL,
        "loaded_from": loaded_from,
        "split": split,
        "task": spec.task,
        "output_images": len(unique_images),
        "output_questions": count,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "image_storage": "sha256-content-addressed-original-bytes",
    }
    summary_path = destination / f"summary_{split}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_rows(args: argparse.Namespace) -> tuple[Iterable[Mapping[str, Any]], str]:
    try:
        from datasets import Image as DatasetImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The JAMMEval adapter needs Hugging Face datasets. "
            "Run it with .venv-data/bin/python."
        ) from exc

    if args.parquet_file:
        files = [str(path.resolve()) for path in args.parquet_file]
        dataset = load_dataset(
            "parquet",
            data_files={args.split: files},
            split=args.split,
        )
        loaded_from = ",".join(files)
    elif args.source_dir is not None:
        config_dir = args.source_dir / args.config
        if not config_dir.is_dir():
            raise FileNotFoundError(
                f"JAMMEval config directory not found: {config_dir}. "
                "Pass the directory containing the four *-Refined folders."
            )
        dataset = load_dataset(str(config_dir), split=args.split)
        loaded_from = str(config_dir.resolve())
    else:
        dataset = load_dataset(
            DATASET_ID,
            args.config,
            split=args.split,
            revision=args.revision,
        )
        loaded_from = f"{DATASET_ID}@{args.revision}/{args.config}"

    if "image" not in dataset.column_names:
        raise ManifestError(f"{args.config}: source dataset has no image column")
    # This is essential: the default decoder returns PIL images and would make
    # byte-exact, no-recompression storage impossible.
    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    return dataset, loaded_from


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", choices=(*DATASET_SPECS, "all"), required=True,
        help="One single-image configuration, or all four.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-dir",
        type=Path,
        help="Local data/JAMMEval directory from the official GitLab clone.",
    )
    source.add_argument(
        "--parquet-file",
        type=Path,
        nargs="+",
        help="One or more local Parquet shards; useful on offline compute nodes.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.dry_run:
        source = args.parquet_file or args.source_dir or f"{DATASET_ID}@{args.revision}"
        print(
            f"would load config={args.config} split={args.split} from {source} "
            f"and write to {args.output_dir}"
        )
        return 0
    configs = list(DATASET_SPECS) if args.config == "all" else [args.config]
    summaries: list[dict[str, Any]] = []
    for config in configs:
        config_args = argparse.Namespace(**{**vars(args), "config": config})
        rows, loaded_from = _load_rows(config_args)
        output_dir = args.output_dir / DATASET_SPECS[config].slug if args.config == "all" else args.output_dir
        summaries.append(
            convert_rows(
                rows,
                config=config,
                split=args.split,
                output_dir=output_dir,
                revision=args.revision,
                limit=args.limit,
                loaded_from=loaded_from,
            )
        )
    print(json.dumps(summaries[0] if len(summaries) == 1 else summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
