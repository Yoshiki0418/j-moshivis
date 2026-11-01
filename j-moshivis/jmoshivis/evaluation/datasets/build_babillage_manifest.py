"""Build paper-compatible manifests from Kyutai's Babillage benchmark.

Babillage contains the spoken prompts used by MoshiVis, but deliberately does
not duplicate the images and ground truth from the three source benchmarks.
This adapter joins the pinned Babillage rows with the pinned source datasets:

* COCO Captions 2014 Karpathy validation/test (CIDEr)
* OCR-VQA validation/test (accuracy)
* VQAv2 validation (official VQA accuracy)

The official question audio is stored byte-for-byte and becomes
``EvalExample.audio_path``.  The source image is also stored without a decode /
re-encode round trip.  COCO rows keep *all* captions for the corresponding
image, while VQAv2 rows keep all ten annotator answers including duplicates;
both details are required by their official metrics.

Running this module without ``--dry-run`` downloads large public datasets.  It
is intentionally not part of the unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..schema import (
    EvalExample,
    ManifestError,
    validate_question_audio_provenance,
    write_manifest,
)


BABILLAGE_DATASET_ID = "kyutai/Babillage"
BABILLAGE_REVISION = "3ede8566da87763fffe3e0a1d9b78cd76bf5b85d"
BABILLAGE_URL = "https://huggingface.co/datasets/kyutai/Babillage"
BABILLAGE_LICENSE = "CC-BY-4.0"
COCO_SOURCE_REVISION = "4d0dfd4a3712a80e7b8498ed76e778f7cea9d21c"
COCO_PARQUET_REVISION = "29bc9e9117edd016db886abecb3c6f5c2e7f7c2b"
COCO_PARQUET_PATH = "2014/coco-{split}-*.parquet"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    config: str
    base_dataset_id: str
    base_revision: str
    base_config: str | None
    valid_splits: tuple[str, ...]
    expected_rows: Mapping[str, int]
    task: str
    metric: str
    base_license: str
    has_reference_audio: bool
    base_storage_revision: str | None = None
    base_storage_format: str | None = None
    base_storage_path: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "coco": DatasetSpec(
        config="coco",
        base_dataset_id="HuggingFaceM4/COCO",
        base_revision=COCO_SOURCE_REVISION,
        base_config="2014",
        valid_splits=("validation", "test"),
        expected_rows={"validation": 25_010, "test": 25_010},
        task="caption",
        metric="cider",
        base_license="CC-BY-4.0",
        has_reference_audio=True,
        base_storage_revision=COCO_PARQUET_REVISION,
        base_storage_format="parquet",
        base_storage_path=COCO_PARQUET_PATH,
    ),
    "ocrvqa": DatasetSpec(
        config="ocrvqa",
        base_dataset_id="howard-hou/OCR-VQA",
        base_revision="88234cc092c5f6d199b5cf3b471e3b490a69c07b",
        base_config=None,
        valid_splits=("validation", "test"),
        expected_rows={"validation": 100_032, "test": 100_424},
        task="vqa",
        metric="ocr_vqa_accuracy",
        # The redistribution currently declares no license in its dataset card.
        base_license="not declared in the howard-hou/OCR-VQA dataset card",
        has_reference_audio=True,
    ),
    "vqav2": DatasetSpec(
        config="vqav2",
        base_dataset_id="lmms-lab/VQAv2",
        base_revision="32665d35052eb4a6d4414851c3c829a72754915a",
        base_config=None,
        valid_splits=("validation",),
        expected_rows={"validation": 214_354},
        task="vqa",
        metric="vqav2_accuracy",
        base_license="CC-BY-4.0",
        has_reference_audio=False,
    ),
}


def _base_storage_uri(spec: DatasetSpec, split: str) -> str | None:
    if spec.base_storage_revision is None:
        return None
    if spec.base_storage_format != "parquet" or spec.base_storage_path is None:
        raise RuntimeError(
            f"{spec.config}: incomplete pinned base storage configuration"
        )
    path = spec.base_storage_path.format(split=split)
    return (
        f"hf://datasets/{spec.base_dataset_id}@{spec.base_storage_revision}/{path}"
    )


@dataclass(slots=True)
class BaseRecord:
    sample_index: int
    image_key: str
    image: Any
    answers: list[str]
    question_type: str
    source: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_babillage_image_id: int | None = None


def _required_int(row: Mapping[str, Any], key: str, row_number: int) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise ManifestError(f"Babillage row {row_number}: invalid {key!r}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"Babillage row {row_number}: missing or invalid {key!r}: {value!r}"
        ) from exc


def _required_text(value: Any, *, label: str) -> str:
    if value is None or not str(value).strip():
        raise ManifestError(f"missing or empty {label}")
    return str(value).strip()


def _nested(row: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read either a flattened ``a.b`` key or nested mappings."""

    if dotted_key in row:
        return row[dotted_key]
    value: Any = row
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return default
        value = value[component]
    return value


def detokenize(tokens: Iterable[Any]) -> str:
    """Reconstruct a readable prompt without losing the original token list."""

    normalized = [_required_text(token, label="transcript token") for token in tokens]
    if not normalized:
        raise ManifestError("audio transcript is empty")
    text = " ".join(normalized)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+(['\N{RIGHT SINGLE QUOTATION MARK}](?:s|t|re|ve|m|ll|d))\b", r"\1", text)
    return text.strip()


def _transcript_and_alignment(
    row: Mapping[str, Any],
    *,
    prefix: str,
    row_number: int,
) -> tuple[list[str], str, list[list[float]]]:
    raw_tokens = row.get(f"{prefix}_transcript")
    if not isinstance(raw_tokens, (list, tuple)):
        raise ManifestError(
            f"Babillage row {row_number}: {prefix}_transcript must be a sequence"
        )
    tokens = [_required_text(token, label=f"{prefix} transcript token") for token in raw_tokens]
    text = detokenize(tokens)

    raw_alignment = row.get(f"{prefix}_alignment")
    if not isinstance(raw_alignment, (list, tuple)) or len(raw_alignment) != len(tokens):
        raise ManifestError(
            f"Babillage row {row_number}: {prefix}_alignment must have one span "
            f"per transcript token ({len(tokens)})"
        )
    alignment: list[list[float]] = []
    for token_index, span in enumerate(raw_alignment):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ManifestError(
                f"Babillage row {row_number}: invalid {prefix} alignment span "
                f"at token {token_index}"
            )
        try:
            start, end = float(span[0]), float(span[1])
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"Babillage row {row_number}: non-numeric {prefix} alignment span "
                f"at token {token_index}"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ManifestError(
                f"Babillage row {row_number}: invalid {prefix} alignment "
                f"[{start}, {end}] at token {token_index}"
            )
        alignment.append([start, end])
    return tokens, text, alignment


def _encoded_bytes(value: Any, *, kind: str, row_number: int) -> tuple[bytes, str | None]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        path_hint = None
    elif isinstance(value, Mapping):
        raw = value.get("bytes")
        path_value = value.get("path")
        path_hint = str(path_value) if path_value else None
        if raw is not None:
            payload = bytes(raw)
        elif path_value:
            path = Path(str(path_value))
            if not path.is_file():
                raise ManifestError(
                    f"Babillage row {row_number}: {kind} path does not exist: {path}"
                )
            payload = path.read_bytes()
        else:
            raise ManifestError(
                f"Babillage row {row_number}: decoded {kind} or missing bytes/path; "
                f"load the column with decode=False"
            )
    elif isinstance(value, Image.Image):
        raise ManifestError(
            f"Babillage row {row_number}: image was decoded; load it with "
            "datasets.Image(decode=False)"
        )
    else:
        raise ManifestError(
            f"Babillage row {row_number}: unsupported {kind} value "
            f"{type(value).__name__}; load it with decode=False"
        )
    if not payload:
        raise ManifestError(f"Babillage row {row_number}: empty {kind} payload")
    return payload, path_hint


def _image_extension(payload: bytes, row_number: int) -> str:
    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise ManifestError(
            f"Babillage row {row_number}: invalid encoded image: {exc}"
        ) from exc
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }
    if image_format not in extensions:
        raise ManifestError(
            f"Babillage row {row_number}: unsupported image format {image_format!r}"
        )
    return extensions[image_format]


def _audio_extension(payload: bytes, path_hint: str | None, row_number: int) -> str:
    if payload.startswith(b"RIFF") and payload[8:12] == b"WAVE":
        return ".wav"
    if payload.startswith(b"OggS"):
        return ".ogg"
    if payload.startswith(b"fLaC"):
        return ".flac"
    if payload.startswith(b"ID3") or payload[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return ".mp3"
    if path_hint:
        suffix = Path(path_hint).suffix.lower()
        if suffix in {".wav", ".ogg", ".flac", ".mp3"}:
            return suffix
    raise ManifestError(
        f"Babillage row {row_number}: unsupported or unrecognised encoded audio"
    )


def _store_blob(
    value: Any,
    directory: Path,
    *,
    kind: str,
    row_number: int,
) -> tuple[Path, str, str]:
    payload, path_hint = _encoded_bytes(value, kind=kind, row_number=row_number)
    digest = hashlib.sha256(payload).hexdigest()
    extension = (
        _image_extension(payload, row_number)
        if kind == "image"
        else _audio_extension(payload, path_hint, row_number)
    )
    destination = directory / digest[:2] / f"{digest}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing != digest:
            raise RuntimeError(f"content-address collision or corrupt asset: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return destination, digest, extension.removeprefix(".")


def _as_reiterable(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    iterator = iter(rows)
    return list(iterator) if iterator is rows else rows


def _coco_records(rows: Iterable[Mapping[str, Any]]) -> Iterable[BaseRecord]:
    rows = _as_reiterable(rows)
    references: dict[int, list[str]] = defaultdict(list)
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ManifestError(f"COCO row {row_number}: expected a mapping")
        try:
            coco_id = int(row["cocoid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"COCO row {row_number}: invalid cocoid") from exc
        caption = _required_text(
            _nested(row, "sentences.raw"), label=f"COCO row {row_number} caption"
        )
        references[coco_id].append(caption)

    reference_positions: dict[int, int] = defaultdict(int)
    for sample_index, row in enumerate(rows):
        coco_id = int(row["cocoid"])
        caption = _required_text(
            _nested(row, "sentences.raw"), label=f"COCO row {sample_index + 1} caption"
        )
        sent_id = _nested(row, "sentences.sentid")
        current_reference_index = reference_positions[coco_id]
        reference_positions[coco_id] += 1
        yield BaseRecord(
            sample_index=sample_index,
            image_key=f"coco:{coco_id}",
            image=row.get("image"),
            answers=list(references[coco_id]),
            question_type="image_captioning",
            expected_babillage_image_id=coco_id,
            source={
                "coco_id": coco_id,
                "image_id": row.get("imgid"),
                "caption_sent_id": sent_id,
                "base_caption": caption,
                "filename": row.get("filename") or row.get("filepath"),
            },
            metadata={
                "reference_count": len(references[coco_id]),
                "current_reference_index": current_reference_index,
                "coco_prompt_policy": (
                    "all_babillage_prompts_each_scored_against_all_image_references"
                ),
                "cider_group_id": coco_id,
            },
        )


def _ocrvqa_records(rows: Iterable[Mapping[str, Any]]) -> Iterable[BaseRecord]:
    sample_index = 0
    for image_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ManifestError(f"OCR-VQA row {image_index + 1}: expected a mapping")
        questions = row.get("questions")
        answers = row.get("answers")
        if not isinstance(questions, (list, tuple)) or not isinstance(answers, (list, tuple)):
            raise ManifestError(
                f"OCR-VQA row {image_index + 1}: questions and answers must be sequences"
            )
        if not questions or len(questions) != len(answers):
            raise ManifestError(
                f"OCR-VQA row {image_index + 1}: questions/answers length mismatch"
            )
        original_image_id = _required_text(
            row.get("image_id"), label=f"OCR-VQA row {image_index + 1} image_id"
        )
        for qa_index, (question_value, answer_value) in enumerate(zip(questions, answers)):
            question = _required_text(
                question_value,
                label=f"OCR-VQA row {image_index + 1} question {qa_index}",
            )
            answer = _required_text(
                answer_value,
                label=f"OCR-VQA row {image_index + 1} answer {qa_index}",
            )
            yield BaseRecord(
                sample_index=sample_index,
                image_key=f"ocrvqa:{image_index}:{original_image_id}",
                image=row.get("image"),
                answers=[answer],
                question_type="ocr",
                expected_babillage_image_id=image_index,
                source={
                    "image_id": original_image_id,
                    "image_row_index": image_index,
                    "qa_index": qa_index,
                    "base_question": question,
                    "image_url": row.get("image_url"),
                    "set_name": row.get("set_name"),
                    "title": row.get("title"),
                    "author_name": row.get("authorName"),
                    "genre": row.get("genre"),
                },
            )
            sample_index += 1


def _vqav2_records(rows: Iterable[Mapping[str, Any]]) -> Iterable[BaseRecord]:
    for sample_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ManifestError(f"VQAv2 row {sample_index + 1}: expected a mapping")
        raw_answers = _vqav2_answer_rows(row, sample_index + 1)
        answers: list[str] = []
        answer_annotations: list[dict[str, Any]] = []
        for answer_index, value in enumerate(raw_answers):
            annotation = dict(value) if isinstance(value, Mapping) else {"answer": value}
            answer = _required_text(
                annotation.get("answer"),
                label=f"VQAv2 row {sample_index + 1} answer {answer_index}",
            )
            answers.append(answer)
            answer_annotations.append(
                {
                    "answer": answer,
                    "answer_confidence": annotation.get("answer_confidence"),
                    "answer_id": annotation.get("answer_id"),
                }
            )
        try:
            image_id = int(row["image_id"])
            question_id = int(row["question_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(
                f"VQAv2 row {sample_index + 1}: invalid image_id/question_id"
            ) from exc
        question = _required_text(
            row.get("question"), label=f"VQAv2 row {sample_index + 1} question"
        )
        question_type = _required_text(
            row.get("question_type"),
            label=f"VQAv2 row {sample_index + 1} question_type",
        )
        yield BaseRecord(
            sample_index=sample_index,
            image_key=f"vqav2:{image_id}",
            image=row.get("image"),
            answers=answers,
            question_type=question_type,
            source={
                "image_id": image_id,
                "question_id": question_id,
                "base_question": question,
            },
            metadata={
                "answer_type": row.get("answer_type"),
                "multiple_choice_answer": row.get("multiple_choice_answer"),
                "answer_annotations": answer_annotations,
            },
        )


def _vqav2_answer_rows(
    row: Mapping[str, Any], row_number: int
) -> list[dict[str, Any]]:
    """Normalize Arrow's supported representations of sequence-of-struct.

    Depending on the datasets/Arrow version and whether ``Dataset.flatten``
    was used, VQAv2 answers can be a list of structs, a struct of lists, or
    three flattened ``answers.*`` columns. Their order and duplicates must be
    preserved for official VQA accuracy.
    """

    raw = row.get("answers")
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ManifestError(f"VQAv2 row {row_number}: answers are empty")
        return [
            dict(value) if isinstance(value, Mapping) else {"answer": value}
            for value in raw
        ]

    if isinstance(raw, Mapping):
        answer_values = raw.get("answer")
        confidence_values = raw.get("answer_confidence")
        id_values = raw.get("answer_id")
    else:
        answer_values = row.get("answers.answer")
        confidence_values = row.get("answers.answer_confidence")
        id_values = row.get("answers.answer_id")

    if not isinstance(answer_values, (list, tuple)) or not answer_values:
        raise ManifestError(f"VQAv2 row {row_number}: answers are missing")

    def optional_sequence(value: Any, name: str) -> list[Any]:
        if value is None:
            return [None] * len(answer_values)
        if not isinstance(value, (list, tuple)) or len(value) != len(answer_values):
            raise ManifestError(
                f"VQAv2 row {row_number}: {name} must match answers length "
                f"{len(answer_values)}"
            )
        return list(value)

    confidences = optional_sequence(confidence_values, "answer_confidence")
    answer_ids = optional_sequence(id_values, "answer_id")
    return [
        {
            "answer": answer,
            "answer_confidence": confidence,
            "answer_id": answer_id,
        }
        for answer, confidence, answer_id in zip(
            answer_values, confidences, answer_ids
        )
    ]


def _base_records(
    config: str, rows: Iterable[Mapping[str, Any]]
) -> Iterable[BaseRecord]:
    if config == "coco":
        return _coco_records(rows)
    if config == "ocrvqa":
        return _ocrvqa_records(rows)
    if config == "vqav2":
        return _vqav2_records(rows)
    raise ValueError(f"unsupported Babillage config: {config!r}")


def _next_pair(
    babillage_iterator: Any,
    base_iterator: Any,
    *,
    row_number: int,
) -> tuple[Mapping[str, Any], BaseRecord] | None:
    missing = object()
    babillage_row = next(babillage_iterator, missing)
    base_record = next(base_iterator, missing)
    if babillage_row is missing and base_record is missing:
        return None
    if babillage_row is missing or base_record is missing:
        exhausted = "Babillage" if babillage_row is missing else "base dataset"
        raise ManifestError(
            f"row {row_number}: {exhausted} ended before its paired dataset"
        )
    if not isinstance(babillage_row, Mapping):
        raise ManifestError(f"Babillage row {row_number}: expected a mapping")
    return babillage_row, base_record


def convert_rows(
    babillage_rows: Iterable[Mapping[str, Any]],
    base_rows: Iterable[Mapping[str, Any]],
    *,
    config: str,
    split: str,
    output_dir: str | Path,
    babillage_revision: str = BABILLAGE_REVISION,
    base_revision: str | None = None,
    limit: int | None = None,
    expected_count: int | None = None,
    loaded_from: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Join decoded-off Babillage/base rows and write a canonical manifest."""

    try:
        spec = DATASET_SPECS[config]
    except KeyError as exc:
        raise ValueError(f"unsupported Babillage config: {config!r}") from exc
    if split not in spec.valid_splits:
        raise ValueError(
            f"Babillage {config!r} has no supported {split!r} split; "
            f"choose one of {spec.valid_splits}"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    base_revision = base_revision or spec.base_revision
    if spec.base_storage_revision is not None and base_revision != spec.base_revision:
        raise ValueError(
            f"{config}: base_revision cannot be overridden because the pinned "
            "auto-converted storage is tied to its source revision"
        )

    destination = Path(output_dir)
    manifest_path = destination / f"manifest_{split}.jsonl"
    images_dir = destination / "images"
    question_audio_dir = destination / "audio" / "questions"
    reference_audio_dir = destination / "audio" / "references"

    image_cache: dict[str, tuple[Path, str]] = {}
    image_hashes: set[str] = set()
    question_audio_hashes: set[str] = set()
    reference_audio_hashes: set[str] = set()
    examples: list[EvalExample] = []

    babillage_iterator = iter(babillage_rows)
    base_iterator = iter(_base_records(config, base_rows))
    row_number = 1
    while limit is None or len(examples) < limit:
        pair = _next_pair(
            babillage_iterator, base_iterator, row_number=row_number
        )
        if pair is None:
            break
        babillage_row, base_record = pair
        babillage_sample_id = _required_int(
            babillage_row, "sample_id", row_number
        )
        if babillage_sample_id != base_record.sample_index:
            raise ManifestError(
                f"Babillage row {row_number}: sample_id {babillage_sample_id} does "
                f"not match base row {base_record.sample_index}; refusing a positional "
                "join with a different revision/order"
            )
        if base_record.expected_babillage_image_id is not None:
            image_id = _required_int(babillage_row, "image_id", row_number)
            if image_id != base_record.expected_babillage_image_id:
                raise ManifestError(
                    f"Babillage row {row_number}: image_id {image_id} does not match "
                    f"base image {base_record.expected_babillage_image_id}"
                )

        sample_id = f"babillage-{config}-{split}-{babillage_sample_id}"
        question_tokens, question, question_alignment = _transcript_and_alignment(
            babillage_row, prefix="question", row_number=row_number
        )
        question_audio_path, question_audio_sha256, question_encoding = _store_blob(
            babillage_row.get("question_audio"),
            question_audio_dir,
            kind="audio",
            row_number=row_number,
        )
        question_audio_hashes.add(question_audio_sha256)

        if base_record.image_key not in image_cache:
            image_path, image_sha256, _ = _store_blob(
                base_record.image,
                images_dir,
                kind="image",
                row_number=row_number,
            )
            image_cache[base_record.image_key] = (image_path, image_sha256)
        image_path, image_sha256 = image_cache[base_record.image_key]
        image_hashes.add(image_sha256)

        question_audio_metadata = {
            "sample_id": sample_id,
            "prompt_text": question,
            "prompt_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            # Existing strict evaluation validation uses this compatibility
            # field name for the encoded asset hash, including OGG assets.
            "wav_sha256": question_audio_sha256,
            "asset_sha256": question_audio_sha256,
            "encoding": question_encoding,
            "transcript_tokens": question_tokens,
            "alignment_sec": question_alignment,
            "speech_start_sec": question_alignment[0][0],
            "speech_end_sec": max(span[1] for span in question_alignment),
            "source_dataset_id": BABILLAGE_DATASET_ID,
            "source_dataset_revision": babillage_revision,
        }

        metadata: dict[str, Any] = {
            "official_metric": spec.metric,
            "moshivis_metric": spec.metric,
            "evaluation_protocol": "moshivis_babillage_official_question_audio",
            "base_ground_truth": base_record.metadata,
            "question_audio": question_audio_metadata,
            "overlap_status": "not_audited",
        }

        if spec.has_reference_audio:
            answer_tokens, answer_text, answer_alignment = _transcript_and_alignment(
                babillage_row, prefix="answer", row_number=row_number
            )
            reference_path, reference_sha256, reference_encoding = _store_blob(
                babillage_row.get("answer_audio"),
                reference_audio_dir,
                kind="audio",
                row_number=row_number,
            )
            reference_audio_hashes.add(reference_sha256)
            metadata["reference_audio"] = {
                "audio_path": str(reference_path.relative_to(manifest_path.parent)),
                "asset_sha256": reference_sha256,
                "encoding": reference_encoding,
                "transcript_text": answer_text,
                "transcript_tokens": answer_tokens,
                "alignment_sec": answer_alignment,
                "speech_start_sec": answer_alignment[0][0],
                "speech_end_sec": max(span[1] for span in answer_alignment),
                "note": "Conversational spoken reference; metric GT is EvalExample.answers",
            }

        source = {
            "dataset_id": BABILLAGE_DATASET_ID,
            "dataset_config": config,
            "dataset_revision": babillage_revision,
            "dataset_url": BABILLAGE_URL,
            "license": BABILLAGE_LICENSE,
            "babillage_sample_id": babillage_sample_id,
            "base_dataset_id": spec.base_dataset_id,
            "base_dataset_config": spec.base_config,
            "base_dataset_revision": base_revision,
            "base_dataset_storage_revision": spec.base_storage_revision,
            "base_dataset_storage_format": spec.base_storage_format,
            "base_dataset_storage_uri": _base_storage_uri(spec, split),
            "base_dataset_url": f"https://huggingface.co/datasets/{spec.base_dataset_id}",
            "base_dataset_license": spec.base_license,
            "image_sha256": image_sha256,
            **base_record.source,
        }
        example = EvalExample(
            sample_id=sample_id,
            benchmark=f"babillage-{config}",
            split=split,
            domain="general_world",
            task=spec.task,
            image_path=str(image_path.relative_to(manifest_path.parent)),
            question=question,
            answers=list(base_record.answers),
            prompt_text=question,
            audio_path=str(question_audio_path.relative_to(manifest_path.parent)),
            question_type=base_record.question_type,
            source=source,
            metadata=metadata,
        )
        # Catch prompt/audio provenance mistakes before anything is written.
        validate_question_audio_provenance(example)
        examples.append(example)
        row_number += 1

    if expected_count is not None and limit is None and len(examples) != expected_count:
        raise ManifestError(
            f"Babillage {config}/{split}: expected {expected_count} rows, got "
            f"{len(examples)}"
        )

    count, manifest_hash = write_manifest(manifest_path, examples)
    summary = {
        "schema_version": "jmoshivis.image-eval.dataset-summary.v1",
        "dataset_id": BABILLAGE_DATASET_ID,
        "dataset_config": config,
        "dataset_revision": babillage_revision,
        "dataset_url": BABILLAGE_URL,
        "base_dataset_id": spec.base_dataset_id,
        "base_dataset_config": spec.base_config,
        "base_dataset_revision": base_revision,
        "base_dataset_storage_revision": spec.base_storage_revision,
        "base_dataset_storage_format": spec.base_storage_format,
        "base_dataset_storage_uri": _base_storage_uri(spec, split),
        "loaded_from": dict(loaded_from or {}),
        "split": split,
        "task": spec.task,
        "official_metric": spec.metric,
        "moshivis_metric": spec.metric,
        "expected_questions": expected_count,
        "output_questions": count,
        "output_images": len(image_hashes),
        "unique_question_audio_assets": len(question_audio_hashes),
        "unique_reference_audio_assets": len(reference_audio_hashes),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "image_storage": "sha256-content-addressed-original-bytes",
        "audio_storage": "sha256-content-addressed-original-bytes",
        "ground_truth_policy": (
            "all_captions_for_image"
            if config == "coco"
            else "all_annotator_answers_with_duplicates"
            if config == "vqav2"
            else "source_answer"
        ),
    }
    if config == "coco":
        summary.update(
            {
                "coco_prompt_policy": (
                    "all_babillage_prompts_each_scored_against_all_image_references"
                ),
                "cider_group_id_field": "metadata.base_ground_truth.cider_group_id",
                "standard_coco_comparison_note": (
                    "25010 Babillage prompt rows are scored; this is not the standard "
                    "single-caption-per-5000-image COCO evaluation unit"
                ),
            }
        )
    summary_path = destination / f"summary_{split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_huggingface_rows(
    spec: DatasetSpec,
    *,
    split: str,
    babillage_revision: str,
    base_revision: str,
) -> tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]], dict[str, str]]:
    try:
        from datasets import Audio as DatasetAudio
        from datasets import Image as DatasetImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Babillage adapter needs Hugging Face datasets. "
            "Run it with .venv-data/bin/python."
        ) from exc

    babillage = load_dataset(
        BABILLAGE_DATASET_ID,
        spec.config,
        split=split,
        revision=babillage_revision,
    )
    babillage = babillage.cast_column(
        "question_audio", DatasetAudio(decode=False)
    )
    if spec.has_reference_audio:
        babillage = babillage.cast_column(
            "answer_audio", DatasetAudio(decode=False)
        )

    storage_uri = _base_storage_uri(spec, split)
    if storage_uri is not None:
        if base_revision != spec.base_revision:
            raise ValueError(
                f"{spec.config}: --base-revision cannot be overridden because the "
                "auto-converted Parquet storage is pinned to its source revision"
            )
        # datasets 5.x no longer executes the historical COCO.py loader.  Use
        # Hugging Face's pinned auto-converted Parquet artifact instead, while
        # recording both its storage commit and the original source commit.
        base = load_dataset(
            "parquet",
            data_files={split: storage_uri},
            split=split,
        )
    else:
        base_arguments: list[str] = [spec.base_dataset_id]
        if spec.base_config is not None:
            base_arguments.append(spec.base_config)
        base = load_dataset(
            *base_arguments,
            split=split,
            revision=base_revision,
        )
    if "image" not in base.column_names:
        raise ManifestError(f"{spec.base_dataset_id}: source has no image column")
    base = base.cast_column("image", DatasetImage(decode=False))
    base_source = (
        f"{spec.base_dataset_id}@{base_revision}/"
        f"{spec.base_config or 'default'}/{split}"
    )
    loaded_from = {
        "babillage": f"{BABILLAGE_DATASET_ID}@{babillage_revision}/{spec.config}/{split}",
        "base": storage_uri or base_source,
        "base_source": base_source,
    }
    if storage_uri is not None:
        loaded_from["base_storage"] = storage_uri
    return babillage, base, loaded_from


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", choices=(*DATASET_SPECS, "all"), required=True
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="validation is available for all configs; test is COCO/OCR-VQA only",
    )
    parser.add_argument("--babillage-revision", default=BABILLAGE_REVISION)
    parser.add_argument(
        "--base-revision",
        help="Override the pinned source revision (single --config only).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.config == "all" and args.base_revision:
        raise ValueError("--base-revision can only be used with one --config")
    configs = list(DATASET_SPECS) if args.config == "all" else [args.config]
    for config in configs:
        spec = DATASET_SPECS[config]
        if args.split not in spec.valid_splits:
            raise ValueError(
                f"Babillage {config!r} has no {args.split!r} split; "
                f"choose one of {spec.valid_splits}"
            )
        if (
            spec.base_storage_revision is not None
            and args.base_revision is not None
            and args.base_revision != spec.base_revision
        ):
            raise ValueError(
                f"{config}: --base-revision cannot be overridden because the "
                "auto-converted Parquet storage is pinned"
            )

    if args.dry_run:
        plan = [
            {
                "config": config,
                "split": args.split,
                "babillage": f"{BABILLAGE_DATASET_ID}@{args.babillage_revision}",
                "base": (
                    f"{DATASET_SPECS[config].base_dataset_id}@"
                    f"{args.base_revision or DATASET_SPECS[config].base_revision}"
                ),
                "base_source_revision": (
                    args.base_revision or DATASET_SPECS[config].base_revision
                ),
                "base_storage_revision": DATASET_SPECS[config].base_storage_revision,
                "base_storage_uri": _base_storage_uri(
                    DATASET_SPECS[config], args.split
                ),
                "expected_rows": DATASET_SPECS[config].expected_rows[args.split],
                "limit": args.limit,
            }
            for config in configs
        ]
        print(json.dumps(plan[0] if len(plan) == 1 else plan, indent=2))
        return 0

    summaries: list[dict[str, Any]] = []
    for config in configs:
        spec = DATASET_SPECS[config]
        base_revision = args.base_revision or spec.base_revision
        babillage, base, loaded_from = _load_huggingface_rows(
            spec,
            split=args.split,
            babillage_revision=args.babillage_revision,
            base_revision=base_revision,
        )
        output_dir = (
            args.output_dir / f"babillage-{config}"
            if args.config == "all"
            else args.output_dir
        )
        summaries.append(
            convert_rows(
                babillage,
                base,
                config=config,
                split=args.split,
                output_dir=output_dir,
                babillage_revision=args.babillage_revision,
                base_revision=base_revision,
                limit=args.limit,
                expected_count=(
                    None if args.limit is not None else spec.expected_rows[args.split]
                ),
                loaded_from=loaded_from,
            )
        )
    print(
        json.dumps(
            summaries[0] if len(summaries) == 1 else summaries,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
