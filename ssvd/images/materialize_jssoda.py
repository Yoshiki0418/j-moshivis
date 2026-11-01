#!/usr/bin/env python3
"""Materialize a locally downloaded ``llm-jp/JSSODa`` parquet snapshot.

This script deliberately does not contact Hugging Face.  It reads the parquet
files and ``source_info.json`` produced by the separate downloader, writes the
embedded image bytes to a content-addressed image tree, and creates one atomic
JSONL manifest per split.

Example::

    python ssvd/images/materialize_jssoda.py \
        --input-dir data/images/jssoda_download \
        --output-dir data/images/jssoda

Re-running the command is safe: images whose SHA-256 already matches are not
rewritten, while each manifest is rebuilt through a temporary file and renamed
only after it has been flushed and fsynced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_ID = "llm-jp/JSSODa"
DATASET_URL = "https://huggingface.co/datasets/llm-jp/JSSODa"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXPECTED_COLUMNS = (
    "id",
    "image",
    "question",
    "text",
    "is_vertical",
    "num_columns",
)
SUPPORTED_SPLITS = ("train", "validation")
OFFICIAL_SPLIT_COUNTS = {"train": 17_991, "validation": 2_246}
HEX_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic rename."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            result = _find_key(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_key(child, key)
            if result is not None:
                return result
    return None


def load_resolved_revision(source_info_path: Path, override: str | None) -> tuple[str, str]:
    """Return ``(resolved_revision, source_info_sha256)``.

    A mutable label such as ``main`` is rejected.  The materialized data must
    remain attributable to an immutable source snapshot.
    """

    if not source_info_path.is_file():
        raise FileNotFoundError(
            f"source provenance file not found: {source_info_path}. "
            "Keep source_info.json from the downloader next to the parquet files."
        )
    try:
        source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source info JSON: {source_info_path}: {exc}") from exc

    revision_value = override if override is not None else _find_key(source_info, "resolved_revision")
    if not isinstance(revision_value, str) or not revision_value.strip():
        raise ValueError(
            f"{source_info_path} does not contain a non-empty resolved_revision"
        )
    revision = revision_value.strip()
    if not HEX_REVISION.fullmatch(revision):
        raise ValueError(
            f"resolved revision must be an immutable hexadecimal commit hash, got {revision!r}"
        )
    return revision, sha256_file(source_info_path)


def discover_parquet_files(input_dir: Path, split: str) -> list[Path]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")

    # Prefer the exact directory layout used by the Hugging Face snapshot.
    for directory in (input_dir / "data", input_dir):
        if not directory.is_dir():
            continue
        matches = sorted(
            path.resolve()
            for path in directory.glob(f"{split}*.parquet")
            if path.is_file()
        )
        if matches:
            return matches

    matches = sorted(
        path.resolve()
        for path in input_dir.rglob(f"{split}*.parquet")
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"no {split!r} parquet files found under {input_dir}"
        )
    return matches


def _require_pyarrow():
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to materialize JSSODa parquet files; "
            "install it in the dataset-build environment"
        ) from exc
    return parquet


def _validate_schema(column_names: Sequence[str], parquet_path: Path) -> None:
    missing = [name for name in EXPECTED_COLUMNS if name not in column_names]
    if missing:
        raise ValueError(
            f"{parquet_path}: missing required JSSODa columns {missing}; "
            f"found {list(column_names)}"
        )


def iter_parquet_rows(
    paths: Sequence[Path], *, batch_size: int
) -> Iterable[tuple[Path, int, Mapping[str, Any]]]:
    parquet = _require_pyarrow()
    for parquet_path in paths:
        parquet_file = parquet.ParquetFile(parquet_path)
        _validate_schema(parquet_file.schema_arrow.names, parquet_path)
        source_row = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=list(EXPECTED_COLUMNS)
        ):
            for row in batch.to_pylist():
                yield parquet_path, source_row, row
                source_row += 1


def extract_image_bytes(value: Any, *, parquet_path: Path, row_number: int) -> bytes:
    """Decode the standard Hugging Face Image parquet representation."""

    payload: Any = None
    source_path: str | None = None
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = value
    elif isinstance(value, Mapping):
        payload = value.get("bytes")
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            source_path = raw_path
    else:
        # Defensive support for a pyarrow StructScalar supplied by a caller.
        as_py = getattr(value, "as_py", None)
        if callable(as_py):
            return extract_image_bytes(
                as_py(), parquet_path=parquet_path, row_number=row_number
            )

    if payload is not None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"{parquet_path} row {row_number}: image.bytes is not binary"
            )
        image_bytes = bytes(payload)
    elif source_path is not None:
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = parquet_path.parent / candidate
        try:
            image_bytes = candidate.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"{parquet_path} row {row_number}: image has no embedded bytes and "
                f"its path cannot be read: {candidate}"
            ) from exc
    else:
        raise ValueError(
            f"{parquet_path} row {row_number}: image contains neither bytes nor path"
        )

    if not image_bytes:
        raise ValueError(f"{parquet_path} row {row_number}: empty image payload")
    return image_bytes


def detect_image_format(payload: bytes) -> tuple[str, str]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if len(payload) >= 12 and payload[:4] in (b"RIFF",) and payload[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff", "image/tiff"
    raise ValueError("unsupported or corrupt encoded image payload")


def _required_text(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context}: {field} must be a non-empty string")
    # Preserve the exact source string: no strip or Unicode normalization here.
    return value


def _required_bool(row: Mapping[str, Any], field: str, context: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be boolean, got {value!r}")
    return value


def _required_int(row: Mapping[str, Any], field: str, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field} must be integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{context}: {field} must be positive, got {value!r}")
    return value


def _store_content_addressed_image(
    output_dir: Path, split: str, payload: bytes
) -> tuple[Path, str, str, int, bool]:
    digest = sha256_bytes(payload)
    extension, mime_type = detect_image_format(payload)
    destination = output_dir / "images" / split / digest[:2] / f"{digest}{extension}"
    reused = False
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"image destination is not a regular file: {destination}")
        existing_digest = sha256_file(destination)
        if existing_digest != digest:
            raise RuntimeError(
                f"content-addressed image is corrupt or collided: {destination}"
            )
        reused = True
    else:
        atomic_write_bytes(destination, payload)
    return destination, digest, mime_type, len(payload), reused


def materialize_split(
    *,
    input_dir: Path,
    output_dir: Path,
    split: str,
    revision: str,
    source_info_path: Path,
    source_info_sha256: str,
    batch_size: int,
    expected_count: int = 0,
) -> dict[str, int]:
    parquet_paths = discover_parquet_files(input_dir, split)
    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_source_ids: set[str] = set()
    reused_images = 0

    input_root = input_dir.resolve()
    output_root = output_dir.resolve()
    for parquet_path, source_row, row in iter_parquet_rows(
        parquet_paths, batch_size=batch_size
    ):
        context = f"{parquet_path} row {source_row}"
        source_id = _required_text(row, "id", context)
        question = _required_text(row, "question", context)
        text = _required_text(row, "text", context)
        is_vertical = _required_bool(row, "is_vertical", context)
        num_columns = _required_int(row, "num_columns", context)
        if source_id in seen_source_ids:
            raise ValueError(f"{context}: duplicate source id {source_id!r} in {split}")
        seen_source_ids.add(source_id)

        payload = extract_image_bytes(
            row.get("image"), parquet_path=parquet_path, row_number=source_row
        )
        image_path, image_sha256, mime_type, num_bytes, reused = (
            _store_content_addressed_image(output_root, split, payload)
        )
        reused_images += int(reused)

        uid = hashlib.sha256(
            f"{DATASET_ID}\0{revision}\0{split}\0{source_id}".encode("utf-8")
        ).hexdigest()
        if uid in seen_uids:
            raise ValueError(f"{context}: duplicate uid {uid}")
        seen_uids.add(uid)

        try:
            parquet_relative = parquet_path.relative_to(input_root).as_posix()
        except ValueError:
            parquet_relative = str(parquet_path)
        image_relative = image_path.relative_to(output_root).as_posix()
        records.append(
            {
                "uid": uid,
                "path": str(image_path),
                "relative_path": image_relative,
                "split": split,
                "source_id": source_id,
                "question": question,
                "text": text,
                "is_vertical": is_vertical,
                "num_columns": num_columns,
                "orientation": "vertical" if is_vertical else "horizontal",
                "layout": {
                    "is_vertical": is_vertical,
                    "orientation": "vertical" if is_vertical else "horizontal",
                    "num_columns": num_columns,
                },
                "source": DATASET_ID,
                "source_url": DATASET_URL,
                "source_revision": revision,
                "source_split": split,
                "source_parquet": parquet_relative,
                "source_row": source_row,
                "source_info_path": str(source_info_path.resolve()),
                "source_info_sha256": source_info_sha256,
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "image_sha256": image_sha256,
                "image_num_bytes": num_bytes,
                "image_mime_type": mime_type,
            }
        )

    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"{split} snapshot is incomplete: expected {expected_count} records, "
            f"found {len(records)}"
        )
    manifest_path = output_root / f"metadata_{split}.jsonl"
    atomic_write_jsonl(manifest_path, records)
    return {
        "records": len(records),
        "new_images": len(records) - reused_images,
        "reused_images": reused_images,
        "parquet_files": len(parquet_paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Local JSSODa snapshot containing data/*.parquet and source_info.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for images/ and metadata_<split>.jsonl",
    )
    parser.add_argument(
        "--source-info",
        type=Path,
        default=None,
        help="Downloader provenance JSON (default: <input-dir>/source_info.json)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override resolved commit hash; source_info.json is still required and hashed",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SUPPORTED_SPLITS,
        default=list(SUPPORTED_SPLITS),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--expected-train-count",
        type=int,
        default=OFFICIAL_SPLIT_COUNTS["train"],
        help="Expected pinned-snapshot row count; set 0 to disable",
    )
    parser.add_argument(
        "--expected-validation-count",
        type=int,
        default=OFFICIAL_SPLIT_COUNTS["validation"],
        help="Expected pinned-snapshot row count; set 0 to disable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.expected_train_count < 0 or args.expected_validation_count < 0:
        raise ValueError("expected counts must be non-negative")
    source_info_path = (
        args.source_info.resolve()
        if args.source_info is not None
        else input_dir / "source_info.json"
    )
    revision, source_info_sha256 = load_resolved_revision(
        source_info_path, args.revision
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source={DATASET_ID}@{revision}")
    for split in args.splits:
        stats = materialize_split(
            input_dir=input_dir,
            output_dir=output_dir,
            split=split,
            revision=revision,
            source_info_path=source_info_path,
            source_info_sha256=source_info_sha256,
            batch_size=args.batch_size,
            expected_count=(
                args.expected_train_count
                if split == "train"
                else args.expected_validation_count
            ),
        )
        print(
            f"{split}: records={stats['records']} "
            f"new_images={stats['new_images']} reused_images={stats['reused_images']} "
            f"parquet_files={stats['parquet_files']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
