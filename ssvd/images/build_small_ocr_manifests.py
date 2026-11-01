#!/usr/bin/env python3
"""Retired historical utility for the discarded JPSC/NDL OCR sources.

This utility is not part of the active v4 data pipeline as of 2026-09-28.
The active OCR source is JSSODa only; keep this file solely to preserve the
provenance of the earlier experiment.

Build provenance-preserving manifests for JPSC1400 and NDL Oneline OCR.

The command consumes already downloaded and extracted datasets.  It never
downloads data and never modifies the source directories.

Example::

    python ssvd/images/build_small_ocr_manifests.py \
        --jpsc-root data/images/raw/JPSC1400 \
        --ndl-root data/images/raw/ocr-ndloneline \
        --output-dir data/images/ocr_small

``--search-root`` can be used instead when an archive has introduced an extra
top-level directory; the script discovers datasets from their official marker
files (``label.txt`` plus ``png/`` for JPSC1400 and ``labeldata_pdm.tsv`` plus
``oneline_tate/``/``oneline_yoko/`` for NDL).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JPSC_SOURCE = "JPSC1400"
JPSC_SOURCE_URL = "https://www.imglab.org/db/"
JPSC_LICENSE_URL = "https://www.imglab.org/db/files/README-JPSC1400.txt"
JPSC_LICENSE_NAME = "JPSC1400 custom permissive license"
JPSC_DEFAULT_REVISION = "20201218"
JPSC_DEFAULT_COUNT = 1_400

NDL_SOURCE = "ndl-lab/ocr-ndloneline"
NDL_SOURCE_URL = "https://github.com/ndl-lab/ocr-ndloneline"
NDL_DATASET_URL = "https://lab.ndl.go.jp/dataset/ocronelinedataset/"
NDL_LICENSE_NAME = "Public Domain Mark 1.0"
NDL_LICENSE_URL = "https://creativecommons.org/publicdomain/mark/1.0/"
NDL_DEFAULT_COUNT = 2_339

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _unique_marker(root: Path, name: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    matches = sorted(
        path.resolve()
        for path in root.rglob(name)
        if path.is_file() and path.name == name
    )
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}")
    if len(matches) != 1:
        rendered = "\n  ".join(str(path) for path in matches)
        raise ValueError(f"multiple {name} files found under {root}:\n  {rendered}")
    return matches[0]


def _has_named_child(root: Path, name: str) -> bool:
    return any(path.is_dir() and path.name == name for path in root.rglob(name))


def discover_jpsc_root(search_roots: Sequence[Path]) -> Path:
    candidates: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for label in search_root.rglob("label.txt"):
            if label.is_file() and _has_named_child(label.parent, "png"):
                candidates.add(label.parent.resolve())
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates)) or "none"
        raise ValueError(f"expected one JPSC1400 root, found: {rendered}")
    return next(iter(candidates))


def discover_ndl_root(search_roots: Sequence[Path]) -> Path:
    candidates: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for label in search_root.rglob("labeldata_pdm.tsv"):
            if not label.is_file():
                continue
            parent = label.parent
            if _has_named_child(parent, "oneline_tate") and _has_named_child(
                parent, "oneline_yoko"
            ):
                candidates.add(parent.resolve())
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates)) or "none"
        raise ValueError(f"expected one NDL Oneline root, found: {rendered}")
    return next(iter(candidates))


def _index_images(root: Path, *, suffixes: set[str] = IMAGE_SUFFIXES) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            by_name[path.name].append(path.resolve())
    for paths in by_name.values():
        paths.sort()
    return dict(by_name)


def _select_unique_image(candidates: Sequence[Path], context: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f"{context}: matching image not found")
    if len(candidates) == 1:
        return candidates[0]

    # Duplicated archive trees are acceptable only when they contain identical
    # bytes.  Select deterministically while retaining the image hash below.
    hashes = {sha256_file(path) for path in candidates}
    if len(hashes) == 1:
        return sorted(candidates)[0]
    rendered = ", ".join(str(path) for path in candidates)
    raise ValueError(f"{context}: ambiguous images with different content: {rendered}")


def _find_jpsc_readme(root: Path, override: Path | None) -> Path:
    if override is not None:
        path = override.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JPSC license/README file not found: {path}")
        return path
    matches = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() == "readme-jpsc1400.txt"
    )
    if not matches:
        # Some releases place a generically named README inside the archive.
        matches = sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file()
            and path.name.lower() in {"readme", "readme.txt"}
            and "jpsc1400" in path.read_text(encoding="utf-8", errors="ignore").lower()
        )
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "none"
        raise ValueError(
            "expected one JPSC1400 README containing the redistribution terms; "
            f"found: {rendered}. Use --jpsc-license-file to disambiguate."
        )
    return matches[0]


def _read_exact_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"expected UTF-8 text file: {path}") from exc


def _jpsc_revision(readme_text: str, override: str | None) -> str:
    if override:
        return override
    match = re.search(r"\bRel\.\s*(\d{8})\b", readme_text)
    if match:
        return match.group(1)
    return JPSC_DEFAULT_REVISION


def _resolve_jpsc_image(
    image_index: Mapping[str, Sequence[Path]], image_id: str, context: str
) -> Path:
    exact_names = [f"{image_id}.png"]
    if image_id.lower().endswith(".png"):
        exact_names.insert(0, image_id)
    exact: list[Path] = []
    for name in exact_names:
        exact.extend(image_index.get(name, ()))
    if exact:
        return _select_unique_image(sorted(set(exact)), context)

    # Be tolerant of numeric IDs whose file names use zero padding.
    normalized_id = image_id[:-4] if image_id.lower().endswith(".png") else image_id
    if normalized_id.isdigit():
        numeric_matches = [
            path
            for name, paths in image_index.items()
            if Path(name).stem.isdigit()
            and int(Path(name).stem) == int(normalized_id)
            for path in paths
        ]
        if numeric_matches:
            return _select_unique_image(sorted(set(numeric_matches)), context)
    raise FileNotFoundError(f"{context}: no PNG image for id {image_id!r}")


def build_jpsc_manifest(
    *,
    root: Path,
    output_path: Path,
    license_file: Path | None,
    revision_override: str | None,
    expected_count: int,
) -> dict[str, Any]:
    root = root.resolve()
    label_path = _unique_marker(root, "label.txt")
    readme_path = _find_jpsc_readme(root, license_file)
    readme_text = _read_exact_text(readme_path)
    if "without restriction and free of charge" not in readme_text:
        raise ValueError(
            f"{readme_path} does not contain the expected JPSC1400 permission notice"
        )
    revision = _jpsc_revision(readme_text, revision_override)
    image_index = _index_images(root, suffixes={".png"})
    label_sha256 = sha256_file(label_path)
    license_sha256 = sha256_file(readme_path)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        _read_exact_text(label_path).splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        fields = raw_line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(
                f"{label_path}:{line_number}: expected image_id, character_code, character"
            )
        image_id, character_code, ground_truth = fields
        if not ground_truth:
            raise ValueError(f"{label_path}:{line_number}: empty ground-truth character")
        if image_id in seen_ids:
            raise ValueError(f"{label_path}:{line_number}: duplicate id {image_id!r}")
        seen_ids.add(image_id)
        context = f"{label_path}:{line_number}"
        image_path = _resolve_jpsc_image(image_index, image_id, context)
        image_sha256 = sha256_file(image_path)
        uid = hashlib.sha256(
            f"{JPSC_SOURCE}\0{revision}\0{image_id}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "uid": uid,
                "path": str(image_path),
                "relative_path": _relative_or_absolute(image_path, root),
                "task": "ocr_character",
                "ground_truth": ground_truth,
                "source_id": image_id,
                "character_code": character_code,
                "orientation": "unspecified",
                "source": JPSC_SOURCE,
                "source_url": JPSC_SOURCE_URL,
                "source_revision": revision,
                "source_label_path": _relative_or_absolute(label_path, root),
                "source_label_line": line_number,
                "source_label_sha256": label_sha256,
                "license": JPSC_LICENSE_NAME,
                "license_url": JPSC_LICENSE_URL,
                "license_text": readme_text,
                "license_file": _relative_or_absolute(readme_path, root),
                "license_sha256": license_sha256,
                "image_sha256": image_sha256,
                "image_num_bytes": image_path.stat().st_size,
            }
        )

    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"JPSC1400 is incomplete: expected {expected_count} records, found {len(records)}"
        )
    atomic_write_jsonl(output_path, records)
    return {
        "records": len(records),
        "revision": revision,
        "manifest": str(output_path),
    }


def _is_ndl_header(fields: Sequence[str]) -> bool:
    if not fields:
        return False
    first = fields[0].strip().lower()
    return first in {"filename", "file_name", "ファイル名"}


def _orientation_in_path(path: Path, orientation: str) -> bool:
    expected = f"oneline_{orientation}"
    return any(part.lower() == expected for part in path.parts)


def _resolve_ndl_image(
    image_index: Mapping[str, Sequence[Path]],
    filename: str,
    orientation: str,
    context: str,
) -> Path:
    candidates = list(image_index.get(filename, ()))
    oriented = [path for path in candidates if _orientation_in_path(path, orientation)]
    if oriented:
        candidates = oriented
    return _select_unique_image(candidates, context)


def build_ndl_manifest(
    *, root: Path, output_path: Path, revision: str | None, expected_count: int
) -> dict[str, Any]:
    root = root.resolve()
    label_path = _unique_marker(root, "labeldata_pdm.tsv")
    label_sha256 = sha256_file(label_path)
    source_revision = revision or f"labeldata-sha256:{label_sha256}"
    image_index = _index_images(root)

    records: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_number, fields in enumerate(reader, start=1):
            if not fields or (len(fields) == 1 and not fields[0].strip()):
                continue
            if line_number == 1 and _is_ndl_header(fields):
                continue
            if len(fields) != 4:
                raise ValueError(
                    f"{label_path}:{line_number}: expected exactly four TSV fields, "
                    f"found {len(fields)}"
                )
            filename, ground_truth, orientation, iiif_url = fields
            if not filename:
                raise ValueError(f"{label_path}:{line_number}: empty filename")
            if not ground_truth:
                raise ValueError(f"{label_path}:{line_number}: empty ground truth")
            if orientation not in {"tate", "yoko"}:
                raise ValueError(
                    f"{label_path}:{line_number}: orientation must be tate or yoko, "
                    f"got {orientation!r}"
                )
            if not iiif_url.startswith(("https://", "http://")):
                raise ValueError(f"{label_path}:{line_number}: invalid IIIF URL {iiif_url!r}")
            if filename in seen_filenames:
                raise ValueError(
                    f"{label_path}:{line_number}: duplicate filename {filename!r}"
                )
            seen_filenames.add(filename)

            context = f"{label_path}:{line_number}"
            image_path = _resolve_ndl_image(
                image_index, filename, orientation, context
            )
            image_sha256 = sha256_file(image_path)
            uid = hashlib.sha256(
                f"{NDL_SOURCE}\0{source_revision}\0{filename}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "uid": uid,
                    "path": str(image_path),
                    "relative_path": _relative_or_absolute(image_path, root),
                    "task": "ocr_line",
                    "ground_truth": ground_truth,
                    "source_id": filename,
                    "orientation": orientation,
                    "orientation_ja": "縦書き" if orientation == "tate" else "横書き",
                    "iiif_url": iiif_url,
                    "source": NDL_SOURCE,
                    "source_url": NDL_SOURCE_URL,
                    "dataset_url": NDL_DATASET_URL,
                    "source_revision": source_revision,
                    "source_label_path": _relative_or_absolute(label_path, root),
                    "source_label_line": line_number,
                    "source_label_sha256": label_sha256,
                    "license": NDL_LICENSE_NAME,
                    "license_url": NDL_LICENSE_URL,
                    "image_sha256": image_sha256,
                    "image_num_bytes": image_path.stat().st_size,
                }
            )

    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"NDL Oneline is incomplete: expected {expected_count} records, "
            f"found {len(records)}"
        )
    atomic_write_jsonl(output_path, records)
    return {
        "records": len(records),
        "revision": source_revision,
        "manifest": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpsc-root", type=Path, default=None)
    parser.add_argument("--ndl-root", type=Path, default=None)
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Root to search for official marker files when explicit roots are omitted",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpsc-license-file", type=Path, default=None)
    parser.add_argument("--jpsc-revision", default=None)
    parser.add_argument("--ndl-revision", default=None)
    parser.add_argument(
        "--expected-jpsc-count",
        type=int,
        default=JPSC_DEFAULT_COUNT,
        help="Expected official row count; set 0 to disable",
    )
    parser.add_argument(
        "--expected-ndl-count",
        type=int,
        default=NDL_DEFAULT_COUNT,
        help="Expected official row count; set 0 to disable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_jpsc_count < 0 or args.expected_ndl_count < 0:
        raise ValueError("expected counts must be non-negative")

    search_roots = [path.resolve() for path in args.search_root]
    jpsc_root = args.jpsc_root.resolve() if args.jpsc_root else None
    ndl_root = args.ndl_root.resolve() if args.ndl_root else None
    if jpsc_root is None and search_roots:
        try:
            jpsc_root = discover_jpsc_root(search_roots)
        except ValueError as exc:
            print(f"JPSC1400 discovery skipped: {exc}", file=sys.stderr)
    if ndl_root is None and search_roots:
        try:
            ndl_root = discover_ndl_root(search_roots)
        except ValueError as exc:
            print(f"NDL Oneline discovery skipped: {exc}", file=sys.stderr)
    if jpsc_root is None and ndl_root is None:
        raise ValueError(
            "no dataset selected; pass --jpsc-root/--ndl-root or a --search-root "
            "containing at least one dataset"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if jpsc_root is not None:
        stats = build_jpsc_manifest(
            root=jpsc_root,
            output_path=output_dir / "metadata_jpsc1400.jsonl",
            license_file=args.jpsc_license_file,
            revision_override=args.jpsc_revision,
            expected_count=args.expected_jpsc_count,
        )
        print(
            f"JPSC1400: records={stats['records']} revision={stats['revision']} "
            f"manifest={stats['manifest']}"
        )
    if ndl_root is not None:
        stats = build_ndl_manifest(
            root=ndl_root,
            output_path=output_dir / "metadata_ndl_oneline.jsonl",
            revision=args.ndl_revision,
            expected_count=args.expected_ndl_count,
        )
        print(
            f"NDL Oneline: records={stats['records']} revision={stats['revision']} "
            f"manifest={stats['manifest']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
