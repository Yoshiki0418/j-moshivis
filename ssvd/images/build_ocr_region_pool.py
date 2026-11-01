#!/usr/bin/env python3
"""Retired historical utility for the discarded NDL/CODH OCR sources.

This utility is not part of the active v4 data pipeline as of 2026-09-28.
The active OCR source is JSSODa only; keep this file solely to preserve the
provenance of the earlier experiment.

Build a dialogue-free, 448 px Japanese OCR region image pool.

The builder consumes the already downloaded NDL PDM part 1, NDL PDM part 2,
and CODH Modern Magazine archives.  It crops only annotation-backed text
regions and writes a provenance-rich JSONL manifest.  It deliberately does
not generate questions, answers, audio, or dialogue turns.

The command is safe with respect to the source datasets: source files are
read-only and existing output images are never overwritten with different
bytes.  An interrupted build can be continued with ``--resume``.

Example::

    python ssvd/images/build_ocr_region_pool.py

    python ssvd/images/build_ocr_region_pool.py --resume

    python ssvd/images/build_ocr_region_pool.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = "jmoshivis.ocr-region-pool.v1"
RECIPE_VERSION = "jmoshivis.ocr-region-crop.448.v1"
SOURCE_REGISTRY_VERSION = "jmoshivis.ocr-source-registry.v1"

DEFAULT_PDM1_ROOT = Path(
    "data/images/ocr/ndl_pdm_part1/extracted/tosho_all_pascalvoc1.1"
)
DEFAULT_PDM2_ROOT = Path("data/images/ocr/ndl_pdm_part2/extracted")
DEFAULT_CODH_ROOT = Path(
    "data/images/ocr/codh_modern_magazine/extracted/mm-ocr-dataset-v1"
)
DEFAULT_OUTPUT = Path("data/images/targeted_v4/ocr_region_pool_v1")

PDM1_SOURCE_ID = "ndl-pdm-ocr-part1"
PDM2_SOURCE_ID = "ndl-pdm-ocr-part2"
CODH_SOURCE_ID = "codh-modern-magazine-ocr-v1"

PDM1_SOURCE_URL = "https://github.com/ndl-lab/pdmocrdataset-part1"
PDM2_SOURCE_URL = "https://github.com/ndl-lab/pdmocrdataset-part2"
CODH_SOURCE_URL = "https://codh.rois.ac.jp/modern-magazine/dataset/"

PDM_LICENSE = {
    "name": "Public Domain Mark 1.0",
    "url": "https://creativecommons.org/publicdomain/mark/1.0/",
    "attribution": "国立国会図書館",
    "modification_notice": (
        "The original page was cropped, padded, and resized by the "
        "J-MoshiVis project; the source institution did not create this crop."
    ),
}
CODH_LICENSE = {
    "name": "CC BY 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "『近代雑誌OCR学習用データセット』（CODH・国語研作成）",
    "modification_notice": (
        "The original page was cropped, padded, and resized by the "
        "J-MoshiVis project."
    ),
}

# These immutable archive hashes were recorded by
# data/images/targeted_download_logs/ocr_real_archives.sha256.
KNOWN_ARCHIVE_SHA256 = {
    PDM1_SOURCE_ID: "860690f1ac1169bdd3a2e04a99593ee0418792a3945b3e0e6b1b99bc3aacd855",
    PDM2_SOURCE_ID: "d26638513c29c9345559b07e0723b2ef5217a16203ae5f162859815982c17cd0",
    CODH_SOURCE_ID: "47875eccf3d7885fc3a7cfd4032e67dde62de29b4ede9faa5923206f16ca3417",
}


@dataclass(frozen=True)
class CropRecipe:
    version: str = RECIPE_VERSION
    target_size: int = 448
    inner_size: int = 416
    min_chars: int = 2
    max_chars: int = 24
    min_cross_axis_px: float = 12.0
    margin_ratio: float = 0.08
    min_margin_px: int = 4
    max_regions_per_page: int = 4
    interpolation: str = "bicubic"
    preserve_aspect_ratio: bool = True
    pad_rgb: tuple[int, int, int] = (255, 255, 255)
    output_format: str = "PNG"
    rotate: bool = False

    def validate(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if not 0 < self.inner_size <= self.target_size:
            raise ValueError("inner_size must be in (0, target_size]")
        if self.min_chars <= 0 or self.max_chars < self.min_chars:
            raise ValueError("invalid character range")
        if self.min_cross_axis_px <= 0:
            raise ValueError("min_cross_axis_px must be positive")
        if self.margin_ratio < 0 or self.min_margin_px < 0:
            raise ValueError("crop margins must be non-negative")
        if self.max_regions_per_page <= 0:
            raise ValueError("max_regions_per_page must be positive")


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    dataset_name: str
    dataset_url: str
    source_revision: str
    archive_path: str
    archive_sha256: str
    license: Mapping[str, str]


@dataclass(frozen=True)
class Region:
    source_id: str
    work_id: str
    page_id: str
    region_index: int
    canonical_transcript: str
    bbox_xyxy: tuple[float, float, float, float]
    orientation: str
    raw_orientation: str
    annotation_width: int
    annotation_height: int
    source_image_path: Path
    source_annotation_path: Path
    source_annotation_sha256: str
    annotation_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Page:
    source_id: str
    work_id: str
    page_id: str
    source_image_path: Path
    annotation_width: int
    annotation_height: int
    regions: tuple[Region, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uid(kind: str, *parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(f"jmoshivis:{kind}:v1".encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


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


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _trim_transcript(text: str) -> str:
    # Keep all internal whitespace and source characters.  Only annotation
    # formatting at the outer boundary is removed.
    return text.strip()


def _normalized_transcript(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _orientation_from_box(width: float, height: float) -> str:
    return "horizontal" if width >= height else "vertical"


def _map_ndl_direction(raw: str, width: float, height: float) -> str:
    return {
        "横": "horizontal",
        "縦": "vertical",
        "右から左": "right_to_left",
    }.get(raw, _orientation_from_box(width, height))


def _part1_image_path(xml_path: Path) -> Path:
    parts = list(xml_path.parts)
    try:
        index = parts.index("Annotations")
    except ValueError as exc:
        raise ValueError(f"part1 XML is not below an Annotations directory: {xml_path}") from exc
    return Path(*parts[:index], "JPEGImages", *parts[index + 1 :]).with_suffix(".jpg")


def iter_pdm1_pages(root: Path) -> Iterator[Page]:
    xml_paths = sorted(root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no NDL PDM part1 XML files under {root}")
    for xml_path in xml_paths:
        xml_bytes = xml_path.read_bytes()
        annotation_sha = sha256_bytes(xml_bytes)
        document = ET.fromstring(xml_bytes)
        size = document.find("size")
        if size is None:
            raise ValueError(f"{xml_path}: missing size")
        annotation_width = int(float(size.findtext("width", "0")))
        annotation_height = int(float(size.findtext("height", "0")))
        image_path = _part1_image_path(xml_path)
        page_id = xml_path.stem
        work_id = page_id.split("_", 1)[0]
        regions: list[Region] = []
        for region_index, obj in enumerate(document.findall("object"), start=1):
            if (obj.findtext("name") or "") != "text":
                continue
            transcript = ""
            for attribute in obj.findall("./attributes/attribute"):
                if (attribute.findtext("name") or "") == "contents":
                    transcript = attribute.findtext("value") or ""
                    break
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            x1 = float(bbox.findtext("xmin", "nan"))
            y1 = float(bbox.findtext("ymin", "nan"))
            x2 = float(bbox.findtext("xmax", "nan"))
            y2 = float(bbox.findtext("ymax", "nan"))
            width, height = x2 - x1, y2 - y1
            orientation = _orientation_from_box(width, height)
            regions.append(
                Region(
                    source_id=PDM1_SOURCE_ID,
                    work_id=work_id,
                    page_id=page_id,
                    region_index=region_index,
                    canonical_transcript=_trim_transcript(transcript),
                    bbox_xyxy=(x1, y1, x2, y2),
                    orientation=orientation,
                    raw_orientation="inferred_from_bbox",
                    annotation_width=annotation_width,
                    annotation_height=annotation_height,
                    source_image_path=image_path,
                    source_annotation_path=xml_path,
                    source_annotation_sha256=annotation_sha,
                    annotation_metadata={"object_name": "text"},
                )
            )
        yield Page(
            source_id=PDM1_SOURCE_ID,
            work_id=work_id,
            page_id=page_id,
            source_image_path=image_path,
            annotation_width=annotation_width,
            annotation_height=annotation_height,
            regions=tuple(regions),
        )


def iter_pdm2_pages(root: Path) -> Iterator[Page]:
    xml_paths = sorted(root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no NDL PDM part2 XML files under {root}")
    for xml_path in xml_paths:
        xml_bytes = xml_path.read_bytes()
        annotation_sha = sha256_bytes(xml_bytes)
        document = ET.fromstring(xml_bytes)
        work_id = xml_path.stem
        image_dir = xml_path.parent.parent / "img"
        for page in document.iter():
            if _local_name(page) != "PAGE":
                continue
            image_name = page.attrib.get("IMAGENAME", "").strip()
            if not image_name:
                raise ValueError(f"{xml_path}: PAGE is missing IMAGENAME")
            annotation_width = int(float(page.attrib.get("WIDTH", "0")))
            annotation_height = int(float(page.attrib.get("HEIGHT", "0")))
            page_id = f"{work_id}/{Path(image_name).stem}"
            image_path = image_dir / image_name
            regions: list[Region] = []
            region_index = 0
            for child in page:
                if _local_name(child) != "LINE":
                    continue
                region_index += 1
                x = float(child.attrib.get("X", "nan"))
                y = float(child.attrib.get("Y", "nan"))
                width = float(child.attrib.get("WIDTH", "nan"))
                height = float(child.attrib.get("HEIGHT", "nan"))
                raw_direction = child.attrib.get("DIRECTION", "")
                regions.append(
                    Region(
                        source_id=PDM2_SOURCE_ID,
                        work_id=work_id,
                        page_id=page_id,
                        region_index=region_index,
                        canonical_transcript=_trim_transcript(
                            child.attrib.get("STRING", "")
                        ),
                        bbox_xyxy=(x, y, x + width, y + height),
                        orientation=_map_ndl_direction(
                            raw_direction, width, height
                        ),
                        raw_orientation=raw_direction or "inferred_from_bbox",
                        annotation_width=annotation_width,
                        annotation_height=annotation_height,
                        source_image_path=image_path,
                        source_annotation_path=xml_path,
                        source_annotation_sha256=annotation_sha,
                        annotation_metadata={
                            "line_type": child.attrib.get("TYPE"),
                            "title": child.attrib.get("TITLE"),
                            "author": child.attrib.get("AUTHOR"),
                        },
                    )
                )
            yield Page(
                source_id=PDM2_SOURCE_ID,
                work_id=work_id,
                page_id=page_id,
                source_image_path=image_path,
                annotation_width=annotation_width,
                annotation_height=annotation_height,
                regions=tuple(regions),
            )


def iter_codh_pages(root: Path) -> Iterator[Page]:
    xml_root = root / "xml"
    image_root = root / "img"
    xml_paths = sorted(xml_root.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no CODH XML files under {xml_root}")
    for xml_path in xml_paths:
        xml_bytes = xml_path.read_bytes()
        annotation_sha = sha256_bytes(xml_bytes)
        document = ET.fromstring(xml_bytes)
        issue_id = xml_path.stem.rsplit("-", 1)[0]
        for page in document.iter():
            if _local_name(page) != "PAGE":
                continue
            image_name = page.attrib.get("IMAGENAME", "").strip()
            if not image_name:
                raise ValueError(f"{xml_path}: PAGE is missing IMAGENAME")
            annotation_width = int(float(page.attrib.get("WIDTH", "0")))
            annotation_height = int(float(page.attrib.get("HEIGHT", "0")))
            page_id = Path(image_name).stem
            image_path = image_root / image_name
            regions: list[Region] = []
            region_index = 0
            for child in page:
                if _local_name(child) != "LINE":
                    continue
                region_index += 1
                x = float(child.attrib.get("X", "nan"))
                y = float(child.attrib.get("Y", "nan"))
                width = float(child.attrib.get("WIDTH", "nan"))
                height = float(child.attrib.get("HEIGHT", "nan"))
                regions.append(
                    Region(
                        source_id=CODH_SOURCE_ID,
                        work_id=issue_id,
                        page_id=page_id,
                        region_index=region_index,
                        canonical_transcript=_trim_transcript(
                            child.attrib.get("STRING", "")
                        ),
                        bbox_xyxy=(x, y, x + width, y + height),
                        orientation=_orientation_from_box(width, height),
                        raw_orientation="inferred_from_bbox",
                        annotation_width=annotation_width,
                        annotation_height=annotation_height,
                        source_image_path=image_path,
                        source_annotation_path=xml_path,
                        source_annotation_sha256=annotation_sha,
                        annotation_metadata={},
                    )
                )
            yield Page(
                source_id=CODH_SOURCE_ID,
                work_id=issue_id,
                page_id=page_id,
                source_image_path=image_path,
                annotation_width=annotation_width,
                annotation_height=annotation_height,
                regions=tuple(regions),
            )


def rejection_reason(region: Region, recipe: CropRecipe) -> str | None:
    text = region.canonical_transcript
    if not text:
        return "empty_transcript"
    if "〓" in text:
        return "replacement_character"
    if len(text) < recipe.min_chars:
        return "too_short"
    if len(text) > recipe.max_chars:
        return "too_long"
    x1, y1, x2, y2 = region.bbox_xyxy
    if not all(math.isfinite(value) for value in region.bbox_xyxy):
        return "nonfinite_bbox"
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return "invalid_bbox"
    cross_axis = height if region.orientation == "horizontal" else width
    if cross_axis < recipe.min_cross_axis_px:
        return "cross_axis_too_small"
    if region.annotation_width <= 0 or region.annotation_height <= 0:
        return "invalid_annotation_dimensions"
    if x2 <= 0 or y2 <= 0 or x1 >= region.annotation_width or y1 >= region.annotation_height:
        return "bbox_outside_annotation"
    return None


def _quality_key(region: Region) -> tuple[float, float, float, int]:
    x1, y1, x2, y2 = region.bbox_xyxy
    width, height = x2 - x1, y2 - y1
    cross_axis = height if region.orientation == "horizontal" else width
    # Prefer crisp regions, then speech-friendly medium-length strings.
    return (-cross_axis, abs(len(region.canonical_transcript) - 12), -(width * height), region.region_index)


def prioritize_regions(regions: Sequence[Region], limit: int) -> list[Region]:
    """Return a deterministic queue with horizontal priority and vertical retention.

    The first ``limit`` entries contain at least one non-horizontal region when
    both orientations are available (for the default limit of four), while up
    to three high-quality horizontal lines are considered first.
    """

    horizontal = sorted(
        (region for region in regions if region.orientation == "horizontal"),
        key=_quality_key,
    )
    other = sorted(
        (region for region in regions if region.orientation != "horizontal"),
        key=lambda region: (
            0 if region.orientation == "vertical" else 1,
            *_quality_key(region),
        ),
    )
    if not horizontal:
        return other
    if not other or limit <= 1:
        return horizontal + other

    horizontal_head = horizontal[: max(1, limit - 1)]
    reserved_other = other[:1]
    selected_ids = {id(region) for region in horizontal_head + reserved_other}
    remaining = sorted(
        (region for region in regions if id(region) not in selected_ids),
        key=lambda region: (
            0 if region.orientation == "horizontal" else 1,
            *_quality_key(region),
        ),
    )
    return horizontal_head + reserved_other + remaining


def _scaled_bbox(region: Region, actual_width: int, actual_height: int) -> tuple[float, float, float, float]:
    scale_x = actual_width / region.annotation_width
    scale_y = actual_height / region.annotation_height
    x1, y1, x2, y2 = region.bbox_xyxy
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def crop_region(
    source_image: Image.Image, region: Region, recipe: CropRecipe
) -> tuple[Image.Image, dict[str, Any]]:
    actual_width, actual_height = source_image.size
    x1f, y1f, x2f, y2f = _scaled_bbox(region, actual_width, actual_height)
    if not all(math.isfinite(value) for value in (x1f, y1f, x2f, y2f)):
        raise ValueError(f"{region.page_id}:{region.region_index}: non-finite scaled bbox")
    if x2f <= x1f or y2f <= y1f:
        raise ValueError(f"{region.page_id}:{region.region_index}: invalid scaled bbox")

    width, height = x2f - x1f, y2f - y1f
    margin_x = max(recipe.min_margin_px, int(round(width * recipe.margin_ratio)))
    margin_y = max(recipe.min_margin_px, int(round(height * recipe.margin_ratio)))
    left = max(0, int(math.floor(x1f)) - margin_x)
    top = max(0, int(math.floor(y1f)) - margin_y)
    right = min(actual_width, int(math.ceil(x2f)) + margin_x)
    bottom = min(actual_height, int(math.ceil(y2f)) + margin_y)
    if right <= left or bottom <= top:
        raise ValueError(f"{region.page_id}:{region.region_index}: empty clamped crop")

    raw_crop = source_image.crop((left, top, right, bottom)).convert("RGB")
    scale = min(recipe.inner_size / raw_crop.width, recipe.inner_size / raw_crop.height)
    resized_width = max(1, min(recipe.inner_size, int(round(raw_crop.width * scale))))
    resized_height = max(1, min(recipe.inner_size, int(round(raw_crop.height * scale))))
    resized = raw_crop.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (recipe.target_size, recipe.target_size), recipe.pad_rgb)
    paste_x = (recipe.target_size - resized_width) // 2
    paste_y = (recipe.target_size - resized_height) // 2
    canvas.paste(resized, (paste_x, paste_y))
    metadata = {
        "annotation_bbox_xyxy": [round(value, 4) for value in region.bbox_xyxy],
        "annotation_dimensions": [region.annotation_width, region.annotation_height],
        "source_image_dimensions": [actual_width, actual_height],
        "coordinate_scale_xy": [actual_width / region.annotation_width, actual_height / region.annotation_height],
        "scaled_bbox_xyxy": [round(value, 4) for value in (x1f, y1f, x2f, y2f)],
        "crop_bbox_xyxy": [left, top, right, bottom],
        "raw_crop_dimensions": [raw_crop.width, raw_crop.height],
        "resized_dimensions": [resized_width, resized_height],
        "paste_xy": [paste_x, paste_y],
    }
    return canvas, metadata


def encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue()


def _path_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _split_group_key(source_id: str, work_id: str) -> str:
    if source_id in {PDM1_SOURCE_ID, PDM2_SOURCE_ID}:
        return f"ndl:pid:{work_id}"
    if source_id == CODH_SOURCE_ID:
        return f"codh:issue:{work_id}"
    raise ValueError(f"unknown source_id: {source_id}")


def _source_registry(
    *, pdm1_root: Path, pdm2_root: Path, codh_root: Path, repo_root: Path
) -> dict[str, SourceInfo]:
    pdm1_revision_path = pdm1_root.parents[1] / "source_revision.txt"
    pdm2_revision_path = pdm2_root.parent / "source_revision.txt"
    pdm1_archive = pdm1_root.parents[1] / "tosho_all_pascalvoc1.1.zip"
    pdm2_archive = pdm2_root.parent / "ndlocr_xml.zip"
    codh_archive = codh_root.parents[1] / "v1.zip"

    for path in (
        pdm1_revision_path,
        pdm2_revision_path,
        pdm1_archive,
        pdm2_archive,
        codh_archive,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"source provenance artifact not found: {path}")
    pdm1_revision = pdm1_revision_path.read_text(encoding="utf-8").strip()
    pdm2_revision = pdm2_revision_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", pdm1_revision):
        raise ValueError(f"invalid part1 source revision: {pdm1_revision!r}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", pdm2_revision):
        raise ValueError(f"invalid part2 source revision: {pdm2_revision!r}")
    return {
        PDM1_SOURCE_ID: SourceInfo(
            source_id=PDM1_SOURCE_ID,
            dataset_name="NDL public-domain OCR dataset part 1",
            dataset_url=PDM1_SOURCE_URL,
            source_revision=pdm1_revision,
            archive_path=_path_for_manifest(pdm1_archive, repo_root),
            archive_sha256=KNOWN_ARCHIVE_SHA256[PDM1_SOURCE_ID],
            license=PDM_LICENSE,
        ),
        PDM2_SOURCE_ID: SourceInfo(
            source_id=PDM2_SOURCE_ID,
            dataset_name="NDL public-domain OCR dataset part 2",
            dataset_url=PDM2_SOURCE_URL,
            source_revision=pdm2_revision,
            archive_path=_path_for_manifest(pdm2_archive, repo_root),
            archive_sha256=KNOWN_ARCHIVE_SHA256[PDM2_SOURCE_ID],
            license=PDM_LICENSE,
        ),
        CODH_SOURCE_ID: SourceInfo(
            source_id=CODH_SOURCE_ID,
            dataset_name="CODH Modern Magazine OCR Dataset v1",
            dataset_url=CODH_SOURCE_URL,
            source_revision="v1",
            archive_path=_path_for_manifest(codh_archive, repo_root),
            archive_sha256=KNOWN_ARCHIVE_SHA256[CODH_SOURCE_ID],
            license=CODH_LICENSE,
        ),
    }


def _ensure_output_safety(output_dir: Path, resume: bool) -> None:
    if not output_dir.exists():
        return
    entries = list(output_dir.iterdir())
    if entries and not resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --resume to verify/reuse "
            "matching derived files. Existing files are never replaced with different bytes."
        )


def build_pool(
    *,
    pages: Iterable[Page],
    sources: Mapping[str, SourceInfo],
    output_dir: Path,
    repo_root: Path,
    recipe: CropRecipe,
    resume: bool,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recipe.validate()
    _ensure_output_safety(output_dir, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_orientation: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    output_pixel_hashes: set[str] = set()
    output_uids: set[str] = set()
    records: list[dict[str, Any]] = []

    for page_number, page in enumerate(pages, start=1):
        counters["pages_seen"] += 1
        counters["regions_seen"] += len(page.regions)
        eligible: list[Region] = []
        for region in page.regions:
            reason = rejection_reason(region, recipe)
            if reason is None:
                eligible.append(region)
            else:
                rejection_counts[reason] += 1
        counters["eligible_regions"] += len(eligible)
        if not eligible:
            continue
        if not page.source_image_path.is_file():
            raise FileNotFoundError(
                f"{page.source_id}:{page.page_id}: source image not found: "
                f"{page.source_image_path}"
            )

        source_bytes = page.source_image_path.read_bytes()
        source_file_sha = sha256_bytes(source_bytes)
        with Image.open(io.BytesIO(source_bytes)) as opened:
            opened.load()
            source_image = opened.convert("RGB")
        parent_image_uid = stable_uid(
            "ocr-parent-image", page.source_id, source_file_sha
        )
        source_page_uid = stable_uid(
            "ocr-source-page", page.source_id, page.work_id, page.page_id
        )
        source_work_uid = stable_uid(
            "ocr-source-work", _split_group_key(page.source_id, page.work_id)
        )
        selected_for_page = 0
        for region in prioritize_regions(eligible, recipe.max_regions_per_page):
            if selected_for_page >= recipe.max_regions_per_page:
                break
            canvas, crop_metadata = crop_region(source_image, region, recipe)
            output_pixel_sha = pixel_sha256(canvas)
            if output_pixel_sha in output_pixel_hashes:
                rejection_counts["duplicate_output_pixels"] += 1
                continue
            source_region_uid = stable_uid(
                "ocr-source-region",
                page.source_id,
                page.work_id,
                page.page_id,
                region.region_index,
                *region.bbox_xyxy,
                region.canonical_transcript,
            )
            image_uid = stable_uid(
                "ocr-region-crop", source_region_uid, recipe.version
            )
            if image_uid in output_uids:
                raise RuntimeError(f"duplicate image UID: {image_uid}")
            png_bytes = encode_png(canvas)
            output_file_sha = sha256_bytes(png_bytes)
            relative_image_path = Path(
                "images", page.source_id, image_uid[:2], f"{image_uid}.png"
            )
            output_path = output_dir / relative_image_path
            if output_path.exists():
                existing_sha = sha256_file(output_path)
                if existing_sha != output_file_sha:
                    raise RuntimeError(
                        f"refusing to overwrite a different existing crop: {output_path}; "
                        f"found {existing_sha}, expected {output_file_sha}"
                    )
                counters["images_reused"] += 1
            else:
                atomic_write_bytes(output_path, png_bytes)
                counters["images_written"] += 1

            source = sources[page.source_id]
            transcript_sha = sha256_bytes(
                region.canonical_transcript.encode("utf-8")
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "uid": image_uid,
                "image_uid": image_uid,
                "image_path": relative_image_path.as_posix(),
                "image_mime_type": "image/png",
                "image_num_bytes": len(png_bytes),
                "image_file_sha256": output_file_sha,
                "image_pixel_sha256": output_pixel_sha,
                "image_width": recipe.target_size,
                "image_height": recipe.target_size,
                "task": "ocr_region_image_pool",
                "canonical_transcript": region.canonical_transcript,
                "normalized_transcript": _normalized_transcript(
                    region.canonical_transcript
                ),
                "canonical_transcript_sha256": transcript_sha,
                "character_count": len(region.canonical_transcript),
                "transcript_provenance": "source_annotation",
                "orientation": region.orientation,
                "raw_orientation": region.raw_orientation,
                "source": {
                    "dataset_id": source.source_id,
                    "dataset_name": source.dataset_name,
                    "dataset_url": source.dataset_url,
                    "source_revision": source.source_revision,
                    "archive_path": source.archive_path,
                    "archive_sha256": source.archive_sha256,
                    "work_id": page.work_id,
                    "page_id": page.page_id,
                    "region_index": region.region_index,
                    "source_work_uid": source_work_uid,
                    "source_page_uid": source_page_uid,
                    "source_region_uid": source_region_uid,
                    "parent_image_uid": parent_image_uid,
                    "source_image_path": _path_for_manifest(
                        page.source_image_path, repo_root
                    ),
                    "source_image_file_sha256": source_file_sha,
                    "source_annotation_path": _path_for_manifest(
                        region.source_annotation_path, repo_root
                    ),
                    "source_annotation_sha256": region.source_annotation_sha256,
                    "annotation_metadata": dict(region.annotation_metadata),
                },
                "geometry": crop_metadata,
                "crop_recipe": asdict(recipe),
                "license": dict(source.license),
                "split": None,
                "split_status": "pending",
                "split_group_key": _split_group_key(
                    page.source_id, page.work_id
                ),
                "overlap_audit": {
                    "status": "pending_post_crop_audit",
                    "targets": ["v3", "frozen_vision_eval"],
                },
                "dialogue": {
                    "status": "pending",
                    "turn_count": None,
                    "generator": None,
                },
            }
            records.append(record)
            output_pixel_hashes.add(output_pixel_sha)
            output_uids.add(image_uid)
            selected_for_page += 1
            by_source[page.source_id] += 1
            by_orientation[region.orientation] += 1
        if selected_for_page:
            counters["pages_selected"] += 1
        counters["selected_regions"] += selected_for_page
        counters["eligible_not_selected_by_page_cap"] += max(
            0, len(eligible) - selected_for_page
        )
        if progress_every > 0 and page_number % progress_every == 0:
            print(
                f"pages={page_number:,} selected={len(records):,} "
                f"written={counters['images_written']:,} reused={counters['images_reused']:,}",
                flush=True,
            )

    records.sort(
        key=lambda row: (
            row["source"]["dataset_id"],
            row["source"]["work_id"],
            row["source"]["page_id"],
            row["source"]["region_index"],
        )
    )
    summary = {
        "counts": dict(sorted(counters.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_by_source": dict(sorted(by_source.items())),
        "selected_by_orientation": dict(sorted(by_orientation.items())),
    }
    return records, summary


def validate_records(
    records: Sequence[Mapping[str, Any]], output_dir: Path, recipe: CropRecipe
) -> dict[str, Any]:
    seen_uids: set[str] = set()
    seen_paths: set[str] = set()
    seen_pixel_hashes: set[str] = set()
    page_counts: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_orientation: Counter[str] = Counter()
    total_bytes = 0

    for line_number, record in enumerate(records, start=1):
        uid = str(record.get("uid", ""))
        image_path_value = str(record.get("image_path", ""))
        if not uid or uid in seen_uids:
            raise ValueError(f"record {line_number}: missing/duplicate UID {uid!r}")
        if not image_path_value or image_path_value in seen_paths:
            raise ValueError(
                f"record {line_number}: missing/duplicate image path {image_path_value!r}"
            )
        transcript = record.get("canonical_transcript")
        if not isinstance(transcript, str) or not (
            recipe.min_chars <= len(transcript) <= recipe.max_chars
        ):
            raise ValueError(f"record {line_number}: invalid transcript length")
        if not transcript.strip() or "〓" in transcript:
            raise ValueError(f"record {line_number}: forbidden transcript")
        dialogue = record.get("dialogue")
        if dialogue != {"status": "pending", "turn_count": None, "generator": None}:
            raise ValueError(f"record {line_number}: dialogue must remain pending/null")
        if record.get("split") is not None or record.get("split_status") != "pending":
            raise ValueError(f"record {line_number}: split must remain pending")

        source = record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"record {line_number}: missing source object")
        page_uid = str(source.get("source_page_uid", ""))
        page_counts[page_uid] += 1
        if page_counts[page_uid] > recipe.max_regions_per_page:
            raise ValueError(
                f"record {line_number}: page exceeds max regions: {page_uid}"
            )

        image_path = output_dir / image_path_value
        if not image_path.is_file():
            raise FileNotFoundError(f"record {line_number}: missing image {image_path}")
        payload = image_path.read_bytes()
        file_hash = sha256_bytes(payload)
        if file_hash != record.get("image_file_sha256"):
            raise ValueError(f"record {line_number}: output file hash mismatch")
        if len(payload) != record.get("image_num_bytes"):
            raise ValueError(f"record {line_number}: output byte count mismatch")
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (recipe.target_size, recipe.target_size):
                raise ValueError(
                    f"record {line_number}: unexpected image size {image.size}"
                )
            pixel_hash = pixel_sha256(image)
        if pixel_hash != record.get("image_pixel_sha256"):
            raise ValueError(f"record {line_number}: output pixel hash mismatch")
        if pixel_hash in seen_pixel_hashes:
            raise ValueError(f"record {line_number}: duplicate output pixels")

        seen_uids.add(uid)
        seen_paths.add(image_path_value)
        seen_pixel_hashes.add(pixel_hash)
        total_bytes += len(payload)
        by_source[str(source.get("dataset_id"))] += 1
        by_orientation[str(record.get("orientation"))] += 1

    return {
        "status": "passed",
        "records": len(records),
        "unique_uids": len(seen_uids),
        "unique_image_paths": len(seen_paths),
        "unique_pixel_hashes": len(seen_pixel_hashes),
        "unique_source_pages": len(page_counts),
        "max_regions_on_one_page": max(page_counts.values(), default=0),
        "image_bytes": total_bytes,
        "by_source": dict(sorted(by_source.items())),
        "by_orientation": dict(sorted(by_orientation.items())),
    }


def _source_registry_json(sources: Mapping[str, SourceInfo], repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_REGISTRY_VERSION,
        "sources": [asdict(sources[key]) for key in sorted(sources)],
        "provenance_note": (
            "Archive SHA-256 values are copied from the immutable download ledger; "
            "the source images retain their stated source rights."
        ),
        "repo_root": str(repo_root.resolve()),
    }


def _iter_all_pages(
    pdm1_root: Path, pdm2_root: Path, codh_root: Path
) -> Iterator[Page]:
    yield from iter_pdm1_pages(pdm1_root)
    yield from iter_pdm2_pages(pdm2_root)
    yield from iter_codh_pages(codh_root)


def _write_outputs(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    build_summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    recipe: CropRecipe,
    sources: Mapping[str, SourceInfo],
    repo_root: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "metadata.jsonl"
    registry_path = output_dir / "sources.json"
    stats_path = output_dir / "metadata.stats.json"
    atomic_write_jsonl(manifest_path, records)
    atomic_write_json(registry_path, _source_registry_json(sources, repo_root))
    stats = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": _path_for_manifest(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path),
        "source_registry": _path_for_manifest(registry_path, repo_root),
        "source_registry_sha256": sha256_file(registry_path),
        "crop_recipe": asdict(recipe),
        "build": dict(build_summary),
        "validation": dict(validation),
        "dialogue_policy": {
            "status": "pending",
            "dialogue_generated": False,
            "turn_count": None,
            "generator": None,
        },
        "split_policy": {
            "status": "pending",
            "rule": "Group NDL by PID and CODH by issue; never split region crops independently.",
        },
        "overlap_policy": {
            "status": "pending_post_crop_audit",
            "targets": ["v3", "frozen_vision_eval"],
            "fingerprints_available": ["file_sha256", "pixel_sha256"],
        },
    }
    atomic_write_json(stats_path, stats)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdm1-root", type=Path, default=DEFAULT_PDM1_ROOT)
    parser.add_argument("--pdm2-root", type=Path, default=DEFAULT_PDM2_ROOT)
    parser.add_argument("--codh-root", type=Path, default=DEFAULT_CODH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-size", type=int, default=448)
    parser.add_argument("--inner-size", type=int, default=416)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=24)
    parser.add_argument("--min-cross-axis-px", type=float, default=12.0)
    parser.add_argument("--margin-ratio", type=float, default=0.08)
    parser.add_argument("--min-margin-px", type=int, default=4)
    parser.add_argument("--max-regions-per-page", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only byte-identical existing derived crops.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _write_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (200, 120), "white")
    pixels = image.load()
    for x in range(20, 170):
        for y in range(20, 34):
            if (x + y) % 5:
                pixels[x, y] = (0, 0, 0)
    for x in range(175, 188):
        for y in range(15, 105):
            if (x + y) % 4:
                pixels[x, y] = (20, 20, 20)
    image.save(path, format="JPEG", quality=95)


def self_test() -> None:
    recipe = CropRecipe(target_size=448, inner_size=416, max_regions_per_page=4)
    with tempfile.TemporaryDirectory(prefix="jmv-ocr-region-selftest-") as temporary:
        root = Path(temporary)
        image_path = root / "page.jpg"
        annotation_path = root / "page.xml"
        _write_test_image(image_path)
        annotation_path.write_text("<annotation/>", encoding="utf-8")
        annotation_sha = sha256_file(annotation_path)
        regions = (
            Region(PDM1_SOURCE_ID, "123", "123_R1", 1, "横書きの文字", (20, 20, 170, 34), "horizontal", "inferred_from_bbox", 200, 120, image_path, annotation_path, annotation_sha, {}),
            Region(PDM1_SOURCE_ID, "123", "123_R1", 2, "縦書き文字", (175, 15, 188, 105), "vertical", "inferred_from_bbox", 200, 120, image_path, annotation_path, annotation_sha, {}),
            Region(PDM1_SOURCE_ID, "123", "123_R1", 3, "〓除外", (20, 45, 100, 60), "horizontal", "inferred_from_bbox", 200, 120, image_path, annotation_path, annotation_sha, {}),
            Region(PDM1_SOURCE_ID, "123", "123_R1", 4, "長" * 25, (20, 70, 170, 85), "horizontal", "inferred_from_bbox", 200, 120, image_path, annotation_path, annotation_sha, {}),
        )
        eligible = [region for region in regions if rejection_reason(region, recipe) is None]
        assert len(eligible) == 2
        queue = prioritize_regions(eligible, recipe.max_regions_per_page)
        assert queue[0].orientation == "horizontal"
        assert any(region.orientation == "vertical" for region in queue[:4])
        with Image.open(image_path) as image:
            crop, geometry = crop_region(image.convert("RGB"), queue[0], recipe)
        assert crop.size == (448, 448)
        assert geometry["crop_bbox_xyxy"][0] < 20
        assert len(encode_png(crop)) > 0

        # Exercise all three XML readers with their official directory shapes.
        p1 = root / "p1" / "tosho_1900_bunkei"
        p1_xml = p1 / "Annotations" / "pdmlinetrainimg" / "tosho_1900_bunkei" / "123_R1.xml"
        p1_img = p1 / "JPEGImages" / "pdmlinetrainimg" / "tosho_1900_bunkei" / "123_R1.jpg"
        _write_test_image(p1_img)
        p1_xml.parent.mkdir(parents=True, exist_ok=True)
        p1_xml.write_text(
            """<annotation><size><width>200</width><height>120</height></size>
            <object><name>text</name><bndbox><xmin>20</xmin><ymin>20</ymin><xmax>170</xmax><ymax>34</ymax></bndbox>
            <attributes><attribute><name>contents</name><value>横書き</value></attribute></attributes></object></annotation>""",
            encoding="utf-8",
        )
        assert next(iter_pdm1_pages(root / "p1")).regions[0].canonical_transcript == "横書き"

        p2 = root / "p2" / "1900" / "456"
        _write_test_image(p2 / "img" / "R1.jpg")
        (p2 / "xml").mkdir(parents=True, exist_ok=True)
        (p2 / "xml" / "456.xml").write_text(
            """<OCRDATASET><PAGE IMAGENAME="R1.jpg" WIDTH="200" HEIGHT="120">
            <LINE DIRECTION="縦" TYPE="本文" STRING="縦書き" X="175" Y="15" WIDTH="13" HEIGHT="90"/></PAGE></OCRDATASET>""",
            encoding="utf-8",
        )
        assert next(iter_pdm2_pages(root / "p2")).regions[0].orientation == "vertical"

        codh = root / "codh"
        _write_test_image(codh / "img" / "kmnt001-001.jpg")
        (codh / "xml").mkdir(parents=True, exist_ok=True)
        (codh / "xml" / "kmnt001-001.xml").write_text(
            """<OCRDATASET><PAGE IMAGENAME="kmnt001-001.jpg" WIDTH="200" HEIGHT="120">
            <LINE STRING="雑誌文字" X="175" Y="15" WIDTH="13" HEIGHT="90"/></PAGE></OCRDATASET>""",
            encoding="utf-8",
        )
        page = next(iter_codh_pages(codh))
        assert page.work_id == "kmnt001"
        assert page.regions[0].orientation == "vertical"
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    repo_root = args.repo_root.resolve()
    pdm1_root = args.pdm1_root.resolve()
    pdm2_root = args.pdm2_root.resolve()
    codh_root = args.codh_root.resolve()
    output_dir = args.output_dir.resolve()
    recipe = CropRecipe(
        target_size=args.target_size,
        inner_size=args.inner_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_cross_axis_px=args.min_cross_axis_px,
        margin_ratio=args.margin_ratio,
        min_margin_px=args.min_margin_px,
        max_regions_per_page=args.max_regions_per_page,
    )
    sources = _source_registry(
        pdm1_root=pdm1_root,
        pdm2_root=pdm2_root,
        codh_root=codh_root,
        repo_root=repo_root,
    )
    records, build_summary = build_pool(
        pages=_iter_all_pages(pdm1_root, pdm2_root, codh_root),
        sources=sources,
        output_dir=output_dir,
        repo_root=repo_root,
        recipe=recipe,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    validation = validate_records(records, output_dir, recipe)
    stats = _write_outputs(
        output_dir=output_dir,
        records=records,
        build_summary=build_summary,
        validation=validation,
        recipe=recipe,
        sources=sources,
        repo_root=repo_root,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
