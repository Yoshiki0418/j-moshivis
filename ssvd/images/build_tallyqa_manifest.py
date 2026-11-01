#!/usr/bin/env python3
"""Build a provenance-preserving manifest from the official TallyQA train split.

This is an indexer only: it never copies images.  TallyQA's COCO-2014 paths
are resolved against the locally available COCO train2017/val2017 images by
their stable numeric COCO image id; Visual Genome paths are resolved below a
Visual Genome root.

The official archive's ``train.json`` currently has no ``issimple`` field,
even though the field is described in the upstream README and is present in
``test.json``.  This script preserves that fact as null instead of inventing a
label.  A separate, conservative audit flag identifies unqualified COCO object
count questions which can overlap with a COCO-instances-derived task.  Passing
one or more ``--coco-instances-json`` files additionally verifies that flag
against the official instance count.

No Japanese translation is fabricated here.  Every row is explicitly marked
``pending`` and unsuitable for Japanese training until a later translation
and validation stage has filled the Japanese fields.

Example::

    python ssvd/images/build_tallyqa_manifest.py \
      --train-json data/images/counting/tallyqa/train.json \
      --coco-root data/images/coco \
      --visual-genome-root data/images/counting/visual_genome \
      --source-archive data/images/counting/tallyqa/tallyqa.zip \
      --output data/images/counting/tallyqa/metadata_train.jsonl \
      --exclude-manifest data/evaluation/vision_v1/manifest_clean.jsonl \
      --coco-instances-json \
        data/images/counting/coco2017/annotations/instances_train2017.json \
      --coco-instances-json \
        data/images/counting/coco2017/annotations/instances_val2017.json

Run the dependency-free synthetic test with ``--self-test``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "jmoshivis.tallyqa_index.v1"
TALLYQA_REPOSITORY = "https://github.com/manoja328/TallyQA_dataset"
TALLYQA_REVISION = "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f"
TALLYQA_ARCHIVE_URL = (
    f"https://raw.githubusercontent.com/manoja328/TallyQA_dataset/"
    f"{TALLYQA_REVISION}/tallyqa.zip"
)
TALLYQA_ANNOTATION_LICENSE = {
    "name": "Apache License 2.0",
    "spdx": "Apache-2.0",
    "url": f"{TALLYQA_REPOSITORY}/blob/{TALLYQA_REVISION}/LICENSE",
    "scope": "TallyQA repository and annotations; does not relicense images",
}
COCO_IMAGE_TERMS = {
    "name": "Original Flickr image terms (per image)",
    "url": "https://cocodataset.org/#termsofuse",
    "note": "TallyQA's Apache-2.0 license does not cover the source image.",
}
VISUAL_GENOME_IMAGE_TERMS = {
    "name": "Visual Genome source-image terms",
    "url": "https://homes.cs.washington.edu/~ranjay/visualgenome/api.html",
    "note": "TallyQA's Apache-2.0 license does not cover the source image.",
}

REQUIRED_FIELDS = {
    "image",
    "answer",
    "data_source",
    "question",
    "image_id",
    "question_id",
}
ALLOWED_IMAGE_PARENTS = {"train2014", "val2014", "VG_100K", "VG_100K_2"}
COCO_FILE_RE = re.compile(
    r"^COCO_(?P<split>train|val)2014_(?P<id>[0-9]{12})\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)


# Only strict, unqualified variants are included.  Subset questions such as
# "people wearing glasses" deliberately do not match.  Conservative false
# negatives are preferable because this field is for de-duplication auditing,
# not for assigning TallyQA's missing train-time ``issimple`` label.
COCO_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("people", "persons", "person"),
    "bicycle": ("bicycles", "bikes", "bicycle", "bike"),
    "car": ("cars", "car"),
    "motorcycle": ("motorcycles", "motorbikes", "motorcycle", "motorbike"),
    "airplane": ("airplanes", "aeroplanes", "planes", "airplane", "aeroplane", "plane"),
    "bus": ("buses", "bus"),
    "train": ("trains", "train"),
    "truck": ("trucks", "truck"),
    "boat": ("boats", "boat"),
    "traffic light": ("traffic lights", "traffic light"),
    "fire hydrant": ("fire hydrants", "fire hydrant"),
    "stop sign": ("stop signs", "stop sign"),
    "parking meter": ("parking meters", "parking meter"),
    "bench": ("benches", "bench"),
    "bird": ("birds", "bird"),
    "cat": ("cats", "cat"),
    "dog": ("dogs", "dog"),
    "horse": ("horses", "horse"),
    "sheep": ("sheep",),
    "cow": ("cows", "cow"),
    "elephant": ("elephants", "elephant"),
    "bear": ("bears", "bear"),
    "zebra": ("zebras", "zebra"),
    "giraffe": ("giraffes", "giraffe"),
    "backpack": ("backpacks", "backpack"),
    "umbrella": ("umbrellas", "umbrella"),
    "handbag": ("handbags", "handbag"),
    "tie": ("ties", "tie"),
    "suitcase": ("suitcases", "suitcase"),
    "frisbee": ("frisbees", "frisbee"),
    "skis": ("skis",),
    "snowboard": ("snowboards", "snowboard"),
    "sports ball": ("sports balls", "sports ball", "balls", "ball"),
    "kite": ("kites", "kite"),
    "baseball bat": ("baseball bats", "baseball bat"),
    "baseball glove": ("baseball gloves", "baseball glove"),
    "skateboard": ("skateboards", "skateboard"),
    "surfboard": ("surfboards", "surfboard"),
    "tennis racket": ("tennis rackets", "tennis racquets", "tennis racket", "tennis racquet"),
    "bottle": ("bottles", "bottle"),
    "wine glass": ("wine glasses", "wine glass"),
    "cup": ("cups", "cup"),
    "fork": ("forks", "fork"),
    "knife": ("knives", "knife"),
    "spoon": ("spoons", "spoon"),
    "bowl": ("bowls", "bowl"),
    "banana": ("bananas", "banana"),
    "apple": ("apples", "apple"),
    "sandwich": ("sandwiches", "sandwich"),
    "orange": ("oranges", "orange"),
    "broccoli": ("broccoli",),
    "carrot": ("carrots", "carrot"),
    "hot dog": ("hot dogs", "hot dog"),
    "pizza": ("pizzas", "pizza"),
    "donut": ("donuts", "doughnuts", "donut", "doughnut"),
    "cake": ("cakes", "cake"),
    "chair": ("chairs", "chair"),
    "couch": ("couches", "sofas", "couch", "sofa"),
    "potted plant": ("potted plants", "potted plant"),
    "bed": ("beds", "bed"),
    "dining table": ("dining tables", "dining table"),
    "toilet": ("toilets", "toilet"),
    "tv": ("televisions", "tvs", "television", "tv"),
    "laptop": ("laptops", "laptop"),
    "mouse": ("computer mice", "computer mouse", "mice", "mouse"),
    "remote": ("remote controls", "remotes", "remote control", "remote"),
    "keyboard": ("keyboards", "keyboard"),
    "cell phone": ("cell phones", "mobile phones", "cell phone", "mobile phone"),
    "microwave": ("microwaves", "microwave"),
    "oven": ("ovens", "oven"),
    "toaster": ("toasters", "toaster"),
    "sink": ("sinks", "sink"),
    "refrigerator": ("refrigerators", "fridges", "refrigerator", "fridge"),
    "book": ("books", "book"),
    "clock": ("clocks", "clock"),
    "vase": ("vases", "vase"),
    "scissors": ("scissors",),
    "teddy bear": ("teddy bears", "teddy bear"),
    "hair drier": ("hair driers", "hair dryers", "hair drier", "hair dryer"),
    "toothbrush": ("toothbrushes", "toothbrush"),
}
ALIAS_TO_CATEGORY = {
    alias: category
    for category, aliases in COCO_CATEGORY_ALIASES.items()
    for alias in aliases
}
SIMPLE_COUNT_RE = re.compile(
    r"^how many (?P<object>.+?)"
    r"(?: (?:are|were) (?:there|pictured|shown|visible|in (?:the )?(?:image|picture|photo)))?\??$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolvedImage:
    path: Path
    exists: bool
    source_dataset: str
    source_split: str
    local_split: str | None
    source_image_id: str
    base_image_uid: str
    original_relative_path: str


@dataclass
class CocoInstancesIndex:
    counts: dict[tuple[int, str], int]
    crowd_counts: dict[tuple[int, str], int]
    image_licenses: dict[int, Mapping[str, Any]]
    image_splits: dict[int, str]
    evidence_files: dict[int, str]
    inputs: list[dict[str, Any]]


@dataclass
class ExclusionIndex:
    base_image_uids: set[str]
    image_sha256s: set[str]
    inputs: list[dict[str, Any]]


VG_URL_RE = re.compile(r"/(?:VG_100K|VG_100K_2)/([0-9]+)\.(?:jpe?g)(?:[?#].*)?$", re.I)
COCO_ID_IN_PATH_RE = re.compile(
    r"(?:COCO_(?:train|val)2014_)?([0-9]{12})\.(?:jpe?g)(?:[?#].*)?$", re.I,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_base_uid(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    match = re.fullmatch(r"coco:0*([0-9]+)", value)
    if match:
        return f"coco:{int(match.group(1)):012d}"
    match = re.fullmatch(r"visual[_ -]?genome:0*([0-9]+)", value)
    if match:
        return f"visual_genome:{int(match.group(1))}"
    return None


def valid_sha256(value: Any, *, field: str, manifest: Path, line_number: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(
            f"{manifest}:{line_number}: {field} must be a 64-character SHA-256 hex string"
        )
    return value.lower()


def ids_from_exclusion_record(record: Mapping[str, Any]) -> set[str]:
    """Extract source-aware image ids without treating arbitrary ids as VG ids."""
    result: set[str] = set()
    for field in ("base_image_uid", "split_group_key"):
        normalized = normalized_base_uid(record.get(field))
        if normalized:
            result.add(normalized)

    source = record.get("source")
    if not isinstance(source, dict):
        source = {}
    image = record.get("image")
    if not isinstance(image, dict):
        image = {}
    for container in (source, image):
        for field in ("base_image_uid", "split_group_key"):
            normalized = normalized_base_uid(container.get(field))
            if normalized:
                result.add(normalized)

    coco_id = source.get("coco_id", image.get("coco_id"))
    if isinstance(coco_id, int) and not isinstance(coco_id, bool) and coco_id >= 0:
        result.add(f"coco:{coco_id:012d}")
    elif isinstance(coco_id, str) and coco_id.isdigit():
        result.add(f"coco:{int(coco_id):012d}")

    text_hints = " ".join(str(value) for value in (
        record.get("benchmark", ""),
        record.get("dataset", ""),
        source.get("dataset", ""),
        source.get("dataset_id", ""),
        source.get("dataset_url", ""),
        source.get("image_url", ""),
        image.get("source_dataset", ""),
        image.get("original_relative_path", ""),
    )).lower()
    image_url = str(source.get("image_url", image.get("url", "")))
    vg_url_match = VG_URL_RE.search(image_url)
    if vg_url_match:
        result.add(f"visual_genome:{int(vg_url_match.group(1))}")
    source_image_id = source.get("image_id", image.get("source_image_id"))
    is_vg = any(token in text_hints for token in ("visual genome", "visual_genome", "ja-vg", "ja_vg"))
    if is_vg and isinstance(source_image_id, int) and not isinstance(source_image_id, bool):
        if source_image_id >= 0:
            result.add(f"visual_genome:{source_image_id}")
    elif is_vg and isinstance(source_image_id, str) and source_image_id.isdigit():
        result.add(f"visual_genome:{int(source_image_id)}")

    # URLs and original paths contain sufficiently strong dataset context.
    for candidate in (
        image_url,
        str(image.get("original_relative_path", "")),
        str(record.get("image_path", "")),
    ):
        vg_match = re.search(r"(?:VG_100K|VG_100K_2)/([0-9]+)\.(?:jpe?g)", candidate, re.I)
        if vg_match:
            result.add(f"visual_genome:{int(vg_match.group(1))}")
        if "coco" in candidate.lower():
            coco_match = COCO_ID_IN_PATH_RE.search(candidate)
            if coco_match:
                result.add(f"coco:{int(coco_match.group(1)):012d}")
    return result


def hashes_from_exclusion_record(
    record: Mapping[str, Any], manifest: Path, line_number: int,
) -> set[str]:
    result: set[str] = set()
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    image = record.get("image") if isinstance(record.get("image"), dict) else {}
    fields = (
        ("image_sha256", record.get("image_sha256")),
        ("source.image_sha256", source.get("image_sha256")),
        ("image.image_sha256", image.get("image_sha256")),
        ("image.sha256", image.get("sha256")),
    )
    for field, value in fields:
        digest = valid_sha256(value, field=field, manifest=manifest, line_number=line_number)
        if digest:
            result.add(digest)

    # If an eval row has no recorded content hash, hash its local image when
    # the manifest makes that image available.  Relative paths are resolved
    # from the manifest directory, matching the evaluation manifest contract.
    if not result:
        path_value = record.get("image_path") or image.get("path") or record.get("path")
        if isinstance(path_value, str) and path_value.strip():
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = manifest.parent / candidate
            if candidate.is_file():
                result.add(sha256_file(candidate))
    return result


def load_exclusion_manifests(paths: Sequence[Path]) -> ExclusionIndex | None:
    if not paths:
        return None
    all_uids: set[str] = set()
    all_hashes: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluation exclusion manifest not found: {path}")
        manifest_uids: set[str] = set()
        manifest_hashes: set[str] = set()
        rows = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: exclusion row must be an object")
                rows += 1
                manifest_uids.update(ids_from_exclusion_record(record))
                manifest_hashes.update(hashes_from_exclusion_record(record, path, line_number))
        if rows == 0:
            raise ValueError(f"evaluation exclusion manifest is empty: {path}")
        all_uids.update(manifest_uids)
        all_hashes.update(manifest_hashes)
        inputs.append({
            "path": str(path.resolve(strict=False)),
            "sha256": sha256_file(path),
            "records": rows,
            "base_image_uids_extracted": len(manifest_uids),
            "image_sha256s_extracted": len(manifest_hashes),
        })
    return ExclusionIndex(
        base_image_uids=all_uids,
        image_sha256s=all_hashes,
        inputs=inputs,
    )


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise


def ensure_int(value: Any, field: str, row_index: int, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"row {row_index}: {field} must be an integer, got {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"row {row_index}: {field} must be nonnegative, got {value}")
    return value


def validate_raw_row(raw: Any, row_index: int) -> Mapping[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"row {row_index}: expected object, got {type(raw).__name__}")
    missing = sorted(REQUIRED_FIELDS - raw.keys())
    if missing:
        raise ValueError(f"row {row_index}: missing required fields: {', '.join(missing)}")
    for field in ("image", "data_source", "question"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"row {row_index}: {field} must be a nonempty string")
    ensure_int(raw["answer"], "answer", row_index)
    ensure_int(raw["image_id"], "image_id", row_index)
    ensure_int(raw["question_id"], "question_id", row_index)
    if "issimple" in raw and not isinstance(raw["issimple"], bool):
        raise ValueError(f"row {row_index}: issimple must be boolean when present")

    relative = PurePosixPath(raw["image"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
        raise ValueError(f"row {row_index}: unsafe or unexpected image path: {raw['image']!r}")
    if relative.parts[0] not in ALLOWED_IMAGE_PARENTS:
        raise ValueError(
            f"row {row_index}: unsupported image source directory {relative.parts[0]!r}"
        )
    if relative.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"row {row_index}: expected JPEG image path: {raw['image']!r}")
    return raw


def load_train_rows(train_json: Path) -> list[Mapping[str, Any]]:
    if train_json.name != "train.json":
        raise ValueError(
            "Only the official TallyQA train split is accepted: input filename "
            f"must be train.json, got {train_json.name!r}. test.json is evaluation-only."
        )
    if not train_json.is_file():
        raise FileNotFoundError(f"TallyQA train JSON not found: {train_json}")
    try:
        payload = json.loads(train_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {train_json}: {error}") from error
    if not isinstance(payload, list):
        raise ValueError("TallyQA train JSON root must be an array")
    if not payload:
        raise ValueError("TallyQA train JSON is empty")

    seen_question_ids: set[int] = set()
    validated: list[Mapping[str, Any]] = []
    for index, item in enumerate(payload):
        raw = validate_raw_row(item, index)
        question_id = int(raw["question_id"])
        if question_id in seen_question_ids:
            raise ValueError(f"row {index}: duplicate question_id {question_id}")
        seen_question_ids.add(question_id)
        validated.append(raw)
    return validated


def infer_coco_local_split(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        if part in {"train2014", "val2014", "train2017", "val2017"}:
            return part
    return None


def first_existing(candidates: Iterable[Path]) -> tuple[Path, bool]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            ordered.append(candidate)
            seen.add(key)
    for candidate in ordered:
        if candidate.is_file():
            return candidate, True
    if not ordered:
        raise AssertionError("image resolver produced no candidates")
    return ordered[0], False


def resolve_image(raw_path: str, coco_root: Path, visual_genome_root: Path) -> ResolvedImage:
    relative = PurePosixPath(raw_path)
    source_parent, file_name = relative.parts
    if source_parent in {"train2014", "val2014"}:
        match = COCO_FILE_RE.fullmatch(file_name)
        if match is None or f"{match.group('split')}2014" != source_parent:
            raise ValueError(f"malformed COCO-2014 image path: {raw_path!r}")
        numeric_id = match.group("id")
        plain_name = f"{numeric_id}.jpg"
        path, exists = first_existing((
            coco_root / source_parent / file_name,
            coco_root / "train2017" / plain_name,
            coco_root / "val2017" / plain_name,
            coco_root / plain_name,
        ))
        return ResolvedImage(
            path=path,
            exists=exists,
            source_dataset="COCO",
            source_split=source_parent,
            local_split=infer_coco_local_split(path) if exists else None,
            source_image_id=numeric_id,
            base_image_uid=f"coco:{numeric_id}",
            original_relative_path=raw_path,
        )

    # Visual Genome archive extraction produces VG_100K and VG_100K_2 below
    # the supplied root.  The second candidate also supports passing either
    # of those directories directly for small tests or custom layouts.
    numeric_stem = PurePosixPath(file_name).stem
    if not numeric_stem.isdigit():
        raise ValueError(f"malformed Visual Genome image path: {raw_path!r}")
    path, exists = first_existing((
        visual_genome_root / source_parent / file_name,
        visual_genome_root / file_name,
    ))
    return ResolvedImage(
        path=path,
        exists=exists,
        source_dataset="Visual Genome",
        source_split=source_parent,
        local_split=source_parent if exists else None,
        source_image_id=numeric_stem,
        base_image_uid=f"visual_genome:{numeric_stem}",
        original_relative_path=raw_path,
    )


def infer_instances_split(path: Path) -> str:
    match = re.search(r"instances_([A-Za-z0-9_-]+)\.json$", path.name)
    return match.group(1) if match else "unknown"


def load_coco_instances(paths: Sequence[Path]) -> CocoInstancesIndex | None:
    if not paths:
        return None
    counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    crowd_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    image_licenses: dict[int, Mapping[str, Any]] = {}
    image_splits: dict[int, str] = {}
    evidence_files: dict[int, str] = {}
    inputs: list[dict[str, Any]] = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"COCO instances JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"COCO instances root must be an object: {path}")
        for field in ("images", "annotations", "categories"):
            if not isinstance(payload.get(field), list):
                raise ValueError(f"COCO instances missing array {field!r}: {path}")
        split = infer_instances_split(path)
        categories = {
            ensure_int(item.get("id"), "COCO category id", index): str(item.get("name", ""))
            for index, item in enumerate(payload["categories"])
            if isinstance(item, dict)
        }
        if not categories or any(not name for name in categories.values()):
            raise ValueError(f"invalid COCO categories in {path}")
        licenses = {
            item.get("id"): item for item in payload.get("licenses", [])
            if isinstance(item, dict) and "id" in item
        }
        for index, image in enumerate(payload["images"]):
            if not isinstance(image, dict):
                raise ValueError(f"COCO image {index} is not an object in {path}")
            image_id = ensure_int(image.get("id"), "COCO image id", index)
            if image_id in image_splits:
                raise ValueError(f"COCO image id {image_id} occurs in multiple instances inputs")
            image_splits[image_id] = split
            evidence_files[image_id] = str(path.resolve(strict=False))
            license_value = licenses.get(image.get("license"))
            if license_value is not None:
                image_licenses[image_id] = dict(license_value)
        for index, annotation in enumerate(payload["annotations"]):
            if not isinstance(annotation, dict):
                raise ValueError(f"COCO annotation {index} is not an object in {path}")
            image_id = ensure_int(annotation.get("image_id"), "COCO annotation image_id", index)
            category_id = ensure_int(
                annotation.get("category_id"), "COCO annotation category_id", index,
            )
            if image_id not in image_splits:
                raise ValueError(f"COCO annotation references unknown image {image_id} in {path}")
            if category_id not in categories:
                raise ValueError(f"COCO annotation references unknown category {category_id} in {path}")
            key = (image_id, categories[category_id])
            if int(annotation.get("iscrowd", 0)):
                crowd_counts[key] += 1
            else:
                counts[key] += 1
        inputs.append({
            "path": str(path.resolve(strict=False)),
            "sha256": sha256_file(path),
            "split": split,
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
        })
        del payload

    return CocoInstancesIndex(
        counts=dict(counts),
        crowd_counts=dict(crowd_counts),
        image_licenses=image_licenses,
        image_splits=image_splits,
        evidence_files=evidence_files,
        inputs=inputs,
    )


def strict_simple_coco_category(question: str) -> str | None:
    normalized = " ".join(question.strip().lower().split())
    match = SIMPLE_COUNT_RE.fullmatch(normalized)
    if match is None:
        return None
    return ALIAS_TO_CATEGORY.get(match.group("object"))


def make_overlap_audit(
    raw: Mapping[str, Any], image: ResolvedImage, coco_index: CocoInstancesIndex | None,
) -> dict[str, Any]:
    category = strict_simple_coco_category(str(raw["question"]))
    candidate = image.source_dataset == "COCO" and category is not None
    result: dict[str, Any] = {
        "candidate": candidate,
        "heuristic": "strict_unqualified_question_and_coco_category_v1",
        "category_en": category if candidate else None,
        "verification_status": "not_candidate",
        "verified_duplicate": False,
        "official_instance_count": None,
        "official_crowd_annotations": None,
        "evidence_instances_json": None,
    }
    if not candidate:
        return result
    if coco_index is None:
        result["verification_status"] = "candidate_unverified"
        return result

    image_id = int(image.source_image_id)
    key = (image_id, category)
    if image_id not in coco_index.image_splits:
        result["verification_status"] = "image_absent_from_instances_inputs"
        return result
    count = coco_index.counts.get(key, 0)
    crowds = coco_index.crowd_counts.get(key, 0)
    result.update({
        "official_instance_count": count,
        "official_crowd_annotations": crowds,
        "evidence_instances_json": coco_index.evidence_files.get(image_id),
    })
    if crowds:
        result["verification_status"] = "ambiguous_due_to_iscrowd"
    elif count == int(raw["answer"]):
        result["verification_status"] = "verified_match"
        result["verified_duplicate"] = True
    else:
        result["verification_status"] = "verified_count_mismatch"
    return result


def image_license(image: ResolvedImage, coco_index: CocoInstancesIndex | None) -> dict[str, Any]:
    if image.source_dataset == "COCO":
        result: dict[str, Any] = dict(COCO_IMAGE_TERMS)
        if coco_index is not None:
            exact = coco_index.image_licenses.get(int(image.source_image_id))
            if exact is not None:
                result["coco_metadata"] = dict(exact)
        return result
    return dict(VISUAL_GENOME_IMAGE_TERMS)


def make_record(
    raw: Mapping[str, Any], image: ResolvedImage, train_json: Path,
    train_json_sha256: str, archive_info: Mapping[str, Any] | None,
    coco_index: CocoInstancesIndex | None,
) -> dict[str, Any]:
    question_id = int(raw["question_id"])
    is_simple_present = "issimple" in raw
    overlap = make_overlap_audit(raw, image, coco_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "uid": f"tallyqa_train_q{question_id}",
        "task": "counting",
        "language": "en",
        "dataset_split": "train",
        "base_image_uid": image.base_image_uid,
        "split_group_key": image.base_image_uid,
        "path": str(image.path),
        "image_path": str(image.path),
        "image": {
            "path": str(image.path),
            "exists_at_build_time": image.exists,
            "source_dataset": image.source_dataset,
            "source_split": image.source_split,
            "resolved_local_split": image.local_split,
            "source_image_id": image.source_image_id,
            "tallyqa_image_id": int(raw["image_id"]),
            "original_relative_path": image.original_relative_path,
            "license": image_license(image, coco_index),
        },
        "question_en": str(raw["question"]),
        "answer_count": int(raw["answer"]),
        "question_complexity": {
            "is_simple": raw.get("issimple"),
            "source_field": "issimple",
            "present_in_source_row": is_simple_present,
            "note": (
                None if is_simple_present else
                "The official TallyQA train.json row has no issimple field; no label was inferred."
            ),
        },
        "coco_instance_overlap": overlap,
        "translation": {
            "status": "pending",
            "question_ja": None,
            "answer_ja": None,
            "ready_for_japanese_training": False,
            "note": "Japanese text must be translated and validated in a later stage.",
        },
        "source": {
            "dataset": "TallyQA",
            "annotation_split": "train",
            "question_id": question_id,
            "image_id": int(raw["image_id"]),
            "data_source": str(raw["data_source"]),
            "train_json": str(train_json.resolve(strict=False)),
            "train_json_sha256": train_json_sha256,
            "repository": TALLYQA_REPOSITORY,
            "repository_revision": TALLYQA_REVISION,
            "archive_url": TALLYQA_ARCHIVE_URL,
            "source_archive": dict(archive_info) if archive_info else None,
            "annotation_license": TALLYQA_ANNOTATION_LICENSE,
            "image_license_is_separate": True,
            "test_split_excluded": True,
        },
        # Preserve every upstream field verbatim, including future additions.
        "tallyqa_original": dict(raw),
    }


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "uid", "task", "language", "dataset_split",
        "base_image_uid", "split_group_key", "image", "question_en",
        "answer_count", "translation", "source", "tallyqa_original",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"generated record missing fields: {', '.join(missing)}")
    if record["dataset_split"] != "train" or record["source"]["annotation_split"] != "train":
        raise ValueError("generated record violated the train-only invariant")
    if record["base_image_uid"] != record["split_group_key"]:
        raise ValueError("base_image_uid and split_group_key must match")
    if record["translation"]["status"] != "pending":
        raise ValueError("untranslated TallyQA record must remain pending")
    if record["translation"]["ready_for_japanese_training"] is not False:
        raise ValueError("untranslated TallyQA record cannot be training-ready")
    ensure_int(record["answer_count"], "generated answer_count", -1)


def build_manifest(
    train_json: Path,
    coco_root: Path,
    visual_genome_root: Path,
    output: Path,
    stats_output: Path,
    source_archive: Path | None = None,
    coco_instances_json: Sequence[Path] = (),
    exclude_manifests: Sequence[Path] = (),
    skip_missing_images: bool = False,
) -> dict[str, Any]:
    if output.resolve(strict=False) == stats_output.resolve(strict=False):
        raise ValueError("output and stats_output must be different files")
    rows = load_train_rows(train_json)
    train_sha = sha256_file(train_json)
    archive_info: dict[str, Any] | None = None
    if source_archive is not None:
        if not source_archive.is_file():
            raise FileNotFoundError(f"TallyQA source archive not found: {source_archive}")
        archive_info = {
            "path": str(source_archive.resolve(strict=False)),
            "sha256": sha256_file(source_archive),
            "bytes": source_archive.stat().st_size,
        }
    coco_index = load_coco_instances(coco_instances_json)
    exclusion_index = load_exclusion_manifests(exclude_manifests)

    image_cache: dict[str, ResolvedImage] = {}
    for raw in rows:
        raw_path = str(raw["image"])
        if raw_path not in image_cache:
            image_cache[raw_path] = resolve_image(raw_path, coco_root, visual_genome_root)
    exclusion_by_base_uid: dict[str, str] = {}
    exclusion_hash_by_base_uid: dict[str, str] = {}
    image_hash_by_base_uid: dict[str, str] = {}
    image_hashes_computed = 0
    if exclusion_index is not None:
        for image in image_cache.values():
            if image.base_image_uid in exclusion_by_base_uid:
                continue
            if image.base_image_uid in exclusion_index.base_image_uids:
                exclusion_by_base_uid[image.base_image_uid] = "base_image_uid"
                continue
            if exclusion_index.image_sha256s and image.exists:
                digest = sha256_file(image.path)
                image_hashes_computed += 1
                image_hash_by_base_uid[image.base_image_uid] = digest
                if digest in exclusion_index.image_sha256s:
                    exclusion_by_base_uid[image.base_image_uid] = "image_sha256"
                    exclusion_hash_by_base_uid[image.base_image_uid] = digest

    missing = [item for item in image_cache.values() if not item.exists]
    blocking_missing = [
        item for item in missing if item.base_image_uid not in exclusion_by_base_uid
    ]
    if blocking_missing and not skip_missing_images:
        examples = "; ".join(
            f"{item.original_relative_path} -> {item.path}" for item in blocking_missing[:10]
        )
        raise FileNotFoundError(
            f"{len(blocking_missing)} non-excluded referenced images are missing. "
            f"Examples: {examples}. "
            "Complete the COCO/Visual Genome downloads or use --skip-missing-images "
            "for an explicitly incomplete manifest."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    records = 0
    skipped_records = 0
    excluded_records = 0
    excluded_records_by_reason: Counter[str] = Counter()
    included_images: set[str] = set()
    by_image_dataset: Counter[str] = Counter()
    by_original_source_split: Counter[str] = Counter()
    by_local_split: Counter[str] = Counter()
    by_data_source: Counter[str] = Counter()
    by_answer: Counter[int] = Counter()
    by_issimple: Counter[str] = Counter()
    by_overlap_status: Counter[str] = Counter()
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            for raw in rows:
                image = image_cache[str(raw["image"])]
                exclusion_reason = exclusion_by_base_uid.get(image.base_image_uid)
                if exclusion_reason is not None:
                    excluded_records += 1
                    excluded_records_by_reason[exclusion_reason] += 1
                    continue
                if not image.exists:
                    skipped_records += 1
                    continue
                record = make_record(
                    raw, image, train_json, train_sha, archive_info, coco_index,
                )
                # The leakage audit already reads every candidate image when
                # hash-based exclusions are configured; keep that provenance
                # instead of discarding the computed digest.
                record["image"]["sha256"] = image_hash_by_base_uid.get(
                    image.base_image_uid
                )
                record["leakage_control"] = {
                    "evaluation_exclusion_manifests_checked": exclusion_index is not None,
                    "matched_evaluation_image": False,
                }
                validate_manifest_record(record)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                records += 1
                included_images.add(image.base_image_uid)
                by_image_dataset[image.source_dataset] += 1
                by_original_source_split[image.source_split] += 1
                by_local_split[image.local_split or "unresolved"] += 1
                by_data_source[str(raw["data_source"])] += 1
                by_answer[int(raw["answer"])] += 1
                if "issimple" not in raw:
                    by_issimple["missing"] += 1
                else:
                    by_issimple[str(bool(raw["issimple"])).lower()] += 1
                by_overlap_status[record["coco_instance_overlap"]["verification_status"]] += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, output)
        temp_name = None
    except BaseException:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise

    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "train_json": str(train_json.resolve(strict=False)),
            "train_json_sha256": train_sha,
            "source_archive": archive_info,
            "raw_rows": len(rows),
            "raw_unique_images": len(image_cache),
            "coco_root": str(coco_root.resolve(strict=False)),
            "visual_genome_root": str(visual_genome_root.resolve(strict=False)),
            "coco_instances": coco_index.inputs if coco_index is not None else [],
            "evaluation_exclusion_manifests": (
                exclusion_index.inputs if exclusion_index is not None else []
            ),
        },
        "configuration": {
            "accepted_annotation_split": "train",
            "test_split_excluded": True,
            "skip_missing_images": skip_missing_images,
            "images_copied": False,
            "japanese_translation_status": "pending",
            "evaluation_image_exclusion_enabled": exclusion_index is not None,
        },
        "image_resolution": {
            "unique_images_found": len(image_cache) - len(missing),
            "unique_images_missing": len(missing),
            "missing_examples": [
                {
                    "original_relative_path": item.original_relative_path,
                    "attempted_path": str(item.path),
                }
                for item in missing[:100]
            ],
        },
        "evaluation_image_exclusion": {
            "manifests": exclusion_index.inputs if exclusion_index is not None else [],
            "base_image_uids_loaded": (
                len(exclusion_index.base_image_uids) if exclusion_index is not None else 0
            ),
            "image_sha256s_loaded": (
                len(exclusion_index.image_sha256s) if exclusion_index is not None else 0
            ),
            "candidate_image_sha256s_computed": image_hashes_computed,
            "excluded_unique_images": len(exclusion_by_base_uid),
            "excluded_records": excluded_records,
            "excluded_records_by_reason": dict(sorted(excluded_records_by_reason.items())),
            "excluded_examples": [
                {
                    "base_image_uid": base_uid,
                    "reason": reason,
                    "matched_sha256": exclusion_hash_by_base_uid.get(base_uid),
                }
                for base_uid, reason in sorted(exclusion_by_base_uid.items())[:100]
            ],
        },
        "output": {
            "jsonl": str(output.resolve(strict=False)),
            "records": records,
            "skipped_records_missing_image": skipped_records,
            "excluded_records_evaluation_overlap": excluded_records,
            "unique_images": len(included_images),
            "by_image_dataset": dict(sorted(by_image_dataset.items())),
            "by_original_source_split": dict(sorted(by_original_source_split.items())),
            "by_resolved_local_split": dict(sorted(by_local_split.items())),
            "by_tallyqa_data_source": dict(sorted(by_data_source.items())),
            "by_answer_count": {str(key): value for key, value in sorted(by_answer.items())},
            "by_original_issimple": dict(sorted(by_issimple.items())),
            "by_coco_overlap_status": dict(sorted(by_overlap_status.items())),
            "sha256": sha256_file(output),
        },
        "notes": {
            "train_test_separation": (
                "Only a file named train.json is accepted. TallyQA test.json is never read."
            ),
            "image_split_safety": (
                "Use split_group_key/base_image_uid for any downstream split so questions "
                "for the same source image cannot cross train/eval boundaries."
            ),
            "evaluation_leakage": (
                "All rows matching an --exclude-manifest by source-aware base image id or "
                "exact image SHA-256 are removed before output."
            ),
            "issimple": (
                "The official train.json currently omits issimple. Missing values remain null; "
                "the COCO-overlap heuristic is an independent de-duplication audit only."
            ),
            "translation": (
                "The source question remains English. Do not train a Japanese model from this "
                "manifest until question_ja/answer_ja are translated and validated."
            ),
        },
    }
    atomic_write_json(stats_output, stats)
    return stats


def self_test() -> None:
    """Exercise validation, resolution, provenance, overlap and atomic output."""
    with tempfile.TemporaryDirectory(prefix="tallyqa_manifest_selftest_") as tmp:
        root = Path(tmp)
        annotation_root = root / "annotations"
        annotation_root.mkdir()
        coco_root = root / "coco"
        (coco_root / "train2017").mkdir(parents=True)
        (coco_root / "train2017" / "000000000001.jpg").write_bytes(b"coco")
        vg_root = root / "visual_genome"
        (vg_root / "VG_100K").mkdir(parents=True)
        (vg_root / "VG_100K" / "4.jpg").write_bytes(b"vg")

        rows = [
            {
                "image": "train2014/COCO_train2014_000000000001.jpg",
                "answer": 2,
                "data_source": "imported_vqa",
                "question": "How many dogs are there?",
                "image_id": 1,
                "question_id": 101,
            },
            {
                "image": "train2014/COCO_train2014_000000000001.jpg",
                "answer": 1,
                "data_source": "amt",
                "question": "How many dogs are wearing collars?",
                "image_id": 1,
                "question_id": 102,
                "issimple": False,
                "future_field": {"preserve": True},
            },
            {
                "image": "VG_100K/4.jpg",
                "answer": 2,
                "data_source": "imported_genome",
                "question": "How many chairs are pictured?",
                "image_id": 90000004,
                "question_id": 103,
            },
        ]
        train_json = annotation_root / "train.json"
        train_json.write_text(json.dumps(rows), encoding="utf-8")
        archive = annotation_root / "tallyqa.zip"
        archive.write_bytes(b"synthetic archive")
        instances = annotation_root / "instances_train2017.json"
        instances.write_text(json.dumps({
            "images": [{
                "id": 1, "file_name": "000000000001.jpg", "license": 1,
            }],
            "categories": [{"id": 18, "name": "dog"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 18, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 18, "iscrowd": 0},
            ],
            "licenses": [{
                "id": 1, "name": "Synthetic image license", "url": "https://example.invalid",
            }],
        }), encoding="utf-8")

        output = root / "manifest.jsonl"
        stats_output = root / "manifest.stats.json"
        stats = build_manifest(
            train_json=train_json,
            coco_root=coco_root,
            visual_genome_root=vg_root,
            output=output,
            stats_output=stats_output,
            source_archive=archive,
            coco_instances_json=[instances],
        )
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 3
        assert records[0]["uid"] == "tallyqa_train_q101"
        assert records[0]["base_image_uid"] == records[1]["base_image_uid"]
        assert records[0]["image"]["resolved_local_split"] == "train2017"
        assert records[0]["question_complexity"]["is_simple"] is None
        assert records[0]["coco_instance_overlap"]["verified_duplicate"] is True
        assert records[0]["coco_instance_overlap"]["official_instance_count"] == 2
        assert records[1]["coco_instance_overlap"]["candidate"] is False
        assert records[1]["question_complexity"]["is_simple"] is False
        assert records[1]["tallyqa_original"]["future_field"] == {"preserve": True}
        assert records[2]["image"]["source_dataset"] == "Visual Genome"
        assert records[2]["coco_instance_overlap"]["candidate"] is False
        assert all(record["translation"]["status"] == "pending" for record in records)
        assert stats["output"]["records"] == 3
        assert stats["output"]["unique_images"] == 2
        assert stats["configuration"]["images_copied"] is False

        # Evaluation leakage is removed by a source-aware Visual Genome id
        # and, independently, by exact image bytes when no source id exists.
        eval_manifest = root / "eval.jsonl"
        eval_rows = [
            {
                "benchmark": "ja-vg-vqa-500",
                "image_path": "images/4.jpg",
                "source": {
                    "dataset_id": "SakanaAI/JA-VG-VQA-500",
                    "image_id": 4,
                    "image_url": "https://example.invalid/VG_100K/4.jpg",
                    "image_sha256": hashlib.sha256(b"vg").hexdigest(),
                },
            },
            {
                "benchmark": "synthetic-byte-overlap",
                "image_sha256": hashlib.sha256(b"coco").hexdigest(),
            },
        ]
        eval_manifest.write_text(
            "".join(json.dumps(item) + "\n" for item in eval_rows), encoding="utf-8",
        )
        excluded_output = root / "manifest_eval_clean.jsonl"
        excluded_stats_path = root / "manifest_eval_clean.stats.json"
        excluded_stats = build_manifest(
            train_json=train_json,
            coco_root=coco_root,
            visual_genome_root=vg_root,
            output=excluded_output,
            stats_output=excluded_stats_path,
            exclude_manifests=[eval_manifest],
        )
        assert excluded_output.read_text(encoding="utf-8") == ""
        assert excluded_stats["evaluation_image_exclusion"]["excluded_unique_images"] == 2
        assert excluded_stats["evaluation_image_exclusion"]["excluded_records"] == 3
        assert excluded_stats["evaluation_image_exclusion"]["excluded_records_by_reason"] == {
            "base_image_uid": 1,
            "image_sha256": 2,
        }
        assert excluded_stats["output"]["excluded_records_evaluation_overlap"] == 3

        # A failed rebuild must not replace a valid existing output.
        bad_root = root / "bad"
        bad_root.mkdir()
        (bad_root / "train.json").write_text(json.dumps([
            rows[0], {**rows[1], "question_id": 101},
        ]), encoding="utf-8")
        original_bytes = output.read_bytes()
        try:
            build_manifest(
                train_json=bad_root / "train.json",
                coco_root=coco_root,
                visual_genome_root=vg_root,
                output=output,
                stats_output=stats_output,
            )
        except ValueError as error:
            assert "duplicate question_id" in str(error)
        else:
            raise AssertionError("duplicate question_id should fail")
        assert output.read_bytes() == original_bytes

        # Accidentally selecting the held-out test split must always fail.
        test_json = annotation_root / "test.json"
        test_json.write_text(json.dumps(rows), encoding="utf-8")
        try:
            load_train_rows(test_json)
        except ValueError as error:
            assert "train.json" in str(error)
        else:
            raise AssertionError("test.json should be rejected")
    print("self-test: OK")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a train-only, provenance-preserving TallyQA manifest.",
    )
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--visual-genome-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stats-output", type=Path,
        help="Default: OUTPUT with .stats.json appended to its filename.",
    )
    parser.add_argument(
        "--source-archive", type=Path,
        help="Optional tallyqa.zip; its SHA-256 is recorded for immutable provenance.",
    )
    parser.add_argument(
        "--coco-instances-json", type=Path, action="append", default=[],
        help=(
            "Optional official instances_train/val JSON (repeatable). Enables exact "
            "verification of simple COCO-count overlap and per-image license metadata."
        ),
    )
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[],
        help=(
            "Evaluation JSONL manifest to exclude by source-aware image id and exact "
            "image SHA-256 (repeatable)."
        ),
    )
    parser.add_argument(
        "--skip-missing-images", action="store_true",
        help="Skip unresolved images and record the omission in stats; default is fail closed.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "--train-json": args.train_json,
        "--coco-root": args.coco_root,
        "--visual-genome-root": args.visual_genome_root,
        "--output": args.output,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Required unless --self-test: {', '.join(missing)}")
    stats_output = args.stats_output or Path(f"{args.output}.stats.json")
    stats = build_manifest(
        train_json=args.train_json,
        coco_root=args.coco_root,
        visual_genome_root=args.visual_genome_root,
        output=args.output,
        stats_output=stats_output,
        source_archive=args.source_archive,
        coco_instances_json=args.coco_instances_json,
        exclude_manifests=args.exclude_manifest,
        skip_missing_images=args.skip_missing_images,
    )
    print(f"wrote {stats['output']['records']} records: {args.output}")
    print(f"unique images: {stats['output']['unique_images']}")
    print(f"stats: {stats_output}")


if __name__ == "__main__":
    main()
