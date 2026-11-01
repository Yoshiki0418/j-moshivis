#!/usr/bin/env python3
"""Build a balanced Japanese counting-task manifest from COCO instances.

This script is intentionally an *indexer*, not a downloader.  It reads an
official ``instances_*.json`` file, checks that each referenced image already
exists below ``--image-root``, and writes JSONL records which point at those
images.  Images are never copied or modified.

Each output row represents one (image, category) counting question.  A pair is
accepted only when all of its COCO annotations are non-crowd and every instance
passes the configured visibility/size checks.  The final set is exactly
balanced over the requested counts (1--10 by default) and approximately
balanced over categories within each count.

Example::

    python ssvd/images/build_coco_counting_manifest.py \
      --instances-json data/images/coco/annotations/instances_train2017.json \
      --image-root data/images/coco/train2017 \
      --output data/manifests/coco_counting_ja_train2017.jsonl

Run the dependency-free synthetic check with ``--self-test``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "jmoshivis.coco_counting_ja.v1"
ANNOTATION_LICENSE = {
    "name": "Creative Commons Attribution 4.0 International",
    "url": "https://creativecommons.org/licenses/by/4.0/",
}


@dataclass(frozen=True)
class JapaneseCategory:
    label: str
    counter: str
    animate: bool = False


# All 80 object categories in COCO 2017.  The labels avoid unnecessary
# transliteration where a familiar Japanese noun exists, and each counter is
# the natural one used in an explicit numerical answer.
COCO_JA: dict[str, JapaneseCategory] = {
    "person": JapaneseCategory("人", "人", True),
    "bicycle": JapaneseCategory("自転車", "台"),
    "car": JapaneseCategory("車", "台"),
    "motorcycle": JapaneseCategory("バイク", "台"),
    "airplane": JapaneseCategory("飛行機", "機"),
    "bus": JapaneseCategory("バス", "台"),
    "train": JapaneseCategory("電車", "台"),
    "truck": JapaneseCategory("トラック", "台"),
    "boat": JapaneseCategory("ボート", "艘"),
    "traffic light": JapaneseCategory("信号機", "基"),
    "fire hydrant": JapaneseCategory("消火栓", "基"),
    "stop sign": JapaneseCategory("一時停止標識", "本"),
    "parking meter": JapaneseCategory("パーキングメーター", "台"),
    "bench": JapaneseCategory("ベンチ", "脚"),
    "bird": JapaneseCategory("鳥", "羽", True),
    "cat": JapaneseCategory("猫", "匹", True),
    "dog": JapaneseCategory("犬", "匹", True),
    "horse": JapaneseCategory("馬", "頭", True),
    "sheep": JapaneseCategory("羊", "頭", True),
    "cow": JapaneseCategory("牛", "頭", True),
    "elephant": JapaneseCategory("象", "頭", True),
    "bear": JapaneseCategory("熊", "頭", True),
    "zebra": JapaneseCategory("シマウマ", "頭", True),
    "giraffe": JapaneseCategory("キリン", "頭", True),
    "backpack": JapaneseCategory("リュック", "個"),
    "umbrella": JapaneseCategory("傘", "本"),
    "handbag": JapaneseCategory("ハンドバッグ", "個"),
    "tie": JapaneseCategory("ネクタイ", "本"),
    "suitcase": JapaneseCategory("スーツケース", "個"),
    "frisbee": JapaneseCategory("フリスビー", "枚"),
    "skis": JapaneseCategory("スキー板", "本"),
    "snowboard": JapaneseCategory("スノーボード", "枚"),
    "sports ball": JapaneseCategory("ボール", "個"),
    "kite": JapaneseCategory("凧", "枚"),
    "baseball bat": JapaneseCategory("野球のバット", "本"),
    "baseball glove": JapaneseCategory("野球のグローブ", "個"),
    "skateboard": JapaneseCategory("スケートボード", "台"),
    "surfboard": JapaneseCategory("サーフボード", "枚"),
    "tennis racket": JapaneseCategory("テニスラケット", "本"),
    "bottle": JapaneseCategory("ボトル", "本"),
    "wine glass": JapaneseCategory("ワイングラス", "脚"),
    "cup": JapaneseCategory("カップ", "個"),
    "fork": JapaneseCategory("フォーク", "本"),
    "knife": JapaneseCategory("ナイフ", "本"),
    "spoon": JapaneseCategory("スプーン", "本"),
    "bowl": JapaneseCategory("ボウル", "個"),
    "banana": JapaneseCategory("バナナ", "本"),
    "apple": JapaneseCategory("りんご", "個"),
    "sandwich": JapaneseCategory("サンドイッチ", "個"),
    "orange": JapaneseCategory("オレンジ", "個"),
    "broccoli": JapaneseCategory("ブロッコリー", "個"),
    "carrot": JapaneseCategory("にんじん", "本"),
    "hot dog": JapaneseCategory("ホットドッグ", "個"),
    "pizza": JapaneseCategory("ピザ", "枚"),
    "donut": JapaneseCategory("ドーナツ", "個"),
    "cake": JapaneseCategory("ケーキ", "個"),
    "chair": JapaneseCategory("椅子", "脚"),
    "couch": JapaneseCategory("ソファ", "台"),
    "potted plant": JapaneseCategory("鉢植え", "個"),
    "bed": JapaneseCategory("ベッド", "台"),
    "dining table": JapaneseCategory("ダイニングテーブル", "台"),
    "toilet": JapaneseCategory("トイレ", "台"),
    "tv": JapaneseCategory("テレビ", "台"),
    "laptop": JapaneseCategory("ノートパソコン", "台"),
    "mouse": JapaneseCategory("マウス", "個"),
    "remote": JapaneseCategory("リモコン", "個"),
    "keyboard": JapaneseCategory("キーボード", "台"),
    "cell phone": JapaneseCategory("携帯電話", "台"),
    "microwave": JapaneseCategory("電子レンジ", "台"),
    "oven": JapaneseCategory("オーブン", "台"),
    "toaster": JapaneseCategory("トースター", "台"),
    "sink": JapaneseCategory("流し台", "台"),
    "refrigerator": JapaneseCategory("冷蔵庫", "台"),
    "book": JapaneseCategory("本", "冊"),
    "clock": JapaneseCategory("時計", "個"),
    "vase": JapaneseCategory("花瓶", "個"),
    "scissors": JapaneseCategory("はさみ", "丁"),
    "teddy bear": JapaneseCategory("テディベア", "体"),
    "hair drier": JapaneseCategory("ヘアドライヤー", "台"),
    "toothbrush": JapaneseCategory("歯ブラシ", "本"),
}

KANJI_NUMBERS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
}

ANIMATE_QUESTIONS = (
    "画像に写っている{label}は何{counter}いますか？",
    "この画像には{label}が全部で何{counter}いますか？",
    "写っている{label}の数を教えてください。",
)
INANIMATE_QUESTIONS = (
    "画像に写っている{label}は何{counter}ありますか？",
    "この画像には{label}が全部で何{counter}ありますか？",
    "写っている{label}の数を教えてください。",
)


@dataclass(frozen=True)
class VisibilityThresholds:
    """Minimum evidence required for *every* counted instance.

    COCO does not expose a true occlusion percentage.  ``area`` is the visible
    segmentation area, while ``bbox`` is its enclosing rectangle.  We retain
    both absolute and image-relative thresholds, a mask/bbox occupancy check,
    and a check that the annotated box lies inside the frame.  These are
    auditable proxies for whether an instance is large and visible enough to
    count reliably.
    """

    min_bbox_width_px: float = 16.0
    min_bbox_height_px: float = 16.0
    min_bbox_area_px: float = 256.0
    min_bbox_area_ratio: float = 0.0005
    min_annotation_area_px: float = 128.0
    min_annotation_area_ratio: float = 0.00025
    min_mask_bbox_ratio: float = 0.05
    min_bbox_in_frame_ratio: float = 0.98

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite non-negative number")
        for key in ("min_bbox_area_ratio", "min_annotation_area_ratio",
                    "min_mask_bbox_ratio", "min_bbox_in_frame_ratio"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be <= 1")


@dataclass(frozen=True)
class Candidate:
    image: Mapping[str, Any]
    category: Mapping[str, Any]
    annotations: tuple[Mapping[str, Any], ...]
    image_path: Path
    count: int

    @property
    def image_id(self) -> int:
        return int(self.image["id"])

    @property
    def category_id(self) -> int:
        return int(self.category["id"])


def stable_digest(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_split(instances_json: Path) -> str:
    stem = instances_json.stem
    prefix = "instances_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_instances(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    required = ("images", "annotations", "categories")
    missing = [key for key in required if not isinstance(data.get(key), list)]
    if missing:
        raise ValueError(f"Not a COCO instances JSON; missing list(s): {missing}")
    return data


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    """Resolve an official COCO ``file_name`` without recursive guessing."""
    candidate = image_root / file_name
    return candidate.resolve(strict=False)


def clipped_bbox_area(bbox: Sequence[float], width: float, height: float) -> float:
    x, y, box_width, box_height = bbox
    x1 = min(max(float(x), 0.0), width)
    y1 = min(max(float(y), 0.0), height)
    x2 = min(max(float(x) + float(box_width), 0.0), width)
    y2 = min(max(float(y) + float(box_height), 0.0), height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def annotation_failures(
    annotation: Mapping[str, Any],
    image: Mapping[str, Any],
    thresholds: VisibilityThresholds,
) -> set[str]:
    """Return machine-readable rejection reasons for one COCO instance."""
    failures: set[str] = set()
    if int(annotation.get("iscrowd", 0)) != 0:
        failures.add("iscrowd")

    try:
        width = float(image["width"])
        height = float(image["height"])
        bbox = [float(value) for value in annotation["bbox"]]
        area = float(annotation["area"])
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError
        if not all(math.isfinite(v) for v in (width, height, area)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        failures.add("malformed_geometry")
        return failures

    if width <= 0 or height <= 0 or bbox[2] <= 0 or bbox[3] <= 0 or area <= 0:
        failures.add("nonpositive_geometry")
        return failures

    image_area = width * height
    bbox_area = bbox[2] * bbox[3]
    if bbox[2] < thresholds.min_bbox_width_px:
        failures.add("bbox_too_narrow")
    if bbox[3] < thresholds.min_bbox_height_px:
        failures.add("bbox_too_short")
    if bbox_area < thresholds.min_bbox_area_px:
        failures.add("bbox_area_too_small")
    if bbox_area / image_area < thresholds.min_bbox_area_ratio:
        failures.add("bbox_area_ratio_too_small")
    if area < thresholds.min_annotation_area_px:
        failures.add("annotation_area_too_small")
    if area / image_area < thresholds.min_annotation_area_ratio:
        failures.add("annotation_area_ratio_too_small")
    if min(area / bbox_area, 1.0) < thresholds.min_mask_bbox_ratio:
        failures.add("mask_bbox_ratio_too_small")
    if clipped_bbox_area(bbox, width, height) / bbox_area < thresholds.min_bbox_in_frame_ratio:
        failures.add("bbox_outside_frame")
    return failures


def group_annotations(
    annotations: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        try:
            key = (int(annotation["image_id"]), int(annotation["category_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        grouped[key].append(annotation)
    return grouped


def collect_candidates(
    coco: Mapping[str, Any],
    image_root: Path,
    thresholds: VisibilityThresholds,
    min_count: int = 1,
    max_count: int = 10,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Collect eligible image/category pairs and pair-level rejection stats."""
    thresholds.validate()
    images = {int(item["id"]): item for item in coco["images"]}
    categories = {int(item["id"]): item for item in coco["categories"]}
    grouped = group_annotations(coco["annotations"])
    candidates: list[Candidate] = []
    rejected_pairs: Counter[str] = Counter()
    rejected_instances: Counter[str] = Counter()
    missing_files: list[str] = []

    for (image_id, category_id), annotations in sorted(grouped.items()):
        count = len(annotations)
        if count < min_count or count > max_count:
            rejected_pairs["count_out_of_range"] += 1
            continue
        image = images.get(image_id)
        category = categories.get(category_id)
        if image is None or category is None:
            rejected_pairs["unknown_image_or_category"] += 1
            continue
        if str(category.get("name", "")) not in COCO_JA:
            rejected_pairs["missing_japanese_category"] += 1
            continue
        image_path = resolve_image_path(image_root, str(image.get("file_name", "")))
        if not image_path.is_file():
            rejected_pairs["missing_image_file"] += 1
            if len(missing_files) < 100:
                missing_files.append(str(image_path))
            continue

        pair_failures: set[str] = set()
        for annotation in annotations:
            failures = annotation_failures(annotation, image, thresholds)
            pair_failures.update(failures)
            rejected_instances.update(failures)
        if pair_failures:
            # A crowd annotation invalidates the complete pair; crowd objects
            # must never be silently omitted from the target count.
            rejected_pairs.update(pair_failures)
            continue

        ordered_annotations = tuple(
            sorted(annotations, key=lambda item: int(item.get("id", -1)))
        )
        candidates.append(Candidate(
            image=image,
            category=category,
            annotations=ordered_annotations,
            image_path=image_path,
            count=count,
        ))

    stats = {
        "grouped_image_category_pairs": len(grouped),
        "eligible_pairs": len(candidates),
        "rejected_pairs_by_reason": dict(sorted(rejected_pairs.items())),
        "rejected_instance_checks_by_reason": dict(sorted(rejected_instances.items())),
        "missing_image_examples": missing_files,
    }
    return candidates, stats


def _category_round_robin(
    candidates: Sequence[Candidate],
    seed: int,
    desired: int,
    image_use: Counter[int],
    max_tasks_per_image: int | None,
) -> list[Candidate]:
    """Deterministically sample while rotating through available categories."""
    by_category: dict[int, deque[Candidate]] = {}
    grouped: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.category_id].append(candidate)
    for category_id, items in grouped.items():
        items.sort(key=lambda item: stable_digest(
            SCHEMA_VERSION, seed, item.count, category_id, item.image_id,
        ))
        by_category[category_id] = deque(items)

    category_ids = sorted(
        by_category,
        key=lambda category_id: stable_digest(
            SCHEMA_VERSION, "category-order", seed, category_id,
        ),
    )
    selected: list[Candidate] = []
    while category_ids and len(selected) < desired:
        next_round: list[int] = []
        made_progress = False
        for category_id in category_ids:
            queue = by_category[category_id]
            chosen: Candidate | None = None
            while queue:
                candidate = queue.popleft()
                if (max_tasks_per_image is None
                        or image_use[candidate.image_id] < max_tasks_per_image):
                    chosen = candidate
                    break
            if chosen is not None:
                selected.append(chosen)
                image_use[chosen.image_id] += 1
                made_progress = True
            if queue:
                next_round.append(category_id)
            if len(selected) >= desired:
                break
        if not made_progress:
            break
        category_ids = next_round
    return selected


def select_balanced(
    candidates: Sequence[Candidate],
    min_count: int = 1,
    max_count: int = 10,
    samples_per_count: int | None = None,
    seed: int = 42,
    max_tasks_per_image: int | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Return an exactly count-balanced, category-round-robin selection.

    If ``samples_per_count`` is omitted, the rarest count bucket determines
    the quota.  If a requested quota is unavailable, all buckets are trimmed
    to the largest common size; therefore the output is never mislabeled as
    balanced while containing unequal count buckets.
    """
    if min_count < 1 or max_count < min_count:
        raise ValueError("Require 1 <= min_count <= max_count")
    if samples_per_count is not None and samples_per_count < 1:
        raise ValueError("samples_per_count must be positive when specified")
    if max_tasks_per_image is not None and max_tasks_per_image < 1:
        raise ValueError("max_tasks_per_image must be positive when specified")

    counts = list(range(min_count, max_count + 1))
    buckets = {count: [item for item in candidates if item.count == count]
               for count in counts}
    missing = [count for count, items in buckets.items() if not items]
    if missing:
        raise ValueError(f"No eligible candidates for count bucket(s): {missing}")

    available = {count: len(items) for count, items in buckets.items()}
    desired = min(available.values())
    if samples_per_count is not None:
        desired = min(desired, samples_per_count)

    # Rotate count processing order too, so an optional per-image cap does not
    # always privilege low counts.  The final trimming restores exact balance.
    count_order = sorted(
        counts,
        key=lambda count: stable_digest(SCHEMA_VERSION, "count-order", seed, count),
    )
    image_use: Counter[int] = Counter()
    picked: dict[int, list[Candidate]] = {}
    for count in count_order:
        picked[count] = _category_round_robin(
            buckets[count], seed, desired, image_use, max_tasks_per_image,
        )
    common_size = min(len(items) for items in picked.values())
    if common_size == 0:
        raise ValueError("Image reuse cap left at least one count bucket empty")
    if common_size < desired:
        picked = {count: items[:common_size] for count, items in picked.items()}

    selected = [item for count in counts for item in picked[count]]
    selected.sort(key=lambda item: stable_digest(
        SCHEMA_VERSION, "output-order", seed, item.count,
        item.category_id, item.image_id,
    ))
    stats = {
        "available_by_count": {str(key): value for key, value in available.items()},
        "selected_per_count": common_size,
        "selected_by_count": {
            str(count): len(picked[count]) for count in counts
        },
        "requested_samples_per_count": samples_per_count,
        "seed": seed,
        "max_tasks_per_image": max_tasks_per_image,
    }
    return selected, stats


def make_uid(split: str, candidate: Candidate) -> str:
    annotation_ids = ",".join(str(item["id"]) for item in candidate.annotations)
    digest = stable_digest(
        SCHEMA_VERSION, split, candidate.image_id, candidate.category_id,
        annotation_ids,
    )
    return f"coco_counting_{digest[:32]}"


def render_qa(candidate: Candidate, uid: str) -> tuple[str, str]:
    name = str(candidate.category["name"])
    ja = COCO_JA[name]
    questions = ANIMATE_QUESTIONS if ja.animate else INANIMATE_QUESTIONS
    template_index = int(stable_digest(uid, "question-template")[:8], 16) % len(questions)
    question = questions[template_index].format(label=ja.label, counter=ja.counter)
    verb = "います" if ja.animate else "あります"
    # An Arabic digit makes the target unambiguous for exact-match evaluation;
    # count_kanji below retains the normal Japanese written representation too.
    answer = f"{candidate.count}{ja.counter}{verb}。"
    return question, answer


def make_record(
    candidate: Candidate,
    split: str,
    license_by_id: Mapping[int, Mapping[str, Any]],
    instances_json: Path,
) -> dict[str, Any]:
    uid = make_uid(split, candidate)
    base_image_uid = f"coco:{candidate.image_id:012d}"
    question, answer = render_qa(candidate, uid)
    category_name = str(candidate.category["name"])
    ja = COCO_JA[category_name]
    image_license_id = candidate.image.get("license")
    try:
        license_key = int(image_license_id)
    except (TypeError, ValueError):
        license_key = -1
    image_license = license_by_id.get(license_key)

    instances = [{
        "annotation_id": int(annotation["id"]),
        "bbox_xywh": [float(value) for value in annotation["bbox"]],
        "segmentation_area_px": float(annotation["area"]),
        "iscrowd": int(annotation.get("iscrowd", 0)),
    } for annotation in candidate.annotations]

    return {
        "schema_version": SCHEMA_VERSION,
        "uid": uid,
        "base_image_uid": base_image_uid,
        "split_group_key": base_image_uid,
        "task": "counting",
        "language": "ja",
        "path": str(candidate.image_path),
        "image_path": str(candidate.image_path),
        "question": question,
        "answer": answer,
        "dialogue": [
            {"role": "user", "text": question},
            {"role": "assistant", "text": answer},
        ],
        "target": {
            "count": candidate.count,
            "count_kanji": KANJI_NUMBERS.get(candidate.count),
            "category_id": candidate.category_id,
            "category_en": category_name,
            "category_ja": ja.label,
            "counter_ja": ja.counter,
            "animate": ja.animate,
        },
        "source": {
            "dataset": "COCO",
            "split": split,
            "instances_json": str(instances_json.resolve(strict=False)),
            "image_id": candidate.image_id,
            "file_name": candidate.image.get("file_name"),
            "width": int(candidate.image["width"]),
            "height": int(candidate.image["height"]),
            "coco_url": candidate.image.get("coco_url"),
            "flickr_url": candidate.image.get("flickr_url"),
            "date_captured": candidate.image.get("date_captured"),
            "image_license": dict(image_license) if image_license else {
                "id": image_license_id,
                "name": None,
                "url": None,
            },
            "annotation_license": ANNOTATION_LICENSE,
            "category": {
                "id": candidate.category_id,
                "name": category_name,
                "supercategory": candidate.category.get("supercategory"),
            },
            "instances": instances,
        },
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def build_manifest(
    instances_json: Path,
    image_root: Path,
    output: Path,
    stats_output: Path,
    thresholds: VisibilityThresholds,
    min_count: int = 1,
    max_count: int = 10,
    samples_per_count: int | None = None,
    seed: int = 42,
    split: str | None = None,
    max_tasks_per_image: int | None = None,
) -> dict[str, Any]:
    if not instances_json.is_file():
        raise FileNotFoundError(f"instances JSON not found: {instances_json}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"image root not found: {image_root}")
    if output.resolve(strict=False) == stats_output.resolve(strict=False):
        raise ValueError("output and stats_output must be different files")

    coco = load_instances(instances_json)
    resolved_split = split or infer_split(instances_json)
    candidates, filter_stats = collect_candidates(
        coco, image_root, thresholds, min_count, max_count,
    )
    selected, selection_stats = select_balanced(
        candidates,
        min_count=min_count,
        max_count=max_count,
        samples_per_count=samples_per_count,
        seed=seed,
        max_tasks_per_image=max_tasks_per_image,
    )
    licenses = {
        int(item["id"]): item for item in coco.get("licenses", [])
        if "id" in item
    }
    records = [
        make_record(item, resolved_split, licenses, instances_json)
        for item in selected
    ]
    record_count = write_jsonl(output, records)

    by_category = Counter(record["target"]["category_en"] for record in records)
    by_license = Counter(
        record["source"]["image_license"].get("name") or "unknown"
        for record in records
    )
    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "instances_json": str(instances_json.resolve(strict=False)),
            "instances_json_sha256": sha256_file(instances_json),
            "image_root": str(image_root.resolve(strict=False)),
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
        },
        "configuration": {
            "split": resolved_split,
            "min_count": min_count,
            "max_count": max_count,
            "samples_per_count": samples_per_count,
            "seed": seed,
            "max_tasks_per_image": max_tasks_per_image,
            "visibility_thresholds": asdict(thresholds),
        },
        "filtering": filter_stats,
        "selection": selection_stats,
        "output": {
            "jsonl": str(output.resolve(strict=False)),
            "records": record_count,
            "unique_images": len({record["source"]["image_id"] for record in records}),
            "by_category_en": dict(sorted(by_category.items())),
            "by_image_license": dict(sorted(by_license.items())),
            "sha256": sha256_file(output),
        },
        "notes": {
            "images_copied": False,
            "balance": "exact over count; category round-robin within each count",
            "visibility": (
                "COCO has no true occlusion fraction. Filtering uses visible "
                "segmentation area, bbox size/area, mask-to-bbox occupancy, and "
                "in-frame bbox fraction as auditable proxies."
            ),
        },
    }
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with stats_output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return stats


def self_test() -> None:
    """Exercise filtering, balancing, provenance, and deterministic output."""
    with tempfile.TemporaryDirectory(prefix="coco_counting_selftest_") as tmp:
        root = Path(tmp)
        image_root = root / "images"
        image_root.mkdir()
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        image_id = 1
        # Two candidates for each count.  No image decoding is required; only
        # official metadata and the existing path are used by this indexer.
        for count in range(1, 4):
            for category_id, category_name in ((1, "person"), (2, "car")):
                file_name = f"{image_id:012d}.jpg"
                (image_root / file_name).touch()
                images.append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": 640,
                    "height": 480,
                    "license": 1,
                    "coco_url": f"https://images.cocodataset.org/{file_name}",
                    "flickr_url": f"https://example.invalid/{file_name}",
                })
                for index in range(count):
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [30.0 + index * 40, 40.0, 32.0, 40.0],
                        "area": 900.0,
                        "iscrowd": 0,
                    })
                    annotation_id += 1
                image_id += 1
        # This crowd pair must be rejected in full, not counted as one object.
        crowd_file = f"{image_id:012d}.jpg"
        (image_root / crowd_file).touch()
        images.append({
            "id": image_id, "file_name": crowd_file,
            "width": 640, "height": 480, "license": 1,
        })
        annotations.append({
            "id": annotation_id, "image_id": image_id, "category_id": 1,
            "bbox": [20, 20, 100, 100], "area": 8000, "iscrowd": 1,
        })

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
            ],
            "licenses": [{"id": 1, "name": "Synthetic License", "url": "https://example.invalid/license"}],
        }
        instances = root / "instances_synthetic.json"
        instances.write_text(json.dumps(coco), encoding="utf-8")
        output_a = root / "a.jsonl"
        stats_a = root / "a.stats.json"
        output_b = root / "b.jsonl"
        stats_b = root / "b.stats.json"
        kwargs = dict(
            instances_json=instances,
            image_root=image_root,
            thresholds=VisibilityThresholds(),
            min_count=1,
            max_count=3,
            samples_per_count=2,
            seed=7,
            split="synthetic",
            max_tasks_per_image=None,
        )
        result = build_manifest(output=output_a, stats_output=stats_a, **kwargs)
        build_manifest(output=output_b, stats_output=stats_b, **kwargs)
        rows = [json.loads(line) for line in output_a.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 6
        assert Counter(row["target"]["count"] for row in rows) == {1: 2, 2: 2, 3: 2}
        assert all(row["language"] == "ja" for row in rows)
        assert all(row["source"]["instances"] for row in rows)
        assert result["filtering"]["rejected_pairs_by_reason"]["iscrowd"] == 1
        assert output_a.read_bytes() == output_b.read_bytes()
        assert result["output"]["sha256"] == sha256_file(output_a)
    print("self-test: OK")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, balanced Japanese COCO counting manifest.",
    )
    parser.add_argument("--instances-json", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Default: OUTPUT with .stats.json appended to its filename.",
    )
    parser.add_argument("--split", help="Override split inferred from instances filename.")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument(
        "--samples-per-count", type=int,
        help="Maximum per count. Default: size of the rarest eligible bucket.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-tasks-per-image", type=int,
        help="Optional image reuse cap. Default: no cap.",
    )
    parser.add_argument("--min-bbox-width-px", type=float, default=16.0)
    parser.add_argument("--min-bbox-height-px", type=float, default=16.0)
    parser.add_argument("--min-bbox-area-px", type=float, default=256.0)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.0005)
    parser.add_argument("--min-annotation-area-px", type=float, default=128.0)
    parser.add_argument("--min-annotation-area-ratio", type=float, default=0.00025)
    parser.add_argument("--min-mask-bbox-ratio", type=float, default=0.05)
    parser.add_argument("--min-bbox-in-frame-ratio", type=float, default=0.98)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "--instances-json": args.instances_json,
        "--image-root": args.image_root,
        "--output": args.output,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Required unless --self-test: {', '.join(missing)}")

    stats_output = args.stats_output or Path(f"{args.output}.stats.json")
    thresholds = VisibilityThresholds(
        min_bbox_width_px=args.min_bbox_width_px,
        min_bbox_height_px=args.min_bbox_height_px,
        min_bbox_area_px=args.min_bbox_area_px,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        min_annotation_area_px=args.min_annotation_area_px,
        min_annotation_area_ratio=args.min_annotation_area_ratio,
        min_mask_bbox_ratio=args.min_mask_bbox_ratio,
        min_bbox_in_frame_ratio=args.min_bbox_in_frame_ratio,
    )
    stats = build_manifest(
        instances_json=args.instances_json,
        image_root=args.image_root,
        output=args.output,
        stats_output=stats_output,
        thresholds=thresholds,
        min_count=args.min_count,
        max_count=args.max_count,
        samples_per_count=args.samples_per_count,
        seed=args.seed,
        split=args.split,
        max_tasks_per_image=args.max_tasks_per_image,
    )
    print(f"wrote {stats['output']['records']} records: {args.output}")
    print(f"stats: {stats_output}")


if __name__ == "__main__":
    main()
