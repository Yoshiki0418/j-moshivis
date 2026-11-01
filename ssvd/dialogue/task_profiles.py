"""Task-aware input preparation and validation for image-dialogue generation.

The authoritative counting/OCR answer is supplied to the VLM before
generation. Validation is a final guard against malformed or accidentally
ungrounded output; it is not the mechanism used to discover the answer.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_MODES = ("auto", "normal", "counting", "ocr")
NORMAL_STRATEGIES = ("standard", "vlm_adaptive")
MINIMUM_COUNT = 2
SELF_GROUNDED_COUNT_MINIMUM = 2
SELF_GROUNDED_COUNT_MAXIMUM = 10
COUNTED_TARGET_MAX_CHARACTERS = 32
OCR_MIN_CHARACTERS = 2
OCR_MAX_CHARACTERS = 24
APPROXIMATE_COUNT_TERMS = (
    "約",
    "およそ",
    "だいたい",
    "くらい",
    "ぐらい",
    "ほど",
    "前後",
    "以上",
    "以下",
    "少なくとも",
    "多くとも",
)
NEGATED_COUNT_TERMS = (
    "いません",
    "いない",
    "ありません",
    "ないです",
    "ではない",
    "じゃない",
)
GENERIC_COUNT_TARGETS = {
    "もの",
    "物",
    "対象",
    "それ",
    "これ",
    "あれ",
    "何か",
    "オブジェクト",
}
COUNT_QUESTION_TERMS = (
    "いくつ",
    "全部で何",
    "数は",
    "何個",
    "何人",
    "何名",
    "何匹",
    "何頭",
    "何羽",
    "何台",
    "何本",
    "何枚",
    "何件",
    "何組",
    "何脚",
    "何杯",
    "何箱",
    "何着",
    "何足",
    "何棟",
    "何軒",
    "何隻",
    "何機",
    "何列",
    "何輪",
    "何冊",
    "何株",
    "何体",
    "何基",
    "何面",
    "何点",
    "何種類",
    "何色",
    "何皿",
    "何切れ",
    "何箇所",
    "何か所",
)
EXPLICIT_COUNT_QUESTION_RE = re.compile(
    r"(?:何(?:人|名|個|台|匹|本|枚|脚|杯|つ|軒|棟|組|隻|羽|冊|頭|件|着|足|箱|皿)(?!か)|いくつ(?!か))"
)
COUNT_COUNTERS = (
    "種類",
    "セット",
    "レベル",
    "フロア",
    "階建て",
    "角形",
    "段階",
    "時間",
    "部屋",
    "車両",
    "切れ",
    "箇所",
    "か所",
    "種",
    "色",
    "通り",
    "階",
    "層",
    "段",
    "室",
    "両",
    "皿",
    "品",
    "辺",
    "人",
    "名",
    "匹",
    "頭",
    "羽",
    "個",
    "つ",
    "台",
    "本",
    "枚",
    "件",
    "組",
    "脚",
    "杯",
    "箱",
    "車",
    "着",
    "足",
    "棟",
    "軒",
    "隻",
    "機",
    "列",
    "輪",
    "冊",
    "株",
    "体",
    "基",
    "面",
    "点",
)
NATIVE_COUNT_FORMS = {
    "ひとり": 1,
    "ひとつ": 1,
    "ふたり": 2,
    "ふたつ": 2,
    "さんにん": 3,
    "みっつ": 3,
    "よにん": 4,
    "よっつ": 4,
    "ごにん": 5,
    "いつつ": 5,
    "ろくにん": 6,
    "むっつ": 6,
    "ななにん": 7,
    "しちにん": 7,
    "ななつ": 7,
    "はちにん": 8,
    "やっつ": 8,
    "きゅうにん": 9,
    "ここのつ": 9,
    "じゅうにん": 10,
    "とお": 10,
}
KANJI_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
COUNT_BODY_PART_RE = re.compile(
    r"\b(legs?|arms?|hands?|feet|foot|eyes?|ears?|fingers?|toes?|teeth|heads?)\b",
    re.IGNORECASE,
)
COUNT_POSSESSION_RE = re.compile(
    r"\b(?:does|do)\b.+\bhave\b", re.IGNORECASE
)
COUNT_EACH_RE = re.compile(r"\b(?:each|per)\b", re.IGNORECASE)
COUNT_VISIBLE_RE = re.compile(
    r"\b(?:visible|shown|pictured|are there|can (?:you )?see|do you see|"
    r"in (?:the )?(?:image|picture|photo))\b",
    re.IGNORECASE,
)


def canonical_task_mode(requested: str, entry: Mapping[str, Any]) -> str:
    if requested not in TASK_MODES:
        raise ValueError(f"unsupported task mode: {requested}")
    if requested != "auto":
        return requested
    capability = entry.get("capability")
    if capability == "counting":
        return "counting"
    if capability == "japanese_ocr":
        return "ocr"
    return "normal"


def _entry_uid(entry: Mapping[str, Any]) -> str:
    uid = entry.get("uid", entry.get("candidate_uid"))
    if not isinstance(uid, (str, int)) or not str(uid):
        raise ValueError("entry requires a non-empty uid or candidate_uid")
    return str(uid)


def _absolute_image_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return os.path.abspath(path)


def _count_question_quality(question: str) -> tuple[int, list[str]]:
    """Classify count questions without discarding fallback-only images."""
    flags = []
    if COUNT_BODY_PART_RE.search(question):
        flags.append("body_part")
    if COUNT_POSSESSION_RE.search(question):
        flags.append("intrinsic_possession")
    if COUNT_EACH_RE.search(question):
        flags.append("each_or_per")
    visible = bool(COUNT_VISIBLE_RE.search(question))
    if visible:
        flags.append("explicitly_visual")
    risky = any(flag != "explicitly_visual" for flag in flags)
    if visible and not risky:
        return 0, flags
    if not risky:
        return 1, flags
    return 2, flags


def _count_fact_rank(fact: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Prefer visible object/person counts, retaining lower-quality fallbacks."""
    source_penalty = 0 if fact.get("data_source") == "imported_genome" else 1
    question = str(fact.get("question_en", ""))
    quality_tier, _ = _count_question_quality(question)
    return quality_tier, source_penalty, len(question), str(fact.get("source_question_id", ""))


def select_counting_grounding(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    facts = entry.get("reference_annotations")
    if not isinstance(facts, list):
        return None
    eligible: list[Mapping[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        answer = fact.get("answer_count")
        question = fact.get("question_en")
        if (
            isinstance(answer, int)
            and not isinstance(answer, bool)
            and answer >= MINIMUM_COUNT
            and isinstance(question, str)
            and question.strip()
        ):
            eligible.append(fact)
    if not eligible:
        return None
    fact = min(eligible, key=_count_fact_rank)
    question = str(fact["question_en"]).strip()
    quality_tier, quality_flags = _count_question_quality(question)
    return {
        "kind": "counting",
        "selection_policy": "visual_object_count_then_non_intrinsic_then_fallback",
        "question_quality_tier": quality_tier,
        "question_quality_flags": quality_flags,
        "source_question_id": fact.get("source_question_id"),
        "question_en": question,
        "answer_count": int(fact["answer_count"]),
        "data_source": fact.get("data_source"),
        "question_turn_index": 0,
        "answer_turn_index": 1,
    }


def select_optional_counting_grounding(
    entry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Select a safe canonical count hint for an adaptive normal dialogue.

    Unlike the dedicated counting task, an adaptive normal dialogue may use a
    count of one. Intrinsic body-part/possession and each/per questions are
    deliberately excluded because they make poor casual follow-up questions.
    """
    facts = entry.get("reference_annotations")
    if not isinstance(facts, list):
        return None
    eligible: list[Mapping[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        answer = fact.get("answer_count")
        question = fact.get("question_en")
        if (
            isinstance(answer, int)
            and not isinstance(answer, bool)
            and answer >= 1
            and isinstance(question, str)
            and question.strip()
            and _count_question_quality(question)[0] <= 1
        ):
            eligible.append(fact)
    if not eligible:
        return None
    fact = min(eligible, key=_count_fact_rank)
    question = str(fact["question_en"]).strip()
    quality_tier, quality_flags = _count_question_quality(question)
    return {
        "kind": "counting",
        "selection_policy": "best_visible_count_at_least_1_quality_tier_0_or_1",
        "question_quality_tier": quality_tier,
        "question_quality_flags": quality_flags,
        "source_question_id": fact.get("source_question_id"),
        "question_en": question,
        "answer_count": int(fact["answer_count"]),
        "data_source": fact.get("data_source"),
    }


def _compact_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


def select_ocr_grounding(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    annotations = entry.get("reference_annotations")
    if not isinstance(annotations, list):
        return None
    eligible: list[tuple[int, int, Mapping[str, Any], str]] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            continue
        if annotation.get("kind") not in (None, "exact_transcription"):
            continue
        text = annotation.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        character_count = len(_compact_text(text))
        if OCR_MIN_CHARACTERS <= character_count <= OCR_MAX_CHARACTERS:
            # Prefer the most informative text within the short-answer limit.
            eligible.append((-character_count, index, annotation, text.strip()))
    if not eligible:
        return None
    _, index, annotation, text = min(eligible, key=lambda item: (item[0], item[1]))
    grounding = {
        "kind": "ocr",
        "selection_policy": "longest_exact_transcription_within_2_to_24_characters",
        "annotation_index": index,
        "text": text,
        "normalized_text": annotation.get("normalized_text"),
        "character_count": len(_compact_text(text)),
        "orientation": annotation.get("orientation"),
        "question_turn_index": 0,
        "answer_turn_index": 1,
    }
    if isinstance(entry.get("geometry"), Mapping):
        grounding["geometry"] = dict(entry["geometry"])
    return grounding


def select_optional_ocr_grounding(
    entry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return only explicitly exact, short OCR ground truth as an optional hint."""
    annotations = entry.get("reference_annotations")
    if not isinstance(annotations, list):
        return None
    eligible: list[tuple[int, int, Mapping[str, Any], str]] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            continue
        if annotation.get("kind") != "exact_transcription":
            continue
        text = annotation.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        character_count = len(_compact_text(text))
        if OCR_MIN_CHARACTERS <= character_count <= OCR_MAX_CHARACTERS:
            eligible.append((-character_count, index, annotation, text.strip()))
    if not eligible:
        return None
    _, index, annotation, text = min(eligible, key=lambda item: (item[0], item[1]))
    grounding = {
        "kind": "ocr",
        "selection_policy": "longest_explicit_exact_transcription_within_2_to_24_characters",
        "annotation_index": index,
        "text": text,
        "normalized_text": annotation.get("normalized_text"),
        "character_count": len(_compact_text(text)),
        "orientation": annotation.get("orientation"),
    }
    if isinstance(entry.get("geometry"), Mapping):
        grounding["geometry"] = dict(entry["geometry"])
    return grounding


def prepare_task_entry(
    raw_entry: Mapping[str, Any],
    requested_mode: str,
    normal_strategy: str = "standard",
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize one input row; return ``(None, reason)`` when ineligible."""
    if normal_strategy not in NORMAL_STRATEGIES:
        raise ValueError(f"unsupported normal strategy: {normal_strategy}")
    task_mode = canonical_task_mode(requested_mode, raw_entry)
    if task_mode != "normal" and normal_strategy != "standard":
        raise ValueError("vlm_adaptive normal strategy requires normal task mode")
    try:
        uid = _entry_uid(raw_entry)
    except ValueError:
        return None, "missing_uid"
    path = raw_entry.get("path")
    if not isinstance(path, str) or not path:
        return None, "missing_image_path"

    entry = dict(raw_entry)
    entry["uid"] = uid
    entry["path"] = path
    entry["absolute_image_path"] = _absolute_image_path(path)
    entry["task_mode"] = task_mode
    # Keep every task variant and every frame from the same source on one side
    # of the eventual train/eval split. Targeted manifests may provide a wider
    # grouping key (for example a video or source-image ID); otherwise UID is
    # the safe image-level fallback.
    entry["split_group_key"] = str(raw_entry.get("split_group_key") or uid)

    if task_mode == "counting":
        grounding = select_counting_grounding(raw_entry)
        if grounding is None:
            return None, "no_count_fact_at_least_2"
        entry["grounding"] = grounding
    elif task_mode == "ocr":
        grounding = select_ocr_grounding(raw_entry)
        if grounding is None:
            return None, "no_exact_ocr_text_2_to_24_characters"
        entry["grounding"] = grounding
    elif normal_strategy == "vlm_adaptive":
        entry["normal_strategy"] = normal_strategy
        optional_groundings: dict[str, dict[str, Any]] = {}
        count_grounding = select_optional_counting_grounding(raw_entry)
        if count_grounding is not None:
            optional_groundings["counting"] = count_grounding
        ocr_grounding = select_optional_ocr_grounding(raw_entry)
        if ocr_grounding is not None:
            optional_groundings["ocr"] = ocr_grounding
        entry["optional_groundings"] = optional_groundings
    return entry, None


def load_task_entries(
    metadata_paths: Iterable[str | Path],
    requested_mode: str,
    normal_strategy: str = "standard",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stats: Counter[str] = Counter()
    for metadata_path in metadata_paths:
        with Path(metadata_path).open(encoding="utf-8", buffering=8 * 1024 * 1024) as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                raw = json.loads(line)
                entry, reason = prepare_task_entry(
                    raw, requested_mode, normal_strategy
                )
                if entry is None:
                    stats[f"skipped_{reason}"] += 1
                    continue
                key = (entry["uid"], entry["task_mode"])
                if key in seen:
                    stats["skipped_duplicate_uid_and_task"] += 1
                    continue
                seen.add(key)
                entries.append(entry)
                stats[f"eligible_{entry['task_mode']}"] += 1
                if entry.get("normal_strategy") == "vlm_adaptive":
                    optional = entry.get("optional_groundings", {})
                    if "counting" in optional:
                        stats["adaptive_optional_counting"] += 1
                    if "ocr" in optional:
                        stats["adaptive_optional_ocr"] += 1
                    if not optional:
                        stats["adaptive_without_canonical_hint"] += 1
    stats["eligible_total"] = len(entries)
    return entries, dict(sorted(stats.items()))


def _validate_dialogue_shape(dialogue: Any) -> str | None:
    if not isinstance(dialogue, list):
        return "dialogue_not_list"
    if not 8 <= len(dialogue) <= 12:
        return "dialogue_turn_count_not_8_to_12"
    for index, turn in enumerate(dialogue):
        if not isinstance(turn, Mapping):
            return f"turn_{index}_not_object"
        expected_speaker = "ユーザー" if index % 2 == 0 else "アシスタント"
        if turn.get("speaker") != expected_speaker:
            return f"turn_{index}_speaker_not_{expected_speaker}"
        if not isinstance(turn.get("text"), str) or not turn["text"].strip():
            return f"turn_{index}_empty_text"
    return None


def _kanji_number(value: str) -> int | None:
    if not value:
        return None
    if "十" in value:
        if value.count("十") != 1:
            return None
        left, right = value.split("十")
        tens = 1 if not left else KANJI_DIGITS.get(left)
        ones = 0 if not right else KANJI_DIGITS.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(value) == 1:
        return KANJI_DIGITS.get(value)
    return None


def _count_numbers(text: str) -> list[int]:
    normalized = unicodedata.normalize("NFKC", text)
    matches: list[tuple[int, int]] = []
    counters = "|".join(re.escape(counter) for counter in COUNT_COUNTERS)
    # Require a counter for Arabic numerals, so an ordinal such as 「2番です」
    # cannot accidentally pass as a count of two visible objects.
    for match in re.finditer(rf"(?<!\d)(\d+)\s*(?:{counters})", normalized):
        matches.append((match.start(), int(match.group(1))))
    for match in re.finditer(rf"([〇零一二三四五六七八九十]+)(?:{counters})", normalized):
        value = _kanji_number(match.group(1))
        if value is not None:
            matches.append((match.start(), value))
    for surface, value in NATIVE_COUNT_FORMS.items():
        for match in re.finditer(re.escape(surface), normalized):
            matches.append((match.start(), value))
    return [value for _, value in sorted(matches)]


def _adaptive_auxiliary_indices(
    auxiliary_task: Mapping[str, Any], dialogue_length: int
) -> tuple[int, int] | str:
    question_index = auxiliary_task.get("question_turn_index")
    answer_index = auxiliary_task.get("answer_turn_index")
    if (
        not isinstance(question_index, int)
        or isinstance(question_index, bool)
        or not isinstance(answer_index, int)
        or isinstance(answer_index, bool)
    ):
        return "adaptive_auxiliary_indices_not_integers"
    # Keep the first exchange an ordinary image question/answer and make the
    # optional ability check a natural follow-up in the middle of the dialogue.
    if question_index < 4 or question_index % 2 != 0:
        return "adaptive_auxiliary_question_not_mid_dialogue_user_turn"
    if answer_index != question_index + 1 or answer_index >= dialogue_length:
        return "adaptive_auxiliary_answer_not_following_assistant_turn"
    return question_index, answer_index


def _contains_japanese(text: str) -> bool:
    return bool(re.search(r"[ぁ-ゟ゠-ヿ一-龯々〆ヶ]", text))


def explicit_count_question_turns(dialogue: Any) -> list[int]:
    """Find clear quantity questions, excluding '何人か'/'いくつか'."""
    if not isinstance(dialogue, list):
        return []
    return [
        index for index, turn in enumerate(dialogue)
        if isinstance(turn, Mapping) and turn.get("speaker") == "ユーザー"
        and EXPLICIT_COUNT_QUESTION_RE.search(
            unicodedata.normalize("NFKC", str(turn.get("text", "")))
        )
    ]


def _validate_adaptive_normal_dialogue(
    entry: Mapping[str, Any],
    dialogue: list[Mapping[str, Any]],
    auxiliary_task: Any,
) -> tuple[bool, str | None]:
    if not isinstance(auxiliary_task, Mapping):
        return False, "adaptive_auxiliary_task_missing_or_not_object"
    kind = auxiliary_task.get("kind")
    if kind not in ("none", "counting", "ocr"):
        return False, "adaptive_auxiliary_kind_not_supported"
    count_questions = explicit_count_question_turns(dialogue)
    if count_questions and kind != "counting":
        return False, "adaptive_count_question_requires_counting_metadata"

    if kind == "none":
        if auxiliary_task.get("grounding_source") != "none":
            return False, "adaptive_none_grounding_source_not_none"
        if auxiliary_task.get("question_turn_index") is not None:
            return False, "adaptive_none_question_turn_index_not_null"
        if auxiliary_task.get("answer_turn_index") is not None:
            return False, "adaptive_none_answer_turn_index_not_null"
        return True, None

    indices = _adaptive_auxiliary_indices(auxiliary_task, len(dialogue))
    if isinstance(indices, str):
        return False, indices
    _, answer_index = indices
    if kind == "counting" and any(index != indices[0] for index in count_questions):
        return False, "adaptive_multiple_count_questions"
    answer_text = str(dialogue[answer_index]["text"])
    optional_groundings = entry.get("optional_groundings")
    if not isinstance(optional_groundings, Mapping):
        optional_groundings = {}

    if kind == "counting":
        source = auxiliary_task.get("grounding_source")
        reported = auxiliary_task.get("answer_count")
        if not isinstance(reported, int) or isinstance(reported, bool):
            return False, "adaptive_counting_answer_count_not_integer"

        if source == "canonical":
            grounding = optional_groundings.get("counting")
            if not isinstance(grounding, Mapping):
                return False, "adaptive_counting_missing_canonical_candidate"
            expected = grounding.get("answer_count")
            if (
                not isinstance(expected, int)
                or isinstance(expected, bool)
                or reported != expected
            ):
                return False, "adaptive_counting_metadata_answer_mismatch"
        elif source == "vlm_visual":
            if not SELF_GROUNDED_COUNT_MINIMUM <= reported <= SELF_GROUNDED_COUNT_MAXIMUM:
                return False, "adaptive_visual_count_not_2_to_10"
            counted_target = auxiliary_task.get("counted_target")
            if not isinstance(counted_target, str) or not counted_target.strip():
                return False, "adaptive_visual_counted_target_missing"
            compact_target = _compact_text(counted_target)
            if (
                len(compact_target) > COUNTED_TARGET_MAX_CHARACTERS
                or not _contains_japanese(counted_target)
                or compact_target in GENERIC_COUNT_TARGETS
            ):
                return False, "adaptive_visual_counted_target_not_concrete_japanese"
            question_index, _ = indices
            question_text = str(dialogue[question_index]["text"])
            if compact_target not in _compact_text(question_text):
                return False, "adaptive_visual_count_question_missing_counted_target"
            if not any(term in question_text for term in COUNT_QUESTION_TERMS):
                return False, "adaptive_visual_count_question_not_quantity_question"
            if _count_numbers(question_text):
                return False, "adaptive_visual_count_question_leaks_answer"
            # The assistant must have introduced the target before the user,
            # who cannot see the image, asks the count follow-up. It must not
            # disclose a count in that setup turn either.
            prior_assistant_text = str(dialogue[question_index - 1]["text"])
            if compact_target not in _compact_text(prior_assistant_text):
                return False, "adaptive_visual_count_target_not_introduced"
            if _count_numbers(prior_assistant_text):
                return False, "adaptive_visual_count_introduced_with_answer_leak"
            expected = reported
        else:
            return False, "adaptive_counting_grounding_source_not_supported"

        numbers = _count_numbers(answer_text)
        if numbers != [expected]:
            return False, (
                f"adaptive_count_answer_mismatch_expected_{expected}_"
                f"observed_{numbers}"
            )
        if any(term in answer_text for term in APPROXIMATE_COUNT_TERMS):
            return False, "adaptive_count_answer_is_approximate_or_bounded"
        if any(term in answer_text for term in NEGATED_COUNT_TERMS):
            return False, "adaptive_count_answer_is_negated"
        return True, None

    source = auxiliary_task.get("grounding_source")
    if source not in ("canonical", "vlm_visual"):
        return False, "adaptive_ocr_grounding_source_not_supported"
    recognized_text = auxiliary_task.get("recognized_text")
    if not isinstance(recognized_text, str) or not recognized_text.strip():
        return False, "adaptive_ocr_recognized_text_missing"
    compact_recognized = _compact_text(recognized_text)
    if not OCR_MIN_CHARACTERS <= len(compact_recognized) <= OCR_MAX_CHARACTERS:
        return False, "adaptive_ocr_text_not_2_to_24_characters"
    if not _contains_japanese(recognized_text):
        return False, "adaptive_ocr_text_has_no_japanese_characters"

    if source == "canonical":
        grounding = optional_groundings.get("ocr")
        if not isinstance(grounding, Mapping):
            return False, "adaptive_ocr_missing_canonical_candidate"
        expected_text = grounding.get("text")
        if (
            not isinstance(expected_text, str)
            or _compact_text(expected_text) != compact_recognized
        ):
            return False, "adaptive_ocr_metadata_text_mismatch"
    if compact_recognized not in _compact_text(answer_text):
        return False, "adaptive_ocr_answer_does_not_contain_recognized_text"
    return True, None


def validate_task_dialogue(
    entry: Mapping[str, Any],
    dialogue: Any,
    auxiliary_task: Any = None,
) -> tuple[bool, str | None]:
    reason = _validate_dialogue_shape(dialogue)
    if reason is not None:
        return False, reason
    task_mode = str(entry.get("task_mode", "normal"))
    if task_mode == "normal":
        if entry.get("normal_strategy") == "vlm_adaptive":
            return _validate_adaptive_normal_dialogue(
                entry, dialogue, auxiliary_task
            )
        return True, None

    grounding = entry.get("grounding")
    if not isinstance(grounding, Mapping):
        return False, "missing_grounding"
    answer_index = grounding.get("answer_turn_index")
    if not isinstance(answer_index, int) or not 0 <= answer_index < len(dialogue):
        return False, "invalid_answer_turn_index"
    answer_text = str(dialogue[answer_index]["text"])

    if task_mode == "counting":
        expected = grounding.get("answer_count")
        if not isinstance(expected, int):
            return False, "missing_answer_count"
        numbers = _count_numbers(answer_text)
        if numbers != [expected]:
            return False, f"count_answer_mismatch_expected_{expected}_observed_{numbers}"
        if any(term in answer_text for term in APPROXIMATE_COUNT_TERMS):
            return False, "count_answer_is_approximate_or_bounded"
        if any(term in answer_text for term in NEGATED_COUNT_TERMS):
            return False, "count_answer_is_negated"
        return True, None

    if task_mode == "ocr":
        expected_text = grounding.get("text")
        if not isinstance(expected_text, str) or not expected_text:
            return False, "missing_ocr_text"
        if _compact_text(expected_text) not in _compact_text(answer_text):
            return False, "ocr_answer_does_not_contain_exact_text"
        return True, None

    return False, f"unsupported_task_mode_{task_mode}"


def public_grounding(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    grounding = entry.get("grounding")
    return dict(grounding) if isinstance(grounding, Mapping) else None


def selected_auxiliary_grounding(
    entry: Mapping[str, Any], auxiliary_task: Any
) -> dict[str, Any] | None:
    """Return code-owned canonical grounding for a validated adaptive choice."""
    if not isinstance(auxiliary_task, Mapping):
        return None
    kind = auxiliary_task.get("kind")
    if (
        kind not in ("counting", "ocr")
        or auxiliary_task.get("grounding_source") != "canonical"
    ):
        return None
    optional_groundings = entry.get("optional_groundings")
    if not isinstance(optional_groundings, Mapping):
        return None
    grounding = optional_groundings.get(kind)
    if not isinstance(grounding, Mapping):
        return None
    result = dict(grounding)
    result["question_turn_index"] = auxiliary_task["question_turn_index"]
    result["answer_turn_index"] = auxiliary_task["answer_turn_index"]
    return result


def auxiliary_verification_status(auxiliary_task: Any) -> str | None:
    """Classify adaptive evidence without promoting VLM observations to truth."""
    if not isinstance(auxiliary_task, Mapping):
        return None
    if auxiliary_task.get("kind") not in ("counting", "ocr"):
        return None
    source = auxiliary_task.get("grounding_source")
    if source == "canonical":
        return "verified_canonical"
    if source == "vlm_visual":
        return "unverified_vlm_visual"
    return None
