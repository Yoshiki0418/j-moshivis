"""CPU-only Japanese VQA metrics and aggregation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable


_PUNCTUATION_RE = re.compile(r"[\s\u3000、。．，！？!?・:：;；'\"“”‘’（）()［］\[\]｛｝{}〈〉《》「」『』【】…‥]+")
_LEADING_LABEL_RE = re.compile(r"^\s*(?:答え|回答)\s*(?:は|:|：)?\s*")
_CHOICE_RE = re.compile(
    r"(?:^|[\s（(]|選択肢|答えは|回答は)([A-ZＡ-Ｚ])(?=$|[\s）).、。]|です|だと思)",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Conservative normalization suitable for Japanese short answers.

    NFKC handles full-width Latin letters/numbers.  We remove punctuation and
    whitespace, but intentionally do not strip Japanese particles or polite
    endings: such transformations can silently turn a wrong answer into a hit.
    """

    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = _LEADING_LABEL_RE.sub("", value)
    return _PUNCTUATION_RE.sub("", value)


def exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(bool(pred) and any(pred == normalize_answer(ref) for ref in references))


def answer_in_response(prediction: str, references: Iterable[str]) -> float:
    """Whether a normalized reference appears in a conversational response.

    This is reported alongside exact match rather than replacing it.  It is
    useful for Moshi's sentence-style answers (``赤です`` vs ``赤``), but can
    over-credit negated statements; paper results should pair it with the
    stored per-sample output and, where needed, a frozen judge protocol.
    """

    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    negative_suffixes = (
        "ではない", "ではありません", "じゃない", "じゃありません",
        "でない", "でありません", "とは限らない",
    )
    for reference in references:
        ref_norm = normalize_answer(reference)
        if not ref_norm:
            continue
        start = 0
        while (position := pred.find(ref_norm, start)) >= 0:
            suffix = pred[position + len(ref_norm):]
            if not any(suffix.startswith(negative) for negative in negative_suffixes):
                return 1.0
            start = position + len(ref_norm)
    return 0.0


def _char_f1_pair(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return float(pred == ref and bool(ref))
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def char_f1(prediction: str, references: Iterable[str]) -> float:
    scores = [_char_f1_pair(prediction, ref) for ref in references]
    return max(scores, default=0.0)


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_similarity(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    scores: list[float] = []
    for reference in references:
        ref = normalize_answer(reference)
        denominator = max(len(pred), len(ref))
        scores.append(0.0 if denominator == 0 else 1.0 - _edit_distance(pred, ref) / denominator)
    return max(scores, default=0.0)


def extract_choice(
    prediction: str,
    choices: list[str] | None = None,
    choice_labels: list[str] | None = None,
) -> str | None:
    normalized = unicodedata.normalize("NFKC", prediction).upper().strip()
    # An answer naming alternatives is invalid even if one label happens to
    # match the stricter extraction pattern below.
    ambiguous = re.findall(r"([A-Z])\s*(?:か|または|もしくは|あるいは)\s*([A-Z])", normalized)
    if ambiguous:
        return None
    alphabetic_labels = {
        unicodedata.normalize("NFKC", match.group(1)).upper()
        for match in _CHOICE_RE.finditer(normalized)
    }
    if len(alphabetic_labels) == 1:
        candidate = next(iter(alphabetic_labels))
        if choice_labels is None or candidate in choice_labels:
            return candidate
    if len(alphabetic_labels) > 1:
        return None
    labels = choice_labels or ([chr(ord("A") + index) for index in range(len(choices))] if choices else [])
    normalized_labels = [unicodedata.normalize("NFKC", label).upper() for label in labels]
    label_hits: list[str] = []
    for label in normalized_labels:
        if label.isdigit():
            # CVQA's primary metric is deliberately strict: accept a bare
            # spoken label or a compact answer phrase such as ``答えは2番です``,
            # but not a number embedded in free text (``猫が2匹います``).
            numeric_answer = re.fullmatch(
                rf"\s*(?:(?:答え|回答|正解|選択肢)\s*(?:は|:|：)?\s*)?"
                rf"{re.escape(label)}\s*(?:番(?:目)?)?\s*"
                rf"(?:(?:です|だ|でしょう|と思います|だと思います))?\s*"
                rf"[。.！!？?]?\s*",
                normalized,
            )
            if numeric_answer:
                label_hits.append(label)
        elif label and label in normalized and len(label) > 1:
            label_hits.append(label)
    if len(set(label_hits)) == 1:
        return label_hits[0]
    if len(set(label_hits)) > 1:
        return None
    # Preserve conventional text-answer handling for non-numeric benchmarks.
    # The audio-adapted CVQA protocol uses all-numeric labels and deliberately
    # bypasses this fallback so its primary metric remains strict 1--4 accuracy.
    if choices and not all(label.isdigit() for label in normalized_labels):
        hits = [
            normalized_labels[index]
            for index, choice in enumerate(choices)
            if normalize_answer(choice)
            and normalize_answer(choice) in normalize_answer(prediction)
        ]
        if len(hits) == 1:
            return hits[0]
    return None


@dataclass(slots=True)
class SampleScore:
    sample_id: str
    exact_match: float
    answer_in_response: float
    char_f1: float
    normalized_edit_similarity: float
    choice_accuracy: float | None = None
    extracted_choice: str | None = None


def score_one(
    sample_id: str,
    prediction: str,
    references: list[str],
    *,
    task: str = "vqa",
    choices: list[str] | None = None,
    choice_labels: list[str] | None = None,
    correct_choice: str | None = None,
) -> SampleScore:
    choice_accuracy: float | None = None
    extracted: str | None = None
    if task == "multiple_choice":
        extracted = extract_choice(prediction, choices, choice_labels)
        target = correct_choice
        if choices and target in choices:
            labels = choice_labels or [chr(ord("A") + index) for index in range(len(choices))]
            target = labels[choices.index(target)]
        target = None if target is None else unicodedata.normalize("NFKC", target).upper()
        choice_accuracy = float(extracted is not None and extracted == target)
    return SampleScore(
        sample_id=sample_id,
        exact_match=exact_match(prediction, references),
        answer_in_response=answer_in_response(prediction, references),
        char_f1=char_f1(prediction, references),
        normalized_edit_similarity=normalized_edit_similarity(prediction, references),
        choice_accuracy=choice_accuracy,
        extracted_choice=extracted,
    )


def wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def aggregate(scores: list[SampleScore]) -> dict[str, Any]:
    if not scores:
        return {"num_samples": 0}
    size = len(scores)
    exact = sum(score.exact_match for score in scores)
    contains = sum(score.answer_in_response for score in scores)
    choices = [score.choice_accuracy for score in scores if score.choice_accuracy is not None]
    result: dict[str, Any] = {
        "num_samples": size,
        "exact_match": exact / size,
        "exact_match_ci95": wilson_interval(exact, size),
        "answer_in_response": contains / size,
        "answer_in_response_ci95": wilson_interval(contains, size),
        "char_f1": sum(score.char_f1 for score in scores) / size,
        "normalized_edit_similarity": sum(score.normalized_edit_similarity for score in scores) / size,
    }
    if choices:
        choice_sum = sum(choices)
        result["choice_accuracy"] = choice_sum / len(choices)
        result["choice_accuracy_ci95"] = wilson_interval(choice_sum, len(choices))
        result["num_multiple_choice"] = len(choices)
    return result


def score_to_dict(score: SampleScore) -> dict[str, Any]:
    return asdict(score)
