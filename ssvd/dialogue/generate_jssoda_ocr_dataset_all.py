"""Full JSSODa entrypoint, including the five short-document edge cases.

The main production generator uses four grounded OCR exchanges.  Five of the
17,991 pages expose fewer than four safe 8-24 character anchors, so this
entrypoint lowers only those pages to one, two, or three OCR exchanges and
uses grounded document questions for the remaining exchanges.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Mapping

from ssvd.dialogue import generate_jssoda_ocr_dataset as production
from ssvd.dialogue import generate_jssoda_ocr_pilot as pilot


def load_all_entries(metadata: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    with metadata.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record.get("candidate_uid") or f"line:{line_number}")
            try:
                reference, annotation = pilot.exact_transcription(record)
            except ValueError as error:
                rejected[uid] = str(error)
                continue
            image_path = Path(str(record.get("path", "")))
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            if not image_path.is_file():
                rejected[uid] = f"image is missing: {image_path}"
                continue
            candidates = production.strict_span_candidates(reference)
            if not candidates:
                rejected[uid] = "no safe OCR candidates"
                continue
            entry = dict(record)
            entry["_reference_text"] = reference
            entry["_annotation"] = dict(annotation)
            entry["_image_path"] = image_path
            entry["_ocr_candidates"] = candidates
            entry["_target_count"] = min(4, len(candidates))
            entries.append(entry)
    entries.sort(key=lambda row: str(row["candidate_uid"]))
    return entries, rejected


def adaptive_prompt(entry: Mapping[str, Any]) -> str:
    target_count = int(entry["_target_count"])
    prompt = production.build_prompt(entry)
    if target_count == 4:
        return prompt
    general_count = 4 - target_count
    return prompt.replace(
        "- 残る4組はすべてOCR質問応答です。",
        f"- 残る4組のうち、{target_count}組をOCR質問応答にします。\n"
        f"- 残り{general_count}組は文書の概要に関する短い質問応答にします。\n"
        "- 概要質問では、段落・行・最初・最後などの位置指定や読み上げ依頼を使いません。",
    )


def adaptive_validate(
    dialogue: Any,
    targets: list[Mapping[str, Any]],
    reference: str,
    candidates: list[Mapping[str, str]],
) -> tuple[bool, str]:
    expected_target_count = min(4, len(candidates))
    if not isinstance(dialogue, list) or len(dialogue) != 10:
        return False, "dialogue must contain exactly 10 turns"
    for index, turn in enumerate(dialogue):
        expected_speaker = "ユーザー" if index % 2 == 0 else "アシスタント"
        if (
            not isinstance(turn, Mapping)
            or turn.get("speaker") != expected_speaker
            or not isinstance(turn.get("text"), str)
            or not turn["text"].strip()
        ):
            return False, f"invalid dialogue turn {index}"
    if len(targets) != expected_target_count:
        return False, (
            f"expected {expected_target_count} grounded OCR targets, got {len(targets)}"
        )

    grounded_questions = {int(row["question_turn_index"]) for row in targets}
    for question_index in (2, 4, 6, 8):
        question = str(dialogue[question_index]["text"])
        looks_like_ocr = any(
            term in question for term in pilot.LOCATION_QUESTION_TERMS
        ) or any(term in question for term in production.READ_REQUEST_TERMS)
        if looks_like_ocr and question_index not in grounded_questions:
            return False, f"OCR-like question {question_index} is not grounded"

    reference_compact = pilot.compact(reference)
    allowed = {pilot.compact(str(row["exact_text"])) for row in candidates}
    prior_assistant = ""
    for turn_index, turn in enumerate(dialogue):
        surface = pilot.compact(str(turn["text"]))
        if turn_index % 2 == 1:
            prior_assistant += surface
            continue
        for candidate in allowed:
            match = difflib.SequenceMatcher(None, candidate, surface).find_longest_match()
            shared = candidate[match.a : match.a + match.size]
            if match.size >= 6 and shared not in prior_assistant:
                return False, f"user turn {turn_index} leaks unseen OCR content"

    seen: set[str] = set()
    for target in targets:
        question_index = int(target["question_turn_index"])
        answer_index = int(target["answer_turn_index"])
        exact = pilot.compact(str(target["exact_text"]))
        question = str(dialogue[question_index]["text"])
        answer = pilot.compact(str(dialogue[answer_index]["text"]))
        answer_without_wrappers = answer.strip("「」『』。！？")
        if answer_index != question_index + 1:
            return False, "target indices are not adjacent"
        if exact in seen:
            return False, "OCR target is duplicated"
        if exact not in allowed or exact not in reference_compact or exact not in answer:
            return False, "OCR target is not grounded in the canonical transcription"
        if not 8 <= len(exact) <= 24:
            return False, "OCR target is outside the 8-24 character limit"
        if len(answer_without_wrappers) < len(exact) + 4:
            return False, "OCR answer is not a complete sentence around the target"
        if answer_without_wrappers not in reference_compact:
            return False, "full OCR answer is not in the canonical transcription"
        if "読みます" in question:
            return False, "OCR question uses an unnatural reading declaration"
        if not any(term in question for term in pilot.LOCATION_QUESTION_TERMS):
            return False, "OCR question does not identify a text position"
        if not any(term in question for term in production.READ_REQUEST_TERMS):
            return False, "OCR question is not a natural reading request"
        question_without_location = pilot.compact(question)
        for term in pilot.LOCATION_QUESTION_TERMS:
            question_without_location = question_without_location.replace(
                pilot.compact(term), ""
            )
        overlap = difflib.SequenceMatcher(
            None, exact, question_without_location
        ).find_longest_match().size
        if overlap >= 4:
            return False, "OCR answer text is leaked in the user question"
        seen.add(exact)
    return True, "ok"


if __name__ == "__main__":
    production.load_entries = load_all_entries
    production.build_prompt = adaptive_prompt
    production.validate = adaptive_validate
    production.main()
