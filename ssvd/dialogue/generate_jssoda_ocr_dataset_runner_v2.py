"""Production runner accepting 10-12 turns while grounding every OCR turn."""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from ssvd.dialogue import generate_jssoda_ocr_dataset as production
from ssvd.dialogue import generate_jssoda_ocr_dataset_all as adaptive
from ssvd.dialogue import generate_jssoda_ocr_dataset_runner as runner
from ssvd.dialogue import generate_jssoda_ocr_pilot as pilot


def flexible_validate(
    dialogue: Any,
    targets: list[Mapping[str, Any]],
    reference: str,
    candidates: list[Mapping[str, str]],
) -> tuple[bool, str]:
    if not isinstance(dialogue, list) or len(dialogue) not in (10, 12):
        return False, "dialogue must contain 10 or 12 turns"
    for index, turn in enumerate(dialogue):
        expected_speaker = "ユーザー" if index % 2 == 0 else "アシスタント"
        if (
            not isinstance(turn, Mapping)
            or turn.get("speaker") != expected_speaker
            or not isinstance(turn.get("text"), str)
            or not turn["text"].strip()
        ):
            return False, f"invalid dialogue turn {index}"

    expected_target_count = min(len(candidates), len(dialogue) // 2 - 1)
    if len(targets) != expected_target_count:
        return False, (
            f"expected {expected_target_count} grounded OCR targets, got {len(targets)}"
        )
    grounded_questions = {int(row["question_turn_index"]) for row in targets}
    for question_index in range(2, len(dialogue), 2):
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
    production.load_entries = adaptive.load_all_entries
    production.build_prompt = runner.fixed_adaptive_prompt
    production.validate = flexible_validate
    production.main()
