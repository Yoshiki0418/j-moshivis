from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


DIALOGUE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DIALOGUE_DIR.parents[1]
if str(DIALOGUE_DIR) not in sys.path:
    sys.path.insert(0, str(DIALOGUE_DIR))

from generate_claude_cli import build_prompt, parse_dialogue as parse_claude  # noqa: E402
from generate_local_vlm import (  # noqa: E402
    load_entries as load_local_entries,
    parse_dialogue as parse_local,
    write_result,
)
from prompt_utils import (  # noqa: E402
    build_instruction,
    collect_done_keys,
    collect_done_uids,
    extract_auxiliary_task,
)
from task_profiles import (  # noqa: E402
    auxiliary_verification_status,
    canonical_task_mode,
    load_task_entries,
    prepare_task_entry,
    select_counting_grounding,
    selected_auxiliary_grounding,
    validate_task_dialogue,
)


def dialogue_with_first_answer(
    answer: str, turns: int = 8,
    first_question: str = "画像には対象がいくつ写っていますか？",
) -> list[dict[str, str]]:
    result = []
    for index in range(turns):
        if index % 2 == 0:
            text = first_question if index == 0 else "ほかには何が見えますか？"
            result.append({"speaker": "ユーザー", "text": text})
        else:
            text = answer if index == 1 else "背景にもいくつか物が見えますね。"
            result.append({"speaker": "アシスタント", "text": text})
    return result


def dialogue_with_auxiliary_answer(
    answer: str, question: str = "それはいくつ見えますか？", turns: int = 8
) -> list[dict[str, str]]:
    result = dialogue_with_first_answer("主な様子を説明しますね。", turns, "画像には何が写っていますか？")
    result[4] = {"speaker": "ユーザー", "text": question}
    result[5] = {"speaker": "アシスタント", "text": answer}
    return result


def counting_row() -> dict:
    return {
        "candidate_uid": "counting:flickr:123",
        "capability": "counting",
        "path": "data/images/example.jpg",
        "split_group_key": "flickr:123",
        "reference_annotations": [
            {
                "source_question_id": 1,
                "question_en": "How many birds are visible in the tree?",
                "answer_count": 1,
                "data_source": "imported_genome",
            },
            {
                "source_question_id": 2,
                "question_en": "How many red cars are parked beside the road?",
                "answer_count": 3,
                "data_source": "amt",
            },
            {
                "source_question_id": 3,
                "question_en": "How many dogs are visible?",
                "answer_count": 2,
                "data_source": "imported_genome",
            },
        ],
    }


class TaskEntryTest(unittest.TestCase):
    def test_auto_mode_uses_capability(self):
        self.assertEqual("counting", canonical_task_mode("auto", {"capability": "counting"}))
        self.assertEqual("ocr", canonical_task_mode("auto", {"capability": "japanese_ocr"}))
        self.assertEqual("normal", canonical_task_mode("auto", {}))

    def test_counting_normalizes_uid_path_and_injects_answer(self):
        row = counting_row()
        entry, reason = prepare_task_entry(row, "counting")
        self.assertIsNone(reason)
        self.assertEqual("counting:flickr:123", entry["uid"])
        self.assertTrue(Path(entry["absolute_image_path"]).is_absolute())
        self.assertEqual(2, entry["grounding"]["answer_count"])
        self.assertEqual(3, entry["grounding"]["source_question_id"])
        self.assertNotIn("uid", row)

    def test_counting_rejects_images_without_answer_at_least_two(self):
        row = counting_row()
        row["reference_annotations"] = [
            {"question_en": "How many people?", "answer_count": 0},
            {"question_en": "How many dogs?", "answer_count": 1},
            {"question_en": "How many cats?", "answer_count": "3"},
            {"question_en": "How many birds?", "answer_count": True},
        ]
        entry, reason = prepare_task_entry(row, "counting")
        self.assertIsNone(entry)
        self.assertEqual("no_count_fact_at_least_2", reason)

    def test_count_selection_is_independent_of_annotation_order(self):
        row = counting_row()
        first, _ = prepare_task_entry(row, "counting")
        row["reference_annotations"].reverse()
        second, _ = prepare_task_entry(row, "counting")
        self.assertEqual(first["grounding"], second["grounding"])

    def test_visible_object_count_beats_intrinsic_body_part_count(self):
        row = counting_row()
        row["reference_annotations"] = [
            {
                "source_question_id": 10,
                "question_en": "How many legs does the girl have?",
                "answer_count": 2,
                "data_source": "imported_genome",
            },
            {
                "source_question_id": 11,
                "question_en": "How many people are visible in the photo?",
                "answer_count": 3,
                "data_source": "amt",
            },
        ]
        entry, reason = prepare_task_entry(row, "counting")
        self.assertIsNone(reason)
        self.assertEqual(11, entry["grounding"]["source_question_id"])
        self.assertEqual(0, entry["grounding"]["question_quality_tier"])
        self.assertIn("explicitly_visual", entry["grounding"]["question_quality_flags"])

    def test_ocr_accepts_short_exact_text_and_rejects_long_page(self):
        short = {
            "candidate_uid": "ocr:sign:1",
            "capability": "japanese_ocr",
            "path": "sign.jpg",
            "reference_annotations": [
                {"kind": "exact_transcription", "text": "営業中"}
            ],
        }
        entry, reason = prepare_task_entry(short, "ocr")
        self.assertIsNone(reason)
        self.assertEqual("営業中", entry["grounding"]["text"])

        short["reference_annotations"][0]["text"] = "あ" * 25
        entry, reason = prepare_task_entry(short, "ocr")
        self.assertIsNone(entry)
        self.assertEqual("no_exact_ocr_text_2_to_24_characters", reason)

    def test_adaptive_normal_collects_only_safe_optional_groundings(self):
        row = counting_row()
        row["reference_annotations"] = [
            {
                "source_question_id": 1,
                "question_en": "How many dogs are visible in the photo?",
                "answer_count": 1,
                "data_source": "imported_genome",
            },
            {
                "source_question_id": 2,
                "question_en": "How many legs does the woman have?",
                "answer_count": 2,
                "data_source": "imported_genome",
            },
            {"kind": "exact_transcription", "text": "営業中"},
        ]
        entry, reason = prepare_task_entry(row, "normal", "vlm_adaptive")
        self.assertIsNone(reason)
        self.assertEqual("normal", entry["task_mode"])
        self.assertEqual("vlm_adaptive", entry["normal_strategy"])
        self.assertEqual(1, entry["optional_groundings"]["counting"]["answer_count"])
        self.assertEqual("営業中", entry["optional_groundings"]["ocr"]["text"])

        row["reference_annotations"] = [
            {
                "question_en": "How many legs does the woman have?",
                "answer_count": 2,
            },
            {"question_en": "How many dogs are absent?", "answer_count": 0},
            {"kind": "exact_transcription", "text": "あ" * 25},
        ]
        entry, reason = prepare_task_entry(row, "normal", "vlm_adaptive")
        self.assertIsNone(reason)
        self.assertEqual({}, entry["optional_groundings"])

    def test_adaptive_strategy_rejects_non_normal_task_mode(self):
        with self.assertRaisesRegex(ValueError, "requires normal task mode"):
            prepare_task_entry(counting_row(), "counting", "vlm_adaptive")


class PromptAndValidationTest(unittest.TestCase):
    def test_adaptive_none_cannot_hide_quantity_question(self):
        entry, _ = prepare_task_entry({"uid": "daily:test", "path": "image.jpg"}, "normal", "vlm_adaptive")
        dialogue = dialogue_with_first_answer("公園です。", first_question="何が見えますか？")
        auxiliary = {"kind": "none", "grounding_source": "none",
                     "question_turn_index": None, "answer_turn_index": None}
        dialogue[4]["text"] = "ベンチは何脚ありますか？"
        self.assertEqual(
            validate_task_dialogue(entry, dialogue, auxiliary),
            (False, "adaptive_count_question_requires_counting_metadata"),
        )
        dialogue[4]["text"] = "木がいくつかありますか？"
        self.assertEqual(validate_task_dialogue(entry, dialogue, auxiliary), (True, None))
        dialogue[4]["text"] = "何人か人がいますか？"
        self.assertEqual(validate_task_dialogue(entry, dialogue, auxiliary), (True, None))

    def test_unannotated_adaptive_prompt_cannot_offer_canonical_output(self):
        prompt = build_instruction("daily:unannotated", normal_strategy="vlm_adaptive")
        self.assertIn("カウントを優先", prompt)
        self.assertIn("canonical は使えません", prompt)
        self.assertNotRegex(prompt, r'"grounding_source"\s*:\s*"canonical"')
        self.assertIn('"counted_target":"ベンチ"', prompt)

    def test_adaptive_prompt_offers_only_available_canonical_kind(self):
        prompt = build_instruction(
            "daily:count-only", normal_strategy="vlm_adaptive",
            optional_groundings={"counting": {"question_en": "How many cats?", "answer_count": 2}},
        )
        self.assertIn("canonical を使える種類は counting だけ", prompt)
        self.assertRegex(prompt, r'"kind": "counting", "grounding_source": "canonical"')
        self.assertNotRegex(prompt, r'"kind": "ocr", "grounding_source": "canonical"')

    def test_non_hex_uid_is_deterministic_for_normal_prompt(self):
        first = build_instruction("counting:flickr:123")
        second = build_instruction("counting:flickr:123")
        self.assertEqual(first, second)

    def test_standard_normal_prompt_is_unchanged_by_default(self):
        prompt = build_instruction("counting:flickr:123")
        self.assertNotIn("画像ごとの補助タスク判断", prompt)
        self.assertNotIn("auxiliary_task", prompt)

    def test_adaptive_prompt_exposes_only_minimal_canonical_candidates(self):
        row = counting_row()
        row["reference_annotations"].append(
            {"kind": "exact_transcription", "text": "営業中"}
        )
        entry, _ = prepare_task_entry(row, "normal", "vlm_adaptive")
        prompt = build_instruction(
            entry["uid"],
            entry["task_mode"],
            None,
            entry["normal_strategy"],
            entry["optional_groundings"],
        )
        self.assertIn("通常対話", prompt)
        self.assertIn("最大1往復", prompt)
        self.assertIn('"answer_count": 2', prompt)
        self.assertIn('"text": "営業中"', prompt)
        self.assertNotIn("source_question_id", prompt)
        self.assertNotIn("selection_policy", prompt)

        answer = json.dumps(
            {
                "auxiliary_task": {
                    "kind": "none",
                    "grounding_source": "none",
                    "question_turn_index": None,
                    "answer_turn_index": None,
                },
                "dialogue": dialogue_with_first_answer("説明しますね。"),
            },
            ensure_ascii=False,
        )
        self.assertEqual("none", extract_auxiliary_task(answer)["kind"])

    def test_counting_prompt_receives_authoritative_question_and_answer(self):
        entry, _ = prepare_task_entry(counting_row(), "counting")
        prompt = build_prompt(
            entry["absolute_image_path"],
            entry["uid"],
            entry["task_mode"],
            entry["grounding"],
        )
        self.assertIn("How many dogs are visible?", prompt)
        self.assertIn('"answer_count": 2', prompt)
        self.assertIn("答えを推測し直さず", prompt)
        self.assertIn("2本あります", prompt)
        self.assertIn("2本います", prompt)
        self.assertNotIn("selection_policy", prompt)
        self.assertNotIn("source_question_id", prompt)

    def test_count_validation_accepts_canonical_fullwidth_number(self):
        entry, _ = prepare_task_entry(counting_row(), "counting")
        for text in ("犬は２匹いますね。", "犬は二匹いますね。", "犬はふたつ見えます。"):
            with self.subTest(text=text):
                valid, reason = validate_task_dialogue(
                    entry, dialogue_with_first_answer(text)
                )
                self.assertTrue(valid, reason)

    def test_count_validation_accepts_additional_count_counters(self):
        for expected, text in (
            (2, "2種類あります。"),
            (2, "２色あります。"),
            (2, "二階建てです。"),
            (2, "2層あります。"),
            (2, "2通りあります。"),
            (2, "2セットあります。"),
            (2, "2時間です。"),
            (2, "2皿あります。"),
            (2, "2切れあります。"),
            (2, "看板は2辺が見えます。"),
            (8, "看板は八角形です。"),
        ):
            with self.subTest(expected=expected, text=text):
                row = counting_row()
                row["reference_annotations"][2]["answer_count"] = expected
                entry, _ = prepare_task_entry(row, "counting")
                self.assertEqual(expected, entry["grounding"]["answer_count"])
                valid, reason = validate_task_dialogue(
                    entry, dialogue_with_first_answer(text)
                )
                self.assertTrue(valid, reason)

    def test_count_validation_rejects_wrong_or_multiple_numbers(self):
        entry, _ = prepare_task_entry(counting_row(), "counting")
        for text in (
            "犬は3匹いますね。",
            "2匹か3匹に見えます。",
            "犬がいますね。",
            "犬は約2匹いますね。",
            "犬は2匹以上いますね。",
            "犬は2匹はいません。",
            "答えは2番です。",
        ):
            with self.subTest(text=text):
                valid, _ = validate_task_dialogue(entry, dialogue_with_first_answer(text))
                self.assertFalse(valid)

    def test_parsers_reject_non_object_turns_without_crashing(self):
        malformed = json.dumps(
            {"dialogue": ["ユーザー", "アシスタント", "ユーザー", "アシスタント"]},
            ensure_ascii=False,
        )
        self.assertIsNone(parse_claude(malformed))
        self.assertIsNone(parse_local(malformed))

    def test_dialogue_shape_is_strict(self):
        entry, _ = prepare_task_entry(counting_row(), "counting")
        valid, _ = validate_task_dialogue(entry, dialogue_with_first_answer("2匹です。", 4))
        self.assertFalse(valid)
        dialogue = dialogue_with_first_answer("2匹です。")
        dialogue[2]["speaker"] = "アシスタント"
        valid, _ = validate_task_dialogue(entry, dialogue)
        self.assertFalse(valid)

    def test_ocr_validation_uses_canonical_text_in_first_answer(self):
        raw = {
            "candidate_uid": "ocr:sign:1",
            "path": "sign.jpg",
            "reference_annotations": [
                {"kind": "exact_transcription", "text": "営業中"}
            ],
        }
        entry, _ = prepare_task_entry(raw, "ocr")
        valid, reason = validate_task_dialogue(
            entry, dialogue_with_first_answer("『営業中』と書いてあります。")
        )
        self.assertTrue(valid, reason)
        valid, _ = validate_task_dialogue(
            entry, dialogue_with_first_answer("準備中と書いてあります。")
        )
        self.assertFalse(valid)

    def test_adaptive_normal_accepts_none_or_later_canonical_count(self):
        row = counting_row()
        row["reference_annotations"] = [
            {
                "question_en": "How many dogs are visible?",
                "answer_count": 1,
                "data_source": "imported_genome",
            }
        ]
        entry, _ = prepare_task_entry(row, "normal", "vlm_adaptive")
        none_task = {
            "kind": "none",
            "grounding_source": "none",
            "question_turn_index": None,
            "answer_turn_index": None,
        }
        valid, reason = validate_task_dialogue(
            entry, dialogue_with_first_answer("犬が見えますね。", first_question="何が見えますか？"), none_task
        )
        self.assertTrue(valid, reason)

        count_task = {
            "kind": "counting",
            "grounding_source": "canonical",
            "question_turn_index": 4,
            "answer_turn_index": 5,
            "answer_count": 1,
        }
        dialogue = dialogue_with_auxiliary_answer("犬は一匹見えますね。")
        valid, reason = validate_task_dialogue(entry, dialogue, count_task)
        self.assertTrue(valid, reason)
        grounding = selected_auxiliary_grounding(entry, count_task)
        self.assertEqual(1, grounding["answer_count"])
        self.assertEqual(5, grounding["answer_turn_index"])

        wrong = dict(count_task, answer_count=2)
        valid, _ = validate_task_dialogue(entry, dialogue, wrong)
        self.assertFalse(valid)
        too_early = dict(count_task, question_turn_index=0, answer_turn_index=1)
        valid, _ = validate_task_dialogue(entry, dialogue, too_early)
        self.assertFalse(valid)
        ungrounded = dict(count_task, grounding_source="vlm_visual")
        valid, _ = validate_task_dialogue(entry, dialogue, ungrounded)
        self.assertFalse(valid)

    def test_adaptive_normal_accepts_self_grounded_clear_visual_count(self):
        raw = {"candidate_uid": "normal:count:1", "path": "apples.jpg"}
        entry, reason = prepare_task_entry(raw, "normal", "vlm_adaptive")
        self.assertIsNone(reason)
        self.assertEqual({}, entry["optional_groundings"])
        visual_task = {
            "kind": "counting",
            "grounding_source": "vlm_visual",
            "question_turn_index": 4,
            "answer_turn_index": 5,
            "counted_target": "リンゴ",
            "answer_count": 2,
        }
        dialogue = dialogue_with_auxiliary_answer(
            "リンゴは2個あります。", "そのリンゴは何個ありますか？"
        )
        dialogue[3] = {
            "speaker": "アシスタント",
            "text": "テーブルの上には赤いリンゴも見えます。",
        }
        valid, validation_reason = validate_task_dialogue(
            entry, dialogue, visual_task
        )
        self.assertTrue(valid, validation_reason)
        self.assertIsNone(selected_auxiliary_grounding(entry, visual_task))
        self.assertEqual(
            "unverified_vlm_visual",
            auxiliary_verification_status(visual_task),
        )

    def test_adaptive_visual_count_rejects_unsafe_metadata_or_answer(self):
        raw = {"candidate_uid": "normal:count:2", "path": "apples.jpg"}
        entry, _ = prepare_task_entry(raw, "normal", "vlm_adaptive")
        base_task = {
            "kind": "counting",
            "grounding_source": "vlm_visual",
            "question_turn_index": 4,
            "answer_turn_index": 5,
            "counted_target": "リンゴ",
            "answer_count": 2,
        }
        valid_dialogue = dialogue_with_auxiliary_answer(
            "リンゴは2個あります。", "そのリンゴは何個ありますか？"
        )
        valid_dialogue[3] = {
            "speaker": "アシスタント",
            "text": "テーブルの上にはリンゴも見えます。",
        }
        cases = (
            (dict(base_task, answer_count=1), valid_dialogue),
            (dict(base_task, answer_count=11), valid_dialogue),
            (dict(base_task, counted_target="objects"), valid_dialogue),
            (dict(base_task, counted_target="もの"), valid_dialogue),
            (
                base_task,
                dialogue_with_auxiliary_answer(
                    "リンゴは2個あります。", "その果物は何個ありますか？"
                ),
            ),
            (
                base_task,
                dialogue_with_auxiliary_answer(
                    "リンゴは3個あります。", "そのリンゴは何個ありますか？"
                ),
            ),
            (
                base_task,
                dialogue_with_auxiliary_answer(
                    "リンゴは約2個あります。", "そのリンゴは何個ありますか？"
                ),
            ),
            (
                base_task,
                dialogue_with_auxiliary_answer(
                    "リンゴは2個ありません。", "そのリンゴは何個ありますか？"
                ),
            ),
        )
        for task, dialogue in cases:
            if dialogue is not valid_dialogue:
                dialogue[3] = {
                    "speaker": "アシスタント",
                    "text": "テーブルの上にはリンゴも見えます。",
                }
            with self.subTest(task=task, answer=dialogue[5]["text"]):
                valid, _ = validate_task_dialogue(entry, dialogue, task)
                self.assertFalse(valid)

    def test_adaptive_normal_validates_visual_and_canonical_ocr(self):
        raw = {"candidate_uid": "normal:1", "path": "sign.jpg"}
        visual_entry, _ = prepare_task_entry(raw, "normal", "vlm_adaptive")
        visual_task = {
            "kind": "ocr",
            "grounding_source": "vlm_visual",
            "question_turn_index": 4,
            "answer_turn_index": 5,
            "recognized_text": "営業中",
        }
        dialogue = dialogue_with_auxiliary_answer(
            "看板には『営業中』とあります。", "その看板には何て書いてありますか？"
        )
        valid, reason = validate_task_dialogue(visual_entry, dialogue, visual_task)
        self.assertTrue(valid, reason)
        self.assertIsNone(selected_auxiliary_grounding(visual_entry, visual_task))

        no_japanese = dict(visual_task, recognized_text="OPEN")
        valid, _ = validate_task_dialogue(visual_entry, dialogue, no_japanese)
        self.assertFalse(valid)

        raw["reference_annotations"] = [
            {"kind": "exact_transcription", "text": "営業中"}
        ]
        canonical_entry, _ = prepare_task_entry(raw, "normal", "vlm_adaptive")
        canonical_task = dict(visual_task, grounding_source="canonical")
        valid, reason = validate_task_dialogue(
            canonical_entry, dialogue, canonical_task
        )
        self.assertTrue(valid, reason)
        self.assertEqual(
            "営業中",
            selected_auxiliary_grounding(canonical_entry, canonical_task)["text"],
        )

    def test_adaptive_local_writer_keeps_normal_mode_and_code_grounding(self):
        row = counting_row()
        row["reference_annotations"] = [
            {
                "question_en": "How many dogs are visible?",
                "answer_count": 1,
                "data_source": "imported_genome",
            }
        ]
        entry, _ = prepare_task_entry(row, "normal", "vlm_adaptive")
        auxiliary_task = {
            "kind": "counting",
            "grounding_source": "canonical",
            "question_turn_index": 4,
            "answer_turn_index": 5,
            "answer_count": 1,
        }
        model_answer = json.dumps(
            {
                "auxiliary_task": auxiliary_task,
                "dialogue": dialogue_with_auxiliary_answer("犬は一匹いますね。"),
            },
            ensure_ascii=False,
        )
        output = io.StringIO()
        self.assertEqual(1, write_result(output, entry, model_answer, "test-vlm", "1/1"))
        record = json.loads(output.getvalue())
        self.assertEqual("normal", record["task_mode"])
        self.assertEqual("vlm_adaptive", record["normal_strategy"])
        self.assertEqual(auxiliary_task, record["auxiliary_task"])
        self.assertEqual("verified_canonical", record["verification_status"])
        self.assertEqual(1, record["grounding"]["answer_count"])

    def test_adaptive_local_writer_marks_visual_count_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "apples.jpg"
            image.write_bytes(b"content-bound-test-image")
            entry, _ = prepare_task_entry(
                {"candidate_uid": "normal:count:writer", "path": str(image)},
                "normal",
                "vlm_adaptive",
            )
            auxiliary_task = {
                "kind": "counting",
                "grounding_source": "vlm_visual",
                "question_turn_index": 4,
                "answer_turn_index": 5,
                "counted_target": "リンゴ",
                "answer_count": 2,
            }
            dialogue = dialogue_with_auxiliary_answer(
                "リンゴは2個あります。", "そのリンゴは何個ありますか？"
            )
            dialogue[3] = {
                "speaker": "アシスタント",
                "text": "テーブルの上にはリンゴも見えます。",
            }
            model_answer = json.dumps(
                {"auxiliary_task": auxiliary_task, "dialogue": dialogue},
                ensure_ascii=False,
            )
            output = io.StringIO()
            self.assertEqual(
                1, write_result(output, entry, model_answer, "test-vlm", "1/1")
            )
            record = json.loads(output.getvalue())
            self.assertEqual("unverified_vlm_visual", record["verification_status"])
            self.assertTrue(record["review_required"])
            self.assertRegex(record["review_id"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["image_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("grounding", record)

    def test_local_writer_attaches_code_owned_grounding(self):
        entry, _ = prepare_task_entry(counting_row(), "counting")
        model_answer = json.dumps(
            {"dialogue": dialogue_with_first_answer("犬は2匹いますね。")},
            ensure_ascii=False,
        )
        output = io.StringIO()
        self.assertEqual(1, write_result(output, entry, model_answer, "test-vlm", "1/1"))
        record = json.loads(output.getvalue())
        self.assertEqual("counting", record["task_mode"])
        self.assertEqual(2, record["grounding"]["answer_count"])
        self.assertEqual("flickr:123", record["split_group_key"])

    def test_registry_is_task_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                json.dumps({"uid": "same", "task_mode": "counting"}) + "\n"
                + json.dumps({"uid": "legacy"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                {("same", "counting"), ("legacy", "normal")},
                collect_done_keys(Path(directory)),
            )
            self.assertEqual(
                {"same", "legacy"}, collect_done_uids(Path(directory))
            )

    def test_registry_scope_uid_excludes_other_task_while_task_scope_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "meta.jsonl"
            metadata.write_text(
                json.dumps(
                    {"candidate_uid": "same", "path": "image.jpg"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "done.jsonl").write_text(
                json.dumps({"uid": "same", "task_mode": "counting"}) + "\n",
                encoding="utf-8",
            )
            task_entries, _, _ = load_local_entries(
                [metadata], root, None, task_mode="normal", registry_scope="task"
            )
            uid_entries, _, _ = load_local_entries(
                [metadata], root, None, task_mode="normal", registry_scope="uid"
            )
            self.assertEqual(["same"], [entry["uid"] for entry in task_entries])
            self.assertEqual([], uid_entries)


class RealManifestContractTest(unittest.TestCase):
    manifest = REPO_ROOT / "data/images/targeted_v4/image_pool_v1/counting_images.jsonl"

    @unittest.skipUnless(manifest.is_file(), "local targeted counting manifest is absent")
    def test_real_manifest_filters_to_3150_counting_images(self):
        entries, stats = load_task_entries([self.manifest], "counting")
        self.assertEqual(6591, stats["input_rows"])
        self.assertEqual(3150, stats["eligible_counting"])
        self.assertEqual(3150, len(entries))
        self.assertTrue(all(2 <= e["grounding"]["answer_count"] <= 15 for e in entries))

    @unittest.skipUnless(manifest.is_file(), "local targeted counting manifest is absent")
    def test_uid_registry_scope_leaves_exact_3441_normal_residual_images(self):
        counting_entries, _ = load_task_entries([self.manifest], "counting")
        self.assertEqual(3150, len(counting_entries))
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory)
            (registry / "counting.jsonl").write_text(
                "".join(
                    json.dumps(
                        {"uid": entry["uid"], "task_mode": "counting"},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for entry in counting_entries
                ),
                encoding="utf-8",
            )
            shard0, _, _ = load_local_entries(
                [self.manifest],
                registry,
                None,
                task_mode="normal",
                normal_strategy="vlm_adaptive",
                registry_scope="uid",
                shard=0,
                num_shards=2,
            )
            shard1, _, _ = load_local_entries(
                [self.manifest],
                registry,
                None,
                task_mode="normal",
                normal_strategy="vlm_adaptive",
                registry_scope="uid",
                shard=1,
                num_shards=2,
            )
        uids0 = {entry["uid"] for entry in shard0}
        uids1 = {entry["uid"] for entry in shard1}
        self.assertFalse(uids0 & uids1)
        self.assertEqual(3441, len(uids0 | uids1))
        self.assertFalse(
            (uids0 | uids1) & {entry["uid"] for entry in counting_entries}
        )


if __name__ == "__main__":
    unittest.main()
