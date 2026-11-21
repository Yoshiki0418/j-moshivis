"""Candidate generation receives the entire conversation; quality findings survive completion."""

import hashlib
import json
import unittest

from ssvd.dialogue.pilot_api import OpenAIPilotAPI
from ssvd.dialogue.regeneration_roles import PROMPT_VERSION, VISUAL_V9, next_step
from ssvd.dialogue.visual_dialogue_prompts import ASSISTANT, USER

IMAGE = "data:image/jpeg;base64,AA=="
OCR = {"kind": "ocr", "target": "入口の看板", "mode": "exact", "written": "AOBA", "reading": "アオバ"}
COUNT = {"kind": "counting", "target": "自転車", "mode": "exact", "count": 2}


def step(replies, *, version=PROMPT_VERSION):
    return next_step("example", "ocr", IMAGE, replies, pairs=4, version=version)


class VisualDialogueTests(unittest.TestCase):
    def test_archived_requests_match_original_generation(self):
        question = {"text": "札には営業時間が書いてあるの？"}
        answer = {"text": "アオバって書いてあるね。", "tasks": [OCR]}
        # Builders: v4/v5 (0d4409f), v6 (d813b51), v7 (acea45d), v8 (40c63e3), v9 (9165aa6).
        # The third request follows an answer with private task metadata.
        expected = {
            "visual_dialogue_v4": [
                "32c834924b4691706ed0da21cdbc07abd834af65a2ce5abafe1bff1b057fbea5",
                "4c2e21298347f3758d404dc528d782948a913d69ad10abf39ceca313b354698a",
                "9af022aaa1aeac372d56e865d7a5c29828d6888aad84af104213df3c28d0e1db",
            ],
            "visual_dialogue_v5": [
                "d511799424f4b67447e46b853c2945700eceedc187d66b2136ee1bb2bf1af1c9",
                "1c61541ae9ec1508665a57dd5d8ddf943d06312cc6efec0c0e0539cb093ef761",
                "c8ad10d67cf0536e4704a07dfb2aa3ba4009ae7fed1bd460e08b73ad538e205b",
            ],
            "visual_dialogue_v6": [
                "2253f840dfbe7ff9d2928602ccf872a2f545b1595c70167b8e0f1b2a574d731c",
                "6930c1db3229c592dbe837c7a4b882bf9da6227add3866ee7985577b2e889de2",
                "9f52610afe0252633994901e0e74d95bbb5316ba01b3a39eca1ee64de8d3078c",
            ],
            "visual_dialogue_v7": [
                "8b713e3dd673f19d75bde618647b70af1bd1279031888b10498f1fe5606f8e12",
                "6930c1db3229c592dbe837c7a4b882bf9da6227add3866ee7985577b2e889de2",
                "02816a52423ac30c75f30bda9fbc09e97b798b9e7e75023b137eedb161bfc7fe",
            ],
            "visual_dialogue_v8": [
                "3d16bdc2c0898ec6f88e757cc8335e36c8f6335cbe2a6cd756a2b4a49ce13c70",
                "069b40d83cfa37d3c77cbbb1694af704d9138d328763e2e21a7ef0fe56e8e49b",
                "a3903b0fa8b3187471c6f1048f54c3d46ae5e0e230afb7d32bea5cb84d962d1f",
            ],
            "visual_dialogue_v9": [
                "0f11e3fa1887aa8a0adda0ff3897707bd7ac8f9daaf71afd46eabb0034634226",
                "0975555aae7de21ff96058abee9ac3972dcf5940de2a126df49d2b0d444ff4d4",
                "27d3fd306e86f60707955369f5d021dcd8baa921e1debb305607aab70832cba4",
            ],
        }
        for version, digests in expected.items():
            for replies, digest in zip(([], [question], [question, answer]), digests):
                with self.subTest(version=version, turn=len(replies)):
                    old = next_step("example", "ocr", IMAGE, replies, pairs=4, version=version)["request"]
                    wire = json.dumps(old, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), digest)

    def test_v7_changed_only_the_v6_user_instructions(self):
        question = {"text": "この看板、どっちを指してる？"}
        answer = {"text": "左を指しているよ。", "tasks": []}
        for replies in ([], [question], [question, answer]):
            with self.subTest(turn=len(replies) + 1):
                current = next_step("example", "ocr", IMAGE, replies, pairs=4, version="visual_dialogue_v7")["request"]
                previous = next_step("example", "ocr", IMAGE, replies, pairs=4, version="visual_dialogue_v6")["request"]
                if len(replies) % 2 == 0:
                    self.assertIn("相手に質問してください", current["instructions"])
                    self.assertNotEqual(current["instructions"], previous["instructions"])
                    self.assertEqual(current, {**previous, "instructions": current["instructions"]})
                else:
                    self.assertEqual(current, previous)

    def test_false_premise_can_be_corrected_and_printed_quantity_stays_ocr(self):
        printed = {"kind": "ocr", "target": "入口の札", "mode": "exact",
                   "written": "定員12名", "reading": "ていいんじゅうにめい"}
        replies = [
            {"text": "入口の札、残り十二席ってこと？"},
            {"text": "定員十二名と書いてあるよ。空席の数までは分からないね。", "tasks": [printed]},
            {"text": "棚の前に見える椅子はいくつ？"},
            {"text": "手前に二脚あるね。", "tasks": [{**COUNT, "target": "棚の前の椅子"}]},
            {"text": "窓から光が差してるね。"}, {"text": "床まで明るいね。", "tasks": []},
            {"text": "一息つきたくなるね。"}, {"text": "窓際が気持ちよさそう。", "tasks": []},
        ]
        result = step(replies, version=VISUAL_V9)
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual([e["task"]["kind"] for e in result["exchanges"]], ["ocr", "counting"])
        self.assertEqual(result["exchanges"][0]["task"]["written"], "定員12名")
        self.assertTrue(result["requires_human_review"])

    def test_personas_receive_image_and_all_public_turns_in_every_outgoing_request(self):
        replies, history = [], []
        api = OpenAIPilotAPI("offline")
        for turn in range(8):
            stage = step(replies)
            counting = next_step("example", "counting", IMAGE, replies, pairs=4)
            self.assertEqual(stage["request"], counting["request"])
            role = "assistant" if turn % 2 else "user"
            self.assertEqual(stage["phase"], role)
            wire = api.prepare(stage["request"])
            previous = step(replies, version=VISUAL_V9)
            previous_wire = api.prepare(previous["request"])
            instructions = wire["instructions"]
            self.assertEqual(wire, {**previous_wire, "instructions": instructions})
            for photo_persona in ("一緒に写真を眺め", "同じ写真を隣で見", "写真を見て"):
                self.assertNotIn(photo_persona, instructions)
            if role == "user":
                self.assertEqual(instructions, USER)
                self.assertEqual(instructions, previous_wire["instructions"])
                self.assertIn("尋ね", instructions)
                self.assertNotIn("tasks", instructions)
            else:
                self.assertEqual(instructions, ASSISTANT)
                self.assertNotEqual(instructions, previous_wire["instructions"])
                self.assertIn("答え", instructions)
                self.assertIn("tasks", instructions)
                self.assertIn("カタカナ", instructions)
                for field in ("written", "reading", "mode", "target", "reason", "unknown"):
                    self.assertNotIn(field, instructions)
            message, = wire["input"]
            prompt, image = message["content"]
            context = json.loads(prompt["text"])
            self.assertEqual(set(context), {"公開された会話", "出力形式"})
            self.assertIn("JSON", context["出力形式"])
            self.assertEqual(context["公開された会話"], history)
            self.assertEqual(image, {"type": "input_image", "image_url": IMAGE, "detail": "high"})
            text = f"発話{turn}"
            reply = {"text": text}
            if role == "assistant":
                reply["tasks"] = ["ocr"] if turn == 1 else []
            replies.append(reply)
            history.append({"role": role, "text": text})
        final = step(replies)
        self.assertEqual(final["status"], "pending_review")
        self.assertEqual(final["dialogue"], history)
        self.assertEqual(final["exchanges"], [{"pair": 1, "task": {"kind": "ocr"}}])

    def test_marks_only_answerers_reading_or_counting_in_the_current_pair(self):
        dialogue = [
            {"text": "三つ並んでいるのって何？"}, {"text": "植木鉢だよ。", "tasks": []},
            {"text": "青い札には何て書いてある？"}, {"text": "小さくて読めないね。", "tasks": ["ocr"]},
            {"text": "札に書かれた定員は？"}, {"text": "十二名って書いてあるよ。", "tasks": ["ocr"]},
            {"text": "受付の札と、並んでる人は見える？"},
            {"text": "アオバと読めるよ。人は少なくとも三人かな。", "tasks": ["ocr", "counting"]},
        ]
        done = step(dialogue)
        self.assertEqual(done["exchanges"], [
            {"pair": 2, "task": {"kind": "ocr"}},
            {"pair": 3, "task": {"kind": "ocr"}},
            {"pair": 4, "task": {"kind": "ocr"}},
            {"pair": 4, "task": {"kind": "counting"}},
        ])
        self.assertEqual(done["review_warnings"], [])
        self.assertTrue(done["requires_human_review"])

    def test_unmarked_or_malformed_markers_do_not_block_review_or_invent_evidence(self):
        dialogue = [
            {"text": "奥に何人いる？"}, {"text": "重なっていて数えきれないね。", "tasks": []},
            {"text": "机は何台？"}, {"text": "手前に二台あるね。"},
            {"text": "看板は読める？"}, {"text": "アオバだよ。", "tasks": [OCR, "other", "ocr", "ocr"]},
            {"text": "目印は？"}, {"text": "入口の青いのぼりかな。", "tasks": "ocr"},
        ]
        done = step(dialogue)
        self.assertEqual(done["status"], "pending_review")
        self.assertEqual(done["exchanges"], [{"pair": 3, "task": {"kind": "ocr"}}])
        self.assertEqual(done["review_warnings"], [
            {"turn": 6, "code": "invalid_task_marker"}, {"turn": 8, "code": "invalid_task_marker"},
        ])
        self.assertTrue(done["requires_human_review"])

    def test_ordinary_conversation_is_complete_without_a_task(self):
        dialogue = [
            {"text": "木陰が気持ちよさそう。"}, {"text": "ベンチまで日陰になっているね。", "tasks": []},
            {"text": "座るならどこがいい？"}, {"text": "川が見える端のほうかな。", "tasks": []},
            {"text": "水面がきらきらしてるね。"}, {"text": "風で光が揺れてるみたい。", "tasks": []},
            {"text": "ベンチの横に飾ってあるのは？"}, {"text": "昔の街並みの写真みたい。", "tasks": []},
        ]
        done = step(dialogue)
        self.assertEqual(done["status"], "pending_review")
        self.assertTrue(done["requires_human_review"])
        self.assertEqual(done["exchanges"], [])
        self.assertEqual(done["review_warnings"], [])
        self.assertEqual(done["dialogue"][-1]["text"], dialogue[-1]["text"])
        for version in ("visual_dialogue_v4", "visual_dialogue_v5"):
            archived = next_step("example", "ocr", IMAGE, dialogue, pairs=4, version=version)
            self.assertEqual(archived["review_warnings"], [{"turn": None, "code": "topic_not_covered"}])

    def test_warns_for_repeated_or_spoiled_questions_and_continues(self):
        replies = [{"text": "どこが気になる？"}, {"text": "アオバって書いてあるね。", "tasks": [OCR]},
                   {"text": "入口の看板は何て書いてある？"}, {"text": "アオバだね。", "tasks": [OCR]},
                   {"text": "この自転車、二台かな。何台ある？"}, {"text": "三台あるね。", "tasks": [COUNT]},
                   {"text": "入口の看板は何て書いてある？"}, {"text": "AOBAだね。", "tasks": [OCR]}]
        done = step(replies, version=VISUAL_V9)
        self.assertEqual(done["status"], "pending_review")
        self.assertTrue(done["requires_human_review"])
        flags = {(w["turn"], w["code"]) for w in done["review_warnings"]}
        self.assertTrue({(3, "previously_answered"), (5, "answer_in_question"), (6, "count_answer_mismatch"),
                         (7, "repeated_question"), (8, "spoken_form")} <= flags)
        self.assertEqual([e["pair"] for e in done["exchanges"]], [1, 2, 3, 4])

    def test_preserves_unknown_reading_and_zero_count_as_given(self):
        task = {"kind": "ocr", "mode": "unknown", "target": "青い看板"}
        sequence = [{"text": "青い看板には何と書いてあるの？"},
                    {"text": "そこまでは分からないね。", "tasks": [task]},
                    {"text": "自転車は何台ある？"}, {"text": "自転車は〇台です。", "tasks": [{**COUNT, "count": 0}]},
                    {"text": "どんな場所かな。"}, {"text": "森に見えるね。", "tasks": []},
                    {"text": "空も暗いね。"}, {"text": "そろそろ夕方かな。", "tasks": []}]
        done = step(sequence, version=VISUAL_V9)
        self.assertEqual(done["review_warnings"], [])
        self.assertEqual([e["task"] for e in done["exchanges"]], [task, {**COUNT, "count": 0}])
        for version in ("visual_dialogue_v4", "visual_dialogue_v5"):
            archived = next_step("example", "ocr", IMAGE, sequence, pairs=4, version=version)
            self.assertIn({"turn": 2, "code": "uncertainty_needs_visual_reason"}, archived["review_warnings"])

    def test_partial_reading_estimate_and_lower_bound_keep_their_meanings(self):
        fragment = {**OCR, "written": "営業", "reading": "えいぎょう", "mode": "partial", "reason": "右端が隠れている"}
        estimate = {**COUNT, "mode": "approximate", "count": 5, "reason": "自転車が重なっている"}
        minimum = {**COUNT, "target": "窓辺の鉢", "mode": "lower_bound", "count": 3}
        dialogue = [
            {"text": "入口の看板には何て書いてある？"}, {"text": "「営業」までは読めるよ。", "tasks": [fragment]},
            {"text": "自転車は何台ありそう？"}, {"text": "五台くらいかな。", "tasks": [estimate]},
            {"text": "窓辺の鉢はいくつある？"}, {"text": "少なくとも三つは見えるよ。", "tasks": [minimum]},
            {"text": "木漏れ日が差してるね。"}, {"text": "明るい窓辺だね。", "tasks": []},
        ]
        done = step(dialogue, version=VISUAL_V9)
        self.assertEqual(done["status"], "pending_review")
        self.assertEqual(done["review_warnings"], [])
        self.assertEqual([e["task"] for e in done["exchanges"]], [fragment, estimate, minimum])

    def test_bad_or_missing_annotations_preserve_text_and_only_missing_utterance_stops(self):
        replies = [{"text": "どんなお店？"}, {"text": "赤い屋根だね。", "tasks": "unknown"},
                   {"text": "青い看板は何て書いてある？"}, {"text": "ぼやけていて読めないね。"},
                   {"text": "近くに何人いる？"}, {"text": "一人かな。", "tasks": [{}]},
                   {"text": "街は静かだね。"}, {"text": "落ち着いた感じだね。", "tasks": []}]
        done = step(replies, version=VISUAL_V9)
        codes = {w["code"] for w in done["review_warnings"]}
        self.assertTrue({"missing_task_metadata", "invalid_task_metadata"} <= codes)
        self.assertEqual(len(done["dialogue"]), 8)
        self.assertEqual(done["status"], "pending_review")
        self.assertEqual(step([{"text": "お店が見えるね。"}, {}])["status"], "hold")
        with self.assertRaises(ValueError):
            step([{}, {"text": "続きを作らない。"}])
