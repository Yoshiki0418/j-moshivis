"""Candidate generation receives the entire conversation; quality findings survive completion."""

import hashlib
import json
import unittest

from ssvd.dialogue.regeneration_roles import next_step

IMAGE = "data:image/jpeg;base64,AA=="
OCR = {"kind": "ocr", "target": "入口の看板", "mode": "exact", "written": "AOBA", "reading": "アオバ"}
COUNT = {"kind": "counting", "target": "自転車", "mode": "exact", "count": 2}


def step(replies):
    return next_step("example", "ocr", IMAGE, replies, pairs=4)


class VisualDialogueTests(unittest.TestCase):
    def test_role_policies_change_without_changing_historical_requests(self):
        question = {"text": "札には営業時間が書いてあるの？"}
        # Taken from the v4 requests before the role instructions changed.
        cases = [
            ([], "位置・色・形", "32c834924b4691706ed0da21cdbc07abd834af65a2ce5abafe1bff1b057fbea5"),
            ([question], "自分の目で確かめる", "4c2e21298347f3758d404dc528d782948a913d69ad10abf39ceca313b354698a"),
        ]
        for replies, cue, digest in cases:
            with self.subTest(replies=replies):
                current = step(replies)["request"]
                old = next_step("example", "ocr", IMAGE, replies, pairs=4, legacy_v4=True)["request"]
                self.assertIn(cue, current["instructions"])
                self.assertNotIn(cue, old["instructions"])
                self.assertEqual(current["input"], old["input"])
                wire = json.dumps(old, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(wire).hexdigest(), digest)

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
        result = step(replies)
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual([e["task"]["kind"] for e in result["exchanges"]], ["ocr", "counting"])
        self.assertEqual(result["exchanges"][0]["task"]["written"], "定員12名")
        self.assertTrue(result["requires_human_review"])

    def test_image_and_all_public_turns_in_every_request_and_no_observer(self):
        replies, history = [], []
        for turn in range(8):
            stage = step(replies)
            role = "assistant" if turn % 2 else "user"
            self.assertEqual(stage["phase"], role)
            prompt, image = stage["request"]["input"][0]["content"]
            context = json.loads(prompt["text"])
            self.assertIn("JSON", context["出力形式"])
            self.assertEqual(context["公開された会話"], history)
            self.assertEqual(image, {"type": "input_image", "image_url": IMAGE, "detail": "high"})
            self.assertNotIn('AOBA', prompt["text"])
            if role == "user" and turn >= 2:
                self.assertNotIn("画像に合えば扱える話題の一例", context)
                self.assertIn(1, context["すでに答えた往復（同じことをもう一度尋ねない）"])
            text = f"発話{turn}"
            reply = {"text": text}
            if role == "assistant":
                reply["tasks"] = [OCR] if turn == 1 else []
            replies.append(reply)
            history.append({"role": role, "text": text})
        final = step(replies)
        self.assertEqual(final["status"], "pending_review")
        self.assertEqual(final["dialogue"], history)
        self.assertEqual(final["exchanges"], [{"pair": 1, "task": OCR}])

    def test_warns_for_repeated_or_spoiled_questions_and_continues(self):
        replies = [{"text": "どこが気になる？"}, {"text": "アオバって書いてあるね。", "tasks": [OCR]},
                   {"text": "入口の看板は何て書いてある？"}, {"text": "アオバだね。", "tasks": [OCR]},
                   {"text": "この自転車、二台かな。何台ある？"}, {"text": "三台あるね。", "tasks": [COUNT]},
                   {"text": "入口の看板は何て書いてある？"}, {"text": "AOBAだね。", "tasks": [OCR]}]
        done = step(replies)
        self.assertEqual(done["status"], "pending_review")
        self.assertTrue(done["requires_human_review"])
        flags = {(w["turn"], w["code"]) for w in done["review_warnings"]}
        self.assertTrue({(3, "previously_answered"), (5, "answer_in_question"), (6, "count_answer_mismatch"),
                         (7, "repeated_question"), (8, "spoken_form")} <= flags)
        self.assertEqual([e["pair"] for e in done["exchanges"]], [1, 2, 3, 4])

    def test_preserves_unknown_reading_and_zero_count_as_given(self):
        task = {"kind": "ocr", "mode": "unknown", "target": "青い看板", "reason": "葉が重なっている"}
        sequence = [{"text": "青い看板には何と書いてあるの？"},
                    {"text": "何と書いてあるか読み取れません。", "tasks": [task]},
                    {"text": "自転車は何台ある？"}, {"text": "自転車は〇台です。", "tasks": [{**COUNT, "count": 0}]},
                    {"text": "どんな場所かな。"}, {"text": "森に見えるね。", "tasks": []},
                    {"text": "空も暗いね。"}, {"text": "そろそろ夕方かな。", "tasks": []}]
        done = step(sequence)
        self.assertEqual(done["review_warnings"], [])
        self.assertEqual(len(done["exchanges"]), 2)

    def test_bad_or_missing_annotations_preserve_text_and_only_missing_utterance_stops(self):
        replies = [{"text": "どんなお店？"}, {"text": "赤い屋根だね。", "tasks": "unknown"},
                   {"text": "青い看板は何て書いてある？"}, {"text": "ぼやけていて読めないね。"},
                   {"text": "近くに何人いる？"}, {"text": "一人かな。", "tasks": [{}]},
                   {"text": "街は静かだね。"}, {"text": "落ち着いた感じだね。", "tasks": []}]
        done = step(replies)
        codes = {w["code"] for w in done["review_warnings"]}
        self.assertTrue({"missing_task_metadata", "invalid_task_metadata", "topic_not_covered"} <= codes)
        self.assertEqual(len(done["dialogue"]), 8)
        self.assertEqual(done["status"], "pending_review")
        self.assertEqual(step([{"text": "お店が見えるね。"}, {}])["status"], "hold")
        with self.assertRaises(ValueError):
            step([{}, {"text": "続きを作らない。"}])
