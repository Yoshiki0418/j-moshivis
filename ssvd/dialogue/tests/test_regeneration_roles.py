"""Offline checks of role isolation and candidate eligibility; naturalness needs human review."""

import json
import unittest

from ssvd.dialogue.regeneration_roles import next_step


# A synthetic 1px image used only to inspect request bodies.
IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+jFj8AAAAASUVORK5CYII="
)
COUNT = {"status": "ok", "scene": "公園にベンチが並ぶ。", "target": "ベンチ", "count": 3}
OCR = {"status": "ok", "scene": "通りに看板がある。", "target": "看板",
       "written": "AOBA 24", "reading": "アオバ にじゅうよん"}


class RegenerationRoleTests(unittest.TestCase):
    def step(self, kind, replies, **kwargs):
        return next_step("synthetic-protocol-example", kind, IMAGE, replies, pairs=4, **kwargs)

    def test_unreadable_or_uncountable_observation_stops_before_dialogue(self):
        for kind in ("ocr", "counting"):
            with self.subTest(kind=kind):
                result = self.step(kind, [{"status": "hold", "reason": "対象を確定できない"}])
                self.assertEqual(result["status"], "hold")
                self.assertTrue(result["requires_human_review"])
                self.assertNotIn("request", result)

    def test_ocr_question_writer_cannot_copy_the_grounded_answer(self):
        question = "看板には何と書いてあるの？"
        asking = self.step("ocr", [OCR], model="gpt-6.1-terra")
        self.assertEqual(asking["phase"], "user")
        self.assertEqual(asking["request"]["model"], "gpt-6.1-terra")
        prompt = json.dumps(asking["request"], ensure_ascii=False)
        self.assertNotIn(OCR["written"], prompt)
        self.assertNotIn(OCR["reading"], prompt)
        self.assertNotIn(IMAGE, prompt)
        self.assertNotIn(OCR["scene"], prompt)
        answering = self.step("ocr", [OCR, {"text": question}])
        self.assertEqual(answering["phase"], "assistant")
        self.assertIn(question, json.dumps(answering["request"], ensure_ascii=False))

    def test_image_observation_and_text_generation_are_separate_steps(self):
        for kind in ("counting", "ocr"):
            with self.subTest(kind=kind):
                result = self.step(kind, [])
                self.assertEqual(result["phase"], "grounding")
                self.assertEqual(result["request"]["model"], "gpt-6-luna")
                self.assertIn(IMAGE, json.dumps(result["request"]))
                self.assertNotIn("dialogue", result)

    def test_spoiling_the_answer_before_the_task_is_held(self):
        for kind, observation, spoiler in (
            ("counting", COUNT, "ベンチは三脚あります。"),
            ("ocr", OCR, "AOBA 24 と書いてあります。"),
        ):
            for history in ([], [{"text": "どんな写真？"}]):
                with self.subTest(kind=kind, previous_turns=len(history)):
                    result = self.step(kind, [observation, *history, {"text": spoiler}])
                    self.assertEqual(result["status"], "hold")
                    self.assertTrue(result["requires_human_review"])

    def two_pair_result(self, kind, observation, introduction, question, answer):
        replies = [observation]
        for utterance in ("どんな写真？", introduction, question, answer):
            step = next_step("synthetic-two-pair", kind, IMAGE, replies, pairs=2)
            if step["status"] == "hold":
                return step
            self.assertEqual(step["status"], "request")
            replies.append({"text": utterance})
        return next_step("synthetic-two-pair", kind, IMAGE, replies, pairs=2)

    def test_user_cannot_ask_about_a_target_never_introduced(self):
        for kind, observation, question, answer in (
            ("counting", COUNT, "ベンチは何脚あるの？", "三脚だよ。"),
            ("ocr", OCR, "看板には何と書いてある？", "アオバにじゅうよんだよ。"),
        ):
            with self.subTest(kind=kind):
                result = self.two_pair_result(kind, observation, "屋外の写真だよ。", question, answer)
                self.assertEqual(result["status"], "hold")

    def test_anaphoric_count_keeps_target_and_counter(self):
        setup = "公園にベンチが並んでるよ。"
        correct = self.two_pair_result("counting", COUNT, setup, "それ、何脚あるの？", "三脚だよ。")
        self.assertEqual(correct["status"], "pending_review")
        different_object = self.two_pair_result("counting", COUNT, setup, "ベンチは何脚あるの？",
                                                "自転車は三台だよ。")
        self.assertEqual(different_object["status"], "hold")

    def test_image_grounded_task_survives_until_human_review(self):
        fixtures = (
            ("counting", COUNT, "ベンチは何脚あるの？", "ベンチは三脚並んでいます。"),
            ("ocr", OCR, "その看板には何と書いてあるの？", "アオバにじゅうよん、と書いてあります。"),
        )
        for kind, observation, question, answer in fixtures:
            with self.subTest(kind=kind):
                replies = [observation]
                turns = []
                target = observation["target"]
                scene_exchanges = [
                    ("どんな写真？", f"外に{target}が見えるよ。"),
                    ("その近くも写ってる？", f"うん、{target}の周りも見えるね。"),
                    ("どこに目が行く？", f"やっぱり{target}かな。"),
                    ("手前の方は？", f"{target}が見えているよ。"),
                ]
                for pair in range(1, 5):
                    ask = self.step(kind, replies)
                    self.assertEqual(ask["phase"], "user")
                    focus = ask["task_pair"]
                    if kind == "ocr" and pair <= focus:
                        user_input = json.dumps(ask["request"], ensure_ascii=False)
                        self.assertNotIn(observation["written"], user_input)
                        self.assertNotIn(observation["reading"], user_input)
                    user_text, assistant_text = (question, answer) if pair == focus else scene_exchanges[pair - 1]
                    replies.append({"text": user_text})
                    response = self.step(kind, replies)
                    self.assertEqual(response["phase"], "assistant")
                    model_input = json.dumps(response["request"], ensure_ascii=False)
                    self.assertIn(user_text, model_input)
                    self.assertNotIn(IMAGE, model_input)
                    if kind == "ocr" and pair == focus:
                        self.assertIn(observation["written"], model_input)
                    replies.append({"text": assistant_text})
                    turns.extend([{"role": "user", "text": user_text},
                                  {"role": "assistant", "text": assistant_text}])
                candidate = self.step(kind, replies)
                self.assertEqual(candidate["status"], "pending_review")
                self.assertIs(candidate["requires_human_review"], True)
                self.assertEqual(candidate["kind"], kind)
                if kind == "ocr":
                    self.assertEqual(candidate["grounding"]["written"], observation["written"])
                    self.assertEqual(candidate["grounding"]["reading"], observation["reading"])
                else:
                    self.assertEqual(candidate["grounding"]["count"], observation["count"])
                self.assertEqual(candidate["dialogue"], turns)
                self.assertEqual(candidate["task_pair"], focus)
                self.assertNotIn("request", candidate)

                # Resuming with the same recorded responses reconstructs the same candidate.
                self.assertEqual(candidate, self.step(kind, json.loads(json.dumps(replies))))

                # A fluent but incorrect answer must not earn a review-ready result.
                index = 2 * focus
                inconsistent = list(replies[:index + 1])
                inconsistent[index] = {"text": "ベンチは四脚です。" if kind == "counting"
                                       else "アオバよんじゅうに、と書いてあります。"}
                self.assertEqual(self.step(kind, inconsistent)["status"], "hold")
                with self.assertRaises(ValueError):
                    self.step(kind, [*inconsistent, {"text": "次の質問は？"}])
