"""Check that the follow-up can replay the first pilot without changing its requests."""

from functools import partial
import copy
import unittest
from unittest.mock import patch

from ssvd.dialogue.preview_regeneration import OBSERVATION_VERSION, preview as preview_dialogue

from ssvd.dialogue.regeneration_roles_v3 import next_step

preview = partial(preview_dialogue, prompt_version=OBSERVATION_VERSION)


IMAGE = "data:image/png;base64,AA=="
SIGN = {"kind": "ocr", "mode": "exact", "target": "入口の看板",
        "written": "青空", "reading": "あおぞら"}
SECOND_SIGN = {**SIGN, "target": "出口の看板", "written": "夕日", "reading": "ゆうひ"}


def observe(*tasks):
    return {"status": "ok", "scene": "建物には入口と出口があり、両方に看板がある。", "tasks": list(tasks)}


def step(replies, *, recovery=False):
    return next_step("synthetic", "ocr", IMAGE, replies, pairs=4, recovery=recovery)


class PilotReplayRulesTests(unittest.TestCase):
    def test_strict_accepted_replies_keep_request_bodies_order_and_outcome(self):
        observation = observe(SIGN)
        replies = []
        request_count = 0
        while True:
            strict = step(replies)
            self.assertEqual(step(replies, recovery=True), strict)
            if strict["status"] != "request":
                break
            request_count += 1
            if strict["phase"] == "grounding":
                replies.append(observation)
                continue
            pair = 1 + (len(replies) - 1) // 2
            if strict["phase"] == "user":
                text = "その看板には何と書いてあるの？" if strict["task_pairs"]["ocr"] == pair else "周りはどんなふう？"
            else:
                text = "青空と書いてあるよ。" if strict["task_pairs"]["ocr"] == pair else "入口の看板が目立つね。"
            replies.append({"text": text})
        self.assertEqual(strict["status"], "pending_review")
        self.assertEqual(request_count, 9)

    def test_duplicate_distinct_observations_pick_first_without_mutating_receipt(self):
        original = observe(SIGN, SECOND_SIGN)
        saved = copy.deepcopy(original)
        self.assertEqual(step([original])["reason"], "invalid_or_duplicate_task")
        self.assertEqual(step([original], recovery=True), step([observe(SIGN)]))
        self.assertEqual(original, saved)
        samples = [original]
        for _ in range(8):
            result = step(samples, recovery=True)
            pair = 1 + (len(samples) - 1) // 2
            due = result["task_pairs"]["ocr"] == pair
            if result["phase"] == "user":
                samples.append({"text": "何と書いてあるの？" if due else "どんな景色？"})
            else:
                samples.append({"text": "青空だよ。" if due else "入口の看板があるよ。"})
        done = step(samples, recovery=True)
        self.assertEqual(done["status"], "pending_review")
        self.assertEqual(done["grounding"]["tasks"], [SIGN])
        self.assertEqual(samples[0], saved)

    def test_reading_separator_is_preserved_as_a_spoken_pause(self):
        original = observe({**SIGN, "written": "北／南", "reading": "きた／みなみ"})
        saved = copy.deepcopy(original)
        self.assertEqual(step([original])["reason"], "ocr_needs_spoken_kana_reading")
        replies = [original]
        while True:
            stage = step(replies, recovery=True)
            if stage["phase"] == "assistant" and (len(replies) - 1) // 2 + 1 == stage["task_pairs"]["ocr"]:
                break
            replies.append({"text": "その看板には何と書いてあるの？"
                            if stage["phase"] == "user" and (len(replies) - 1) // 2 + 1 == stage["task_pairs"]["ocr"]
                            else ("どんな景色？" if stage["phase"] == "user" else "入口の看板があるよ。")})
        prompt = stage["request"]["input"][0]["content"][0]["text"]
        self.assertIn("きた、みなみ", prompt)
        self.assertIn("北／南", prompt)
        self.assertEqual(original, saved)

    def test_no_guesses_about_contradictions_unreadable_data_or_alternate_readings(self):
        for evidence in (
            observe(SIGN, {**SECOND_SIGN, "target": SIGN["target"]}),
            observe(SIGN, {**SECOND_SIGN, "reading": ""}),
            observe({**SIGN, "reading": "あおぞら／せいくう"}),
            observe({**SIGN, "written": "青／空", "reading": "青／そら"}),
            {**observe(SIGN), "scene": "青空と書かれた入口の看板がある。"},
            {"status": "hold", "reason": "invalid_or_duplicate_task"},
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(step([evidence], recovery=True), step([evidence]))
                self.assertEqual(step([evidence], recovery=True)["status"], "hold")

    def test_duplicate_observations_need_distinct_referents(self):
        for left, right in (
            ("正面の大きな看板", "大きい看板"),
            ("駅舎の白い看板", "駅舎の白い看板の駅名"),
            ("駅名", "隣の駅名"),
            ("入口の看板", "入り口の看板"),
            ("赤い飲食店の看板", "白い飲食店の看板"),
            ("左側にある看板", "右側を向いた文字"),
            ("入口と出口の看板", "出口の看板"),
        ):
            for example in (observe({**SIGN, "target": left}, {**SECOND_SIGN, "target": right}),
                            observe({"kind": "counting", "mode": "exact", "target": left, "count": 3},
                                    {"kind": "counting", "mode": "exact", "target": right, "count": 4})):
                with self.subTest(targets=(left, right), kind=example["tasks"][0]["kind"]):
                    self.assertEqual(step([example], recovery=True)["reason"], "invalid_or_duplicate_task")
        for left, right in (("左側にある看板", "右側にある看板"),
                            ("赤い飲食店の看板の店名", "緑色のスーパーの看板の店名")):
            first = {**SIGN, "target": left}
            example = observe(first, {**SECOND_SIGN, "target": right})
            self.assertEqual(step([example], recovery=True), step([observe(first)]))

    def test_reference_to_an_introduced_ocr_target_does_not_require_recovery(self):
        task = {**SIGN, "target": "案内板中央の白い文字"}
        replies = [observe(task)]
        while True:
            stage = step(replies)
            pair = (len(replies) - 1) // 2 + 1
            if stage["phase"] == "assistant" and pair + 1 == stage["task_pairs"]["ocr"]:
                break
            replies.append({"text": "どんな様子？" if stage["phase"] == "user" else "建物があるよ。"})
        introduction = "案内板の中央には白い文字があるよ。"
        replies.append({"text": introduction})
        request = step(replies)
        self.assertEqual(request, step(replies, recovery=True))
        self.assertEqual(request["phase"], "user")
        self.assertIn(introduction, request["request"]["input"][0]["content"][0]["text"])
        replies.append({"text": "何と書いてあるの？"})
        self.assertEqual(step(replies)["phase"], "assistant")

    def test_preview_keeps_first_transcript_identity_and_opt_in(self):
        source = {"image": "image.png", "image_sha256": "b" * 64}
        with patch("ssvd.dialogue.preview_regeneration._load_source", return_value=(source, "ocr", IMAGE, "a" * 64)):
            first = preview(None, "synthetic", pairs=4)
            self.assertEqual(first, preview(None, "synthetic", pairs=4, recovery=True))
            transcript = {"identity": first["identity"], "replies": [observe(SIGN, SECOND_SIGN)]}
            strict = preview(None, "synthetic", pairs=4, transcript=transcript)
            recovered = preview(None, "synthetic", pairs=4, transcript=transcript, recovery=True)
            self.assertEqual(strict["status"], "hold")
            self.assertEqual(recovered["identity"], strict["identity"])
            self.assertEqual(recovered["phase"], "user")
