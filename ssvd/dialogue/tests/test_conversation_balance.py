"""Regression coverage for the v11 conversation pilot and unchanged v10 requests."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ssvd.dialogue import preview_regeneration
from ssvd.dialogue.regeneration_roles import PROMPT_VERSION, VISUAL_V11, next_step
from ssvd.review_store import content_fingerprint


IMAGE = "data:image/png;base64,eA=="
MODEL = "gpt-6.1-terra"


def request(replies, version=VISUAL_V11):
    return next_step("fixture", "ocr", IMAGE, replies, pairs=4, model=MODEL, version=version)["request"]


def hint(user_texts):
    replies = []
    for text in user_texts:
        replies.extend([{"text": text}, {"text": "お店の前だね。", "tasks": []}])
    content = request(replies)["input"][0]["content"]
    return json.loads(content[0]["text"])["今回の会話のヒント"]


class ConversationBalanceTests(unittest.TestCase):
    def test_v10_requests_still_match_frozen_digests(self):
        self.assertEqual(PROMPT_VERSION, "visual_dialogue_v10")
        q = {"text": "青いのは何？"}
        a = {"text": "青い標識だね。", "tasks": []}
        digests = (
            "5bd24f3febd39475eeb78095a0e6fc9ef9034df7f2df5563e2b0e7e39f0b2f1d",
            "8b561962bf7212eecc4401d14bfeea5a926e2e79de14f0084223159899a3acd1",
            "c14167ad897482be1365832ad83981352d4aaaa417f2d4f49a911c17c331230a",
        )
        for prior, expected in zip(([], [q], [q, a]), digests):
            wire = json.dumps(request(prior, PROMPT_VERSION), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(hashlib.sha256(wire.encode()).hexdigest(), expected)

    def test_cues_follow_actual_user_turns_and_leave_room_for_a_third_question(self):
        asks = ["入口はどこ？", "右の看板について教えて。"]
        comments = ["甘い香りがしそう。", "私は抹茶味が好き。"]
        self.assertEqual(len({hint([]), hint(asks), hint(comments), hint([asks[0], comments[1]])}), 4)
        self.assertIn("それだけで番を渡せ", hint(asks))
        self.assertIn("尋ね", hint(comments))
        # A cue does not reject, regenerate, or even relabel a legitimate further question.
        replies = []
        for text in asks + ["緑の幕は何？", "緑は落ち着く色だね。"]:
            replies.extend([{"text": text}, {"text": "葉っぱのような色だね。", "tasks": []}])
        result = next_step("fixture", "ocr", IMAGE, replies, model=MODEL, pairs=4, version=VISUAL_V11)
        self.assertEqual(result["status"], "pending_review")
        self.assertFalse(result["exchanges"])

    def test_assistant_question_invites_a_reply_after_two_comments(self):
        history = [{"text": "海沿いっていいね。"}, {"text": "きれいだね。", "tasks": []},
                   {"text": "休日にゆっくり散歩したいな。"}, {"text": "海と山ならどっちが好き？", "tasks": []}]
        context = json.loads(request(history)["input"][0]["content"][0]["text"])
        self.assertIn("自分の考えや好みを返して", context["今回の会話のヒント"])

    def test_full_preview_preserves_image_history_markers_and_version_boundary(self):
        pairs = [
            ("青い看板に何て書いてある？", "ミナトって書いてあるよ。", ["ocr"]),
            ("こういう字の形、好きだな。", "丸い線に親しみがわくね。", []),
            ("手書きのカードにも使ってみたい。", "線を少し太くすると、短いひと言が映えそう。", []),
            ("奥にある赤い鉢はいくつ？", "二つ見えるよ。", ["counting"]),
        ]
        source = dict(uid="fixture", image="image.png", image_sha256="a" * 64)
        with patch.object(preview_regeneration, "_load_source", return_value=(source, "ocr", IMAGE, "b" * 64)):
            def preview(transcript, version=VISUAL_V11):
                return preview_regeneration.preview(Path("."), "fixture", model=MODEL, pairs=len(pairs),
                                                    transcript=transcript, prompt_version=version)

            identity = preview(None)["identity"]
            transcript = {"identity": identity, "replies": []}
            with self.assertRaisesRegex(ValueError, "different source or configuration"):
                preview(transcript, PROMPT_VERSION)
            history = []
            for user, assistant, markers in pairs:
                for role, reply in (("user", {"text": user}), ("assistant", {"text": assistant, "tasks": markers})):
                    stage = preview(transcript)
                    self.assertEqual(stage["phase"], role)
                    text, image = stage["request"]["input"][0]["content"]
                    context = json.loads(text["text"])
                    self.assertEqual(context["公開された会話"], history)
                    self.assertEqual(image, {"type": "input_image", "image_url": IMAGE, "detail": "high"})
                    self.assertEqual("今回の会話のヒント" in context, role == "user")
                    history.append({"role": role, "text": reply["text"]})
                    transcript["replies"].append(reply)
            final = preview(transcript)
        self.assertEqual(final["status"], "pending_review")
        candidate = final["candidate"]
        self.assertTrue(candidate["review_required"])
        self.assertEqual(candidate["prompt_version"], VISUAL_V11)
        self.assertEqual(candidate["review_id"], content_fingerprint(candidate))
        self.assertEqual([turn["text"] for turn in candidate["dialogue"]], [turn["text"] for turn in history])
        aux = candidate["auxiliary_task"]
        self.assertEqual((aux["kind"], aux["answer_turn_index"]), ("ocr", 1))
        self.assertNotIn("recognized_text", aux)
        self.assertEqual(aux["additional_tasks"], [dict(kind="counting", grounding_source="vlm_visual",
                                                       question_turn_index=6, answer_turn_index=7)])
