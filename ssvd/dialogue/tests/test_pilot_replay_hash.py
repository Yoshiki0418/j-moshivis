"""An image-backed request must have the same wire body after a ledger round trip."""

import hashlib
import json
import unittest

from ssvd.dialogue.pilot import encode
from ssvd.dialogue.pilot_api import OpenAIPilotAPI
from ssvd.dialogue.regeneration_roles_v3 import next_step


def assistant_body(task, *, image="data:image/png;base64,AA==", greeting="どんな写真？"):
    observation = {"scene": "入口に札が掛かっている", "status": "ok", "tasks": [task]}
    replies = [observation]
    while True:
        stage = next_step("synthetic", "ocr", image, replies, pairs=4)
        pair = (len(replies) - 1) // 2 + 1
        active = pair == stage["task_pairs"]["ocr"]
        if stage["phase"] == "assistant" and active:
            return OpenAIPilotAPI("offline").prepare(stage["request"])
        if stage["phase"] == "user":
            text = "入口の札には何と書いてあるの？" if active else greeting
        else:
            text = "入口に札が掛かっているね。"
        replies.append({"text": text})


class ReplayHashTests(unittest.TestCase):
    def test_reloading_sorted_task_keys_preserves_the_paid_request(self):
        task = dict(target="入口の札", mode="partial", kind="ocr", written="北", reading="きた", reason="端が切れている")
        from_ledger = json.loads(encode(task))
        self.assertNotEqual(list(task), list(from_ledger))
        self.assertEqual(assistant_body(task), assistant_body(from_ledger))

    def test_digest_still_binds_image_evidence_and_dialogue(self):
        task = dict(target="入口の札", mode="exact", kind="ocr", written="北", reading="きた")

        def digest(body):
            return hashlib.sha256(encode(body).encode()).hexdigest()

        original = digest(assistant_body(task))
        for changed in (
            assistant_body(task, image="data:image/png;base64,AQ=="),
            assistant_body({**task, "written": "南", "reading": "みなみ"}),
            assistant_body(task, greeting="ここはどんな場所？"),
        ):
            self.assertNotEqual(digest(changed), original)
