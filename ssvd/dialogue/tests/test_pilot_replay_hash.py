"""The fallback matches the bytes sent, never just a semantically similar request."""

import copy
import json
import unittest

from ssvd.dialogue.pilot_api import OpenAIPilotAPI
from ssvd.dialogue.pilot_replay_hash import matches_saved_request, request_digest
from ssvd.dialogue.regeneration_roles import next_step


class ReplayHashTests(unittest.TestCase):
    def prompt(self, task):
        observation = {"scene": "入口に札が掛かっている", "status": "ok", "tasks": [task]}
        replies = [observation]
        while True:
            stage = next_step("synthetic", "ocr", "data:image/png;base64,AA==", replies, pairs=4)
            pair = 1 + (len(replies) - 1) // 2
            active = pair == stage["task_pairs"]["ocr"]
            if stage["phase"] == "assistant" and active:
                return OpenAIPilotAPI("offline").prepare(stage["request"])
            if stage["phase"] == "user":
                replies.append({"text": "何と書いてあるの？" if active else "どんな様子？"})
            else:
                replies.append({"text": "入口に札が掛かっているね。"})

    def test_original_key_order_reproduces_digest_but_new_values_do_not(self):
        task = dict(kind="ocr", target="札", mode="partial", written="北", reading="きた", reason="端が切れている")
        body = self.prompt(task)
        original = copy.deepcopy(body)
        digest = request_digest(self.prompt(dict(reversed(list(task.items())))))
        self.assertNotEqual(request_digest(body), digest)
        self.assertTrue(matches_saved_request(body, digest))
        self.assertEqual(body, original)
        for change in (dict(target="入口の札"), dict(written="南", reading="みなみ"), dict(reason="陰になる")):
            self.assertFalse(matches_saved_request(self.prompt({**task, **change}), digest))
        different_model = {**body, "model": "different"}
        self.assertFalse(matches_saved_request(different_model, digest))

    def test_does_not_accept_reordered_dialogue_or_expand_large_or_unknown_task_keys(self):
        task = dict(kind="ocr", target="札", mode="exact", written="北", reading="きた")
        body = self.prompt(task)
        original_text = body["input"][0]["content"][0]["text"]
        cue, payload, last = original_text.split("\n")
        for field in ("history", "extra"):
            context = json.loads(payload)
            if field == "history":
                turns = context["公開された会話"]
                turns[0] = dict(reversed(list(turns[0].items())))
            else:
                context["画像の観察情報"]["この返答の根拠"]["extra"] = "unknown"
            modified = copy.deepcopy(body)
            modified["input"][0]["content"][0]["text"] = "\n".join((cue, json.dumps(context, ensure_ascii=False), last))
            self.assertFalse(matches_saved_request(modified, request_digest(body)))
        self.assertFalse(matches_saved_request(body, "no-digest"))

    def test_exact_hash_for_any_phase(self):
        body = {"instructions": "synthetic grounding", "input": []}
        self.assertTrue(matches_saved_request(body, request_digest(body)))
        self.assertFalse(matches_saved_request(body, "0" * 64))
