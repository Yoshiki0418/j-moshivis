"""Offline integration of CLI turn replay, human review protection and quota handling."""

from contextlib import closing
import hashlib
import os
import threading
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, call

from ssvd.dialogue import codex_production as runner, preview_regeneration
from ssvd.dialogue.codex_transport import CodexInvocationError, CodexRateLimit, CodexUsageLimit
from ssvd.dialogue.regeneration_roles import default_pair_count
from ssvd.dialogue.visual_dialogue_prompts_v11 import USER
from ssvd.review_store import content_fingerprint


class FixtureTransport:
    def __init__(self):
        self.history_lengths = []
        self.on_generate = lambda: None

    def preflight(self):
        pass

    def generate(self, request, image):
        text, media = request["input"][0]["content"]
        assert media["image_url"] == "data:image/png;base64,eA=="
        self.history_lengths.append(len(json.loads(text["text"])["公開された会話"]))
        self.on_generate()
        reply = {"text": "道の向こうにお店が見えるね。"}
        if request["instructions"] != USER:
            reply = {"text": "赤い看板が目を引くね。", "tasks": []}
        return reply, dict(model=runner.MODEL, auth_method="chatgpt", transport="codex_cli")


class CLIRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / runner.CLI_MANIFEST
        self.path.parent.mkdir(parents=True)
        self.events = self.root / runner.DEFAULT_REVIEW_EVENTS
        self.events.touch()
        self.transport = FixtureTransport()
        self.uids = ["fixture1", "fixture3"]
        samples = []
        sources = {}
        for uid in self.uids:
            image_sha = hashlib.sha256(uid.encode()).hexdigest()
            base = hashlib.sha256((uid + "-base").encode()).hexdigest()
            image = uid + ".png"
            sources[uid] = (dict(image=image, image_sha256=image_sha), "ocr", "data:image/png;base64,eA==", base)
            samples.append(dict(uid=uid, image_path=image, kind="ocr", pairs=default_pair_count(uid),
                                image_sha256=image_sha, base_fingerprint=base))
        manifest = dict(schema_version=1, model=runner.MODEL, collection=runner.COLLECTION,
                        prompt_version=runner.VISUAL_V11, backend="codex_cli", excluded_api_manifest_sha256="a"*64,
                        sample_size=len(samples), pairs=sum(s["pairs"] for s in samples), sampling_target={"ocr": 2},
                        samples=samples)
        raw = json.dumps(manifest).encode()
        self.path.write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.steps = [s["pairs"] * 2 for s in samples]
        for mock in (patch.object(runner, "verify"),
                     patch.object(preview_regeneration, "_load_source", side_effect=lambda _, uid: sources[uid])):
            mock.start()
            self.addCleanup(mock.stop)

    def generate(self, **kwargs):
        return runner.run(self.root, self.digest, workers=1, transport=self.transport, **kwargs)

    def test_refused_turn_resumes_and_first_segment_publishes_pending_only(self):
        def limit():
            if len(self.transport.history_lengths) == 3:
                raise CodexUsageLimit()
        self.transport.on_generate = limit
        first = self.generate(limit=1)
        self.assertEqual((first["state"], first["received_turns"], first["outstanding_turns"]), ("usage_limit", 2, 0))
        segment = self.generate(limit=1)
        self.assertEqual((segment["state"], segment["outcomes"]), ("segment_complete", {"pending_review": 1}))
        self.assertEqual(self.transport.history_lengths, [0, 1, 2, *range(2, self.steps[0])])
        full = self.generate()
        self.assertEqual((full["state"], full["received_turns"]), ("queue_complete", sum(self.steps)))
        self.assertEqual(self.generate()["received_turns"], sum(self.steps))
        self.assertEqual(self.events.read_bytes(), b"")
        with closing(sqlite3.connect(self.path.parent / runner.DB_NAME)) as db:
            result = json.loads(db.execute("SELECT outcome FROM outcomes LIMIT 1").fetchone()[0])
        cand = result["candidate"]
        self.assertTrue(cand["review_required"])
        self.assertEqual((result["generation_backend"], cand["generation_backend"]), ("codex_cli", "codex_cli"))
        self.assertEqual(cand["review_id"], content_fingerprint(cand))

    def test_human_decision_during_final_turn_wins(self):
        def act():
            if len(self.transport.history_lengths) == self.steps[0]:
                runner.ReviewStore(self.events).append(fingerprint="f"*64, decision="approved",
                                                       uid=self.uids[0], collection=runner.COLLECTION)
        self.transport.on_generate = act
        self.assertEqual(self.generate(limit=1)["outcomes"], {"protected": 1})

    def test_temporary_throttle_retries_only_the_refused_turn(self):
        def burst():
            if len(self.transport.history_lengths) == 3:
                raise CodexRateLimit("slow down")
        self.transport.on_generate = burst
        self.assertEqual(self.generate(limit=1, retry_delays=(0,))["state"], "segment_complete")
        self.assertEqual(self.transport.history_lengths, [0, 1, 2, *range(2, self.steps[0])])

    def test_unknown_dispatch_is_held_without_double_generation(self):
        def fail():
            if len(self.transport.history_lengths) == 2:
                raise CodexInvocationError("no confirmed reply")
        self.transport.on_generate = fail
        with self.assertRaisesRegex(CodexInvocationError, "confirmed"):
            self.generate()
        self.assertEqual(runner.report(self.root, self.digest)["outstanding_turns"], 1)
        self.assertEqual(self.generate()["outcomes"], {"hold": 1, "pending_review": 1})
        self.assertEqual(self.transport.history_lengths, [0, 1, *range(self.steps[1])])

    def test_partial_quota_stops_service_and_preserves_unconfirmed_turn(self):
        def hit_quota():
            raise runner.CodexIncompleteAtLimit("quota after partial output")
        self.transport.on_generate = hit_quota
        result = runner.run(self.root, self.digest, workers=8, transport=self.transport)
        self.assertEqual(result["state"], "usage_limit")
        self.assertGreater(result["outstanding_turns"], 0)
        self.assertEqual(result["received_turns"], 0)

    def test_wrong_digest_prevents_dispatch(self):
        self.digest = "a"*64
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.generate()
        self.assertFalse((self.path.parent / runner.DB_NAME).exists())

class ServiceTests(unittest.TestCase):
    def test_warmup_then_eight_workers(self):
        stop = threading.Event()
        with patch.object(runner, "run", side_effect=[{"state": "segment_complete"}, {"state": "queue_complete"}]) as run:
            self.assertEqual(runner.serve(Path("."), "digest", 8, stop), 0)
        self.assertEqual(run.call_args_list, [
            call(Path("."), "digest", workers=1, limit=1, stop=stop),
            call(Path("."), "digest", workers=8, stop=stop),
        ])

    def test_service_restart_policy(self):
        for outcome, code in [
            ({"state": "usage_limit"}, 0), ({"state": "operator_stop"}, 0),
            ({"state": "rate_limited"}, os.EX_TEMPFAIL),
            (runner.CodexTransientError("temporary 403"), os.EX_TEMPFAIL),
            (ValueError("manifest checksum"), os.EX_CONFIG),
        ]:
            with self.subTest(outcome=outcome), patch.object(runner, "run", side_effect=[outcome]) as run:
                self.assertEqual(runner.serve(Path("."), "digest", 8, threading.Event()), code)
                self.assertEqual(run.call_count, 1)
