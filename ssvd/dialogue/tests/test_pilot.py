"""Offline integration of both pilot arms with the existing role-by-role generator."""

from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore


class ExampleAPI:
    def __init__(self, reserve=.001, error=None):
        self.reserve = reserve
        self.error = error
        self.sent = []
        self.before_quote_returns = lambda: None

    def prepare(self, body):
        return body

    def reserve_usd(self, body):
        self.before_quote_returns()
        return self.reserve

    def generate(self, body, *, request_id):
        self.sent.append(request_id)
        if self.error:
            raise self.error
        parts = body["input"][0]["content"]
        if len(parts) == 2:
            reply = {"status": "ok", "scene": "公園にベンチがある。", "tasks": []}
        else:
            reply = {"text": "公園の様子を教えて。" if "聞き手" in body["instructions"] else "木の下にベンチがあるよ。"}
        return reply, {"response_id": "resp_example", "usage": {"input_tokens": 50, "output_tokens": 10},
                       "cost_usd": .0005}


class PilotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        events = self.root / DEFAULT_REVIEW_EVENTS
        events.parent.mkdir(parents=True)
        events.touch()
        self.run_dir = self.root / "private-run"
        self.samples = [dict(uid=f"sample-{i}", kind="ocr" if i < 14 else "counting",
                             image_sha256="a"*64, base_fingerprint="b"*64, pairs=4)
                        for i in range(20)]
        manifest = {"schema_version": 1, "samples": self.samples}
        with patch.object(pilot, "prepare_manifest", return_value=manifest):
            pilot.prepare(self.root, self.run_dir)
        def source(root, uid):
            sample = next(s for s in self.samples if s["uid"] == uid)
            return (dict(uid=uid, image="example.jpg", image_sha256=sample["image_sha256"]),
                    sample["kind"], "data:image/jpeg;base64,QQ==", sample["base_fingerprint"])
        self.source = self.enter_patch("ssvd.dialogue.preview_regeneration._load_source", side_effect=source)
        self.review = self.enter_patch("ssvd.dialogue.pilot.write_review")

    def enter_patch(self, target, **kwargs):
        ctx = patch(target, **kwargs)
        self.addCleanup(ctx.stop)
        return ctx.start()

    def run_with(self, api):
        with redirect_stdout(StringIO()):
            pilot.run_pilot(self.root, self.run_dir, api)

    def test_real_roles_and_idempotent_resume(self):
        api = ExampleAPI()
        self.run_with(api)
        self.assertEqual(len(set(api.sent)), 40 * 9)
        result = pilot.status(self.run_dir)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(result["generation_cost_usd"], .18)
        for model in pilot.MODELS:
            self.assertEqual(result["models"][model]["outcomes"], {"pending_review": 20})
            self.assertEqual(result["models"][model]["generated_tasks"], {"none": 20})
        self.run_with(api)
        self.assertEqual(len(api.sent), 40 * 9)

    def test_replay_paid_call_after_interruption_without_resending(self):
        api = ExampleAPI()
        real_preview = pilot.preview
        def stop(root, uid, **kwargs):
            if kwargs["transcript"]["replies"]:
                raise KeyboardInterrupt
            return real_preview(root, uid, **kwargs)
        with patch.object(pilot, "preview", side_effect=stop), self.assertRaises(KeyboardInterrupt):
            self.run_with(api)
        with patch.object(pilot, "BUDGET_USD", .002):
            with self.assertRaisesRegex(RuntimeError, "spending limit"):
                self.run_with(api)
        result = pilot.status(self.run_dir)
        self.assertEqual(result["settled_requests"], 3)
        self.assertEqual(len(api.sent), 3)
        self.assertEqual(result["unresolved_requests"], 0)

    def test_uncertain_call_is_never_replayed_and_reserve_is_retained(self):
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            self.run_with(ExampleAPI(error=RuntimeError("unknown")))
        with patch.object(pilot, "BUDGET_USD", .001):
            after = ExampleAPI()
            with self.assertRaisesRegex(RuntimeError, "spending limit"):
                self.run_with(after)
            self.assertFalse(after.sent)
        report = pilot.status(self.run_dir)
        self.assertEqual(report["models"][pilot.MODELS[0]]["outcomes"], {"hold": 1})
        self.assertEqual(report["unresolved_reserve_usd"], .001)

    def test_definite_rejection_releases_reservation(self):
        with self.assertRaises(pilot.PilotRequestRejected):
            self.run_with(ExampleAPI(error=pilot.PilotRequestRejected("rejected")))
        self.assertEqual(pilot.status(self.run_dir)["unresolved_requests"], 0)

    def test_protection_and_budget_checked_before_submit(self):
        api = ExampleAPI(reserve=3.0)
        with self.assertRaisesRegex(RuntimeError, "spending limit"):
            self.run_with(api)
        self.assertFalse(api.sent)
        api.reserve = .001
        def decision():
            ReviewStore(self.root / DEFAULT_REVIEW_EVENTS).append(
                fingerprint="b"*64, decision="approved", collection=pilot.COLLECTION, uid=self.samples[0]["uid"])
            api.before_quote_returns = lambda: None
        api.before_quote_returns = decision
        with closing(pilot.open_db(self.run_dir)) as db:
            out = pilot.generate_one(self.root, db, self.samples[0], pilot.MODELS[0], api)
        self.assertEqual(out["status"], "protected")
        self.assertFalse(api.sent)
