"""Exercise recovery with a fixed original cohort and no network access."""

from contextlib import closing, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot, pilot_recovery
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore


class FakeAPI:
    def __init__(self):
        self.sent = []
        self.before_quote_returns = lambda: None

    def prepare(self, body):
        return body

    def reserve_usd(self, body):
        self.before_quote_returns()
        return .001

    def generate(self, body, *, request_id):
        self.sent.append((body["index"], request_id))
        return {"index": body["index"]}, {"usage": {"input_tokens": 30, "output_tokens": 5}, "cost_usd": .0002}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run_dir = self.root / "pilot"
        history = self.root / DEFAULT_REVIEW_EVENTS
        history.parent.mkdir(parents=True)
        history.touch()
        self.samples = [dict(uid=f"item-{i}", kind="ocr" if i < 14 else "counting", pairs=4,
                             image_sha256="a"*64, base_fingerprint="b"*64) for i in range(20)]
        with patch.object(pilot, "prepare_manifest", return_value={"schema_version": 1, "samples": self.samples}):
            pilot.prepare(self.root, self.run_dir)
        self.first = (self.samples[0]["uid"], pilot.MODELS[0])
        self.complete = {"status": "pending_review", "grounding": {"tasks": []}}
        self.hold = {"status": "hold", "reason": "initial validation"}
        self.api = FakeAPI()
        with closing(pilot.open_db(self.run_dir)) as db, db:
            for sample in self.samples:
                for model in pilot.MODELS:
                    key = (sample["uid"], model)
                    db.execute("INSERT INTO outcomes VALUES(?,?,?)",
                               (*key, pilot.encode(self.hold if key == self.first else self.complete)))
            for step in range(2):
                body = self.body(step)
                digest = hashlib.sha256(pilot.encode(body).encode()).hexdigest()
                db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved,charged,reply,receipt) "
                           "VALUES(?,?,?,?,?,?,?,?,?)", (*self.first, step, digest, f"old-{step}", 1000000, 200000,
                           pilot.encode({"index": step}), '{"usage":{"input_tokens":30,"output_tokens":5}}'))
        self.original_report = pilot.status(self.run_dir)
        self.preview = self.install(pilot, "preview", side_effect=self.stage)
        records = {s["uid"]: s for s in self.samples}
        self.source = self.install(pilot_recovery, "_load_source", side_effect=lambda root, uid:
                                   (records[uid], records[uid]["kind"], "", records[uid]["base_fingerprint"]))
        self.write_review = self.install(pilot_recovery, "write_review", return_value=self.run_dir / "recovery/review.html")

    def install(self, obj, name, **kwargs):
        ctx = patch.object(obj, name, **kwargs)
        self.addCleanup(ctx.stop)
        return ctx.start()

    def body(self, step):
        return {"uid": self.first[0], "model": self.first[1], "index": step,
                "history": [{"index": i} for i in range(step)]}

    def stage(self, root, uid, *, model, pairs, transcript, recovery):
        self.assertTrue(recovery)
        self.assertEqual((uid, model), self.first)
        step = len(transcript["replies"])
        self.assertEqual(transcript["replies"], self.body(step)["history"])
        if step == 4:
            return self.complete
        return {"status": "request", "request": self.body(step)}

    def recover(self):
        with redirect_stdout(StringIO()):
            return pilot_recovery.recover(self.root, self.run_dir, self.api)

    def test_resume_only_new_steps_and_keep_original_report_and_candidates(self):
        pre = pilot_recovery.recovery_status(self.run_dir)
        self.assertEqual((pre["remaining"], pre["additional_requests"]), (40, 0))
        (self.run_dir / "review.html").write_text("original")
        self.recover()
        self.recover()
        self.assertEqual([i for i, _ in self.api.sent], [2, 3])
        self.assertEqual((self.run_dir / "review.html").read_text(), "original")
        self.assertEqual(self.write_review.call_args.args[1], self.run_dir / "recovery")
        with closing(pilot.open_db(self.run_dir)) as db:
            self.assertEqual(pilot.read_outcomes(db)[self.first], self.hold)
            self.assertEqual(pilot_recovery._outcomes(db)[self.first], self.complete)
            self.assertEqual(db.execute("SELECT count(*) FROM recovery_outcomes").fetchone()[0], 40)
        result = pilot_recovery.recovery_status(self.run_dir)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(result["additional_requests"], 2)
        self.assertEqual(result["total_generation_cost_usd"], .0008)
        self.assertEqual(result["additional_generation_cost_usd"], .0004)
        self.assertEqual(result["models"][self.first[1]]["transitions"], {"pending_review->pending_review": 19,
                                                                         "hold->pending_review": 1})
        initial = pilot.status(self.run_dir)
        self.assertEqual(initial["shared_budget"], result["shared_budget"])
        self.assertEqual(initial.pop("shared_budget"), {"legacy_committed_usd": 0, "committed_usd": .0008,
                                                       "remaining_usd": 1.9992})
        self.assertEqual(initial, {k: v for k, v in self.original_report.items() if k != "shared_budget"})

    def test_interrupt_after_settlement_replays_prefix_from_both_phases(self):
        def interrupt(*args, **kwargs):
            if len(kwargs["transcript"]["replies"]) == 3:
                raise KeyboardInterrupt
            return self.stage(*args, **kwargs)
        self.preview.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.recover()
        self.preview.side_effect = self.stage
        self.recover()
        self.assertEqual([i for i, _ in self.api.sent], [2, 3])

    def test_incomplete_original_or_wrong_hash_prevents_sending(self):
        with closing(pilot.open_db(self.run_dir)) as db, db:
            db.execute("UPDATE calls SET request_hash='wrong' WHERE step=0")
        with self.assertRaisesRegex(RuntimeError, "changed request"):
            self.recover()
        with closing(pilot.open_db(self.run_dir)) as db, db:
            db.execute("DELETE FROM outcomes WHERE uid=? AND model=?", self.first)
        with self.assertRaisesRegex(RuntimeError, "40 original"):
            self.recover()
        self.assertFalse(self.api.sent)

    def test_missing_original_observation_cannot_trigger_a_new_image_request(self):
        with closing(pilot.open_db(self.run_dir)) as db, db:
            db.execute("DELETE FROM calls")
        with self.assertRaisesRegex(RuntimeError, "retained initial image observation"):
            self.recover()
        self.assertFalse(self.api.sent)

    def test_retired_existing_candidate_is_removed_from_followup(self):
        self.source.side_effect = FileNotFoundError()
        self.recover()
        with closing(pilot.open_db(self.run_dir)) as db:
            results = pilot_recovery._outcomes(db)
        self.assertEqual(results[(self.samples[1]["uid"], pilot.MODELS[0])]["status"], "protected")

    def test_uncertain_does_not_send_or_release_reserve(self):
        with closing(pilot.open_db(self.run_dir)) as db, db:
            db.execute("UPDATE calls SET charged=NULL,reply=NULL WHERE step=1")
        self.recover()
        result = pilot_recovery.recovery_status(self.run_dir)
        self.assertEqual(result["unresolved_reserve_usd"], .001)
        self.assertEqual(result["models"][self.first[1]]["transitions"]["hold->hold"], 1)
        self.assertFalse(self.api.sent)

    def test_shared_budget_includes_initial_cost_and_new_human_decision_wins(self):
        with patch.object(pilot, "BUDGET_USD", .0012), self.assertRaisesRegex(RuntimeError, "spending limit"):
            self.recover()
        def decide():
            ReviewStore(self.root / DEFAULT_REVIEW_EVENTS).append(
                fingerprint="b"*64, decision="approved", collection=pilot.COLLECTION, uid=self.first[0])
        self.api.before_quote_returns = decide
        self.recover()
        self.assertFalse(self.api.sent)
        with closing(pilot.open_db(self.run_dir)) as db:
            for model in pilot.MODELS:
                self.assertEqual(pilot_recovery._outcomes(db)[(self.first[0], model)]["status"], "protected")

    def test_gap_or_premature_terminal_state_cannot_reuse_different_indices(self):
        self.preview.side_effect = None
        self.preview.return_value = self.complete
        with self.assertRaisesRegex(RuntimeError, "before consuming"):
            self.recover()
        with closing(pilot.open_db(self.run_dir)) as db, db:
            db.execute("DELETE FROM calls WHERE step=0")
        with self.assertRaisesRegex(RuntimeError, "missing steps"):
            self.recover()
        self.assertFalse(self.api.sent)
