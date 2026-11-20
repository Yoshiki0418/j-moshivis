"""Offline integration of both pilot arms with the existing role-by-role generator."""

from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot, pilot_recovery
from ssvd.dialogue.regeneration_roles import _USER
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
        if body["instructions"] == _USER:
            reply = {"text": "公園の様子を教えて。"}
        else:
            reply = {"text": "木の下にベンチがあるよ。", "tasks": []}
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
        self.assertEqual(len(set(api.sent)), 40 * 8)
        result = pilot.status(self.run_dir)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(result["generation_cost_usd"], .16)
        for model in pilot.MODELS:
            self.assertEqual(result["models"][model]["outcomes"], {"pending_review": 20})
            self.assertEqual(result["models"][model]["generated_tasks"], {"none": 20})
            self.assertEqual(result["models"][model]["candidates_with_warnings"], 20)
            self.assertEqual(result["models"][model]["review_warnings_by_code"]["repeated_question"], 60)
        with closing(pilot.open_db(self.run_dir)) as db:
            sample = pilot.read_outcomes(db)[self.samples[0]["uid"], pilot.MODELS[0]]["candidate"]
        dialogue = [(turn["speaker"], turn["text"]) for turn in sample["dialogue"]]
        self.assertEqual(dialogue, [("ユーザー", "公園の様子を教えて。"), ("アシスタント", "木の下にベンチがあるよ。")] * 4)
        self.run_with(api)
        self.assertEqual(len(api.sent), 40 * 8)

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

    def test_new_cohort_reuses_only_frozen_images_and_shares_legacy_budget_and_lock(self):
        legacy = self.run_dir
        current = legacy / pilot.PROMPT_VERSION
        with closing(pilot.open_db(legacy)) as db, db:
            db.execute("UPDATE cohort SET prompt=?", (pilot.LEGACY_VERSION,))
            for step, charge in enumerate((10_000_000, None)):
                db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved,charged,receipt) "
                           "VALUES(?,?,?,?,?,?,?,?)",
                           (self.samples[0]["uid"], pilot.MODELS[0], step, "old", str(step),
                            12_000_000, charge, '{"usage":{"input_tokens":1,"output_tokens":1}}'))
        with patch.object(pilot, "prepare_manifest", side_effect=AssertionError("cohort must stay frozen")):
            prepared = pilot.prepare(self.root, current)
        self.assertEqual(prepared["samples"], self.samples)
        report = pilot.status(current)
        self.assertEqual(report["settled_requests"], 0)
        self.assertEqual(report["shared_budget"],
                         {"legacy_committed_usd": .022, "committed_usd": .022, "remaining_usd": 1.978})
        history = pilot.status(legacy, prompt_version=pilot.LEGACY_VERSION)
        self.assertEqual((history["generation_cost_usd"], history["unresolved_reserve_usd"]), (.01, .012))
        api = ExampleAPI()
        with pilot.operation(legacy), self.assertRaisesRegex(RuntimeError, "Another pilot operation"):
            pilot.run_pilot(self.root, current, api)
        with patch.object(pilot, "BUDGET_USD", .0225), self.assertRaisesRegex(RuntimeError, "spending limit"):
            pilot.run_pilot(self.root, current, api)
        with self.assertRaisesRegex(RuntimeError, "prompt version"):
            pilot.run_pilot(self.root, legacy, api)
        self.assertFalse(api.sent)
        self.assertEqual(history, pilot.status(legacy, prompt_version=pilot.LEGACY_VERSION))
        with closing(pilot.open_db(current)) as db, db:
            db.executemany("INSERT INTO outcomes VALUES(?,?,?)",
                           ((s["uid"], model, '{"status":"protected"}') for s in self.samples for model in pilot.MODELS))
            db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved) VALUES(?,?,?,?,?,?)",
                       (self.samples[0]["uid"], pilot.MODELS[0], 0, "new", "new", 3_000_000))
        expected = {"legacy_committed_usd": .022, "committed_usd": .025, "remaining_usd": 1.975}
        self.assertEqual(pilot_recovery.recovery_status(current)["shared_budget"], expected)
        self.assertEqual(pilot.status(current)["shared_budget"], expected)
        (legacy / "pilot.sqlite3").rename(legacy / "backup.sqlite3")
        with self.assertRaisesRegex(RuntimeError, "ledger missing"):
            pilot.prepare(self.root, current)
        (legacy / "manifest.json").rename(legacy / "backup.json")
        with self.assertRaisesRegex(RuntimeError, "must be restored"):
            pilot.run_pilot(self.root, current, api)
        self.assertFalse(api.sent)

    def test_both_archives_count_toward_the_limit_and_must_match_the_cohort(self):
        base = self.run_dir
        middle = base / pilot.OBSERVATION_VERSION
        current = base / pilot.PROMPT_VERSION
        with closing(pilot.open_db(base)) as db, db:
            db.execute("UPDATE cohort SET prompt=?", (pilot.LEGACY_VERSION,))
            db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved) "
                       "VALUES('a',?,0,'v2','v2',12000000)", (pilot.MODELS[0],))
        pilot.prepare(self.root, middle)
        with closing(pilot.open_db(middle)) as db, db:
            manifest = pilot.load_manifest(db)
            manifest.pop("budget_predecessors")
            manifest["budget_predecessor"] = pilot.LEGACY_VERSION
            db.execute("UPDATE cohort SET prompt=?, manifest=?", (pilot.OBSERVATION_VERSION, pilot.encode(manifest)))
            db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved,charged,receipt) "
                       "VALUES('b',?,0,'v3','v3',15000000,10000000,?)", (pilot.MODELS[1], pilot.encode(
                           {"usage": {"input_tokens": 4, "output_tokens": 1}})))
            db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved) "
                       "VALUES('b',?,1,'unsettled','unknown',3000000)", (pilot.MODELS[1],))
        with patch.object(pilot, "prepare_manifest", side_effect=AssertionError("cohort must stay frozen")):
            selected = pilot.prepare(self.root, current)
        self.assertEqual(selected["budget_predecessors"], list(pilot.ARCHIVED_VERSIONS))
        self.assertEqual(selected["samples"], self.samples)
        budget = pilot.status(current)["shared_budget"]
        self.assertEqual((budget["legacy_committed_usd"], budget["remaining_usd"]), (.025, 1.975))
        middle_report = pilot.status(middle, prompt_version=pilot.OBSERVATION_VERSION)
        self.assertEqual(middle_report["shared_budget"]["legacy_committed_usd"], .012)
        api = ExampleAPI()
        with patch.object(pilot, "BUDGET_USD", .025), self.assertRaisesRegex(RuntimeError, "spending limit"):
            pilot.run_pilot(self.root, current, api)
        with closing(pilot.open_db(middle)) as db, db:
            manifest["samples"][0]["uid"] = "different"
            db.execute("UPDATE cohort SET manifest=?", (pilot.encode(manifest),))
        with self.assertRaisesRegex(RuntimeError, "frozen comparison cohort"):
            pilot.run_pilot(self.root, current, api)
        middle.rename(base / "removed-archive")
        with self.assertRaisesRegex(RuntimeError, "must be restored"):
            pilot.run_pilot(self.root, current, api)
        self.assertFalse(api.sent)

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
            out = pilot.generate_one(self.root, db, self.samples[0], pilot.MODELS[0], api, previous_commitment=0)
        self.assertEqual(out["status"], "protected")
        self.assertFalse(api.sent)
