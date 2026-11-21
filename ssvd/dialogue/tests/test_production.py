"""Exercise production selection-to-review boundaries with synthetic sources and a fake transport."""

from contextlib import ExitStack, closing, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot, production, production_budget
from ssvd.dialogue.pilot_api import OpenAIPilotAPI, PilotRequestRejected
from ssvd.dialogue.preview_regeneration import COLLECTION
from ssvd.dialogue.regeneration_roles import VISUAL_V11, default_pair_count
from ssvd.dialogue.visual_dialogue_prompts import USER as LEGACY_USER
from ssvd.dialogue.visual_dialogue_prompts_v11 import USER
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, review_metadata


class OfflineTerra(OpenAIPilotAPI):
    def __init__(self):
        super().__init__("offline-test", service_tier="flex")
        self.attempts = []
        self.on_generate = lambda: None

    def reserve_usd(self, body):
        return 0.001

    def generate(self, body, *, request_id):
        self.attempts.append(request_id)
        self.on_generate()
        reply = {"text": "何が見える？"} if body["instructions"] in (LEGACY_USER, USER) else {"text": "リンゴが見えるよ。", "tasks": []}
        return reply, {"response_id": "resp_" + request_id, "model": body["model"], "service_tier": "flex",
                       "usage": {"input_tokens": 1000, "output_tokens": 20}, "cost_usd": 0.00055}


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.run_dir = self.root / production.RUN_PATH
        self.run_dir.mkdir(parents=True)
        self.api = OfflineTerra()
        self.events = self.root / DEFAULT_REVIEW_EVENTS
        self.events.touch()
        viewer = self.root / "data/.dataset_viewer"
        viewer.mkdir()
        self.catalog = viewer / "catalog.sqlite3"
        with closing(sqlite3.connect(self.catalog)) as db, db:
            db.executescript("""
                CREATE TABLE sources(source_id, path);
                CREATE TABLE items(collection, uid, review_required, review_status, has_image, has_dialogue,
                    auxiliary_kind, image_path, review_id, dialogue_offset, dialogue_length, dialogue_source);
                INSERT INTO sources VALUES ('fixture', 'data/source.jsonl');
            """)
        self.uid = "fixture1"
        self.sample = self.add_source(self.uid)
        self.freeze(self.sample)
        # The private fixed evaluation corpus and historical ledgers have separate integration tests.
        self.verify_fresh = self.resources.enter_context(patch.object(production, "verify_manifest"))
        self.prior = self.resources.enter_context(patch.object(production, "prior_commitment", return_value=0))
        for module in (production, production_budget):
            self.resources.enter_context(patch.object(module, "CUTOFF", 4_102_444_800))

    def add_source(self, uid):
        image_path = f"data/{uid}.jpg"
        raw = uid.encode()
        (self.root / image_path).write_bytes(raw)
        image_sha = hashlib.sha256(raw).hexdigest()
        source = dict(uid=uid, image=image_path, image_sha256=image_sha, generator="fixture-Qwen",
                      task_mode="normal", dialogue=[{"speaker": "ユーザー", "text": "元の発話"}],
                      verification_status="unverified_vlm_visual", auxiliary_task={"kind": "ocr"})
        source.update(review_metadata(source))
        content = json.dumps(source).encode() + b"\n"
        with (self.root / "data/source.jsonl").open("ab") as stream:
            offset = stream.tell()
            stream.write(content)
        with closing(sqlite3.connect(self.catalog)) as db, db:
            db.execute("INSERT INTO items VALUES (?, ?, 1, 'pending', 1, 1, 'ocr', ?, ?, ?, ?, 'fixture')",
                       (COLLECTION, uid, image_path, source["review_id"], offset, len(content)))
        return dict(uid=uid, image_path=image_path, image_sha256=image_sha, pairs=default_pair_count(uid),
                    kind="ocr", base_fingerprint=source["review_id"])

    def freeze(self, *samples):
        manifest = dict(schema_version=1, model=production.MODEL, collection=COLLECTION,
                        prompt_version=pilot.PROMPT_VERSION, sample_size=len(samples),
                        pairs=sum(s["pairs"] for s in samples), samples=samples, sampling_target={"ocr": len(samples)})
        manifest_bytes = json.dumps(manifest).encode()
        (self.run_dir / "manifest.json").write_bytes(manifest_bytes)
        self.digest = hashlib.sha256(manifest_bytes).hexdigest()

    def generate(self, **options):
        with redirect_stdout(StringIO()):
            return production.run(self.root, self.digest, lambda: self.api, workers=1,
                                  retry_delays=(0, 0), **options)

    def summary(self):
        return json.loads((self.run_dir / "summary.json").read_text())

    def reject_attempt(self, attempt):
        def out_of_quota():
            if len(self.api.attempts) == attempt:
                reason = PilotRequestRejected("OpenAI rejected the request (HTTP 429)")
                reason.code = "insufficient_quota"
                raise reason
        self.api.on_generate = out_of_quota

    def test_rollout_keeps_previous_candidates_and_defers_partial_legacy_dialogues(self):
        partial = self.add_source("fixture3")
        fresh = self.add_source("fixture5")
        self.freeze(self.sample, partial, fresh)
        self.reject_attempt(11)  # Complete the first conversation; retain two paid turns in the second.
        with self.assertRaises(RuntimeError):
            self.generate()
        ledger = self.run_dir / "production.sqlite3"
        with closing(sqlite3.connect(ledger)) as db:
            original = db.execute("SELECT outcome FROM outcomes WHERE uid=?", (self.uid,)).fetchone()
            receipts = db.execute("SELECT * FROM calls WHERE uid=?", (partial["uid"],)).fetchall()

        report = self.generate(new_dialogue_prompt=VISUAL_V11)
        self.assertEqual(report["state"], "previous_prompt_deferred")
        self.assertEqual(report["outcomes"], {"pending_review": 2})
        self.assertEqual(report["deferred_previous_prompt"], 1)
        self.assertEqual(report["unprocessed"], 1)
        self.assertEqual(report["new_dialogue_prompt"], VISUAL_V11)
        self.assertEqual(len(self.api.attempts), 11 + fresh["pairs"] * 2)
        with closing(sqlite3.connect(ledger)) as db:
            self.assertEqual(db.execute("SELECT outcome FROM outcomes WHERE uid=?", (self.uid,)).fetchone(), original)
            self.assertEqual(db.execute("SELECT * FROM calls WHERE uid=?", (partial["uid"],)).fetchall(), receipts)
            self.assertEqual(db.execute("SELECT * FROM dialogue_prompt_versions").fetchall(), [(fresh["uid"], VISUAL_V11)])
            saved = json.loads(db.execute("SELECT outcome FROM outcomes WHERE uid=?", (fresh["uid"],)).fetchone()[0])
            self.assertEqual(saved["identity"]["prompt_version"], VISUAL_V11)
            self.assertEqual(saved["candidate"]["prompt_version"], VISUAL_V11)
            # An old worker does not know this new prompt. Even fresh UID/step keys cannot pay for v10 after rollout.
            with self.assertRaisesRegex(sqlite3.OperationalError, "generation_prompt"):
                db.execute("""INSERT INTO calls(uid,model,step,request_hash,client_id,reserved)
                              VALUES(?,?,99,'test','outdated-writer',1)""", (fresh["uid"], production.MODEL))
        self.generate()
        self.assertEqual(len(self.api.attempts), 11 + fresh["pairs"] * 2)
        status = production.inspect_run(self.root, self.digest)
        self.assertEqual(status["deferred_previous_prompt"], 1)
        self.assertEqual(status["new_dialogue_prompt"], VISUAL_V11)

    def test_v11_partial_resume_reuses_receipts_and_remembers_the_rollout(self):
        self.reject_attempt(3)
        with self.assertRaises(RuntimeError):
            self.generate(new_dialogue_prompt=VISUAL_V11)
        self.assertEqual(self.summary()["paid_receipts"], 2)
        self.assertEqual(self.summary()["deferred_previous_prompt"], 0)
        report = self.generate()
        self.assertEqual(report["new_dialogue_prompt"], VISUAL_V11)
        self.assertEqual(report["state"], "queue_complete")
        self.assertEqual(report["paid_receipts"], 8)
        self.assertEqual(len(self.api.attempts), 9)
        self.assertEqual(json.loads((self.run_dir / "pending_candidates.jsonl").read_text())["prompt_version"], VISUAL_V11)
        self.generate(new_dialogue_prompt=VISUAL_V11)
        self.assertEqual(len(self.api.attempts), 9)

    def test_stale_backup_cannot_silently_undo_a_published_rollout(self):
        # No paid turns between the two snapshots: invoice totals alone cannot detect a rollback.
        stop = threading.Event()
        stop.set()
        self.generate(stop=stop)
        ledger = self.run_dir / "production.sqlite3"
        with closing(sqlite3.connect(":memory:")) as backup:
            with closing(sqlite3.connect(ledger)) as db:
                db.backup(backup)
            self.generate(new_dialogue_prompt=VISUAL_V11, stop=stop)
            with closing(sqlite3.connect(ledger)) as db:
                backup.backup(db)
        with self.assertRaisesRegex(RuntimeError, "published prompt rollout"):
            self.generate()
        self.assertEqual(self.api.attempts, [])

    def test_partial_receipts_resume_without_repaying_and_yield_a_reviewable_record(self):
        self.reject_attempt(3)
        with self.assertRaises(RuntimeError):
            self.generate()
        self.assertEqual(self.summary()["paid_receipts"], 2)
        self.assertEqual(self.summary()["outstanding_calls"], 0)
        self.api.on_generate = lambda: None
        report = self.generate()
        self.assertEqual(report["outcomes"], {"pending_review": 1})
        self.assertEqual(report["paid_receipts"], 8)
        self.assertEqual(len(self.api.attempts), 9)
        self.assertEqual(report["charged_usd"], 0.0044)
        candidate = json.loads((self.run_dir / "pending_candidates.jsonl").read_text())
        self.assertEqual(candidate["generator"], "gpt-6.1-terra")
        self.assertEqual(len(candidate["dialogue"]), 8)
        self.assertTrue(candidate["review_required"])
        self.assertEqual(self.events.read_bytes(), b"")
        self.generate()
        self.assertEqual(len(self.api.attempts), 9)
        self.verify_fresh.assert_called_once()

    def test_fresh_queue_with_evaluation_overlap_fails_before_any_paid_work(self):
        self.verify_fresh.side_effect = ValueError("Evaluation image overlaps frozen source")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.generate()
        self.assertEqual(self.api.attempts, [])
        self.assertFalse((self.run_dir / "production.sqlite3").exists())

    def test_preflight_checks_images_while_status_is_a_lightweight_read(self):
        self.verify_fresh.return_value = {"verified_samples": 1}
        self.assertEqual(production.inspect_run(self.root, self.digest)["state"], "prepared")
        self.verify_fresh.assert_not_called()
        report = production.inspect_run(self.root, self.digest, verify_sources=True)
        self.assertEqual(report["cohort"], {"verified_samples": 1})
        self.assertFalse((self.run_dir / "production.sqlite3").exists())

    def test_budget_stops_before_any_submission(self):
        self.prior.return_value = production_budget.CREDIT - pilot.nanos(0.0009)
        report = self.generate()
        self.assertEqual(report["state"], "budget_reached")
        self.assertEqual(self.api.attempts, [])

    def test_uncertain_request_stays_reserved_across_restart(self):
        def lost_response():
            raise RuntimeError("unknown outcome")
        self.api.on_generate = lost_response
        with self.assertRaises(RuntimeError):
            self.generate()
        report = self.generate()
        self.assertEqual(len(self.api.attempts), 1)
        self.assertEqual(report["outcomes"], {"hold": 1})
        self.assertEqual(report["outstanding_calls"], 1)
        self.assertEqual(report["reserved_usd"], 0.001)
        self.assertEqual((self.run_dir / "pending_candidates.jsonl").read_bytes(), b"")

    def test_a_decision_during_the_last_request_protects_the_original(self):
        def human_reviews():
            if len(self.api.attempts) == 8:
                ReviewStore(self.events).append(fingerprint=self.sample["base_fingerprint"], decision="approved",
                                                uid=self.uid, collection=COLLECTION)
        self.api.on_generate = human_reviews
        self.assertEqual(self.generate()["outcomes"], {"protected": 1})
        self.assertEqual((self.run_dir / "pending_candidates.jsonl").read_bytes(), b"")
        self.assertEqual(len(self.events.read_text().splitlines()), 1)

    def test_capacity_retries_are_bounded_and_release_each_reservation(self):
        def congested():
            raise PilotRequestRejected("OpenAI rejected the request (HTTP 429)")
        self.api.on_generate = congested
        with self.assertRaises(RuntimeError):
            self.generate()
        self.assertEqual(len(self.api.attempts), 3)
        self.assertEqual(self.summary()["reserved_usd"], 0)
        self.assertEqual(self.summary()["unprocessed"], 1)

    def test_existing_writer_lock_blocks_a_second_run_before_a_request(self):
        with pilot.operation(self.run_dir):
            with self.assertRaisesRegex(RuntimeError, "running"):
                self.generate()
        self.assertEqual(self.api.attempts, [])
        self.assertFalse((self.run_dir / "production.sqlite3").exists())
