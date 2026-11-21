"""Test production accounting without account credentials, provider calls or real ledgers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot, production_budget as budget


class ProductionBudgetTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "data/curation/gpt_regeneration_issue13/production_v10_terra"
        self.run.mkdir(parents=True)
        self.digest = "f" * 64
        future = patch.object(budget, "CUTOFF", 4_000_000_000)
        future.start()
        self.addCleanup(future.stop)

    @staticmethod
    def reserve(db, uid, amount):
        with db:
            db.execute("""INSERT INTO calls(uid,model,step,request_hash,client_id,reserved)
                       VALUES(?,?,0,'request',?,?)""", (uid, budget.MODEL, uid, amount))

    def test_preflight_never_creates_missing_files_and_recovers_empty_init(self):
        budget.validate_existing(self.run, self.digest, 7)
        self.assertEqual(list(self.run.iterdir()), [])
        (self.run / "production.sqlite3").touch()
        budget.setup(self.run, self.digest, 7)
        with closing(budget.connect(self.run, readonly=True)) as db:
            self.assertEqual(db.execute("SELECT prior, available FROM limits").fetchone(), (7, budget.CREDIT - 7))
        self.assertEqual((self.run / "billing-contract.json").stat().st_mode & 0o777, 0o600)

    def test_atomic_cap_across_workers_and_preserve_unresolved_on_restart(self):
        start = budget.CREDIT - 30
        budget.setup(self.run, self.digest, start)
        ready = Barrier(2)

        def reserve(uid):
            with closing(budget.connect(self.run)) as db:
                ready.wait(timeout=3)
                try:
                    self.reserve(db, uid, 20)
                except sqlite3.IntegrityError as error:
                    self.assertIn("production_budget_limit", str(error))
                    return False
                return True

        with ThreadPoolExecutor(2) as pool:
            self.assertCountEqual(pool.map(reserve, ("one", "two")), [True, False])
        budget.setup(self.run, self.digest, start + 2)
        with closing(budget.connect(self.run)) as db:
            self.assertEqual(pilot.committed(db), 20)
            outcome = db.execute("SELECT outcome FROM outcomes").fetchone()
            self.assertEqual(json.loads(outcome[0])["reason"], "unresolved_billing")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "production_budget_limit"):
                self.reserve(db, "three", 9)
        with self.assertRaises(RuntimeError):
            budget.validate_existing(self.run, self.digest, start)  # Must not widen the contract.

    def test_expired_credit_is_refused_at_insert_even_if_worker_started_previously(self):
        with patch.object(budget, "CUTOFF", 1):
            budget.setup(self.run, self.digest, 10)
            with closing(budget.connect(self.run)) as db:
                with self.assertRaisesRegex(sqlite3.IntegrityError, "production_expired"):
                    self.reserve(db, "one", 1)

    def test_missing_mutated_or_older_ledgers_refuse_resume(self):
        budget.setup(self.run, self.digest, 10)
        file = self.run / "billing-contract.json"
        saved = file.read_text()
        file.write_text(saved.replace('"service_tier": "flex"', '"service_tier": "default"'))
        with self.assertRaisesRegex(RuntimeError, "contract differs"):
            budget.setup(self.run, self.digest, 10)
        file.write_text(saved)
        with closing(budget.connect(self.run)) as db:
            self.reserve(db, "one", 20)
            with db:
                db.execute("UPDATE calls SET charged=-1")
            with self.assertRaisesRegex(RuntimeError, "Reconcile production spending"):
                budget.validate_existing(self.run, self.digest, 10)
            with db:
                db.execute("UPDATE calls SET charged=1")
        file.unlink()
        with self.assertRaisesRegex(RuntimeError, "Restore the production billing contract"):
            budget.setup(self.run, self.digest, 10)
        file.write_text(saved)
        (self.run / "summary.json").write_text('{"paid_receipts":2,"charged_usd":0.0}')
        with self.assertRaisesRegex(RuntimeError, "older"):
            budget.setup(self.run, self.digest, 10)
        (self.run / "production.sqlite3").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing production ledger"):
            budget.setup(self.run, self.digest, 10)
        with self.assertRaisesRegex(RuntimeError, "valid remaining project budget"):
            budget.setup(self.run, self.digest, budget.CREDIT + 1)

    def _historical_ledgers(self):
        pilots = self.root / "data/curation/gpt_pilot_issue12"
        first = self.run.parent / "first100_v10"
        first.mkdir()
        rows = []
        for version in (pilot.LEGACY_VERSION, pilot.PROMPT_VERSION):
            directory = pilots if version == pilot.LEGACY_VERSION else pilots / version
            directory.mkdir()
            path = directory / "pilot.sqlite3"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE cohort(id,manifest,prompt,models)")
                manifest = {"samples": []}
                if version != pilot.LEGACY_VERSION:
                    manifest["budget_predecessors"] = [pilot.LEGACY_VERSION]
                db.execute("INSERT INTO cohort VALUES(1,?,?,?)", (pilot.encode(manifest), version, pilot.encode(pilot.MODELS)))
            rows.append((path, version, None if version == pilot.LEGACY_VERSION else 5))
        quota = 100
        prior = 10 + 5  # One unresolved reservation and one settled call.
        contract = dict(model=budget.MODEL, all_runs_max_nano_usd=budget.CREDIT, manifest_sha=self.digest,
                        shards=4, allocated_nano_usd=[quota]*4, issue13_max_nano_usd=4*quota,
                        pilot_committed_nano_usd=prior)
        (first / "billing-contract.json").write_text(json.dumps(contract))
        for i in range(4):
            path = first / f"shard{i}.sqlite3"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE limits(id,manifest_sha,model,shard,cap,prior)")
                db.execute("INSERT INTO limits VALUES(1,?,?,?,?,?)", (self.digest,budget.MODEL,i,quota,prior))
            rows.append((path, str(i), 5))
        for path, request, charge in rows:
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE calls(client_id,reserved,charged,receipt)")
                receipt = None if charge is None else json.dumps({"response_id": "resp_"+request, "cost_usd": charge/1e9})
                db.execute("INSERT INTO calls VALUES(?,10,?,?)", (request,charge,receipt))
        (first / "summary.json").write_text(json.dumps(dict(remaining=0, paid_receipts=4, pairs=2,
                    receipt_usd=20/1e9, unresolved_reserve_usd=0)))
        backup = pilots / "initial_snapshot/pilot.sqlite3"
        backup.parent.mkdir()
        shutil.copyfile(rows[0][0], backup)
        return pilots, first

    def test_both_prior_wave_and_unresolved_pilot_are_counted_once(self):
        pilots, first = self._historical_ledgers()
        self.assertEqual(budget.prior_commitment(self.root,self.run), 35)
        with closing(sqlite3.connect(pilots / "initial_snapshot/pilot.sqlite3")) as db, db:
            db.execute("INSERT INTO calls VALUES('snapshot-only',10,NULL,NULL)")
        with self.assertRaisesRegex(RuntimeError, "only in the pilot snapshot"):
            budget.prior_commitment(self.root, self.run)
        with closing(sqlite3.connect(pilots / "initial_snapshot/pilot.sqlite3")) as db, db:
            db.execute("DELETE FROM calls WHERE client_id='snapshot-only'")
        with closing(sqlite3.connect(first/"shard3.sqlite3")) as db, db:
            db.execute("UPDATE calls SET client_id='1'")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            budget.prior_commitment(self.root,self.run)
        (first / "shard2.sqlite3").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            budget.prior_commitment(self.root,self.run)
        (pilots / "pilot.sqlite3").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing pilot"):
            budget.prior_commitment(self.root,self.run)
