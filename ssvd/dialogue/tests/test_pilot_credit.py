"""The earlier pilot/recovery budget still applies after the production wave takes over."""

from contextlib import ExitStack, closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue import pilot, production_budget
from ssvd.dialogue.production_cohort import DEFAULT_MANIFEST, PILOTS


class SharedCreditTests(unittest.TestCase):
    def setUp(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        self.root = Path(resources.enter_context(tempfile.TemporaryDirectory()))
        self.pilot_dir = self.root / PILOTS / pilot.PROMPT_VERSION
        self.db = resources.enter_context(closing(sqlite3.connect(":memory:")))
        self.db.execute("CREATE TABLE calls(charged INTEGER, reserved INTEGER)")
        self.db.execute("INSERT INTO calls VALUES(300000000,900000000)")
        self.wave = (self.root / DEFAULT_MANIFEST).parent
        self.wave.mkdir(parents=True)
        (self.wave / "manifest.json").write_bytes(b"fixture")
        resources.enter_context(patch.object(production_budget, "prior_commitment", return_value=pilot.nanos("5.3")))
        resources.enter_context(patch.object(production_budget, "validate_existing"))

    def limits(self):
        return pilot.pilot_credit_limits(self.root, self.pilot_dir, self.db, previous_commitment=pilot.nanos("1.2"))

    def test_historical_budget_and_production_settled_and_unresolved_calls(self):
        # The pilots alone have already spent $1.50 ($1.20 predecessor + $0.30 here).
        self.assertEqual(self.limits(), (pilot.nanos("2"), None))
        with closing(production_budget.connect(self.wave)) as wave, wave:
            wave.execute("CREATE TABLE calls(charged INTEGER, reserved INTEGER)")
            wave.execute("INSERT INTO calls VALUES(?,?)", (pilot.nanos("64.2"), pilot.nanos("64.5")))
            wave.execute("INSERT INTO calls VALUES(NULL,?)", (pilot.nanos("0.4"),))
        cap, deadline = self.limits()
        # $64.60 + $5.30 = $69.90 already committed, leaving only $0.10 for the pilot.
        self.assertEqual(cap, pilot.nanos("1.60"))
        with patch.object(production_budget, "CUTOFF", 0), self.assertRaises(pilot.SpendingLimitReached):
            deadline()
        with closing(production_budget.connect(self.wave)) as wave, wave:
            wave.execute("DELETE FROM calls")
        self.assertEqual(self.limits()[0], pilot.nanos("2"))
        with self.assertRaisesRegex(RuntimeError, "canonical pilot ledger"):
            pilot.pilot_credit_limits(self.root, self.root / "other-pilot", self.db, previous_commitment=0)


if __name__ == "__main__":
    unittest.main()
