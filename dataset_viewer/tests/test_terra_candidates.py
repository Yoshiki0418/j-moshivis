import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dataset_viewer.terra_candidates import COLLECTION, MODEL, TerraCandidateStore
from ssvd.review_store import REVIEW_REQUIRED_STATUS, ReviewConflictError, review_metadata


def create_ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE outcomes(uid TEXT PRIMARY KEY, model TEXT NOT NULL, outcome TEXT NOT NULL);
        CREATE TABLE calls(uid TEXT, model TEXT, step INTEGER, charged INTEGER, finished_at TEXT,
                           PRIMARY KEY(uid,model,step));
    """)
    return db


def outcome(uid, text="こんにちは", *, version="visual_dialogue_v10"):
    image = "b" * 64
    c = {
        "uid": uid, "generator": MODEL, "image": "image.jpg", "image_sha256": image,
        "prompt_version": version, "task_mode": "normal",
        "auxiliary_task": {"kind": "counting", "additional_tasks": [{"kind": "ocr"}]},
        "verification_status": REVIEW_REQUIRED_STATUS,
        "dialogue": [{"speaker": "アシスタント" if i % 2 else "ユーザー", "text": text} for i in range(8)],
    }
    c.update(review_metadata(c))
    return {"candidate": c, "status": "pending_review",
            "identity": {"uid": uid, "model": MODEL, "collection": COLLECTION, "image_sha256": image,
                         "pairs": 4, "prompt_version": c["prompt_version"], "base_fingerprint": "a" * 64}}


def add(db, uid, value=None):
    db.execute("INSERT INTO outcomes VALUES(?,?,?)", (uid, MODEL, json.dumps(value or outcome(uid))))
    db.execute("INSERT INTO calls VALUES(?,?,7,0,'2026-11-21 10:00:00')", (uid, MODEL))


class TerraCandidateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dir = self.root / "data/curation/gpt_regeneration_issue13"
        self.store = TerraCandidateStore(self.root)

    def create(self, name):
        db = create_ledger(self.dir / name)
        self.addCleanup(db.close)
        return db

    def get(self, uid, **overrides):
        key = dict(collection=COLLECTION, uid=uid, base_fingerprint="a"*64, image_sha256="b"*64)
        return self.store.get(**{**key, **overrides})

    def test_committed_first_wave_and_live_production_without_modifying_sources(self):
        self.assertEqual(self.store.keys(), ())
        self.assertFalse(self.dir.exists())
        with self.create("first100_v10/shard3.sqlite3") as first:
            add(first, "first")
        prod = self.create("production_v10_terra/production.sqlite3")
        prod.execute("PRAGMA journal_mode=WAL")
        prod.execute("PRAGMA wal_autocheckpoint=0")
        with prod:
            add(prod, "live")
        self.assertEqual({x[1] for x in self.store.keys()}, {"first", "live"})
        before = self.store.version()
        add(prod, "next")
        self.assertIsNone(self.get("next"))
        self.assertEqual(self.store.version(), before)
        prod.commit()
        new = self.get("next")
        self.assertGreater(self.store.version(), before)
        self.assertEqual(new["source"], "production_v10_terra")
        self.assertEqual(new["created_at"], "2026-11-21T10:00:00Z")
        self.assertEqual(self.get("first")["source"], "first100_v10")
        self.assertEqual(self.store.queue_keys()[-1][-2:], (1, 1))
        self.assertIsNone(self.get("first", base_fingerprint="c"*64))
        self.assertIsNone(self.get("first", image_sha256="c"*64))
        new["candidate"]["dialogue"].clear()
        self.assertEqual(len(self.get("next")["candidate"]["dialogue"]), 8)

    def test_only_complete_canonical_candidates_and_no_conflicting_versions(self):
        with self.create("production_v10_terra/production.sqlite3") as prod:
            add(prod, "held", {"status": "hold"})
            add(prod, "protected", {"status": "protected"})
            bad = outcome("corrupt")
            bad["candidate"]["dialogue"][1]["text"] = "changed"
            add(prod, "corrupt", bad)
            other = outcome("wrong-model")
            other["identity"]["model"] = "local"
            add(prod, "wrong-model", other)
            add(prod, "conflict")
            plain = outcome("plain")
            plain["candidate"]["auxiliary_task"] = {"kind": "none"}
            plain["candidate"].update(review_metadata(plain["candidate"]))
            add(prod, "plain", plain)
        with self.create("first100_v10/shard0.sqlite3") as first:
            add(first, "conflict", outcome("conflict", "another version"))
        with self.assertLogs("dataset_viewer.terra_candidates", level="WARNING"):
            keys = self.store.queue_keys()
        self.assertEqual([(k[1], k[4:]) for k in keys], [("plain", (0, 0))])
        self.assertIsNone(self.get("conflict"))

    def test_shared_production_ledger_requires_the_uid_assignment_for_v11(self):
        prod = self.create("production_v10_terra/production.sqlite3")
        v11 = "visual_dialogue_v11"
        with prod:
            add(prod, "legacy")
        self.assertIsNotNone(self.get("legacy"))
        schema = "CREATE TABLE dialogue_prompt_versions(uid TEXT PRIMARY KEY, prompt_version TEXT NOT NULL)"
        with prod:
            prod.execute(schema)
            prod.executemany("INSERT INTO dialogue_prompt_versions VALUES(?,?)",
                             [("new", v11), ("duplicate", v11), ("wrong-candidate", v11),
                              ("future", "visual_dialogue_v99")])
            add(prod, "new", outcome("new", version=v11))
            add(prod, "duplicate", outcome("duplicate", version=v11))
            add(prod, "unassigned", outcome("unassigned", version=v11))
            add(prod, "wrong-candidate", outcome("wrong-candidate"))
            add(prod, "future", outcome("future", version="visual_dialogue_v99"))
            mismatch = outcome("mismatch")
            mismatch["identity"]["prompt_version"] = v11
            add(prod, "mismatch", mismatch)
        with self.create("first100_v10/shard0.sqlite3") as first:
            first.execute(schema)
            first.execute("INSERT INTO dialogue_prompt_versions VALUES('forged',?)", (v11,))
            add(first, "forged", outcome("forged", version=v11))
            add(first, "duplicate")
        with self.assertLogs("dataset_viewer.terra_candidates", level="WARNING"):
            keys = self.store.queue_keys()
        self.assertEqual({(key[1], key[4:]) for key in keys}, {("legacy", (1, 1)), ("new", (1, 1))})
        self.assertEqual(self.get("new")["candidate"]["prompt_version"], v11)

        # Check the current assignment when a preview is reopened for approval.
        with prod:
            prod.execute("DELETE FROM dialogue_prompt_versions WHERE uid='new'")
        with self.assertLogs("dataset_viewer.terra_candidates", level="WARNING"):
            self.assertIsNone(self.get("new"))
        self.assertEqual([key[1] for key in self.store.keys()], ["legacy"])

    def test_cli_and_api_share_the_queue_but_require_their_own_version_and_source(self):
        v11 = "visual_dialogue_v11"
        prod = self.create("production_v10_terra/production.sqlite3")
        with prod:
            add(prod, "existing-api")
            prod.execute("CREATE TABLE dialogue_prompt_versions(uid TEXT PRIMARY KEY, prompt_version TEXT NOT NULL)")
            prod.execute("INSERT INTO dialogue_prompt_versions VALUES('duplicate',?)", (v11,))
            add(prod, "duplicate", outcome("duplicate", version=v11))
            forged_api = outcome("wrong-api-source")
            forged_api["candidate"]["generation_backend"] = "codex_cli"
            add(prod, "wrong-api-source", forged_api)

        path = self.root / "data/curation/codex_regeneration_terra_v11/generation.sqlite3"
        path.parent.mkdir(parents=True)
        cli = sqlite3.connect(path)
        self.addCleanup(cli.close)
        cli.execute("PRAGMA journal_mode=WAL")
        cli.execute("CREATE TABLE outcomes(uid TEXT PRIMARY KEY, model TEXT, outcome TEXT, finished_at TEXT)")
        when = "2026-11-21 13:00:00"
        def insert(uid, version=v11, root_tag="codex_cli", candidate_tag="codex_cli"):
            row = outcome(uid, version=version)
            if root_tag:
                row["generation_backend"] = root_tag
            if candidate_tag:
                row["candidate"]["generation_backend"] = candidate_tag
            cli.execute("INSERT INTO outcomes VALUES(?,?,?,?)", (uid, MODEL, json.dumps(row), when))

        with cli:
            insert("new-cli")
            insert("duplicate")  # Same content and version from two producers must not be silently relabelled.
            insert("no-root-tag", root_tag=None)
            insert("no-candidate-tag", candidate_tag=None)
            insert("wrong-source-tag", candidate_tag="openai_api")
            insert("wrong-version", version="visual_dialogue_v10")
        with self.assertLogs("dataset_viewer.terra_candidates", level="WARNING"):
            keys = self.store.queue_keys()
        self.assertEqual({key[1] for key in keys}, {"existing-api", "new-cli"})
        self.assertEqual(self.get("existing-api")["generation_backend"], "openai_api")
        draft = self.get("new-cli")
        self.assertEqual(draft["generation_backend"], "codex_cli")
        self.assertEqual(draft["created_at"], "2026-11-21T13:00:00Z")
        insert("just-generated")
        self.assertIsNone(self.get("just-generated"))
        cli.commit()
        self.assertEqual(self.get("just-generated")["candidate"]["prompt_version"], v11)

    def test_selected_candidate_is_rechecked_after_in_place_repair_or_deletion(self):
        prod = self.create("production_v10_terra/production.sqlite3")
        with prod:
            add(prod, "repaired")
            add(prod, "deleted")
        original_id = self.get("repaired")["candidate"]["review_id"]
        with prod:
            repaired = outcome("repaired", "訂正済み")
            prod.execute("UPDATE outcomes SET outcome=? WHERE uid='repaired'", (json.dumps(repaired),))
        self.assertNotEqual(self.get("repaired")["candidate"]["review_id"], original_id)
        with prod:
            prod.execute("UPDATE outcomes SET outcome=? WHERE uid='repaired'", (json.dumps({"status": "hold"}),))
        self.assertIsNone(self.get("repaired"))
        with prod:
            prod.execute("DELETE FROM outcomes WHERE uid='deleted'")
        self.assertFalse(self.store.keys())

    def test_temporarily_locked_candidate_does_not_fall_back_to_local_dialogue(self):
        prod = self.create("production_v10_terra/production.sqlite3")
        with prod:
            add(prod, "locked")
        self.assertIsNotNone(self.get("locked"))
        prod.execute("BEGIN EXCLUSIVE")
        try:
            with self.assertRaises(ReviewConflictError):
                self.get("locked")
        finally:
            prod.rollback()
        self.assertIsNotNone(self.get("locked"))

    def test_missing_or_replaced_private_ledger_removes_its_stale_candidates(self):
        p = self.dir / "production_v10_terra/production.sqlite3"
        with self.create("production_v10_terra/production.sqlite3") as db:
            add(db, "old")
        self.assertIsNotNone(self.get("old"))
        p.rename(p.with_suffix(".backup"))
        with self.create("production_v10_terra/production.sqlite3") as db:
            add(db, "new")
        self.assertEqual([k[1] for k in self.store.keys()], ["new"])
        p.unlink()
        self.assertFalse(self.store.keys())


if __name__ == "__main__":
    unittest.main()
