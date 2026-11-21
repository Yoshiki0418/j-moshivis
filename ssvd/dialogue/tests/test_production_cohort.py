"""Exercise selection and immutable publication without using production data or APIs."""

from contextlib import closing
import json
import sqlite3
from unittest import TestCase, mock

from ssvd.dialogue import production_cohort as cohort
from ssvd.dialogue.tests import test_pilot_selection as fixtures
from ssvd.images.collect_japanese_daily_v1 import image_fingerprints
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore


class ProductionSelectionTests(TestCase):
    def setUp(self):
        # Same held-out uid, alias, exact, near, benchmark and human rejection as the pilot tests.
        data = fixtures.PilotSelectionTests()
        data.setUp()
        self.addCleanup(data.doCleanups)
        self.root = data.root
        self.path = self.root / cohort.DEFAULT_MANIFEST
        self.protected = set(data.restricted)
        with closing(sqlite3.connect(data.catalog)) as db, (self.root / "images.jsonl").open("wb") as out:
            for name in ("metadata_source", "metadata_offset", "metadata_length"):
                db.execute(f"ALTER TABLE items ADD COLUMN {name}")
            db.execute("INSERT INTO sources VALUES ('images','images.jsonl')")
            for uid, path in db.execute("SELECT uid,image_path FROM items").fetchall():
                content = json.dumps(dict(uid=uid, path=path, fingerprints=image_fingerprints(self.root / path))).encode()
                db.execute("UPDATE items SET metadata_source='images',metadata_offset=?,metadata_length=? WHERE uid=?",
                           (out.tell(), len(content), uid))
                out.write(content + b"\n")
            db.commit()

    def test_queue_mixes_every_prefix_and_excludes_history_and_eval(self):
        with mock.patch.object(cohort, "_previous", return_value={}):
            initial = cohort.prepare_manifest(self.root, self.path, 20)
            first = initial["samples"][0]
            previous = {first["uid"]: first}
        with mock.patch.object(cohort, "_previous", return_value=previous):
            manifest = cohort.prepare_manifest(self.root, self.path, 24)
            self.assertEqual(manifest["samples"], cohort.prepare_manifest(self.root, self.path, 24)["samples"])
            ids = {r["uid"] for r in manifest["samples"]}
            self.assertFalse(ids & (self.protected | previous.keys()))
            for n in range(1, 25):
                count = sum(s["kind"] == "counting" for s in manifest["samples"][:n])
                self.assertEqual(count, (n * manifest["sampling_target"]["counting"] + 12) // 24)
            digest = cohort._publish(self.path, manifest)
            self.assertEqual(cohort.load_manifest(self.path, digest), manifest)
            with self.assertRaises(FileExistsError):
                cohort._publish(self.path, {"samples": []})
            with self.assertRaises(ValueError):
                cohort.load_manifest(self.path, "0" * 64)
            uid = manifest["samples"][0]["uid"]
            store = ReviewStore(self.root / DEFAULT_REVIEW_EVENTS)
            store.append(uid=uid, collection=cohort.COLLECTION, fingerprint="b" * 64, decision="approved")
            result = cohort.verify_manifest(self.root, self.path, digest)
            self.assertEqual(result["verified_samples"], 24)
            self.assertEqual(result["human_decisions_since_freeze"], 1)
            self.assertEqual(result["eligible_in_catalog"], 23)

    def test_review_on_an_unrelated_image_does_not_invalidate_selection(self):
        human = cohort._touched(self.root)
        with mock.patch.object(cohort, "_previous", return_value={}):
            samples = cohort.prepare_manifest(self.root, self.path, 4)["samples"]
            # A held-out UID is outside this queue and already blocked, including pixel matches.
            with mock.patch.object(cohort, "_touched", side_effect=[human, human | {next(iter(self.protected - human))}]):
                self.assertEqual(cohort.prepare_manifest(self.root, self.path, 4)["samples"], samples)
            with mock.patch.object(cohort, "_touched", side_effect=[human, human | {samples[0]["uid"]}]):
                with self.assertRaisesRegex(ValueError, "during selection"):
                    cohort.prepare_manifest(self.root, self.path, 4)

    def test_changed_source_and_unavailable_history_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "first-100"):
            cohort._previous(self.root, self.path)
        with mock.patch.object(cohort, "_previous", return_value={}):
            manifest, digest = cohort.freeze_manifest(self.root, self.path, 4, cohort.SEED)
            with self.assertRaises(FileExistsError):
                cohort.freeze_manifest(self.root, self.path, 4, cohort.SEED)
            (self.root / manifest["samples"][0]["image_path"]).write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "image identity has changed"):
                cohort.verify_manifest(self.root, self.path, digest)
            self.path.unlink()
            (self.path.parent / "billing-contract.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Restore the original manifest"):
                cohort.freeze_manifest(self.root, self.path, 4, cohort.SEED)

    def test_prior_unfinished_calls_and_neighboring_queues_are_protected(self):
        pilot = self.root / cohort.PILOTS
        pilot.mkdir(parents=True)
        first = self.root / cohort.WAVES / "first100_v10"
        first.mkdir(parents=True)
        earlier = first.parent / "previous_production"
        earlier.mkdir()
        rows = [dict(uid=str(i), image_path=f"{i}.jpg", image_sha256=f"{i:064x}") for i in range(102)]
        (pilot / "manifest.json").write_text(json.dumps({"samples": rows[:1]}))
        (first / "manifest.json").write_text(json.dumps({"samples": rows[1:101]}))
        (earlier / "manifest.json").write_text(json.dumps({"samples": rows[101:]}))
        path = earlier / "production.sqlite3"
        with closing(sqlite3.connect(path)) as db:
            db.executescript("CREATE TABLE calls(uid); CREATE TABLE outcomes(uid); INSERT INTO calls VALUES ('101');")
        self.assertEqual(set(cohort._previous(self.root, self.path)), {r["uid"] for r in rows})
        with closing(sqlite3.connect(path)) as db:
            db.execute("INSERT INTO calls VALUES ('orphan')")
            db.commit()
        with self.assertRaisesRegex(ValueError, "no corresponding source"):
            cohort._previous(self.root, self.path)
