from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from ssvd.ai_suggestion_store import (
    AISuggestionConflictError,
    AISuggestionIntegrityError,
    AISuggestionStore,
    MAX_PAYLOAD_BYTES,
    default_ai_suggestion_store,
    human_touched_uids,
)
from ssvd.review_store import ReviewStore


class AISuggestionStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = default_ai_suggestion_store(self.root)
        self.key = dict(collection="adaptive", uid="example:1",
                        base_fingerprint="a" * 64, image_sha256="b" * 64)
        self.proposal = dict(
            **self.key,
            dialogue=[{"speaker": "ユーザー", "text": "何が見えますか？"},
                      {"speaker": "アシスタント", "text": "犬が見えます。"}],
            auxiliary_task=None, model="gpt-6-luna", verdict="corrected",
            changes=[{"turn_index": 1, "reason": "画像で確認できる対象に修正"}],
            concerns=[], provenance={"cli_version": "test-only"},
        )

    def test_missing_reads_create_no_directory_or_database(self):
        self.assertIsNone(self.store.get_suggestion(**self.key))
        self.assertEqual(0, self.store.count())
        self.assertEqual([], list(self.root.iterdir()))

    def test_round_trip_is_detached_immutable_and_idempotent(self):
        saved = self.store.put_suggestion(**self.proposal)
        self.assertEqual(saved, self.store.get_suggestion(**self.key))
        self.assertEqual(saved, self.store.put_suggestion(**self.proposal))
        content = {k: v for k, v in saved.items() if k not in {"suggestion_id", "created_at"}}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, saved["suggestion_id"])
        saved["dialogue"][1]["text"] = "returned value changed"
        self.proposal["dialogue"][1]["text"] = "caller input changed"
        self.assertEqual("犬が見えます。", self.store.get_suggestion(**self.key)["dialogue"][1]["text"])
        self.assertEqual(1, self.store.count())
        self.assertFalse((self.root / "data/curation/dialogue_review_events.jsonl").exists())

    def test_source_identity_changes_do_not_reuse_suggestions(self):
        self.store.put_suggestion(**self.proposal)
        for name, value in (("collection", "other"), ("uid", "other"),
                            ("base_fingerprint", "c" * 64), ("image_sha256", "c" * 64)):
            with self.subTest(name=name):
                key = dict(self.key, **{name: value})
                self.assertIsNone(self.store.get_suggestion(**key))
                self.store.put_suggestion(**dict(self.proposal, **{name: value}))
        self.assertEqual(5, self.store.count())
        self.assertEqual(1, self.store.count(collection="other"))

    def test_different_payload_conflicts_without_changing_saved_row(self):
        saved = self.store.put_suggestion(**self.proposal)
        for field, value in (("model", "another-model"), ("verdict", "needs_attention"),
                             ("concerns", ["要確認"]), ("provenance", {"different": True})):
            with self.subTest(field=field):
                with self.assertRaises(AISuggestionConflictError):
                    self.store.put_suggestion(**dict(self.proposal, **{field: value}))
                self.assertEqual(saved, self.store.get_suggestion(**self.key))

    def test_concurrent_writers_keep_one_identical_result(self):
        # Initialize once, then exercise separate connections as HTTP workers do.
        saved = self.store.put_suggestion(**self.proposal)
        def write(_):
            return AISuggestionStore(self.store.path).put_suggestion(**self.proposal)
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write, range(24)))
        self.assertTrue(all(result == saved for result in results))
        self.assertEqual(1, self.store.count())
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            self.assertEqual("wal", connection.execute("PRAGMA journal_mode").fetchone()[0])

    def test_concurrent_first_writers_cannot_overwrite_each_other(self):
        def write(index):
            proposal = dict(self.proposal, uid="contended", concerns=[str(index)])
            try:
                return AISuggestionStore(self.store.path).put_suggestion(**proposal)
            except AISuggestionConflictError:
                return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write, range(12)))
        self.assertEqual(1, sum(result is not None for result in results))
        self.assertEqual(1, self.store.count())

    def test_invalid_inputs_fail_before_creating_database(self):
        invalid = [
            {"base_fingerprint": "z" * 64}, {"image_sha256": "A" * 64},
            {"uid": ""}, {"collection": "\x00"}, {"model": ""},
            {"dialogue": []}, {"dialogue": [{"speaker": "system", "text": "x"}]},
            {"dialogue": [{"speaker": "ユーザー", "text": ""}]},
            {"dialogue": [{"speaker": "ユーザー", "text": "x", "extra": 1}]},
            {"dialogue": [{"speaker": "ユーザー", "text": "x" * 16001}]},
            {"changes": [{"turn_index": True, "reason": "bad index"}]},
            {"changes": [{"turn_index": 2, "reason": "bad index"}]},
            {"changes": [{"turn_index": 1, "reason": "", "extra": 1}]},
            {"concerns": "wrong type"}, {"concerns": [4]}, {"verdict": "approved"},
            {"auxiliary_task": []}, {"provenance": {"bad": float("nan")}},
            {"provenance": {3: "bad key"}}, {"provenance": {"bad": {1, 2}}},
        ]
        for update in invalid:
            with self.subTest(update=repr(update)[:120]):
                with self.assertRaises(ValueError):
                    self.store.put_suggestion(**dict(self.proposal, **update))
        self.assertFalse(self.store.path.exists())

    def test_metadata_depth_and_total_byte_limits(self):
        nested = {}
        for _ in range(15):
            nested = {"nested": nested}
        huge = [{"speaker": "ユーザー", "text": "あ" * 16000} for _ in range(12)]
        self.assertGreater(len(json.dumps(huge, ensure_ascii=False).encode()), MAX_PAYLOAD_BYTES)
        for update in ({"provenance": nested}, {"dialogue": huge}):
            with self.assertRaises(ValueError):
                self.store.put_suggestion(**dict(self.proposal, **update))
        self.assertFalse(self.store.path.exists())

    def test_tampered_payload_fails_closed_on_get_and_idempotent_put(self):
        self.store.put_suggestion(**self.proposal)
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("UPDATE suggestions SET payload_json='{}'")
        with self.assertRaises(AISuggestionIntegrityError):
            self.store.get_suggestion(**self.key)
        with self.assertRaises(AISuggestionIntegrityError):
            self.store.put_suggestion(**self.proposal)

    def test_modified_payload_with_recomputed_storage_hash_still_fails_content_hash(self):
        saved = self.store.put_suggestion(**self.proposal)
        saved["dialogue"][1]["text"] = "別の内容です。"
        raw = json.dumps(saved, ensure_ascii=False)
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("UPDATE suggestions SET payload_json=?, payload_sha256=?",
                               (raw, hashlib.sha256(raw.encode()).hexdigest()))
        with self.assertRaises(AISuggestionIntegrityError):
            self.store.get_suggestion(**self.key)

    def test_corrupt_database_fails_closed(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_bytes(b"not a database")
        with self.assertRaises(AISuggestionIntegrityError):
            self.store.get_suggestion(**self.key)

    def test_full_human_history_preserves_same_fingerprint_across_collections(self):
        review = ReviewStore(self.root / "review.jsonl")
        self.assertEqual(set(), human_touched_uids(review))
        self.assertFalse(review.path.exists())
        for collection, decision in (("first", "approved"), ("second", "pending")):
            review.append(fingerprint="a" * 64, decision=decision,
                          uid="same-uid", collection=collection)
        self.assertEqual(1, len(review.latest_decisions()))
        self.assertEqual({("first", "same-uid"), ("second", "same-uid")},
                         human_touched_uids(review))

    def test_invalid_review_fragments_do_not_create_human_history(self):
        review = ReviewStore(self.root / "review.jsonl")
        valid = dict(fingerprint="a" * 64, decision="pending", uid="target", collection="first")
        invalid = [dict(valid, fingerprint="z" * 64), dict(valid, decision="automatic"),
                   dict(valid, uid=None), dict(valid, collection=[]), []]
        review.path.write_text("".join(json.dumps(row) + "\n" for row in invalid) + "{unfinished\n")
        before = review.path.read_bytes()
        self.assertEqual(set(), human_touched_uids(review))
        self.assertEqual(before, review.path.read_bytes())

    def test_human_history_cache_reuses_parse_and_invalidates_on_append_and_replace(self):
        review = ReviewStore(self.root / "review.jsonl")
        review.append(fingerprint="a" * 64, decision="pending", uid="first", collection="c")
        with mock.patch("ssvd.ai_suggestion_store.json.loads", wraps=json.loads) as loads:
            result = human_touched_uids(review)
            self.assertEqual(1, loads.call_count)
            result.clear()
            self.assertEqual({("c", "first")}, human_touched_uids(review))
            self.assertEqual(1, loads.call_count)
            review.append(fingerprint="b" * 64, decision="approved", uid="second", collection="c")
            self.assertEqual({("c", "first"), ("c", "second")}, human_touched_uids(review))
            self.assertEqual(3, loads.call_count)
            replacement = self.root / "replacement.jsonl"
            replacement.write_text(json.dumps(dict(fingerprint="c" * 64, decision="rejected", uid="third", collection="c")) + "\n")
            replacement.replace(review.path)
            self.assertEqual({("c", "third")}, human_touched_uids(review))
            self.assertEqual(4, loads.call_count)


if __name__ == "__main__":
    unittest.main()
