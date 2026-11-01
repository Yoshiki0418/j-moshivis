from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ssvd.review_store import (
    ReviewConflictError,
    MAX_REVIEW_NOTE_CHARS,
    ReviewStore,
    content_fingerprint,
    image_file_sha256,
    requires_review,
    review_metadata,
)


class ReviewStoreTests(unittest.TestCase):
    def test_revision_is_durable_content_bound_and_rejection_keeps_text(self):
        store = ReviewStore(self.path)
        original = content_fingerprint(self.record)
        revised = json.loads(json.dumps(self.record))
        revised["dialogue"][1]["text"] = "3個です。"
        revised["auxiliary_task"]["answer_count"] = 3
        fingerprint = content_fingerprint(revised)
        store.append(fingerprint=fingerprint, decision="approved", uid="sample-1", collection="counting_v4",
                     revised_record=revised, base_fingerprint=original, parent_fingerprint=original)
        self.assertEqual(ReviewStore(self.path).resolve_record(self.record), revised)
        store.append(fingerprint=fingerprint, decision="rejected", uid="sample-1", collection="counting_v4")
        self.assertEqual(store.resolve_record(self.record), revised)
        self.assertEqual(store.latest_decisions()[fingerprint]["decision"], "rejected")
        with self.assertRaises(ReviewConflictError):
            store.append(fingerprint=fingerprint, decision="approved", uid="sample-1", collection="counting_v4",
                         revised_record=revised, base_fingerprint=original, parent_fingerprint=original)
        changed = dict(self.record, image_sha256="c" * 64)
        self.assertEqual(store.resolve_record(changed), changed)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "data/curation/dialogue_review_events.jsonl"
        self.record = {
            "uid": "sample-1",
            "image": "data/images/a.jpg",
            "image_sha256": "a" * 64,
            "task_mode": "normal",
            "dialogue": [
                {"speaker": "ユーザー", "text": "リンゴはいくつ？"},
                {"speaker": "アシスタント", "text": "2個です。"},
            ],
            "auxiliary_task": {
                "kind": "counting",
                "grounding_source": "vlm_visual",
                "answer_count": 2,
            },
            "verification_status": "unverified_vlm_visual",
            "generator": "test-vlm",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fingerprint_is_content_bound_and_image_hashable(self) -> None:
        image = self.root / "a.jpg"
        image.write_bytes(b"image bytes")
        self.assertEqual(image_file_sha256(image), image_file_sha256(image))

        first = content_fingerprint(self.record)
        changed = dict(self.record)
        changed["dialogue"] = [*self.record["dialogue"]]
        changed["dialogue"][1] = {
            "speaker": "アシスタント",
            "text": "3個です。",
        }
        self.assertNotEqual(first, content_fingerprint(changed))
        self.assertEqual(first, review_metadata(self.record)["review_id"])

    def test_missing_status_still_fails_closed_for_vlm_visual(self) -> None:
        record = dict(self.record)
        record.pop("verification_status")
        self.assertTrue(requires_review(record))
        self.assertTrue(review_metadata(record)["review_required"])
        for auxiliary in (
            {"kind": "count", "grounding_source": "vlm_visual"},
            {"grounding_source": "vlm_visual"},
        ):
            malformed = dict(record)
            malformed["auxiliary_task"] = auxiliary
            self.assertTrue(requires_review(malformed))
            self.assertTrue(review_metadata(malformed)["review_required"])

    def test_append_only_latest_exact_decision_wins(self) -> None:
        store = ReviewStore(self.path)
        fingerprint = content_fingerprint(self.record)
        store.append(
            fingerprint=fingerprint,
            decision="approved",
            uid="sample-1",
            collection="counting_v4",
            note="確認済み",
        )
        store.append(
            fingerprint=fingerprint,
            decision="rejected",
            uid="sample-1",
            collection="counting_v4",
            note="再確認で誤り",
        )
        with self.path.open("ab") as stream:
            stream.write(b"{partial")
        other = "b" * 64
        store.append(
            fingerprint=other,
            decision="approved",
            uid="sample-2",
            collection="counting_v4",
        )
        latest = store.latest_decisions()
        self.assertEqual(latest[fingerprint]["decision"], "rejected")
        self.assertEqual(latest[other]["decision"], "approved")
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 4)

    def test_note_limit(self) -> None:
        with self.assertRaises(ValueError):
            ReviewStore(self.path).append(
                fingerprint=content_fingerprint(self.record),
                decision="approved",
                uid="sample-1",
                collection="counting_v4",
                note="x" * (MAX_REVIEW_NOTE_CHARS + 1),
            )


if __name__ == "__main__":
    unittest.main()
