"""Content binding and human eligibility at the offline preview entry point."""

import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ssvd.dialogue.preview_regeneration import COLLECTION, preview
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, review_metadata


class PreviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data/.dataset_viewer").mkdir(parents=True)
        self.image = self.root / "image.jpg"
        self.image.write_bytes(b"synthetic image fixture")
        self.source = dict(uid="test", image="image.jpg", task_mode="normal", generator="test-Qwen",
                           image_sha256=hashlib.sha256(self.image.read_bytes()).hexdigest(),
                           dialogue=[{"speaker": "ユーザー", "text": "OLD_DIALOGUE"}],
                           auxiliary_task={"kind": "ocr", "grounding_source": "vlm_visual",
                                           "recognized_text": "OLD_TRANSCRIPTION"},
                           verification_status="unverified_vlm_visual", split_group_key="test-group")
        self.source.update(review_metadata(self.source))
        raw = json.dumps(self.source).encode()
        (self.root / "source.jsonl").write_bytes(raw)
        with closing(sqlite3.connect(self.root / "data/.dataset_viewer/catalog.sqlite3")) as db:
            db.executescript("""
                CREATE TABLE sources (source_id, path);
                CREATE TABLE items (collection, uid, review_required, review_status, has_image,
                    has_dialogue, auxiliary_kind, image_path, review_id, dialogue_offset,
                    dialogue_length, dialogue_source);
                INSERT INTO sources VALUES ('source', 'source.jsonl');
                """)
            db.execute("INSERT INTO items VALUES (?, ?, 1, 'pending', 1, 1, 'ocr', ?, ?, 0, ?, 'source')",
                       (COLLECTION, "test", self.image.name, self.source["review_id"], len(raw)))
            db.commit()

    def test_only_image_and_task_sent_to_model(self):
        with self.assertRaisesRegex(ValueError, "4 to 6"):
            preview(self.root, "test", pairs=7)
        step = preview(self.root, "test")
        wire = json.dumps(step["request"])
        for private in ("OLD_DIALOGUE", "OLD_TRANSCRIPTION", "image.jpg", self.source["review_id"]):
            self.assertNotIn(private, wire)
        identity = dict(step["identity"], uid="other")
        with self.assertRaisesRegex(ValueError, "different source"):
            preview(self.root, "test", transcript={"identity": identity, "replies": []})

    def test_prior_human_event_and_image_replacement(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            preview(self.root, "test")
        store = ReviewStore(self.root / DEFAULT_REVIEW_EVENTS)
        store.append(fingerprint="a" * 64, decision="rejected", collection=COLLECTION, uid="test")
        with self.assertRaisesRegex(ValueError, "human"):
            preview(self.root, "test")

    def test_new_ocr_version_uses_existing_review_gate(self):
        finished = dict(status="pending_review", task_pair=2,
                        grounding=dict(written="SAMPLE", reading="サンプル", target="看板"),
                        dialogue=[{"role": r, "text": t} for r, t in [
                            ("user", "どんな場所？"), ("assistant", "看板が見えるよ。"),
                            ("user", "なんて書いてある？"), ("assistant", "サンプルだよ。")]])
        with patch("ssvd.dialogue.preview_regeneration.next_step", return_value=finished):
            result = preview(self.root, "test")
        candidate = result["candidate"]
        self.assertEqual(candidate["auxiliary_task"]["recognized_text"], "SAMPLE")
        self.assertEqual(candidate["auxiliary_task"]["spoken_reading"], "サンプル")
        self.assertEqual(candidate["auxiliary_task"]["question_turn_index"], 2)
        self.assertEqual(candidate["dialogue"][1]["speaker"], "アシスタント")
        self.assertEqual(candidate["split_group_key"], "test-group")
        self.assertTrue(candidate["review_required"])
        self.assertNotEqual(candidate["review_id"], self.source["review_id"])
        self.assertEqual(result["status"], "pending_review")
