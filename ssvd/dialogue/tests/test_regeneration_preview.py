"""Content binding and human eligibility at the offline preview entry point."""

import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ssvd.dialogue.preview_regeneration import COLLECTION, preview
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, content_fingerprint, review_metadata


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
        with self.assertRaisesRegex(ValueError, "4 to 8"):
            preview(self.root, "test", pairs=9)
        step = preview(self.root, "test")
        wire = json.dumps(step["request"])
        for private in ("OLD_DIALOGUE", "OLD_TRANSCRIPTION", "image.jpg", self.source["review_id"]):
            self.assertNotIn(private, wire)
        identity = dict(step["identity"], uid="other")
        with self.assertRaisesRegex(ValueError, "different source"):
            preview(self.root, "test", transcript={"identity": identity, "replies": []})
        for old_version in ("grounded_roles_v1", "grounded_roles_v2", "grounded_roles_v3"):
            earlier = dict(step["identity"], prompt_version=old_version)
            with self.assertRaisesRegex(ValueError, "different source"):
                preview(self.root, "test", transcript={"identity": earlier, "replies": []})

    def test_prior_human_event_and_image_replacement(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            preview(self.root, "test")
        store = ReviewStore(self.root / DEFAULT_REVIEW_EVENTS)
        store.append(fingerprint="a" * 64, decision="rejected", collection=COLLECTION, uid="test")
        with self.assertRaisesRegex(ValueError, "human"):
            preview(self.root, "test")

    def replay(self, tasks, exchanges, pairs):
        identity = preview(self.root, "test", pairs=pairs)["identity"]
        replies = []
        transcript = {"identity": identity, "replies": replies}
        for pair in range(1, pairs + 1):
            task = tasks[pair - 1] if pair <= len(tasks) else None
            q, a = exchanges[task["kind"]] if task else ("どんな様子？", "看板と自転車が見えるよ。")
            replies.extend([{"text": q}, {"text": a, "tasks": [task] if task else []}])
            step = preview(self.root, "test", pairs=pairs, transcript=transcript)
            self.assertNotEqual(step["status"], "hold", step.get("reason"))
        self.assertEqual(step["status"], "pending_review")
        return step

    def test_dual_task_fingerprint_and_no_exact_answer_for_lower_bound(self):
        ocr = dict(kind="ocr", mode="exact", written="SAMPLE", reading="サンプル", target="看板")
        count = dict(kind="counting", mode="lower_bound", count=3, target="自転車", reason="奥が隠れている")
        result = self.replay([ocr, count], {
            "ocr": ("看板にはなんて書いてある？", "サンプルって書いてあるよ。"),
            "counting": ("自転車は何台ある？", "少なくとも三台は見えるよ。")}, pairs=8)
        candidate = result["candidate"]
        auxiliary = candidate["auxiliary_task"]
        self.assertEqual(auxiliary["recognized_text"], "SAMPLE")
        self.assertEqual(auxiliary["spoken_reading"], "サンプル")
        self.assertEqual(auxiliary["question_turn_index"], 0)
        second = auxiliary["additional_tasks"][0]
        self.assertEqual(second["minimum_count"], 3)
        self.assertNotIn("answer_count", second)
        self.assertEqual(len(candidate["dialogue"]), 16)
        self.assertEqual(candidate["dialogue"][1]["speaker"], "アシスタント")
        self.assertEqual(candidate["split_group_key"], "test-group")
        self.assertTrue(candidate["review_required"])
        self.assertNotEqual(candidate["review_id"], self.source["review_id"])
        second["minimum_count"] = 4
        self.assertNotEqual(content_fingerprint(candidate), candidate["review_id"])

    def test_ordinary_candidate_from_ocr_source_still_requires_review(self):
        candidate = self.replay([], {}, pairs=7)["candidate"]
        self.assertEqual(candidate["auxiliary_kind"], "none")
        self.assertEqual(len(candidate["dialogue"]), 14)
        self.assertTrue(candidate["review_required"])

    def test_uncertain_observations_keep_only_supported_values(self):
        for kind, mode, values, answer, expected in [
            ("counting", "approximate", {"count": 8}, "八台くらいに見えるよ。", {"estimated_count": 8}),
            ("counting", "unknown", {}, "重なってて台数は分からないな。", {}),
            ("ocr", "partial", {"written": "A7", "reading": "エーなな"},
             "エーなな、って部分だけ読めるよ。", {"visible_fragment": "A7", "spoken_reading": "エーなな"}),
            ("ocr", "unknown", {}, "ぼやけていて文字までは読めないな。", {}),
        ]:
            with self.subTest(kind=kind, mode=mode):
                target, question = ("看板", "看板には何て書いてある？") if kind == "ocr" else ("自転車", "自転車は何台ある？")
                task = dict(kind=kind, mode=mode, target=target, reason="一部が隠れている", **values)
                result = self.replay([task], {kind: (question, answer)}, pairs=4)
                auxiliary = result["candidate"]["auxiliary_task"]
                self.assertTrue(result["candidate"]["review_required"])
                self.assertEqual(auxiliary["answer_mode"], mode)
                self.assertEqual(auxiliary["reason"], task["reason"])
                value_fields = {"answer_count", "minimum_count", "estimated_count",
                                "recognized_text", "visible_fragment", "spoken_reading"}
                self.assertEqual({k: auxiliary[k] for k in value_fields if k in auxiliary}, expected)
