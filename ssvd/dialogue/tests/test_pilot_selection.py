"""Regression tests for pilot sampling, protection and fixed-evaluation exclusion."""

from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

from PIL import Image

from ssvd.dialogue.pilot_selection import QUOTAS, SEED, prepare_manifest
from ssvd.dialogue.preview_regeneration import COLLECTION
from ssvd.dialogue.regeneration_roles import next_step
from ssvd.images.collect_japanese_daily_v1 import image_fingerprints
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, review_metadata


class PilotSelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.catalog = self.root / "data/.dataset_viewer/catalog.sqlite3"
        self.catalog.parent.mkdir(parents=True)
        self.locks = self.root / "j-moshivis/evaluation"
        self.locks.mkdir(parents=True)
        self.index = self.root / "data/evaluation/v3_image_index.jsonl"
        self.index.parent.mkdir()
        eval_manifest = self.root / "data/speech_v3/eval.jsonl"
        eval_manifest.parent.mkdir()
        uids = sorted((f"candidate-{i}" for i in range(68)),
                      key=lambda uid: hashlib.sha256(f"{SEED}:{uid}".encode()).digest())
        self.restricted = uids[:6]
        eval_uid, reviewed_uid, exact_uid, near_uid, bench_uid, group_uid = self.restricted
        store = ReviewStore(self.root / DEFAULT_REVIEW_EVENTS)
        store.append(fingerprint="b" * 64, collection=COLLECTION, uid=reviewed_uid, decision="rejected")

        def make_image(name, seed):
            path = self.root / f"{name}.png"
            pixels = random.Random(seed).randbytes(32 * 32 * 3)
            Image.frombytes("RGB", (32, 32), pixels).save(path)
            return path

        frozen = make_image("heldout", 1000)
        benchmark = make_image("benchmark", 1001)
        self.index.write_text(json.dumps(dict(uid=eval_uid, path=str(frozen), **image_fingerprints(frozen))))
        eval_manifest.write_text(json.dumps(dict(uid=eval_uid)))
        raw = self.root / "bench.jsonl"
        raw.write_text(json.dumps(dict(image_path=str(benchmark), source=dict(image_sha256=self.sha(benchmark)))))
        self.lock = {
            "training_overlap_reference": {
                "v3_eval_manifest_sha256": self.sha(eval_manifest),
                "fingerprint_index_sha256": self.sha(self.index),
            },
            "manifests": {"raw": {"path": str(raw), "sha256": self.sha(raw)}},
        }
        self.write_lock()
        (self.locks / "stair_caption_ja_val_clean_v1.lock.json").write_text(
            json.dumps({"manifest": self.lock["manifests"]["raw"]}))

        with closing(sqlite3.connect(self.catalog)) as db, (self.root / "source.jsonl").open("wb") as source:
            db.executescript("""
                CREATE TABLE sources (source_id, path);
                CREATE TABLE items (collection, uid, source_uid, review_required, review_status, has_image,
                    has_dialogue, auxiliary_kind, image_path, review_id, dialogue_offset, dialogue_length,
                    dialogue_source, summary_json);
                INSERT INTO sources VALUES ('source','source.jsonl');
                """)
            for i, uid in enumerate(uids):
                kind = "ocr" if i % 2 == 0 else "counting"
                path = make_image(uid, i)
                if uid == exact_uid:
                    path.write_bytes(frozen.read_bytes())
                elif uid == bench_uid:
                    path.write_bytes(benchmark.read_bytes())
                elif uid == near_uid:
                    with Image.open(frozen) as image:
                        image.putpixel((0, 0), (7, 7, 7))
                        image.save(path)
                aux = {"kind": kind, "grounding_source": "vlm_visual", "recognized_text": "PRIVATE SHOP"}
                row = dict(uid=uid, image=path.name, image_sha256=self.sha(path), generator="Qwen-test",
                           task_mode="normal", auxiliary_task=aux, split_group_key=eval_uid if uid == group_uid else uid,
                           verification_status="unverified_vlm_visual",
                           dialogue=[{"speaker": "ユーザー", "text": "小さな文字や重なった対象の旧入力"}])
                row.update(review_metadata(row))
                encoded = json.dumps(row).encode() + b"\n"
                db.execute("INSERT INTO items VALUES (?, ?, ?, 1, 'pending', 1, 1, ?, ?, ?, ?, ?, 'source', ?)",
                           (COLLECTION, uid, uid, kind, path.name, row["review_id"], source.tell(), len(encoded),
                            json.dumps({"generator": "Qwen-test", "image_integrity_status": "verified",
                                        "collection_buckets": ["count_friendly_everyday", "japanese_text_rich"]})))
                source.write(encoded)
            db.commit()

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_lock(self):
        (self.locks / "suite_v1.lock.json").write_text(json.dumps(self.lock))

    def test_paired_manifest_protects_human_and_fixed_images(self):
        manifest = prepare_manifest(self.root)
        self.assertEqual(manifest, prepare_manifest(self.root))
        samples = manifest["samples"]
        self.assertEqual(len(samples), 20)
        self.assertFalse(set(s["uid"] for s in samples).intersection(self.restricted))
        self.assertEqual(manifest["overlap_exclusions_encountered"],
                         {"heldout_identity": 2, "excluded_exact_overlap": 2, "excluded_near_overlap": 1})
        self.assertEqual(Counter((s["kind"], s["sampling_hint"]) for s in samples),
                         {(k, hint): n for k, quotas in QUOTAS.items() for hint, n in quotas.items()})
        self.assertNotIn("PRIVATE SHOP", json.dumps(manifest))
        self.assertNotIn("旧入力", json.dumps(manifest, ensure_ascii=False))
        for sample in samples:
            self.assertEqual(sample["pairs"], next_step(sample["uid"], sample["kind"], "data:image/png;base64,", [])["pair_count"])
            self.assertEqual(self.sha(self.root / sample["image_path"]), sample["image_sha256"])

    def test_changed_or_missing_fixed_index_stops_selection(self):
        self.index.write_text("")
        with self.assertRaisesRegex(ValueError, "differs from its lock"):
            prepare_manifest(self.root)
        self.lock["training_overlap_reference"]["fingerprint_index_sha256"] = self.sha(self.index)
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "Missing held-out"):
            prepare_manifest(self.root)
        self.index.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_manifest(self.root)

    def test_missing_human_history_stops_selection(self):
        (self.root / DEFAULT_REVIEW_EVENTS).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "human decision history"):
            prepare_manifest(self.root)
