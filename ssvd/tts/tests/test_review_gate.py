from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


TTS_DIR = Path(__file__).resolve().parents[1]
SSVD_DIR = TTS_DIR.parent
DIALOGUE_DIR = SSVD_DIR / "dialogue"
TEST_IMAGE_ASSET_SHA256 = "b" * 64
for directory in (TTS_DIR, SSVD_DIR, DIALOGUE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_speech_dataset import (  # noqa: E402
    ReviewAssetError,
    StaleSpeechMetadataError,
    build_one,
)
from generate_local_vlm import write_result  # noqa: E402
from make_train_jsonl import _review_gate, main as make_train_main  # noqa: E402
from review_store import (  # noqa: E402
    ReviewStore,
    content_fingerprint,
    image_file_sha256,
)
from task_profiles import prepare_task_entry  # noqa: E402


def reviewed_meta() -> tuple[dict, str]:
    auxiliary = {
        "kind": "counting",
        "grounding_source": "vlm_visual",
        "question_turn_index": 4,
        "answer_turn_index": 5,
        "counted_target": "リンゴ",
        "answer_count": 2,
    }
    dialogue = [
        {"speaker": "ユーザー", "text": "何が見えますか？"},
        {"speaker": "アシスタント", "text": "果物が見えます。"},
        {"speaker": "ユーザー", "text": "どんな果物ですか？"},
        {"speaker": "アシスタント", "text": "赤いリンゴです。"},
        {"speaker": "ユーザー", "text": "リンゴは何個ありますか？"},
        {"speaker": "アシスタント", "text": "リンゴは2個あります。"},
    ]
    record = {
        "uid": "source-1",
        "image": "data/images/apples.jpg",
        "image_sha256": "a" * 64,
        "task_mode": "normal",
        "dialogue": dialogue,
        "auxiliary_task": auxiliary,
        "verification_status": "unverified_vlm_visual",
        "generator": "test-vlm",
    }
    review_id = content_fingerprint(record)
    meta = {
        "uid": "source-1",
        "source_uid": "source-1",
        "image": record["image"],
        "image_sha256": record["image_sha256"],
        "image_asset_sha256": TEST_IMAGE_ASSET_SHA256,
        "task_mode": "normal",
        "auxiliary_task": auxiliary,
        "verification_status": "unverified_vlm_visual",
        "review_required": True,
        "review_id": review_id,
        "generator": "test-vlm",
        "segments": [
            {
                "speaker": "USER" if turn["speaker"] == "ユーザー" else "ASSISTANT",
                "text": turn["text"],
            }
            for turn in dialogue
        ],
    }
    return meta, review_id


class FakeVoice:
    def __init__(self, name: str):
        self.name = name
        self.styles = ["default"]
        self.default_style = "default"

    def synth(self, _text: str, **_kwargs):
        return 24_000, np.full(2_400, 0.2, dtype=np.float32)


def fake_tts_inputs() -> tuple[dict, dict]:
    voices = {
        "assistant": FakeVoice("assistant"),
        "user": FakeVoice("user"),
    }
    config = {
        "assistant": {"name": "assistant"},
        "user": [{"name": "user", "weight": 1.0}],
        "assistant_policy": {
            "base": {},
            "intonation": {},
            "happy": {},
        },
        "user_variation": {
            "intonation_scale": [1.0, 1.0],
            "length": [1.0, 1.0],
            "sdp_ratio": [0.2, 0.2],
            "alt_style_prob": 0.0,
        },
    }
    return voices, config


def reviewed_record(image: Path, *, uid: str = "source-1") -> dict:
    meta, _ = reviewed_meta()
    dialogue = [
        {
            "speaker": "ユーザー" if segment["speaker"] == "USER" else "アシスタント",
            "text": segment["text"],
        }
        for segment in meta["segments"]
    ]
    record = {
        "uid": uid,
        "image": str(image),
        "image_sha256": image_file_sha256(image),
        "task_mode": "normal",
        "dialogue": dialogue,
        "auxiliary_task": meta["auxiliary_task"],
        "verification_status": "unverified_vlm_visual",
        "review_required": True,
        "generator": "test-vlm",
    }
    record["review_id"] = content_fingerprint(record)
    return record


def approve_for_tts(root: Path, record: dict) -> ReviewStore:
    store = ReviewStore(root / "tts_reviews.jsonl")
    image_hash = image_file_sha256(record["image"]) if record.get("image") else None
    fingerprint = content_fingerprint(record, image_sha256=image_hash)
    store.append(fingerprint=fingerprint, decision="approved", uid=record["uid"], collection="counting_v4")
    return store


class ReviewGateTests(unittest.TestCase):
    def test_approval_round_trip_preserves_optional_review_warnings(self):
        marker = {"kind": "counting", "grounding_source": "vlm_visual", "question_turn_index": 4, "answer_turn_index": 5}
        cases = (
            ("legacy", {}),
            ("unmarked", {"auxiliary_task": {"kind": "none", "grounding_source": "none"}, "review_warnings": []}),
            ("marked", {"auxiliary_task": marker, "review_warnings": [{"turn": 6, "code": "spoken_form"}]}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12)).save(image)
            store = ReviewStore(root / "reviews.jsonl")
            voices, config = fake_tts_inputs()
            out = root / "speech"
            for uid, fields in cases:
                with self.subTest(uid=uid):
                    record = reviewed_record(image, uid=uid)
                    record.update(fields)
                    review_id = content_fingerprint(record)
                    record["review_id"] = review_id
                    self.assertEqual(build_one(record, voices, config, out, "cpu", review_store=store), "review_blocked")
                    self.assertFalse((out / uid).exists())
                    store.append(fingerprint=review_id, decision="approved", uid=uid, collection="japanese_daily_v1")
                    self.assertEqual(build_one(record, voices, config, out, "cpu", review_store=store), "ok")
                    self.assertEqual(build_one(record, voices, config, out, "cpu", review_store=store), "skip")
                    meta = json.loads((out / uid / "dialogue.json").read_text())
                    self.assertEqual(meta["review_id"], review_id)
                    if "review_warnings" in fields:
                        self.assertEqual(meta["review_warnings"], fields["review_warnings"])
                    else:
                        self.assertNotIn("review_warnings", meta)
                    decisions = store.latest_decisions()
                    image_hash = image_file_sha256(out / uid / "image.jpg")
                    ok, reason, evidence = _review_gate(meta, decisions, {uid}, image_hash)
                    self.assertTrue(ok, reason)
                    self.assertEqual(evidence["review_id"], review_id)
                    if "review_warnings" in meta:
                        del meta["review_warnings"]
                    else:
                        meta["review_warnings"] = []
                    self.assertEqual(_review_gate(meta, decisions, {uid}, image_hash)[1], "review_stale")

    def test_cli_blocks_required_before_loading_voice_pool(self):
        from build_speech_dataset import main as tts_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            source = root / "input.jsonl"
            source.write_text(json.dumps(record) + "\n")
            args = ["build_speech_dataset.py", "--dialogues", str(source),
                    "--outdir", str(root / "speech"), "--device", "cpu"]
            with mock.patch("build_speech_dataset.ROOT", root), mock.patch.object(sys, "argv", args), \
                    mock.patch("build_speech_dataset.load_pool") as pool, redirect_stdout(io.StringIO()):
                tts_main()
                pool.assert_not_called()
            self.assertFalse((root / "speech").exists())

    def test_required_unapproved_or_rejected_never_creates_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            voices, config = fake_tts_inputs()
            store = ReviewStore(root / "reviews.jsonl")
            for status in (None, "pending", "rejected"):
                if status:
                    store.append(fingerprint=record["review_id"], decision=status, uid=record["uid"], collection="counting_v4")
                self.assertEqual(build_one(record, voices, config, root / "speech", "cpu", review_store=store), "review_blocked")
                self.assertFalse((root / "speech").exists())
            store.append(fingerprint=record["review_id"], decision="approved", uid=record["uid"], collection="counting_v4")
            self.assertEqual(build_one(record, voices, config, root / "speech", "cpu", review_store=store), "ok")
            Image.new("RGB", (16, 12), (20, 20, 180)).save(image)
            self.assertEqual(build_one(record, voices, config, root / "speech-new", "cpu", review_store=store), "review_blocked")
            self.assertFalse((root / "speech-new").exists())

    def test_optional_revision_stays_optional_and_old_legacy_audio_is_excluded(self):
        from dataset_keys import speech_sample_uid
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            record.update(review_required=False, verification_status=None, auxiliary_task=None)
            record.pop("task_mode")  # Older normal records need exact missing-field fingerprints too.
            record.pop("review_id")
            base = content_fingerprint(record)
            voices, config = fake_tts_inputs()
            store = ReviewStore(root / "reviews.jsonl")
            self.assertEqual(build_one(record, voices, config, root / "speech", "cpu", review_store=store), "ok")
            revised = json.loads(json.dumps(record))
            revised["dialogue"][-1]["text"] = "リンゴは3個あります。"
            new_id = content_fingerprint(revised)
            revised["human_revision"] = {"review_id": new_id, "base_fingerprint": base}
            store.append(fingerprint=new_id, decision="approved", uid=record["uid"], collection="counting_v4",
                         revised_record=revised, base_fingerprint=base, parent_fingerprint=base)
            self.assertEqual(build_one(revised, voices, config, root / "speech", "cpu", review_store=store), "ok")
            path = root / "speech" / speech_sample_uid(revised)
            meta = json.loads((path / "dialogue.json").read_text())
            self.assertFalse(meta["review_required"])
            allowed, reason, evidence = _review_gate(meta, store.latest_decisions(), {record["uid"]},
                image_file_sha256(path / "image.jpg"), store.latest_revisions())
            self.assertTrue(allowed, reason)
            self.assertEqual(evidence["status"], "approved")
            legacy = {"uid": record["uid"], "source_uid": record["uid"], "task_mode": "normal"}
            self.assertEqual(_review_gate(legacy, store.latest_decisions(), {record["uid"]}, None, store.latest_revisions())[1], "review_superseded")

    def test_human_revision_speech_is_separate_and_old_audio_is_superseded(self):
        from dataset_keys import speech_sample_uid
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "apples.jpg"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            original = reviewed_record(image)
            store = ReviewStore(root / "reviews.jsonl")
            base = content_fingerprint(original)
            store.append(fingerprint=base, decision="approved", uid=original["uid"], collection="counting_v4")
            voices, config = fake_tts_inputs()
            out = root / "speech"
            self.assertEqual(build_one(original, voices, config, out, "cpu", review_store=store), "ok")
            original_wav = (out / speech_sample_uid(original) / "stereo_dialogue.wav").read_bytes()
            revised = json.loads(json.dumps(original))
            revised["dialogue"][-1]["text"] = "リンゴは3個あります。"
            revised["auxiliary_task"]["answer_count"] = 3
            new_id = content_fingerprint(revised)
            revised["human_revision"] = {"base_fingerprint": base, "parent_fingerprint": base, "review_id": new_id}
            store.append(fingerprint=new_id, decision="approved", uid=original["uid"], collection="counting_v4",
                         revised_record=revised, base_fingerprint=base, parent_fingerprint=base)
            effective = store.resolve_record(original, image_sha256=image_file_sha256(image))
            self.assertEqual(build_one(effective, voices, config, out, "cpu", review_store=store), "ok")
            self.assertNotEqual(speech_sample_uid(original), speech_sample_uid(effective))
            self.assertEqual((out / speech_sample_uid(original) / "stereo_dialogue.wav").read_bytes(), original_wav)
            for record, expected_reason in ((original, "review_superseded"), (effective, None)):
                path = out / speech_sample_uid(record)
                meta = json.loads((path / "dialogue.json").read_text())
                eligible, reason, _ = _review_gate(meta, store.latest_decisions(), {original["uid"]},
                    image_file_sha256(path / "image.jpg"), store.latest_revisions())
                self.assertEqual(reason, expected_reason)
                self.assertEqual(eligible, expected_reason is None)

    def test_generator_to_tts_fingerprint_survives_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "./apples.jpg"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            entry, reason = prepare_task_entry(
                {"candidate_uid": "source-integrated", "path": str(image)},
                "normal",
                "vlm_adaptive",
            )
            self.assertIsNone(reason)
            auxiliary = {
                "kind": "counting",
                "grounding_source": "vlm_visual",
                "question_turn_index": 4,
                "answer_turn_index": 5,
                "counted_target": "リンゴ",
                "answer_count": 2,
            }
            dialogue = [
                {"speaker": "ユーザー", "text": "どんな写真ですか？"},
                {"speaker": "アシスタント", "text": "テーブルの写真です。"},
                {"speaker": "ユーザー", "text": "何が置かれていますか？"},
                {"speaker": "アシスタント", "text": "赤いリンゴが置かれています。"},
                {"speaker": "ユーザー", "text": "そのリンゴは何個ありますか？"},
                {"speaker": "アシスタント", "text": "リンゴは2個あります。"},
                {"speaker": "ユーザー", "text": "どんな色ですか？"},
                {"speaker": "アシスタント", "text": "鮮やかな赤色です。"},
            ]
            answer = json.dumps(
                {"auxiliary_task": auxiliary, "dialogue": dialogue},
                ensure_ascii=False,
            )
            output = io.StringIO()
            self.assertEqual(
                1, write_result(output, entry, answer, "test-vlm", "1/1")
            )
            record = json.loads(output.getvalue())

            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )
            meta = json.loads(
                (speech_root / record["uid"] / "dialogue.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["image"], meta["image"])
            self.assertEqual(record["review_id"], meta["review_id"])
            self.assertEqual(record["image_sha256"], meta["image_sha256"])
            self.assertEqual(
                image_file_sha256(speech_root / record["uid"] / "image.jpg"),
                meta["image_asset_sha256"],
            )
            event = {
                "uid": record["uid"],
                "review_id": record["review_id"],
                "decision": "approved",
            }
            allowed, gate_reason, evidence = _review_gate(
                meta,
                {record["review_id"]: event},
                {record["uid"]},
                image_file_sha256(speech_root / record["uid"] / "image.jpg"),
            )
            self.assertTrue(allowed, gate_reason)
            self.assertEqual("approved", evidence["status"])

    def test_canonical_sample_does_not_require_a_ledger(self) -> None:
        allowed, reason, evidence = _review_gate(
            {"uid": "canonical", "verification_status": "verified_canonical"},
            {},
            set(),
        )
        self.assertTrue(allowed)
        self.assertIsNone(reason)
        self.assertIsNone(evidence)

    def test_exact_latest_approval_is_the_only_accepted_decision(self) -> None:
        meta, review_id = reviewed_meta()
        approved = {
            "uid": "source-1",
            "review_id": review_id,
            "decision": "approved",
            "reviewed_at": "2026-09-28T00:00:00+00:00",
        }
        allowed, reason, evidence = _review_gate(
            meta, {review_id: approved}, {"source-1"}, TEST_IMAGE_ASSET_SHA256
        )
        self.assertTrue(allowed)
        self.assertIsNone(reason)
        self.assertEqual("approved", evidence["status"])
        self.assertEqual(review_id, evidence["review_id"])

        rejected = dict(approved, decision="rejected")
        allowed, reason, _ = _review_gate(
            meta, {review_id: rejected}, {"source-1"}, TEST_IMAGE_ASSET_SHA256
        )
        self.assertFalse(allowed)
        self.assertEqual("review_rejected", reason)

        pending = dict(approved, decision="pending")
        allowed, reason, _ = _review_gate(
            meta, {review_id: pending}, {"source-1"}, TEST_IMAGE_ASSET_SHA256
        )
        self.assertFalse(allowed)
        self.assertEqual("review_pending", reason)

    def test_regeneration_invalidates_an_old_approval(self) -> None:
        meta, old_review_id = reviewed_meta()
        meta["segments"][5]["text"] = "リンゴは3個あります。"
        allowed, reason, evidence = _review_gate(
            meta,
            {
                old_review_id: {
                    "uid": "source-1",
                    "review_id": old_review_id,
                    "decision": "approved",
                }
            },
            {"source-1"},
            TEST_IMAGE_ASSET_SHA256,
        )
        self.assertFalse(allowed)
        self.assertEqual("review_stale", reason)
        self.assertIsNone(evidence)

    def test_unreviewed_new_content_is_pending(self) -> None:
        meta, _ = reviewed_meta()
        allowed, reason, _ = _review_gate(
            meta, {}, set(), TEST_IMAGE_ASSET_SHA256
        )
        self.assertFalse(allowed)
        self.assertEqual("review_pending", reason)

    def test_source_image_replacement_gets_a_new_pending_review_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            old_image_sha256 = record["image_sha256"]
            old_review_id = record["review_id"]

            Image.new("RGB", (16, 12), (20, 20, 180)).save(image)
            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )

            meta = json.loads(
                (speech_root / record["uid"] / "dialogue.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotEqual(old_image_sha256, meta["image_sha256"])
            self.assertEqual(image_file_sha256(image), meta["image_sha256"])
            self.assertNotEqual(old_review_id, meta["review_id"])
            copied = speech_root / record["uid"] / "image.jpg"
            with Image.open(copied) as copied_image:
                red, _, blue = copied_image.convert("RGB").getpixel((0, 0))
            self.assertGreater(blue, red)

            old_approval = {
                "uid": record["uid"],
                "review_id": old_review_id,
                "decision": "approved",
            }
            allowed, reason, evidence = _review_gate(
                meta,
                {old_review_id: old_approval},
                {record["uid"]},
                image_file_sha256(copied),
            )
            self.assertFalse(allowed)
            self.assertEqual("review_stale", reason)
            self.assertIsNone(evidence)

    def test_existing_image_only_change_preserves_wav_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            old_review_id = record["review_id"]
            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )
            sample = speech_root / record["uid"]
            wav_before = (sample / "stereo_dialogue.wav").read_bytes()
            alignment = sample / "assistant_dialogue.refined.json"
            alignment.write_bytes(b"existing alignment must survive")

            Image.new("RGB", (16, 12), (20, 180, 20)).save(image)
            self.assertEqual(
                "review_updated",
                build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record)),
            )
            self.assertEqual(wav_before, (sample / "stereo_dialogue.wav").read_bytes())
            self.assertEqual(b"existing alignment must survive", alignment.read_bytes())
            meta = json.loads((sample / "dialogue.json").read_text(encoding="utf-8"))
            self.assertEqual(image_file_sha256(image), meta["image_sha256"])
            self.assertNotEqual(old_review_id, meta["review_id"])
            self.assertEqual(
                "skip", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )

    def test_existing_dialogue_change_stops_without_touching_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )
            sample = speech_root / record["uid"]
            alignment = sample / "assistant_dialogue.refined.json"
            alignment.write_bytes(b"existing alignment must survive")
            before = {
                name: (sample / name).read_bytes()
                for name in (
                    "stereo_dialogue.wav",
                    "dialogue.json",
                    "image.jpg",
                    "assistant_dialogue.refined.json",
                )
            }

            changed = json.loads(json.dumps(record, ensure_ascii=False))
            changed["dialogue"][-1]["text"] = "リンゴは3個あります。"
            with self.assertRaisesRegex(
                StaleSpeechMetadataError, "different dialogue"
            ):
                build_one(changed, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, changed))
            for name, payload in before.items():
                self.assertEqual(payload, (sample / name).read_bytes(), name)

    def test_unreadable_review_image_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "broken.jpg"
            image.write_bytes(b"not an image")
            meta, _ = reviewed_meta()
            record = {
                "uid": "broken",
                "image": str(image),
                "image_sha256": "a" * 64,
                "task_mode": "normal",
                "dialogue": [
                    {
                        "speaker": "ユーザー"
                        if segment["speaker"] == "USER"
                        else "アシスタント",
                        "text": segment["text"],
                    }
                    for segment in meta["segments"]
                ],
                "auxiliary_task": meta["auxiliary_task"],
                "verification_status": "unverified_vlm_visual",
                "review_required": True,
                "generator": "test-vlm",
            }
            voices, config = fake_tts_inputs()
            with self.assertRaisesRegex(ReviewAssetError, "cannot decode"):
                build_one(record, voices, config, root / "speech", "cpu")
            self.assertFalse((root / "speech" / "broken").exists())

    def test_malformed_vlm_visual_auxiliary_still_sets_review_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image, uid="malformed")
            record.pop("review_required")
            record.pop("verification_status")
            record["auxiliary_task"]["kind"] = "unexpected"
            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )
            meta = json.loads(
                (speech_root / "malformed" / "dialogue.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIs(meta["review_required"], True)
            self.assertRegex(meta["review_id"], r"^[0-9a-f]{64}$")

    def test_missing_or_changed_copied_image_fails_closed(self) -> None:
        meta, review_id = reviewed_meta()
        approved = {
            "uid": "source-1",
            "review_id": review_id,
            "decision": "approved",
        }
        allowed, reason, _ = _review_gate(
            meta, {review_id: approved}, {"source-1"}
        )
        self.assertFalse(allowed)
        self.assertEqual("review_asset_missing", reason)

        allowed, reason, _ = _review_gate(
            meta, {review_id: approved}, {"source-1"}, "c" * 64
        )
        self.assertFalse(allowed)
        self.assertEqual("review_asset_stale", reason)

    def test_train_jsonl_preserves_and_verifies_copied_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (16, 12), (180, 20, 20)).save(image)
            record = reviewed_record(image)
            voices, config = fake_tts_inputs()
            speech_root = root / "speech"
            self.assertEqual(
                "ok", build_one(record, voices, config, speech_root, "cpu", review_store=approve_for_tts(root, record))
            )
            sample = speech_root / record["uid"]
            (sample / "assistant_dialogue.refined.json").write_text(
                "{}", encoding="utf-8"
            )
            meta = json.loads((sample / "dialogue.json").read_text(encoding="utf-8"))
            ledger = root / "review.jsonl"
            ReviewStore(ledger).append(
                fingerprint=meta["review_id"],
                decision="approved",
                uid=record["uid"],
                collection="counting_v4",
            )
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            argv = [
                "make_train_jsonl.py",
                "--datadir",
                str(speech_root),
                "--out_train",
                str(train),
                "--out_eval",
                str(eval_path),
                "--eval_ratio",
                "0",
                "--review-ledger",
                str(ledger),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                make_train_main()
            rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, len(rows))
            self.assertEqual(meta["image_asset_sha256"], rows[0]["image_asset_sha256"])
            self.assertEqual("approved", rows[0]["human_review"]["status"])

            # An in-place copied-image change invalidates the export even though
            # the dialogue review decision itself remains approved.
            Image.new("RGB", (16, 12), (20, 20, 180)).save(sample / "image.jpg")
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                make_train_main()
            self.assertEqual("", train.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
