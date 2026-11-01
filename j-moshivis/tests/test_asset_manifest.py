from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from PIL import Image

from jmoshivis.datasets import asset_manifest as manifest_module
from jmoshivis.datasets.asset_manifest import (
    AssetManifestError,
    build_asset_manifest,
    verify_asset_manifest,
)


class AssetManifestTests(unittest.TestCase):
    def _make_sample(
        self,
        root: Path,
        uid: str,
        *,
        channels: int = 2,
        frames: int = 2_400,
        duration: float | None = None,
    ) -> dict[str, object]:
        directory = root / uid
        directory.mkdir(parents=True)
        audio = directory / "stereo_dialogue.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(b"\x00\x00" * channels * frames)
        image = directory / "image.jpg"
        Image.new("RGB", (4, 3), (20, 40, 60)).save(image, format="JPEG")
        alignment = {
            "format": "test",
            "token_alignments": [
                [[1, 2], [0.0, 0.05], "USER"],
                [[3, 4], [0.05, 0.1], "ASSISTANT"],
            ],
        }
        (directory / "assistant_dialogue.refined.json").write_text(
            json.dumps(alignment), encoding="utf-8"
        )
        return {
            "uid": uid,
            "path": str(audio.resolve()),
            "image": str(image.resolve()),
            "duration": frames / 24_000 if duration is None else duration,
            "text": "test",
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _dataset(self, root: Path) -> tuple[Path, Path]:
        train = root / "train.jsonl"
        eval_path = root / "eval.jsonl"
        self._write_jsonl(train, [self._make_sample(root, "train-uid")])
        self._write_jsonl(eval_path, [self._make_sample(root, "eval-uid")])
        return train, eval_path

    def test_generation_is_deterministic_and_both_modes_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            sources = {"eval": eval_path, "train": train}
            first = root / "first.manifest.jsonl"
            second = root / "second.manifest.jsonl"

            result = build_asset_manifest(sources, first, workers=2)
            build_asset_manifest(sources, second, workers=1)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(result["sample_count"], 2)
            header, train_record, eval_record = [
                json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(header["asset_count"], 6)
            self.assertEqual(train_record["split"], "train")
            self.assertEqual(eval_record["split"], "eval")
            self.assertEqual(
                train_record["dialogue_roles"], ["ASSISTANT", "USER"]
            )
            self.assertEqual(
                [asset["role"] for asset in train_record["assets"]],
                ["audio", "image", "alignment"],
            )
            self.assertEqual(train_record["assets"][0]["channels"], 2)
            self.assertEqual(train_record["assets"][0]["sample_rate"], 24_000)
            self.assertEqual(train_record["assets"][0]["frames"], 2_400)
            self.assertEqual(train_record["assets"][0]["duration_seconds"], 0.1)
            self.assertEqual(
                verify_asset_manifest(first, sources, mode="fast")["status"],
                "verified",
            )
            self.assertEqual(
                verify_asset_manifest(first, sources, mode="full")[
                    "verification_mode"
                ],
                "full",
            )

    def test_generation_and_verification_report_flushed_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            sources = {"train": train, "eval": eval_path}
            manifest = root / "assets.manifest.jsonl"

            with (
                mock.patch.object(
                    manifest_module, "PROGRESS_INTERVAL_SAMPLES", 1
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                build_asset_manifest(sources, manifest, workers=1)

            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[asset-manifest] generate: 0/2 samples (0.0%)",
                        flush=True,
                    ),
                    mock.call(
                        "[asset-manifest] generate: 1/2 samples (50.0%)",
                        flush=True,
                    ),
                    mock.call(
                        "[asset-manifest] generate: 2/2 samples (100.0%)",
                        flush=True,
                    ),
                ]
            )

            with (
                mock.patch.object(
                    manifest_module, "PROGRESS_INTERVAL_SAMPLES", 1
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                verify_asset_manifest(manifest, sources, mode="full")

            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[asset-manifest] verify (full): 0/2 samples (0.0%)",
                        flush=True,
                    ),
                    mock.call(
                        "[asset-manifest] verify (full): 1/2 samples (50.0%)",
                        flush=True,
                    ),
                    mock.call(
                        "[asset-manifest] verify (full): 2/2 samples (100.0%)",
                        flush=True,
                    ),
                ]
            )

    def test_full_mode_detects_same_stat_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            sources = {"train": train, "eval": eval_path}
            manifest = root / "assets.manifest.jsonl"
            build_asset_manifest(sources, manifest, workers=2)

            record = json.loads(
                manifest.read_text(encoding="utf-8").splitlines()[1]
            )
            audio_record = record["assets"][0]
            audio = Path(audio_record["path"])
            payload = bytearray(audio.read_bytes())
            payload[-1] ^= 1
            audio.write_bytes(payload)
            os.utime(
                audio,
                ns=(audio_record["mtime_ns"], audio_record["mtime_ns"]),
            )

            # Fast startup checks are intentionally stat-based.
            verify_asset_manifest(manifest, sources, mode="fast")
            with self.assertRaisesRegex(AssetManifestError, "sha256"):
                verify_asset_manifest(manifest, sources, mode="full")

    def test_missing_asset_and_duplicate_uid_fail_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            missing = root / "train-uid" / "image.jpg"
            missing.unlink()
            output = root / "assets.manifest.jsonl"
            with self.assertRaisesRegex(AssetManifestError, "missing image"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path}, output, workers=2
                )
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._make_sample(root, "same-uid")
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(train, [row])
            self._write_jsonl(eval_path, [row])
            with self.assertRaisesRegex(AssetManifestError, "duplicate sample uid"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "assets.manifest.jsonl",
                    workers=1,
                )

    def test_conflicting_declared_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._make_sample(root, "uid")
            row["image"] = str(root / "wrong.jpg")
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(train, [row])
            self._write_jsonl(eval_path, [])
            with self.assertRaisesRegex(AssetManifestError, "conflicts"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "assets.manifest.jsonl",
                    workers=1,
                )

    def test_unverified_visual_sample_requires_exact_approval_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._make_sample(root, "reviewed-uid")
            row.update(
                {
                    "verification_status": "unverified_vlm_visual",
                    "auxiliary_task": {
                        "kind": "counting",
                        "grounding_source": "vlm_visual",
                    },
                }
            )
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(eval_path, [])

            self._write_jsonl(train, [row])
            with self.assertRaisesRegex(AssetManifestError, "review_required"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "missing-marker.manifest.jsonl",
                    workers=1,
                )

            review_id = "a" * 64
            row.update({"review_required": True, "review_id": review_id})
            self._write_jsonl(train, [row])
            with self.assertRaisesRegex(AssetManifestError, "human_review"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "missing-evidence.manifest.jsonl",
                    workers=1,
                )

            row["human_review"] = {
                "review_id": review_id,
                "status": "approved",
            }
            self._write_jsonl(train, [row])
            with self.assertRaisesRegex(
                AssetManifestError, "image_asset_sha256"
            ):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "missing-image-hash.manifest.jsonl",
                    workers=1,
                )

            row["image_asset_sha256"] = "b" * 64
            self._write_jsonl(train, [row])
            with self.assertRaisesRegex(AssetManifestError, "copied image hash"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "wrong-image-hash.manifest.jsonl",
                    workers=1,
                )

            copied_image = Path(str(row["path"])).parent / "image.jpg"
            row["image_asset_sha256"] = hashlib.sha256(
                copied_image.read_bytes()
            ).hexdigest()
            self._write_jsonl(train, [row])
            approved_manifest = root / "approved.manifest.jsonl"
            result = build_asset_manifest(
                {"train": train, "eval": eval_path},
                approved_manifest,
                workers=1,
            )
            self.assertEqual(1, result["sample_count"])

            image_record = json.loads(
                approved_manifest.read_text(encoding="utf-8").splitlines()[1]
            )["assets"][1]
            image_path = Path(image_record["path"])
            payload = bytearray(image_path.read_bytes())
            payload[-1] ^= 1
            image_path.write_bytes(payload)
            os.utime(
                image_path,
                ns=(image_record["mtime_ns"], image_record["mtime_ns"]),
            )
            with self.assertRaisesRegex(
                AssetManifestError, "review copied image sha256"
            ):
                verify_asset_manifest(
                    approved_manifest,
                    {"train": train, "eval": eval_path},
                    mode="fast",
                )

    def test_any_vlm_visual_grounding_requires_review_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._make_sample(root, "malformed-review-uid")
            row["auxiliary_task"] = {
                "kind": "unexpected",
                "grounding_source": "vlm_visual",
            }
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(train, [row])
            self._write_jsonl(eval_path, [])

            with self.assertRaisesRegex(AssetManifestError, "review_required"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path},
                    root / "malformed-review.manifest.jsonl",
                    workers=1,
                )

    def test_non_stereo_audio_is_rejected_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(
                train, [self._make_sample(root, "mono-uid", channels=1)]
            )
            self._write_jsonl(eval_path, [])
            output = root / "assets.manifest.jsonl"

            with self.assertRaisesRegex(AssetManifestError, "must be stereo"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path}, output, workers=1
                )
            self.assertFalse(output.exists())

    def test_jsonl_wav_duration_mismatch_is_rejected_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(
                train,
                [self._make_sample(root, "duration-uid", duration=0.2)],
            )
            self._write_jsonl(eval_path, [])
            output = root / "assets.manifest.jsonl"

            with self.assertRaisesRegex(AssetManifestError, "duration mismatch"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path}, output, workers=1
                )
            self.assertFalse(output.exists())

    def test_fast_mode_validates_recorded_audio_header_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            sources = {"train": train, "eval": eval_path}
            manifest = root / "assets.manifest.jsonl"
            build_asset_manifest(sources, manifest, workers=1)
            lines = manifest.read_text(encoding="utf-8").splitlines()
            sample = json.loads(lines[1])
            sample["assets"][0]["channels"] = 1
            lines[1] = json.dumps(
                sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(AssetManifestError, "audio channels"):
                verify_asset_manifest(manifest, sources, mode="fast")

    def test_malformed_alignment_is_rejected_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._make_sample(root, "bad-alignment")
            alignment_path = (
                Path(str(row["path"])).parent
                / "assistant_dialogue.refined.json"
            )
            alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
            alignment["token_alignments"][1][1] = [0.08, 0.04]
            alignment_path.write_text(json.dumps(alignment), encoding="utf-8")
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            self._write_jsonl(train, [row])
            self._write_jsonl(eval_path, [])
            output = root / "assets.manifest.jsonl"

            with self.assertRaisesRegex(AssetManifestError, "reversed"):
                build_asset_manifest(
                    {"train": train, "eval": eval_path}, output, workers=1
                )
            self.assertFalse(output.exists())

    def test_source_mutation_during_build_prevents_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            output = root / "assets.manifest.jsonl"
            original_iterator = manifest_module._iter_completed_bounded

            def mutate_after_first_record(jobs, *, workers):
                for index, record in enumerate(
                    original_iterator(jobs, workers=workers)
                ):
                    yield record
                    if index == 0:
                        row = json.loads(train.read_text(encoding="utf-8"))
                        row["text"] = "changed with the same record count"
                        self._write_jsonl(train, [row])

            with mock.patch.object(
                manifest_module,
                "_iter_completed_bounded",
                side_effect=mutate_after_first_record,
            ):
                with self.assertRaisesRegex(AssetManifestError, "source JSONL changed"):
                    build_asset_manifest(
                        {"train": train, "eval": eval_path}, output, workers=1
                    )
            self.assertFalse(output.exists())

    def test_fast_mode_rejects_malformed_recorded_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            sources = {"train": train, "eval": eval_path}
            manifest = root / "assets.manifest.jsonl"
            build_asset_manifest(sources, manifest, workers=1)
            lines = manifest.read_text(encoding="utf-8").splitlines()
            sample = json.loads(lines[1])
            sample["assets"][0]["sha256"] = "not-a-sha"
            lines[1] = json.dumps(
                sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(AssetManifestError, "invalid audio sha256"):
                verify_asset_manifest(manifest, sources, mode="fast")

    def test_asset_mutation_during_hash_prevents_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, eval_path = self._dataset(root)
            output = root / "assets.manifest.jsonl"
            target = root / "train-uid" / "stereo_dialogue.wav"
            original_hash = manifest_module._sha256_file
            mutated = False

            def hash_then_mutate(path):
                nonlocal mutated
                digest = original_hash(path)
                if Path(path) == target and not mutated:
                    with target.open("ab") as handle:
                        handle.write(b"\x00")
                    mutated = True
                return digest

            with mock.patch.object(
                manifest_module, "_sha256_file", side_effect=hash_then_mutate
            ):
                with self.assertRaisesRegex(AssetManifestError, "changed while hashing"):
                    build_asset_manifest(
                        {"train": train, "eval": eval_path}, output, workers=1
                    )
            self.assertTrue(mutated)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
