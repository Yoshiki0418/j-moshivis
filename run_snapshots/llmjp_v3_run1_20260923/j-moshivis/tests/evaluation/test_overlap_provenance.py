from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from jmoshivis.evaluation.overlap import build_index as build_index_module
from jmoshivis.evaluation.overlap.audit import main as audit_main
from jmoshivis.evaluation.overlap.build_index import (
    image_index_metadata_path,
    main as build_index_main,
)
from jmoshivis.evaluation.schema import EvalExample, write_manifest


class OverlapProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        train_image = root / "train.png"
        eval_image = root / "eval.png"
        Image.new("RGB", (8, 8), "red").save(train_image)
        Image.new("RGB", (8, 8), "blue").save(eval_image)

        train_manifest = root / "train.jsonl"
        train_manifest.write_text(
            json.dumps({"uid": "train-1", "image": train_image.name}) + "\n",
            encoding="utf-8",
        )
        eval_manifest = root / "eval.jsonl"
        write_manifest(
            eval_manifest,
            [
                EvalExample(
                    sample_id="eval-1",
                    benchmark="fixture",
                    split="test",
                    domain="general_world",
                    task="vqa",
                    image_path=eval_image.name,
                    question="何色ですか？",
                    answers=["青"],
                )
            ],
        )
        index = root / "index.jsonl"
        self._run_build(train_manifest, index)
        return train_manifest, eval_manifest, index

    def _run_build(self, train_manifest: Path, index: Path, *extra: str) -> int:
        argv = [
            "build_index",
            "--train-manifest",
            str(train_manifest),
            "--output",
            str(index),
            "--workers",
            "1",
            *extra,
        ]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            return build_index_main()

    def _run_audit(self, train_manifest: Path, eval_manifest: Path, index: Path) -> int:
        argv = [
            "audit",
            "--manifest",
            str(eval_manifest),
            "--train-manifest",
            str(train_manifest),
            "--image-index",
            str(index),
            "--clean-manifest",
            str(eval_manifest.parent / "clean.jsonl"),
            "--report",
            str(eval_manifest.parent / "report.jsonl"),
            "--dry-run",
        ]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            return audit_main()

    def test_audit_accepts_current_and_legacy_v1_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_manifest, eval_manifest, index = self._fixture(Path(directory))
            metadata_path = image_index_metadata_path(index)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input_manifests"][0]["record_count"], 1)
            self.assertEqual(self._run_audit(train_manifest, eval_manifest, index), 0)

            del metadata["input_manifests"][0]["record_count"]
            metadata.pop("state")
            metadata.pop("updated_unix")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(self._run_audit(train_manifest, eval_manifest, index), 0)

    def test_audit_rejects_missing_or_mismatched_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_manifest, eval_manifest, index = self._fixture(Path(directory))
            metadata_path = image_index_metadata_path(index)
            index_bytes = index.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            metadata_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "metadata sidecar is missing"):
                self._run_audit(train_manifest, eval_manifest, index)

            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            index.write_bytes(index_bytes + b"\n")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self._run_audit(train_manifest, eval_manifest, index)
            index.write_bytes(index_bytes)

            cases = [
                (
                    "fingerprint settings",
                    lambda value: value["fingerprint"].__setitem__("dhash", "changed"),
                    "fingerprint index provenance",
                ),
                (
                    "manifest path",
                    lambda value: value["input_manifests"][0].__setitem__(
                        "path", str(Path(directory) / "other.jsonl")
                    ),
                    "training manifest path",
                ),
                (
                    "manifest hash",
                    lambda value: value["input_manifests"][0].__setitem__(
                        "sha256", "0" * 64
                    ),
                    "training manifest path",
                ),
                (
                    "manifest record count",
                    lambda value: value["input_manifests"][0].__setitem__(
                        "record_count", 2
                    ),
                    "training manifest path",
                ),
                (
                    "unique UID count",
                    lambda value: value.__setitem__("num_expected_uids", 2),
                    "unique image UID count",
                ),
                (
                    "successful UID count",
                    lambda value: value.__setitem__("num_successful_uids", 2),
                    "successful/error UID counts",
                ),
                (
                    "error UID count",
                    lambda value: value.__setitem__("num_error_uids", 1),
                    "successful/error UID counts",
                ),
                (
                    "unfinished transaction field",
                    lambda value: value.__setitem__("pending_append", None),
                    "unfinished build state",
                ),
            ]
            for label, mutate, error_pattern in cases:
                with self.subTest(label=label):
                    changed = copy.deepcopy(metadata)
                    mutate(changed)
                    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, error_pattern):
                        self._run_audit(train_manifest, eval_manifest, index)

    def test_resume_rejects_untrusted_sidecar_without_modifying_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train_manifest, _, index = self._fixture(Path(directory))
            metadata_path = image_index_metadata_path(index)
            index_bytes = index.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            metadata_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "required file is missing"):
                self._run_build(train_manifest, index, "--resume")
            self.assertEqual(index.read_bytes(), index_bytes)

            metadata["fingerprint"]["phash"] = "changed"
            changed_metadata = json.dumps(metadata).encode("utf-8")
            metadata_path.write_bytes(changed_metadata)
            with self.assertRaisesRegex(RuntimeError, "fingerprint index provenance"):
                self._run_build(train_manifest, index, "--resume")
            self.assertEqual(index.read_bytes(), index_bytes)
            self.assertEqual(metadata_path.read_bytes(), changed_metadata)

    def test_resume_recovers_interrupted_first_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "train.png"
            Image.new("RGB", (8, 8), "red").save(image)
            train_manifest = root / "train.jsonl"
            train_manifest.write_text(
                json.dumps({"uid": "train-1", "image": image.name}) + "\n",
                encoding="utf-8",
            )
            index = root / "index.jsonl"

            original_write = build_index_module._write_image_index_metadata
            write_count = 0

            def interrupt_before_commit(
                index_path: Path, metadata: dict[str, object]
            ) -> None:
                nonlocal write_count
                write_count += 1
                if write_count == 4:
                    raise RuntimeError("simulated process interruption")
                original_write(index_path, metadata)

            with patch.object(
                build_index_module,
                "_write_image_index_metadata",
                side_effect=interrupt_before_commit,
            ), self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                self._run_build(train_manifest, index)

            metadata_path = image_index_metadata_path(index)
            interrupted = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(interrupted["state"], "building")
            self.assertIsNotNone(interrupted["pending_append"])
            self.assertEqual(len(index.read_text(encoding="utf-8").splitlines()), 1)

            interrupted_bytes = index.read_bytes()
            index.write_bytes(b"!" + interrupted_bytes[1:])
            with self.assertRaisesRegex(RuntimeError, "pending bytes differ"):
                self._run_build(train_manifest, index, "--resume")
            self.assertEqual(index.read_bytes(), b"!" + interrupted_bytes[1:])
            index.write_bytes(interrupted_bytes)

            self.assertEqual(self._run_build(train_manifest, index, "--resume"), 0)
            completed = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(completed["state"], "complete")
            self.assertEqual(completed["num_successful_uids"], 1)
            self.assertEqual([row["uid"] for row in rows], ["train-1"])


if __name__ == "__main__":
    unittest.main()
