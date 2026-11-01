from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jmoshivis.evaluation.audio_protocol import ASSISTANT_PRELUDE_PROTOCOL
from jmoshivis.evaluation.freeze_moshivis_suite import (
    build_suite_lock,
    write_new_suite_lock,
)
from jmoshivis.evaluation.run_inference import (
    _asset_inventory,
    _verify_frozen_asset_inventory,
    main as run_inference_main,
)
from jmoshivis.evaluation.schema import (
    EvalExample,
    ManifestError,
    read_manifest,
    write_manifest,
)
from jmoshivis.evaluation.suite_moshivis import (
    SUITE_LOCK_SCHEMA_V2,
    suite_allows_paper_input,
    verify_evaluation_suite,
)


def _fixture_manifest(root: Path, *, benchmark: str = "ocr-vqa") -> Path:
    image = root / "image.jpg"
    audio = root / "question.wav"
    image.write_bytes(b"fixture image bytes")
    audio.write_bytes(b"fixture audio bytes")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    audio_sha256 = hashlib.sha256(audio.read_bytes()).hexdigest()
    question = "What is shown?"
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    task = "caption" if benchmark == "coco-captions" else "vqa"
    answers = (
        ["a red car", "a car that is red"]
        if task == "caption"
        else ["Title"]
    )
    examples = [
        EvalExample(
            sample_id="fixture-1",
            benchmark=benchmark,
            split="validation",
            domain="general_world",
            task=task,
            image_path=image.name,
            audio_path=audio.name,
            question=question,
            answers=answers,
            source={"image_sha256": image_sha256},
            metadata={
                "question_audio": {
                    "sample_id": "fixture-1",
                    "prompt_text": question,
                    "prompt_sha256": question_sha256,
                    "wav_sha256": audio_sha256,
                }
            },
        )
    ]
    manifest = root / "manifest.jsonl"
    write_manifest(manifest, examples)
    return manifest


class MoshiVisSuiteV2Tests(unittest.TestCase):
    def test_freeze_and_verify_v2_records_the_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            output = root / "suite.lock.json"

            lock = build_suite_lock(
                manifest,
                output,
                suite_id="babillage_fixture_v2",
                frozen_on="2026-09-30",
            )
            write_new_suite_lock(output, lock)
            contract = verify_evaluation_suite(output, manifest)

            self.assertEqual(lock["schema_version"], SUITE_LOCK_SCHEMA_V2)
            self.assertEqual(lock["scope"]["paper_generation"], "greedy")
            self.assertEqual(lock["scope"]["text_max_new_tokens"], 96)
            self.assertEqual(
                lock["scope"]["paper_input_modalities"], ["audio", "text"]
            )
            self.assertEqual(lock["manifest"]["questions"], 1)
            self.assertEqual(
                lock["manifest"]["frozen_asset_inventories"]["audio"][
                    "num_unique_images"
                ],
                1,
            )
            self.assertEqual(
                lock["manifest"]["frozen_asset_inventories"]["audio"][
                    "num_unique_audio"
                ],
                1,
            )
            self.assertEqual(
                lock["manifest"]["frozen_asset_inventories"]["text"][
                    "num_unique_audio"
                ],
                0,
            )
            self.assertEqual(
                lock["manifest"]["benchmark_composition"]["ocr-vqa"][
                    "questions"
                ],
                1,
            )
            self.assertEqual(contract["frozen_num_questions"], 1)
            self.assertEqual(
                contract["frozen_asset_inventories"],
                lock["manifest"]["frozen_asset_inventories"],
            )
            self.assertEqual(contract["text_max_new_tokens"], 96)
            self.assertEqual(contract["scoring_protocol"], lock["scoring_protocol"])
            examples = read_manifest(manifest)
            self.assertEqual(
                _asset_inventory(examples, manifest, require_audio=True),
                contract["frozen_asset_inventories"]["audio"],
            )
            self.assertEqual(
                _asset_inventory(examples, manifest, require_audio=False),
                contract["frozen_asset_inventories"]["text"],
            )
            self.assertTrue(suite_allows_paper_input(contract, "audio"))
            self.assertTrue(suite_allows_paper_input(contract, "text"))

    def test_atomic_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            output = root / "suite.lock.json"
            lock = build_suite_lock(manifest, output, suite_id="fixture")
            write_new_suite_lock(output, lock)
            original = output.read_bytes()

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_new_suite_lock(output, lock)

            self.assertEqual(output.read_bytes(), original)

    def test_verifier_rejects_count_protocol_and_composition_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            base = build_suite_lock(
                manifest, root / "unused.json", suite_id="fixture"
            )
            mutations = (
                ("count", lambda lock: lock["manifest"].update(questions=2), "count"),
                (
                    "generation",
                    lambda lock: lock["scope"].update(paper_generation="sampling"),
                    "paper_generation",
                ),
                (
                    "composition",
                    lambda lock: lock["manifest"]["benchmark_composition"][
                        "ocr-vqa"
                    ].update(questions=2),
                    "composition",
                ),
                (
                    "text_tokens",
                    lambda lock: lock["scope"].update(text_max_new_tokens=0),
                    "text_max_new_tokens",
                ),
            )
            for name, mutate, message in mutations:
                lock = copy.deepcopy(base)
                mutate(lock)
                path = root / f"{name}.lock.json"
                path.write_text(json.dumps(lock), encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ManifestError, message):
                        verify_evaluation_suite(path, manifest)

    def test_freeze_rejects_cider_without_official_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root, benchmark="coco-captions")
            missing = {
                "implementation_file_sha256": {},
                "dependency_versions": {"pycocoevalcap": None, "java": None},
                "dependency_artifact_sha256": {
                    "stanford-corenlp-3.4.1.jar": None
                },
            }
            with patch(
                "jmoshivis.evaluation.score_moshivis.scoring_provenance",
                return_value=missing,
            ):
                with self.assertRaisesRegex(RuntimeError, "official scoring"):
                    build_suite_lock(
                        manifest, root / "suite.json", suite_id="cider_fixture"
                    )

    def test_freeze_rejects_missing_declared_image_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            example = read_manifest(manifest)[0]
            example.source.pop("image_sha256")
            write_manifest(manifest, [example])

            with self.assertRaisesRegex(ManifestError, "image_sha256"):
                build_suite_lock(manifest, root / "suite.json", suite_id="fixture")

    def test_freeze_rejects_diagnostic_only_assistant_prelude_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            prelude = root / "assistant-prelude.wav"
            prelude.write_bytes(b"diagnostic prelude bytes")

            with self.assertRaisesRegex(
                ValueError, "cannot be frozen as a paper suite.*--diagnostic"
            ):
                build_suite_lock(
                    manifest,
                    root / "suite.json",
                    suite_id="fixture",
                    audio_dialogue_protocol=ASSISTANT_PRELUDE_PROTOCOL,
                    assistant_prelude_wav=prelude,
                )

    def test_verifier_rejects_handcrafted_assistant_prelude_paper_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            lock = build_suite_lock(
                manifest, root / "unused.json", suite_id="fixture"
            )
            lock["scope"]["audio_dialogue_protocol"] = ASSISTANT_PRELUDE_PROTOCOL
            lock["scope"]["assistant_prelude"] = {
                "wav_sha256": "a" * 64,
            }
            lock_path = root / "handcrafted.lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            with self.assertRaisesRegex(
                ManifestError, "paper audio protocol must be bare_question_v1"
            ):
                verify_evaluation_suite(lock_path, manifest)

    def test_assistant_prelude_is_allowed_only_for_diagnostic_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            lock_path = root / "suite.lock.json"
            write_new_suite_lock(
                lock_path,
                build_suite_lock(manifest, lock_path, suite_id="fixture"),
            )
            prelude = root / "assistant-prelude.wav"
            prelude.write_bytes(b"diagnostic prelude bytes")
            common = [
                "run_inference",
                "--manifest", str(manifest),
                "--suite-lock", str(lock_path),
                "--output", str(root / "predictions.jsonl"),
                "--checkpoint", str(root / "unused-checkpoint.pt"),
                "--input-modality", "audio",
                "--audio-dialogue-protocol", ASSISTANT_PRELUDE_PROTOCOL,
                "--assistant-prelude-wav", str(prelude),
                "--dry-run",
            ]

            with patch("sys.argv", common):
                with self.assertRaisesRegex(ValueError, "use --diagnostic"):
                    run_inference_main()
            with patch("sys.argv", [*common, "--diagnostic"]):
                self.assertEqual(run_inference_main(), 0)

    def test_asset_mutation_is_checked_once_by_inference_not_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            lock_path = root / "suite.lock.json"
            lock = build_suite_lock(manifest, lock_path, suite_id="fixture")
            write_new_suite_lock(lock_path, lock)
            image = root / "image.jpg"
            image.write_bytes(b"mutated image bytes")

            contract = verify_evaluation_suite(lock_path, manifest)
            examples = read_manifest(manifest)
            # Isolate the aggregate lock comparison from the per-row declared
            # checksum check: both protections run during real inference.
            examples[0].source["image_sha256"] = hashlib.sha256(
                image.read_bytes()
            ).hexdigest()
            actual = _asset_inventory(
                examples,
                manifest,
                require_audio=False,
            )
            with self.assertRaisesRegex(ValueError, "frozen suite"):
                _verify_frozen_asset_inventory(contract, "text", actual)

    def test_v2_lock_allows_non_diagnostic_text_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture_manifest(root)
            lock_path = root / "suite.lock.json"
            lock = build_suite_lock(
                manifest,
                lock_path,
                suite_id="text_fixture",
                paper_input_modalities=("audio", "text"),
            )
            write_new_suite_lock(lock_path, lock)
            argv = [
                "run_inference",
                "--manifest", str(manifest),
                "--suite-lock", str(lock_path),
                "--output", str(root / "predictions.jsonl"),
                "--checkpoint", str(root / "unused-checkpoint.pt"),
                "--input-modality", "text",
                "--dry-run",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(run_inference_main(), 0)

            argv.extend(["--max-new-tokens", "95"])
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                    run_inference_main()

    def test_v1_dispatch_preserves_audio_only_contract(self) -> None:
        project = Path(__file__).resolve().parents[2]
        workspace = project.parent
        lock_path = project / "evaluation/suite_v1.lock.json"
        manifest = workspace / "data/evaluation/vision_v1/manifest_audio.jsonl"

        contract = verify_evaluation_suite(lock_path, manifest)

        self.assertEqual(contract["suite_id"], "vision_v1")
        self.assertTrue(suite_allows_paper_input(contract, "audio"))
        self.assertFalse(suite_allows_paper_input(contract, "text"))


if __name__ == "__main__":
    unittest.main()
