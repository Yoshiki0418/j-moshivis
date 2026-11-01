from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jmoshivis.evaluation.schema import EvalExample, ManifestError, write_manifest
from jmoshivis.evaluation.metrics_moshivis import (
    normalize_ocr_vqa_answer,
    normalize_vqav2_answer as normalize_vqa_answer,
    ocr_vqa_exact_match as ocr_vqa_accuracy,
    vqav2_accuracy as vqa_accuracy,
)
from jmoshivis.evaluation.score_moshivis import evaluate_files


def _example(
    sample_id: str,
    benchmark: str,
    task: str,
    answers: list[str],
) -> EvalExample:
    return EvalExample(
        sample_id=sample_id,
        benchmark=benchmark,
        split="validation",
        domain="general_world",
        task=task,
        image_path="unused.jpg",
        question="fixture question",
        answers=answers,
    )


class MoshiVisMetricTests(unittest.TestCase):
    def test_ocr_vqa_is_case_insensitive_exact_match(self) -> None:
        self.assertEqual(
            normalize_ocr_vqa_answer("  Science-Fiction  "),
            "  science-fiction  ",
        )
        self.assertEqual(ocr_vqa_accuracy("RED", ["red", "blue"]), 1.0)
        self.assertEqual(ocr_vqa_accuracy("red.", ["red"]), 0.0)
        self.assertEqual(ocr_vqa_accuracy("the answer is red", ["red"]), 0.0)

    def test_vqav2_uses_official_normalization(self) -> None:
        self.assertEqual(normalize_vqa_answer("The two."), "2")
        self.assertEqual(normalize_vqa_answer("2,000."), "2000")
        self.assertEqual(normalize_vqa_answer("isnt"), "isn't")

    def test_vqav2_consensus_is_leave_one_out(self) -> None:
        # The official score ladder for ten annotations is 0.3/0.6/0.9/1.0,
        # not the simplified 1/3, 2/3, 1 formula.
        for matches, expected in ((1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0)):
            references = ["cat"] * matches + ["dog"] * (10 - matches)
            self.assertAlmostEqual(vqa_accuracy("cat", references), expected)

    def test_scorer_keeps_metrics_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            examples = [
                _example("ocr", "ocr-vqa", "vqa", ["Title"]),
                _example("vqa", "vqav2", "vqa", ["two"] * 3 + ["three"] * 7),
                _example(
                    "caption", "coco-captions", "caption",
                    ["a red car", "a car is red", "the red automobile"],
                ),
            ]
            write_manifest(manifest, examples)
            predictions.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"sample_id": "ocr", "answer_text": "title"},
                        {"sample_id": "vqa", "answer_text": "2"},
                        {"sample_id": "caption", "answer_text": "a red car"},
                    )
                ),
                encoding="utf-8",
            )
            with patch(
                "jmoshivis.evaluation.score_moshivis._compute_cider",
                return_value=(1.25, {"caption": 1.25}),
            ):
                rows, summary = evaluate_files(manifest, predictions)

            self.assertEqual(len(rows), 3)
            self.assertEqual(summary["by_benchmark"]["ocr-vqa"]["paper_score"], 100.0)
            self.assertAlmostEqual(
                summary["by_benchmark"]["vqav2"]["paper_score"], 90.0
            )
            self.assertEqual(
                summary["by_benchmark"]["coco-captions"]["paper_score"], 125.0
            )
            self.assertFalse(summary["ready_for_scoring"])

    def test_vqav2_requires_ten_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(manifest, [_example("vqa", "vqav2", "vqa", ["cat"] * 9)])
            predictions.write_text(
                json.dumps({"sample_id": "vqa", "answer_text": "cat"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "exactly 10"):
                evaluate_files(manifest, predictions)

    def test_cider_requires_multiple_references_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(
                manifest,
                [_example("caption", "coco-captions", "caption", ["a cat"])],
            )
            predictions.write_text(
                json.dumps({"sample_id": "caption", "answer_text": "a cat"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "multiple reference"):
                evaluate_files(manifest, predictions)

    def test_multiple_cider_benchmarks_are_scored_as_separate_corpora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(
                manifest,
                [
                    _example("first", "coco-captions", "caption", ["a cat", "cat"]),
                    _example("second", "cococo", "caption", ["a dog", "dog"]),
                ],
            )
            predictions.write_text(
                "".join(
                    json.dumps({"sample_id": sample_id, "answer_text": answer}) + "\n"
                    for sample_id, answer in (("first", "a cat"), ("second", "a dog"))
                ),
                encoding="utf-8",
            )

            def fake_cider(items: dict[str, str], _: dict[str, list[str]]):
                sample_id = next(iter(items))
                score = 1.0 if sample_id == "first" else 2.0
                return score, {sample_id: score}

            with patch(
                "jmoshivis.evaluation.score_moshivis._compute_cider",
                side_effect=fake_cider,
            ) as scorer:
                _, summary = evaluate_files(manifest, predictions)

            self.assertEqual(scorer.call_count, 2)
            self.assertEqual(summary["by_benchmark"]["coco-captions"]["paper_score"], 100.0)
            self.assertEqual(summary["by_benchmark"]["cococo"]["paper_score"], 200.0)

    def test_paper_ready_requires_matching_suite_run_and_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            suite_lock = root / "suite.lock.json"
            write_manifest(manifest, [_example("ocr", "ocr-vqa", "vqa", ["title"])])
            predictions.write_text(
                json.dumps({"sample_id": "ocr", "answer_text": "title"}) + "\n",
                encoding="utf-8",
            )
            provenance = {
                "implementation_file_sha256": {},
                "dependency_versions": {},
                "dependency_artifact_sha256": {},
            }
            frozen_inventory = {
                "asset_inventory_sha256": "d" * 64,
                "num_unique_images": 1,
                "num_unique_audio": 0,
            }
            contract = {
                "suite_id": "moshivis-fixture-v1",
                "suite_lock_sha256": "a" * 64,
                "frozen_num_questions": 1,
                "paper_generation": "greedy",
                "paper_input_modalities": ["audio", "text"],
                "text_max_new_tokens": 96,
                "frozen_asset_inventories": {"text": frozen_inventory},
                "response_tail_seconds": 10.0,
                "response_overlap_seconds": 0.32,
                "scoring_protocol": provenance,
            }
            run = {
                "_sidecar_path": str(root / "predictions.jsonl.run.json"),
                "_sidecar_sha256": "b" * 64,
                "status": "complete",
                "suite_id": contract["suite_id"],
                "suite_lock_sha256": contract["suite_lock_sha256"],
                "generation": "greedy",
                "input_modality": "text",
                "max_new_tokens": 96,
                "response_seconds": 10.0,
                "response_overlap_seconds": 0.32,
                "diagnostic": False,
                "checkpoint_hash_skipped": False,
                "checkpoint_sha256": "c" * 64,
                "checkpoint": "fixture.safetensors",
                **frozen_inventory,
            }
            with (
                patch(
                    "jmoshivis.evaluation.score_moshivis.verify_evaluation_suite",
                    return_value=contract,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.validate_prediction_run",
                    return_value=run,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.scoring_provenance",
                    return_value=provenance,
                ),
            ):
                _, summary = evaluate_files(
                    manifest,
                    predictions,
                    require_run_metadata=True,
                    suite_lock_path=suite_lock,
                )

            self.assertTrue(summary["ready_for_scoring"])
            self.assertTrue(summary["scoring_protocol_verified"])
            self.assertTrue(summary["text_generation_limit_verified"])
            self.assertTrue(summary["asset_inventory_verified"])
            self.assertTrue(summary["paper_ready"])
            self.assertIn("asset inventory", summary["paper_guard"])

            run["max_new_tokens"] = 95
            with (
                patch(
                    "jmoshivis.evaluation.score_moshivis.verify_evaluation_suite",
                    return_value=contract,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.validate_prediction_run",
                    return_value=run,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.scoring_provenance",
                    return_value=provenance,
                ),
            ):
                _, mismatch = evaluate_files(
                    manifest,
                    predictions,
                    require_run_metadata=True,
                    suite_lock_path=suite_lock,
                )

            self.assertFalse(mismatch["text_generation_limit_verified"])
            self.assertFalse(mismatch["paper_ready"])

            run["max_new_tokens"] = 96
            run["asset_inventory_sha256"] = "e" * 64
            with (
                patch(
                    "jmoshivis.evaluation.score_moshivis.verify_evaluation_suite",
                    return_value=contract,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.validate_prediction_run",
                    return_value=run,
                ),
                patch(
                    "jmoshivis.evaluation.score_moshivis.scoring_provenance",
                    return_value=provenance,
                ),
            ):
                _, asset_mismatch = evaluate_files(
                    manifest,
                    predictions,
                    require_run_metadata=True,
                    suite_lock_path=suite_lock,
                )

            self.assertFalse(asset_mismatch["asset_inventory_verified"])
            self.assertFalse(asset_mismatch["paper_ready"])
            self.assertIn("asset inventory", asset_mismatch["paper_guard"])


if __name__ == "__main__":
    unittest.main()
