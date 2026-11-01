from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jmoshivis.evaluation.judge import (
    DEFAULT_JUDGE_MODEL,
    prepare_batch,
    score_batch,
)
from jmoshivis.evaluation.schema import EvalExample, write_manifest


class JudgeTests(unittest.TestCase):
    def test_prepare_and_score_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "red").save(root / "image.png")
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            requests = root / "requests.jsonl"
            results = root / "results.jsonl"
            item = EvalExample(
                sample_id="jammeval-fixture-1",
                benchmark="jammeval-cc-ocr-ja-refined",
                split="test",
                domain="japan",
                task="vqa",
                image_path="image.png",
                question="時刻は何時ですか？",
                prompt_text="時刻は何時ですか？\n簡潔に答えてください。",
                answers=["20:36"],
            )
            write_manifest(manifest, [item])
            predictions.write_text(
                json.dumps({"sample_id": item.sample_id, "answer_text": "20時36分です", "error": None}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            metadata = prepare_batch(manifest, predictions, requests)
            self.assertEqual(metadata["num_requests"], 3)
            request_rows = [
                json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()
            ]
            request = request_rows[0]
            prompt = request["body"]["input"][1]["content"][0]["text"]
            self.assertIn(item.prompt_text, prompt)
            self.assertIn("20時36分です", prompt)
            body = {
                "model": DEFAULT_JUDGE_MODEL,
                "output": [
                    {"content": [{"type": "output_text", "text": "reasoning: equivalent\ncorrect: yes"}]}
                ]
            }
            results.write_text(
                "".join(
                    json.dumps(
                        {
                            "custom_id": request_row["custom_id"],
                            "response": {"status_code": 200, "body": body},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for request_row in request_rows
                ),
                encoding="utf-8",
            )
            rows, summary = score_batch(manifest, predictions, results, requests)
            self.assertEqual(rows[0]["score"], 1.0)
            self.assertEqual(summary["overall"]["mean_accuracy_official_compatible"], 1.0)
            self.assertEqual(summary["overall"]["n_repeats"], 3)
            self.assertTrue(summary["protocol_ready"])
            # A fixture without the frozen suite lock and inference sidecar is
            # useful for parser tests, but can never be promoted to a paper result.
            self.assertFalse(summary["ready_for_jammeval_judge"])

    def test_nonstandard_protocol_requires_explicit_diagnostic_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "red").save(root / "image.png")
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            requests = root / "requests.jsonl"
            item = EvalExample(
                sample_id="jammeval-fixture-1",
                benchmark="jammeval-cc-ocr-ja-refined",
                split="test",
                domain="japan",
                task="vqa",
                image_path="image.png",
                question="何ですか？",
                answers=["時計"],
            )
            write_manifest(manifest, [item])
            predictions.write_text(
                json.dumps({"sample_id": item.sample_id, "answer_text": "時計"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-standard"):
                prepare_batch(
                    manifest,
                    predictions,
                    requests,
                    model="not-the-frozen-model",
                    n_repeats=1,
                )


if __name__ == "__main__":
    unittest.main()
