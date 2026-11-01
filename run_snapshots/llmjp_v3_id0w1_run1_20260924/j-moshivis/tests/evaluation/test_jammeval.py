from __future__ import annotations

import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from jmoshivis.evaluation.datasets.jammeval import (
    SHORT_ANSWER_PROMPT_SUFFIX,
    convert_rows,
)
from jmoshivis.evaluation.schema import ManifestError, read_manifest


def encoded_png(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (9, 7), color).save(buffer, format="PNG")
    return buffer.getvalue()


class JammevalDatasetTests(unittest.TestCase):
    def test_short_answer_configs_preserve_bytes_and_provenance(self) -> None:
        payload = encoded_png()
        fixtures = {
            "CC-OCR-JA-Refined": (
                {
                    "original_id": 4,
                    "image": {"bytes": payload, "path": None},
                    "question": "車両はいつ発車しますか？",
                    "answer": "20:36",
                    "ocr": "快速 20:36 千葉",
                    "image_name": "Japanese_test_4.jpg",
                },
                {"ocr": "快速 20:36 千葉", "license": "MIT"},
            ),
            "Heron-Bench-Refined": (
                {
                    "original_id": 8,
                    "image": {"bytes": payload, "path": "001.jpg"},
                    "image_category": "anime",
                    "context": "机の上に青色の羽が置かれている。",
                    "question": "この作品の監督は誰ですか？",
                    "answer": "宮崎駿",
                },
                {"context": "机の上に青色の羽が置かれている。", "image_category": "anime"},
            ),
            "JA-VLM-Bench-Refined": (
                {
                    "original_id": 2,
                    "image": {"bytes": payload, "path": None},
                    "question": "この場所は何区ですか？",
                    "answer": "渋谷区",
                    "image_url": "https://images.unsplash.com/example.jpg",
                    "page_url": "https://unsplash.com/photos/example",
                },
                {
                    "license": "Unsplash License",
                    "page_url": "https://unsplash.com/photos/example",
                },
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config, (row, expected_source) in fixtures.items():
                with self.subTest(config=config):
                    output = root / config
                    summary = convert_rows(
                        [row],
                        config=config,
                        split="test",
                        output_dir=output,
                        revision="fixture-revision",
                        loaded_from="fixture",
                    )
                    self.assertEqual(summary["output_questions"], 1)
                    self.assertEqual(summary["output_images"], 1)
                    manifest = output / "manifest_test.jsonl"
                    example = read_manifest(manifest, check_files=True)[0]
                    self.assertEqual(
                        example.prompt_text,
                        row["question"] + SHORT_ANSWER_PROMPT_SUFFIX,
                    )
                    self.assertEqual(example.resolve_image(manifest).read_bytes(), payload)
                    self.assertEqual(
                        example.source["image_sha256"], hashlib.sha256(payload).hexdigest()
                    )
                    for key, value in expected_source.items():
                        self.assertEqual(example.source[key], value)

    def test_cvqa_uses_spoken_numbers_and_keeps_official_letters(self) -> None:
        payload = encoded_png("blue")
        row = {
            "image": {"bytes": payload, "path": "Japan_fixture.jpg"},
            "original_id": "5919991124277882622_0",
            "question": "赤いジャケットを着ているキャラクターは誰ですか？",
            "options": ["相良宗介", "モンキー・D・ルフィ", "セル", "コナン"],
            "answer": 1,
            "Category": "Public Figure and pop culture",
            "Image Type": "Self",
            "Image Source": "Self-open",
            "License": "CC BY-SA",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            convert_rows(
                [row],
                config="CVQA-JA-Refined",
                split="test",
                output_dir=root,
                revision="fixture-revision",
            )
            example = read_manifest(root / "manifest_test.jsonl", check_files=True)[0]

            self.assertEqual(example.choice_labels, ["1", "2", "3", "4"])
            self.assertEqual(example.correct_choice, "2")
            self.assertEqual(example.answers, ["モンキー・D・ルフィ"])
            self.assertIn("1. 相良宗介", example.prompt_text or "")
            self.assertIn("4. コナン", example.prompt_text or "")
            self.assertEqual(example.metadata["official_choice_labels"], ["A", "B", "C", "D"])
            self.assertEqual(example.metadata["official_choice_map"]["B"], "モンキー・D・ルフィ")
            self.assertEqual(example.metadata["official_correct_choice"], "B")
            self.assertEqual(example.source["license"], "CC BY-SA")
            self.assertEqual(example.source["image_source"], "Self-open")

    def test_shared_images_are_stored_once_by_content_hash(self) -> None:
        payload = encoded_png("green")
        rows = [
            {
                "original_id": index,
                "image": {"bytes": payload, "path": "001.jpg"},
                "image_category": "art",
                "context": "同じ画像の説明",
                "question": f"質問{index}ですか？",
                "answer": f"回答{index}",
            }
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = convert_rows(
                rows,
                config="Heron-Bench-Refined",
                split="test",
                output_dir=root,
                revision="fixture-revision",
            )
            examples = read_manifest(root / "manifest_test.jsonl", check_files=True)
            self.assertEqual(summary["output_questions"], 2)
            self.assertEqual(summary["output_images"], 1)
            self.assertEqual(examples[0].image_path, examples[1].image_path)
            self.assertEqual(len(list((root / "images").rglob("*.png"))), 1)

    def test_decoded_pil_image_is_rejected_to_prevent_recompression(self) -> None:
        row = {
            "original_id": 1,
            "image": Image.new("RGB", (4, 4), "red"),
            "question": "何色ですか？",
            "answer": "赤",
            "ocr": "赤",
            "image_name": "fixture.png",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ManifestError, "decode=False"):
                convert_rows(
                    [row],
                    config="CC-OCR-JA-Refined",
                    split="test",
                    output_dir=directory,
                    revision="fixture-revision",
                )


if __name__ == "__main__":
    unittest.main()
