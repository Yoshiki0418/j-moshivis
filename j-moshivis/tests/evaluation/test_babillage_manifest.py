from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
import types
import unittest
import wave
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import Mock, call, patch

from PIL import Image

from jmoshivis.evaluation.datasets.build_babillage_manifest import (
    BABILLAGE_DATASET_ID,
    BABILLAGE_REVISION,
    COCO_PARQUET_REVISION,
    COCO_SOURCE_REVISION,
    DATASET_SPECS,
    _base_storage_uri,
    _load_huggingface_rows,
    _vqav2_answer_rows,
    convert_rows,
    main,
)
from jmoshivis.evaluation.schema import (
    ManifestError,
    read_manifest,
    validate_question_audio_provenance,
)


def encoded_png(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (9, 7), color).save(buffer, format="PNG")
    return buffer.getvalue()


def encoded_wav(value: int = 0) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(int(value).to_bytes(2, "little", signed=True) * 240)
    return buffer.getvalue()


def audio_fields(prefix: str, tokens: list[str], payload: bytes) -> dict[str, object]:
    return {
        f"{prefix}_audio": {"bytes": payload, "path": f"fixture-{prefix}.wav"},
        f"{prefix}_transcript": tokens,
        f"{prefix}_alignment": [
            [index * 0.1, (index + 1) * 0.1] for index in range(len(tokens))
        ],
    }


class BabillageManifestTests(unittest.TestCase):
    def test_coco_keeps_official_audio_and_all_image_references(self) -> None:
        image = encoded_png("green")
        base_rows = [
            {
                "image": {"bytes": image, "path": "same.png"},
                "filepath": "COCO_val2014_000000000123.jpg",
                "filename": "COCO_val2014_000000000123.jpg",
                "imgid": 7,
                "cocoid": 123,
                "sentences": {"raw": "A green square.", "sentid": 10},
            },
            {
                "image": {"bytes": image, "path": "same.png"},
                "filepath": "COCO_val2014_000000000123.jpg",
                "filename": "COCO_val2014_000000000123.jpg",
                "imgid": 7,
                "cocoid": 123,
                "sentences": {"raw": "A small colored tile.", "sentid": 11},
            },
        ]
        question_audio = encoded_wav(1)
        answer_audio = encoded_wav(2)
        babillage_rows = []
        for sample_id in range(2):
            row: dict[str, object] = {"sample_id": sample_id, "image_id": 123}
            row.update(audio_fields("question", ["What", "do", "you", "see", "?"], question_audio))
            row.update(audio_fields("answer", ["I", "see", "green", "."], answer_audio))
            babillage_rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = convert_rows(
                babillage_rows,
                base_rows,
                config="coco",
                split="validation",
                output_dir=root,
                expected_count=2,
            )
            manifest = root / "manifest_validation.jsonl"
            examples = read_manifest(manifest, check_files=True, require_audio=True)

            self.assertEqual(summary["output_questions"], 2)
            self.assertEqual(summary["output_images"], 1)
            self.assertEqual(summary["ground_truth_policy"], "all_captions_for_image")
            self.assertEqual(
                summary["coco_prompt_policy"],
                "all_babillage_prompts_each_scored_against_all_image_references",
            )
            self.assertEqual(
                examples[0].answers,
                ["A green square.", "A small colored tile."],
            )
            self.assertEqual(examples[0].question, "What do you see?")
            self.assertEqual(examples[0].task, "caption")
            self.assertEqual(examples[0].source["caption_sent_id"], 10)
            self.assertEqual(examples[1].source["caption_sent_id"], 11)
            self.assertEqual(examples[0].metadata["moshivis_metric"], "cider")
            self.assertEqual(
                examples[0].metadata["base_ground_truth"]["cider_group_id"], 123
            )
            self.assertEqual(
                examples[0].metadata["base_ground_truth"]["coco_prompt_policy"],
                "all_babillage_prompts_each_scored_against_all_image_references",
            )
            self.assertEqual(examples[0].source["dataset_revision"], BABILLAGE_REVISION)
            self.assertEqual(
                examples[0].source["base_dataset_revision"],
                DATASET_SPECS["coco"].base_revision,
            )
            self.assertEqual(
                examples[0].source["base_dataset_storage_revision"],
                COCO_PARQUET_REVISION,
            )
            self.assertEqual(
                examples[0].source["base_dataset_storage_uri"],
                _base_storage_uri(DATASET_SPECS["coco"], "validation"),
            )
            self.assertEqual(summary["base_dataset_revision"], COCO_SOURCE_REVISION)
            self.assertEqual(
                summary["base_dataset_storage_revision"], COCO_PARQUET_REVISION
            )
            self.assertEqual(examples[0].resolve_audio(manifest).read_bytes(), question_audio)
            reference_path = root / examples[0].metadata["reference_audio"]["audio_path"]
            self.assertEqual(reference_path.read_bytes(), answer_audio)
            self.assertEqual(
                examples[0].metadata["reference_audio"]["transcript_text"],
                "I see green.",
            )
            self.assertEqual(examples[0].image_path, examples[1].image_path)
            self.assertEqual(len(list((root / "images").rglob("*.png"))), 1)
            validate_question_audio_provenance(examples[0])

    def test_ocrvqa_flattens_qas_and_uses_base_answers_as_metric_gt(self) -> None:
        image = encoded_png("blue")
        base_rows = [
            {
                "image": {"bytes": image, "path": "book.png"},
                "image_id": "000638692X",
                "questions": ["Who wrote this book?", "What is the title?"],
                "answers": ["Philip Marsden", "The Bronski House"],
                "title": "The Bronski House",
                "authorName": "Philip Marsden",
                "genre": "Travel",
                "image_url": "https://example.invalid/book.jpg",
                "set_name": "validation",
            }
        ]
        babillage_rows = []
        for sample_id, (question, answer) in enumerate(
            [
                (["Who", "wrote", "this", "book", "?"], ["Philip", "Marsden"]),
                (["What", "is", "its", "title", "?"], ["The", "Bronski", "House"]),
            ]
        ):
            row: dict[str, object] = {"sample_id": sample_id, "image_id": 0}
            row.update(audio_fields("question", question, encoded_wav(sample_id + 1)))
            row.update(audio_fields("answer", answer, encoded_wav(sample_id + 3)))
            babillage_rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = convert_rows(
                babillage_rows,
                base_rows,
                config="ocrvqa",
                split="validation",
                output_dir=root,
                expected_count=2,
            )
            examples = read_manifest(
                root / "manifest_validation.jsonl",
                check_files=True,
                require_audio=True,
            )

            self.assertEqual(summary["output_images"], 1)
            self.assertEqual(summary["ground_truth_policy"], "source_answer")
            self.assertEqual(examples[0].answers, ["Philip Marsden"])
            self.assertEqual(examples[1].answers, ["The Bronski House"])
            self.assertEqual(examples[1].question, "What is its title?")
            self.assertEqual(examples[1].source["base_question"], "What is the title?")
            self.assertEqual(examples[0].source["image_id"], "000638692X")
            self.assertEqual(examples[0].image_path, examples[1].image_path)

    def test_vqav2_keeps_all_annotator_answers_including_duplicates(self) -> None:
        answers = ["down"] * 7 + ["at table", "skateboard", "table"]
        base_rows = [
            {
                "question_type": "none of the above",
                "multiple_choice_answer": "down",
                "answers": [
                    {
                        "answer": answer,
                        "answer_confidence": "yes",
                        "answer_id": index + 1,
                    }
                    for index, answer in enumerate(answers)
                ],
                "image_id": 262148,
                "answer_type": "other",
                "question_id": 262148000,
                "question": "Where is he looking?",
                "image": {"bytes": encoded_png("yellow"), "path": "vqa.png"},
            }
        ]
        babillage_rows = [
            {
                "sample_id": 0,
                **audio_fields(
                    "question",
                    ["Where", "is", "he", "looking", "?"],
                    encoded_wav(4),
                ),
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = convert_rows(
                babillage_rows,
                base_rows,
                config="vqav2",
                split="validation",
                output_dir=root,
                expected_count=1,
            )
            example = read_manifest(
                root / "manifest_validation.jsonl",
                check_files=True,
                require_audio=True,
            )[0]

            self.assertEqual(example.answers, answers)
            self.assertEqual(example.answers.count("down"), 7)
            self.assertEqual(example.metadata["moshivis_metric"], "vqav2_accuracy")
            self.assertEqual(example.source["question_id"], 262148000)
            self.assertEqual(example.source["base_question"], "Where is he looking?")
            self.assertEqual(
                example.metadata["base_ground_truth"]["multiple_choice_answer"],
                "down",
            )
            self.assertNotIn("reference_audio", example.metadata)
            self.assertEqual(summary["ground_truth_policy"], "all_annotator_answers_with_duplicates")

    def test_vqav2_accepts_struct_of_lists_and_flattened_answer_columns(self) -> None:
        struct = {
            "answers": {
                "answer": ["yes", "yes", "no"],
                "answer_confidence": ["yes", "yes", "maybe"],
                "answer_id": [1, 2, 3],
            }
        }
        flattened = {
            "answers.answer": ["two", "2"],
            "answers.answer_confidence": ["yes", "yes"],
            "answers.answer_id": [1, 2],
        }

        self.assertEqual(
            [item["answer"] for item in _vqav2_answer_rows(struct, 1)],
            ["yes", "yes", "no"],
        )
        self.assertEqual(
            [item["answer"] for item in _vqav2_answer_rows(flattened, 2)],
            ["two", "2"],
        )

    def test_join_refuses_different_sample_order(self) -> None:
        base_rows = [
            {
                "question_type": "what",
                "multiple_choice_answer": "red",
                "answers": [{"answer": "red", "answer_id": 1, "answer_confidence": "yes"}],
                "image_id": 1,
                "answer_type": "other",
                "question_id": 2,
                "question": "What color?",
                "image": {"bytes": encoded_png(), "path": "image.png"},
            }
        ]
        babillage_rows = [
            {
                "sample_id": 99,
                **audio_fields("question", ["What", "color", "?"], encoded_wav()),
            }
        ]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ManifestError, "refusing a positional join"
        ):
            convert_rows(
                babillage_rows,
                base_rows,
                config="vqav2",
                split="validation",
                output_dir=directory,
            )

    def test_question_audio_hash_is_bound_to_encoded_asset(self) -> None:
        question_audio = encoded_wav(5)
        base_rows = [
            {
                "question_type": "what",
                "multiple_choice_answer": "red",
                "answers": [{"answer": "red", "answer_id": 1, "answer_confidence": "yes"}],
                "image_id": 1,
                "answer_type": "other",
                "question_id": 2,
                "question": "What color?",
                "image": {"bytes": encoded_png(), "path": "image.png"},
            }
        ]
        babillage_rows = [
            {
                "sample_id": 0,
                **audio_fields("question", ["What", "color", "?"], question_audio),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            convert_rows(
                babillage_rows,
                base_rows,
                config="vqav2",
                split="validation",
                output_dir=root,
            )
            example = read_manifest(root / "manifest_validation.jsonl")[0]
            expected = hashlib.sha256(question_audio).hexdigest()
            self.assertEqual(example.metadata["question_audio"]["wav_sha256"], expected)
            self.assertEqual(example.metadata["question_audio"]["asset_sha256"], expected)

    def test_coco_loader_uses_pinned_auto_converted_parquet(self) -> None:
        class FakeDataset:
            def __init__(self, columns: list[str]) -> None:
                self.column_names = columns
                self.cast_calls: list[tuple[str, object]] = []

            def cast_column(self, name: str, feature: object) -> "FakeDataset":
                self.cast_calls.append((name, feature))
                return self

        class FakeFeature:
            def __init__(self, *, decode: bool) -> None:
                self.decode = decode

        babillage = FakeDataset(["question_audio", "answer_audio"])
        base = FakeDataset(["image", "cocoid", "sentences"])
        load_dataset = Mock(side_effect=[babillage, base])
        datasets_module = types.ModuleType("datasets")
        datasets_module.Audio = FakeFeature
        datasets_module.Image = FakeFeature
        datasets_module.load_dataset = load_dataset

        with patch.dict(sys.modules, {"datasets": datasets_module}):
            loaded_babillage, loaded_base, loaded_from = _load_huggingface_rows(
                DATASET_SPECS["coco"],
                split="validation",
                babillage_revision=BABILLAGE_REVISION,
                base_revision=COCO_SOURCE_REVISION,
            )

        storage_uri = (
            "hf://datasets/HuggingFaceM4/COCO@"
            f"{COCO_PARQUET_REVISION}/2014/coco-validation-*.parquet"
        )
        self.assertIs(loaded_babillage, babillage)
        self.assertIs(loaded_base, base)
        self.assertEqual(
            load_dataset.call_args_list,
            [
                call(
                    BABILLAGE_DATASET_ID,
                    "coco",
                    split="validation",
                    revision=BABILLAGE_REVISION,
                ),
                call(
                    "parquet",
                    data_files={"validation": storage_uri},
                    split="validation",
                ),
            ],
        )
        self.assertEqual(loaded_from["base_storage"], storage_uri)
        self.assertNotIn("trust_remote_code", repr(load_dataset.call_args_list))

    def test_dry_run_reports_coco_source_and_storage_revisions(self) -> None:
        text_output = StringIO()
        argv = [
            "build_babillage_manifest",
            "--output-dir",
            "/tmp/not-created-by-dry-run",
            "--config",
            "coco",
            "--split",
            "validation",
            "--dry-run",
        ]
        with patch("sys.argv", argv), contextlib.redirect_stdout(text_output):
            self.assertEqual(main(), 0)
        plan = json.loads(text_output.getvalue())
        self.assertEqual(plan["base_source_revision"], COCO_SOURCE_REVISION)
        self.assertEqual(plan["base_storage_revision"], COCO_PARQUET_REVISION)
        self.assertEqual(
            plan["base_storage_uri"],
            _base_storage_uri(DATASET_SPECS["coco"], "validation"),
        )

    def test_coco_rejects_source_revision_inconsistent_with_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "auto-converted storage"
        ):
            convert_rows(
                [],
                [],
                config="coco",
                split="validation",
                output_dir=directory,
                base_revision="0" * 40,
            )


if __name__ == "__main__":
    unittest.main()
