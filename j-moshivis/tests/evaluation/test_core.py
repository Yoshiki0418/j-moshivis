from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from jmoshivis.evaluation.datasets.ja_vg_vqa import _save_image, select_core
from jmoshivis.evaluation.metrics import (
    answer_in_response,
    exact_match,
    extract_choice,
    normalize_answer,
    score_one,
)
from jmoshivis.evaluation.merge_manifests import merge
from jmoshivis.evaluation.prepare_tts import (
    main as prepare_tts_main,
    render_example_prompt,
    render_prompt,
)
from jmoshivis.evaluation.run_inference import (
    RUN_INVARIANT_KEYS,
    _append_prediction_row,
    _asset_inventory,
    _prediction_state,
    _repair_prediction_tail,
    _run_fingerprint,
    _scan_predictions,
    _strict_json_loads,
    _write_run_metadata,
)
from jmoshivis.evaluation.overlap.audit import BKTree, coco_uid
from jmoshivis.evaluation.overlap.fingerprints import hamming, image_fingerprints
from jmoshivis.evaluation.response import extract_answer_events
from jmoshivis.evaluation.schema import (
    EvalExample,
    ManifestError,
    read_manifest,
    sha256_file,
    validate_question_audio_provenance,
    write_manifest,
)
from jmoshivis.evaluation.score import evaluate_files, validate_prediction_run
from jmoshivis.evaluation.validate_manifest import main as validate_manifest_main


def example(sample_id: str = "sample-1") -> EvalExample:
    return EvalExample(
        sample_id=sample_id,
        benchmark="fixture",
        split="test",
        domain="general_world",
        task="vqa",
        image_path="image.png",
        question="車は何色ですか？",
        answers=["赤"],
        question_type="color",
    )


class SchemaTests(unittest.TestCase):
    def test_round_trip_and_relative_file_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), "red").save(root / "image.png")
            count, digest = write_manifest(root / "manifest.jsonl", [example()])
            self.assertEqual(count, 1)
            self.assertEqual(len(digest), 64)
            loaded = read_manifest(root / "manifest.jsonl", check_files=True)
            self.assertEqual(loaded[0].question, "車は何色ですか？")

    def test_duplicate_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ManifestError):
                write_manifest(Path(directory) / "manifest.jsonl", [example(), example()])

    def test_text_asset_inventory_does_not_require_question_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            item = example()
            item.source["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, [item])

            second = example("sample-2")
            second.source["image_sha256"] = item.source["image_sha256"]
            with patch(
                "jmoshivis.evaluation.run_inference.sha256_file",
                wraps=sha256_file,
            ) as hash_file:
                inventory = _asset_inventory(
                    [item, second], manifest, require_audio=False
                )

            self.assertEqual(inventory["num_unique_images"], 1)
            self.assertEqual(inventory["num_unique_audio"], 0)
            hash_file.assert_called_once_with(image.resolve())

    def test_merge_rebases_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (8, 8), "red").save(source / "image.png")
            manifest = source / "manifest.jsonl"
            write_manifest(manifest, [example()])
            output = root / "combined" / "manifest.jsonl"
            merged = merge([manifest], output)
            write_manifest(output, merged)
            loaded = read_manifest(output, check_files=True)[0]
            self.assertTrue(loaded.resolve_image(output).is_file())
            self.assertFalse(Path(loaded.metadata["source_manifest"]["path"]).is_absolute())

    def test_merge_rebases_nested_reference_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            references = source / "audio" / "references"
            references.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(source / "image.png")
            reference_wav = references / "answer.wav"
            reference_wav.write_bytes(b"reference audio bytes")
            item = example()
            item.metadata["reference_audio"] = {
                "audio_path": "audio/references/answer.wav",
                "wav_sha256": hashlib.sha256(reference_wav.read_bytes()).hexdigest(),
            }
            manifest = source / "manifest.jsonl"
            write_manifest(manifest, [item])
            output = root / "combined" / "manifest.jsonl"

            merged = merge([manifest], output)
            write_manifest(output, merged)
            loaded = read_manifest(output, check_files=True)[0]
            rebased = Path(loaded.metadata["reference_audio"]["audio_path"])

            self.assertFalse(rebased.is_absolute())
            self.assertEqual((output.parent / rebased).resolve(), reference_wav.resolve())
            self.assertTrue((output.parent / rebased).is_file())

    def test_question_audio_provenance_binds_rendered_prompt(self) -> None:
        item = example()
        item.prompt_text = render_prompt(item.rendered_prompt, "short_answer")
        prompt_digest = hashlib.sha256(item.rendered_prompt.encode("utf-8")).hexdigest()
        item.metadata["question_audio"] = {
            "sample_id": item.sample_id,
            "prompt_text": item.rendered_prompt,
            "prompt_style": "short_answer",
            # Existing prepare_tts manifests use this compatibility spelling.
            "question_sha256": prompt_digest,
        }
        validate_question_audio_provenance(item)

        item.metadata["question_audio"]["prompt_text"] = item.question
        with self.assertRaisesRegex(ManifestError, "prompt_text"):
            validate_question_audio_provenance(item)

    def test_question_audio_provenance_checks_ids_and_all_prompt_hashes(self) -> None:
        item = example()
        prompt_digest = hashlib.sha256(item.rendered_prompt.encode("utf-8")).hexdigest()
        item.metadata["question_audio"] = {
            "sample_id": "another-sample",
            "prompt_text": item.rendered_prompt,
            "question_sha256": prompt_digest,
        }
        with self.assertRaisesRegex(ManifestError, "sample_id mismatch"):
            validate_question_audio_provenance(item)

        item.metadata["question_audio"].update(
            sample_id=item.sample_id,
            prompt_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ManifestError, "prompt_sha256"):
            validate_question_audio_provenance(item)

        item.metadata["question_audio"] = {
            "sample_id": item.sample_id,
            "prompt_text": item.rendered_prompt,
            "prompt_sha256": prompt_digest,
        }
        validate_question_audio_provenance(item)

    def test_asset_inventory_rejects_audio_for_a_different_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            audio = root / "question.wav"
            Image.new("RGB", (8, 8), "red").save(image)
            audio.write_bytes(b"fixture wav bytes")
            item = example()
            item.audio_path = audio.name
            item.source["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            item.metadata["question_audio"] = {
                "sample_id": item.sample_id,
                "prompt_text": "別の質問です。",
                "question_sha256": hashlib.sha256("別の質問です。".encode("utf-8")).hexdigest(),
                "wav_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(ManifestError, "prompt_text"):
                _asset_inventory([item], root / "manifest.jsonl")

    def test_strict_manifest_validation_checks_optional_audio_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            audio = root / "question.wav"
            manifest = root / "manifest.jsonl"
            Image.new("RGB", (8, 8), "red").save(image)
            audio.write_bytes(b"fixture wav bytes")
            item = example()
            item.audio_path = audio.name
            item.source["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            item.metadata["question_audio"] = {
                "sample_id": item.sample_id,
                "prompt_text": "別の質問です。",
                "question_sha256": hashlib.sha256("別の質問です。".encode("utf-8")).hexdigest(),
                "wav_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            }
            write_manifest(manifest, [item])
            argv = ["validate_manifest", "--manifest", str(manifest), "--strict-checksums"]
            with patch("sys.argv", argv), self.assertRaisesRegex(ManifestError, "prompt_text"):
                validate_manifest_main()


class MetricTests(unittest.TestCase):
    def test_japanese_normalization_and_conversational_answer(self) -> None:
        self.assertEqual(normalize_answer(" 回答は：赤。 "), "赤")
        self.assertEqual(normalize_answer("コーヒー"), "コーヒー")
        self.assertEqual(exact_match("赤です", ["赤"]), 0.0)
        self.assertEqual(answer_in_response("赤です", ["赤"]), 1.0)

    def test_common_negation_is_not_a_hit(self) -> None:
        prediction = "白ではありません。赤です。"
        self.assertEqual(answer_in_response(prediction, ["白"]), 0.0)
        self.assertEqual(answer_in_response(prediction, ["赤"]), 1.0)

    def test_mc_requires_one_unambiguous_label(self) -> None:
        self.assertEqual(extract_choice("答えはＢです。"), "B")
        self.assertIsNone(extract_choice("AかBです。"))
        score = score_one("x", "りんごです", ["B"], task="multiple_choice", choices=["みかん", "りんご"], correct_choice="B")
        self.assertEqual(score.choice_accuracy, 1.0)
        numeric = score_one(
            "y", "2番です", ["りんご"], task="multiple_choice",
            choices=["みかん", "りんご"], choice_labels=["1", "2"], correct_choice="2",
        )
        self.assertEqual(numeric.choice_accuracy, 1.0)
        by_text = score_one(
            "z", "りんごです", ["りんご"], task="multiple_choice",
            choices=["みかん", "りんご"], choice_labels=["1", "2"], correct_choice="2",
        )
        self.assertEqual(by_text.choice_accuracy, 0.0)
        self.assertIsNone(by_text.extracted_choice)
        self.assertEqual(extract_choice("答えは2番です。", ["猫", "犬"], ["1", "2"]), "2")
        self.assertIsNone(extract_choice("犬が2匹います。", ["猫", "犬"], ["1", "2"]))
        self.assertIsNone(extract_choice("10個あります", ["猫", "犬"], ["1", "2"]))


class ResponseTests(unittest.TestCase):
    def test_window_filters_special_ids_but_keeps_spm_space(self) -> None:
        events = [
            {"frame": 8, "token_id": 100},
            {"frame": 9, "token_id": 3},
            {"frame": 10, "token_id": 9},
            {"frame": 11, "token_id": 123},
            {"frame": 12, "token_id": 0},
            {"frame": 13, "token_id": 456},
            {"frame": 14, "token_id": 2},
            {"frame": 15, "token_id": 789},
        ]
        selected = extract_answer_events(events, question_end_frame=11, overlap_frames=1)
        self.assertEqual([row["token_id"] for row in selected], [9, 123, 456])


class DatasetTests(unittest.TestCase):
    def test_tts_refuses_to_replace_a_frozen_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest_audio.jsonl"
            output.write_text("frozen\n", encoding="utf-8")
            argv = [
                "prepare_tts",
                "--manifest", str(Path(directory) / "clean.jsonl"),
                "--output-manifest", str(output),
                "--audio-dir", str(Path(directory) / "audio"),
            ]
            with patch("sys.argv", argv), self.assertRaisesRegex(
                FileExistsError, "refusing to replace"
            ):
                prepare_tts_main()

    def test_ja_vg_preserves_source_encoded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"
            source = b"\xff\xd8\xffsource-jpeg-payload"
            digest = _save_image({"bytes": source, "path": None}, destination)
            self.assertEqual(destination.read_bytes(), source)
            self.assertEqual(digest, hashlib.sha256(source).hexdigest())

    def test_core_is_deterministic_and_one_question_per_image(self) -> None:
        rows = [
            {
                "image_id": image_id,
                "qas": [
                    {"qa_id": image_id * 10, "question": "何色ですか？", "answer": "赤"},
                    {"qa_id": image_id * 10 + 1, "question": "どこですか？", "answer": "公園"},
                ],
            }
            for image_id in range(10)
        ]
        first = select_core(rows, 42)
        second = select_core(rows, 42)
        self.assertEqual([qa["qa_id"] for _, qa in first], [qa["qa_id"] for _, qa in second])
        self.assertEqual(len(first), len(rows))
        self.assertEqual(len({row["image_id"] for row, _ in first}), len(rows))

    def test_prompt_rendering_is_explicit_and_stable(self) -> None:
        self.assertEqual(render_prompt("何色ですか？", "verbatim"), "何色ですか？")
        self.assertEqual(
            render_prompt("何色ですか？", "short_answer"),
            "何色ですか？ 一言で簡潔に答えてください。",
        )
        item = example()
        item.prompt_text = "車は何色ですか？\n1. 赤\n2. 青"
        self.assertEqual(render_example_prompt(item, "verbatim"), item.prompt_text)


class OverlapTests(unittest.TestCase):
    def test_fingerprints_survive_lossy_resave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.zeros((64, 64, 3), dtype=np.uint8)
            array[:, :32] = (240, 20, 20)
            array[:, 32:] = (20, 20, 240)
            image = Image.fromarray(array)
            image.save(root / "a.png")
            image.save(root / "b.jpg", quality=70)
            first = image_fingerprints(root / "a.png")
            second = image_fingerprints(root / "b.jpg")
            self.assertLessEqual(hamming(str(first["phash"]), str(second["phash"])), 8)
            self.assertLessEqual(hamming(str(first["dhash"]), str(second["dhash"])), 4)

    def test_bk_tree_and_coco_uid(self) -> None:
        tree = BKTree()
        tree.add("0000000000000000")
        tree.add("ffffffffffffffff")
        self.assertEqual(tree.query("0000000000000001", 1)[0][0], 1)
        expected = hashlib.sha256(b"coco_000000000123").hexdigest()
        self.assertEqual(coco_uid(123), expected)


class ScoringTests(unittest.TestCase):
    def test_prediction_run_sidecar_binds_manifest_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(manifest, [example("one")])
            fingerprint = "a" * 64
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "one",
                        "answer_text": "赤",
                        "error": None,
                        "run_fingerprint": fingerprint,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            sidecar = predictions.with_suffix(predictions.suffix + ".run.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": "jmoshivis.image-eval.run.v2",
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        "run_fingerprint": fingerprint,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            validate_prediction_run(manifest, predictions, required=True)
            predictions.write_text(
                predictions.read_text(encoding="utf-8").replace("赤", "青"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "checksum"):
                validate_prediction_run(manifest, predictions, required=True)

    def test_complete_prediction_sidecar_cannot_hide_missing_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(manifest, [example("one"), example("two")])
            fingerprint = "b" * 64
            predictions.write_text(
                json.dumps(
                    {
                        "sample_id": "one",
                        "answer_text": "赤",
                        "error": None,
                        "run_fingerprint": fingerprint,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.with_suffix(predictions.suffix + ".run.json").write_text(
                json.dumps(
                    {
                        "schema_version": "jmoshivis.image-eval.run.v2",
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        "run_fingerprint": fingerprint,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "marked complete"):
                validate_prediction_run(manifest, predictions, required=True)

    def test_prediction_rows_after_success_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"sample_id": "one", "error": None},
                        {"sample_id": "one", "error": "late failure"},
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "after a successful"):
                _prediction_state(predictions)

    def test_resume_truncates_only_an_incomplete_final_prediction_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            committed = (
                json.dumps(
                    {"sample_id": "one", "answer_text": "赤", "error": None},
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            torn = b'{"sample_id":"two","answer_text":"\xe9'
            predictions.write_bytes(committed + torn)

            with self.assertRaises(ManifestError):
                _prediction_state(predictions)
            scan = _scan_predictions(predictions, allow_incomplete_final_row=True)
            self.assertEqual(scan.tail_action, "truncate")
            self.assertEqual(scan.completed, frozenset({"one"}))
            self.assertEqual(_repair_prediction_tail(predictions, scan), "truncate")
            self.assertEqual(predictions.read_bytes(), committed)
            self.assertEqual(_prediction_state(predictions)[0], {"one"})

    def test_resume_never_skips_interior_prediction_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            contents = (
                b'{"sample_id":"one","error":null}\n'
                b'{"sample_id":"broken"\n'
                b'{"sample_id":"last"'
            )
            predictions.write_bytes(contents)
            with self.assertRaisesRegex(ManifestError, r":2:"):
                _scan_predictions(predictions, allow_incomplete_final_row=True)
            self.assertEqual(predictions.read_bytes(), contents)

            predictions.write_bytes(
                b'{"sample_id":"one","error":null}\n\n'
                b'{"sample_id":"last"'
            )
            with self.assertRaisesRegex(ManifestError, "empty prediction row"):
                _scan_predictions(predictions, allow_incomplete_final_row=True)

    def test_resume_keeps_complete_final_row_missing_only_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            first = b'{"sample_id":"one","error":null}\n'
            second = b'{"sample_id":"two","error":null}'
            predictions.write_bytes(first + second)

            finalized_scan = _scan_predictions(predictions)
            self.assertEqual(finalized_scan.tail_action, "append_newline")
            self.assertEqual(predictions.read_bytes(), first + second)
            scan = _scan_predictions(predictions, allow_incomplete_final_row=True)
            self.assertEqual(scan.tail_action, "append_newline")
            self.assertEqual(scan.completed, frozenset({"one", "two"}))
            self.assertEqual(_repair_prediction_tail(predictions, scan), "append_newline")
            self.assertEqual(predictions.read_bytes(), first + second + b"\n")
            self.assertEqual(_prediction_state(predictions)[0], {"one", "two"})

    def test_tail_repair_refuses_a_concurrent_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_bytes(b'{"sample_id":"one"')
            scan = _scan_predictions(predictions, allow_incomplete_final_row=True)
            with predictions.open("ab") as handle:
                handle.write(b" ")
            changed = predictions.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                _repair_prediction_tail(predictions, scan)
            self.assertEqual(predictions.read_bytes(), changed)

    def test_durable_prediction_append_writes_one_strict_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            _append_prediction_row(
                predictions,
                {"sample_id": "one", "answer_text": "日本語", "error": None},
            )
            self.assertTrue(predictions.read_bytes().endswith(b"\n"))
            self.assertEqual(_prediction_state(predictions)[0], {"one"})

    def test_prediction_writer_and_resume_reject_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                _append_prediction_row(
                    predictions,
                    {
                        "sample_id": "nan-model",
                        "answer_text": "",
                        "error": None,
                        "gate": float("nan"),
                    },
                )
            self.assertFalse(predictions.exists())

            predictions.write_text(
                '{"sample_id":"nan-model","error":null,"gate":NaN}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "non-finite JSON"):
                _scan_predictions(predictions, allow_incomplete_final_row=True)

            sidecar = Path(directory) / "predictions.jsonl.run.json"
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                _write_run_metadata(sidecar, {"status": "running", "value": float("inf")})
            self.assertFalse(sidecar.exists())
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                _strict_json_loads('{"status":"running","value":Infinity}')

    def test_runtime_identity_is_bound_into_run_fingerprint(self) -> None:
        self.assertTrue(
            {
                "python",
                "python_executable",
                "python_implementation",
                "platform",
                "runtime_environment",
            }.issubset(RUN_INVARIANT_KEYS)
        )
        base = {key: None for key in RUN_INVARIANT_KEYS}
        base.update(
            python="3.13 fixture",
            python_executable="/fixture/python",
            python_implementation="CPython",
            platform="Linux-fixture",
            runtime_environment={
                "torch_build": {"cuda_build": "12.8"},
                "nvidia_smi": {"gpus": ["GPU-A"]},
            },
        )
        original = _run_fingerprint(base)
        changed = dict(base)
        changed["runtime_environment"] = {
            "torch_build": {"cuda_build": "12.8"},
            "nvidia_smi": {"gpus": ["GPU-B"]},
        }
        self.assertNotEqual(original, _run_fingerprint(changed))

    def test_missing_prediction_stays_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(manifest, [example("one"), example("two")])
            predictions.write_text(
                json.dumps({"sample_id": "one", "answer_text": "赤です"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows, summary = evaluate_files(manifest, predictions)
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["num_missing_predictions"], 1)
            self.assertEqual(summary["overall_micro"]["answer_in_response"], 0.5)

    def test_failed_attempt_is_retried_and_latest_success_is_scored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_manifest(manifest, [example("one")])
            records = [
                {"sample_id": "one", "answer_text": "", "error": "RuntimeError: transient", "attempt": 1},
                {"sample_id": "one", "answer_text": "赤", "error": None, "attempt": 2},
            ]
            predictions.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            completed, attempts = _prediction_state(predictions)
            self.assertEqual(completed, {"one"})
            self.assertEqual(attempts, {"one": 2})
            _, summary = evaluate_files(manifest, predictions)
            self.assertEqual(summary["overall_micro"]["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
