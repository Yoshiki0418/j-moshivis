from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from jmoshivis.train import (
    _file_fingerprint,
    build_run_metadata,
    initialize_training_seed,
    rank_local_resume_sample_index,
    resolve_resume_path,
    validate_run_directory,
)
from jmoshivis.trainer import TRAINING_CHECKPOINT_COMPLETE


class TrainingRuntimeTests(unittest.TestCase):
    def test_seed_reproduces_python_numpy_and_torch_initialization(self) -> None:
        first_state = initialize_training_seed(31415)
        first_numpy = np.random.standard_normal(4)
        first_layer = torch.nn.Linear(3, 2)
        first_weights = first_layer.weight.detach().clone()

        second_state = initialize_training_seed(31415)
        second_numpy = np.random.standard_normal(4)
        second_layer = torch.nn.Linear(3, 2)

        self.assertEqual(first_state, second_state)
        np.testing.assert_array_equal(first_numpy, second_numpy)
        torch.testing.assert_close(
            first_weights, second_layer.weight.detach(), rtol=0.0, atol=0.0
        )

    def test_seed_never_uses_global_cuda_seed_apis(self) -> None:
        with patch(
            "jmoshivis.train.torch.manual_seed",
            side_effect=AssertionError("global torch.manual_seed is forbidden"),
        ) as torch_seed, patch(
            "jmoshivis.train.torch.cuda.manual_seed_all",
            side_effect=AssertionError("all-device CUDA seeding is forbidden"),
        ) as cuda_seed_all:
            initialize_training_seed(9, device=torch.device("cpu"))
        torch_seed.assert_not_called()
        cuda_seed_all.assert_not_called()

    def test_cuda_seed_targets_only_the_assigned_generator(self) -> None:
        generators = [MagicMock(), MagicMock()]
        assigned_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch.object(torch.cuda, "default_generators", generators), patch(
            "jmoshivis.train.torch.cuda.get_rng_state",
            return_value=assigned_state,
        ) as get_state, patch(
            "jmoshivis.train.torch.manual_seed",
            side_effect=AssertionError("global torch.manual_seed is forbidden"),
        ) as torch_seed, patch(
            "jmoshivis.train.torch.cuda.manual_seed_all",
            side_effect=AssertionError("all-device CUDA seeding is forbidden"),
        ) as cuda_seed_all:
            state = initialize_training_seed(17, device=torch.device("cuda:1"))

        generators[0].manual_seed.assert_not_called()
        generators[1].manual_seed.assert_called_once_with(17)
        self.assertEqual(get_state.call_count, 2)
        self.assertEqual(state["seed"], 17)
        self.assertIsNotNone(state["cuda_state_sha256"])
        torch_seed.assert_not_called()
        cuda_seed_all.assert_not_called()

    def test_binary_fingerprint_is_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"\x00\xff" * 4096)
            first = _file_fingerprint(path, count_lines=False)
            path.write_bytes(b"\x00\xfe" * 4096)
            second = _file_fingerprint(path, count_lines=False)
            self.assertEqual(first["bytes"], second["bytes"])
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_run_metadata_fingerprints_inputs_and_normalizes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            tokenizer = root / "tokenizer.model"
            model_config = root / "model.yaml"
            base_model = root / "model.safetensors"
            mimi = root / "mimi.safetensors"
            train.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
            eval_path.write_text('{"id": 3}\n', encoding="utf-8")
            tokenizer.write_bytes(b"tokenizer")
            model_config.write_text("dim: 8\n", encoding="utf-8")
            base_model.write_bytes(b"base-model")
            mimi.write_bytes(b"mimi-model")
            args = OmegaConf.create(
                {
                    "paths": {
                        "tokenizer": str(tokenizer),
                        "kyuteye_config": str(model_config),
                        "moshivis_weight": str(base_model),
                    },
                    "data": {
                        "train_data": str(train),
                        "eval_data": str(eval_path),
                    },
                    "train": {"resume_from": "latest", "seed": 7},
                }
            )

            class MainProcess:
                is_main_process = True
                num_processes = 2
                mixed_precision = "bf16"

            metadata = build_run_metadata(
                args,
                MainProcess(),
                {"shape": (1, 2)},
                mimi_weight=mimi,
                seed_state={"seed": 7},
                asset_manifest_verification={
                    "status": "verified",
                    "verification_mode": "fast",
                    "manifest": {"sha256": "a" * 64},
                },
            )

            self.assertEqual(metadata["datasets"]["train"]["records"], 2)
            self.assertEqual(metadata["datasets"]["eval"]["records"], 1)
            self.assertEqual(metadata["visual_adapter_init_audit"]["shape"], [1, 2])
            self.assertNotIn("resume_from", metadata["resolved_config"]["train"])
            self.assertEqual(metadata["base_model"]["bytes"], len(b"base-model"))
            self.assertEqual(metadata["mimi_model"]["bytes"], len(b"mimi-model"))
            self.assertEqual(
                metadata["asset_manifest_verification"]["status"], "verified"
            )
            self.assertIn("models/loaders.py", metadata["implementation"])
            runtime = metadata["runtime"]
            self.assertIn("python_version", runtime)
            self.assertIn("platform", runtime)
            self.assertIn("torch", runtime["packages"])
            self.assertIn("accelerate", runtime["packages"])
            self.assertIn("moshi", runtime["packages"])
            self.assertIn("torch_cuda_version", runtime)
            self.assertIn("cudnn_version", runtime)
            self.assertEqual(runtime["devices"][0]["rank"], 0)
            self.assertEqual(json.loads(json.dumps(runtime)), runtime)
            sampling = metadata["data_sampling_policy"]
            self.assertEqual(
                sampling["train_row_sharding"],
                "shuffle_then_drop_remainder_rank_stride",
            )
            self.assertEqual(sampling["eval_row_sharding"], "no_drop_rank_stride")
            self.assertEqual(sampling["train_rows_dropped_per_epoch"], 0)
            self.assertEqual(
                sampling["modality_schedule"],
                "rank_balanced_per_optimizer_window",
            )
            self.assertEqual(
                metadata["data_resume_policy"]["name"],
                "exact_rank_stream_from_micro_step",
            )
            self.assertEqual(len(metadata["run_fingerprint"]), 64)

    def test_resume_sample_cursor_uses_micro_steps_including_overflow_windows(self) -> None:
        # global_step=7 represents seven successful updates.  Ten complete
        # accumulation windows were consumed because three AMP attempts
        # overflowed; data resume must follow micro_step, not global_step.
        state = {
            "global_step": 7,
            "micro_step": 80,
            "accumulation_steps": 8,
        }
        self.assertEqual(
            rank_local_resume_sample_index(
                state, local_batch_size=2, accumulation_steps=8
            ),
            160,
        )
        self.assertEqual(
            rank_local_resume_sample_index(
                None, local_batch_size=2, accumulation_steps=8
            ),
            0,
        )

    def test_resume_cursor_rejects_partial_or_impossible_optimizer_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer-window boundary"):
            rank_local_resume_sample_index(
                {
                    "global_step": 1,
                    "micro_step": 9,
                    "accumulation_steps": 8,
                },
                local_batch_size=1,
                accumulation_steps=8,
            )
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            rank_local_resume_sample_index(
                {
                    "global_step": 2,
                    "micro_step": 8,
                    "accumulation_steps": 8,
                },
                local_batch_size=1,
                accumulation_steps=8,
            )

    def test_resume_is_opt_in_and_latest_ignores_incomplete_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step_9").mkdir()
            complete = root / "step_8"
            complete.mkdir()
            (complete / TRAINING_CHECKPOINT_COMPLETE).write_text("{}")

            with self.assertRaisesRegex(ValueError, "resume_from"):
                validate_run_directory(root, {"run_fingerprint": "x"}, None)
            self.assertEqual(resolve_resume_path(root, "latest"), complete)


if __name__ == "__main__":
    unittest.main()
