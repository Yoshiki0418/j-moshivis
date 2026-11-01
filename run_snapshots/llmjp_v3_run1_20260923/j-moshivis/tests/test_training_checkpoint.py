from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import accelerate.checkpointing as accelerate_checkpointing
from accelerate import Accelerator
from safetensors.torch import load_file

from jmoshivis.trainer import (
    ASSIGNED_DEVICE_RNG_SCHEMA_VERSION,
    JmoshiVisTrainer,
    TRAINING_CHECKPOINT_COMPLETE,
    _load_accelerate_state_without_global_cuda_rng,
    _read_rank_rng_state,
    _restore_rank_rng_state,
    _save_accelerate_state_without_global_cuda_rng,
    _save_rank_rng_state,
    read_training_checkpoint_metadata,
)


class TrainingCheckpointTests(unittest.TestCase):
    def _make_trainer(self, checkpoint_dir: str):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
        model = torch.nn.Linear(2, 2)
        image = torch.nn.Linear(2, 2)
        audit = {
            "schema_version": "jmoshivis.visual-adapter-init.v1",
            "mode": "test",
            "visual_adapter_sampled_sha256": "adapter-sha",
        }
        model.visual_adapter_init_audit = audit
        image.visual_adapter_init_audit = audit
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *image.parameters()], lr=0.1
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 - 0.05 * min(step, 10)
        )
        # The production scheduler is intentionally not Accelerator-wrapped:
        # one optimizer update must equal one scheduler update on every world size.
        accelerator.register_for_checkpointing(scheduler)
        model, image, optimizer = accelerator.prepare(model, image, optimizer)
        args = SimpleNamespace(
            target_batch_size=1,
            batch_size=1,
            max_steps=20,
            save_every=1,
            keep_last=2,
            checkpoint_dir=checkpoint_dir,
            log_interval=1,
        )
        metadata = {
            "run_fingerprint": "run-sha",
            "visual_adapter_init_audit": audit,
            "datasets": {"train": {"sha256": "data-sha", "records": 2}},
        }
        trainer = JmoshiVisTrainer(
            model,
            optimizer,
            torch.device("cpu"),
            args,
            accelerator,
            image_embedder=image,
            scheduler=scheduler,
            run_metadata=metadata,
        )
        return trainer, accelerator, scheduler

    @staticmethod
    def _optimizer_update(trainer: JmoshiVisTrainer) -> None:
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in trainer.model.parameters())
        loss = loss + sum(
            parameter.square().sum()
            for parameter in trainer.image_embedder.parameters()
        )
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.global_step += 1
        trainer.micro_step += 1

    def test_complete_state_restores_model_optimizer_scheduler_step_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, scheduler = self._make_trainer(directory)
            self._optimizer_update(trainer)
            trainer.epoch = 3
            saved_weight = next(trainer.model.parameters()).detach().clone()
            saved_optimizer_step = next(iter(trainer.optimizer.state.values()))[
                "step"
            ].clone()
            saved_scheduler_step = scheduler.last_epoch
            torch.random.default_generator.manual_seed(12345)
            checkpoint = trainer.save_training_checkpoint()
            expected_random = torch.rand(4)

            with torch.no_grad():
                next(trainer.model.parameters()).add_(100)
            trainer.global_step = 99
            trainer.micro_step = 999
            trainer.epoch = 9
            trainer.optimizer.state.clear()
            for _ in range(3):
                scheduler.step()
            torch.random.default_generator.manual_seed(999)

            observed_global_cuda_states = []

            def observe_empty_global_cuda_state(states):
                observed_global_cuda_states.append(states)
                self.assertEqual(states, [])

            with patch(
                "jmoshivis.trainer.torch.cuda.set_rng_state_all",
                side_effect=observe_empty_global_cuda_state,
            ) as set_all:
                state = trainer.load_training_checkpoint(checkpoint)
            set_all.assert_called_once_with([])
            self.assertEqual(observed_global_cuda_states, [[]])

            self.assertTrue((checkpoint / TRAINING_CHECKPOINT_COMPLETE).is_file())
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(trainer.micro_step, 1)
            self.assertEqual(trainer.epoch, 3)
            self.assertTrue(
                torch.equal(next(trainer.model.parameters()).detach(), saved_weight)
            )
            restored_optimizer_step = next(iter(trainer.optimizer.state.values()))[
                "step"
            ]
            self.assertTrue(torch.equal(restored_optimizer_step, saved_optimizer_step))
            self.assertEqual(scheduler.last_epoch, saved_scheduler_step)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))

    def test_accelerate_save_never_queries_all_visible_cuda_rngs(self) -> None:
        observed_cuda_availability = []

        class FakeAccelerator:
            def save_state(self, path, safe_serialization):
                del path, safe_serialization
                observed_cuda_availability.append(
                    accelerate_checkpointing.is_cuda_available()
                )
                if accelerate_checkpointing.is_cuda_available():
                    torch.cuda.get_rng_state_all()

        with tempfile.TemporaryDirectory() as directory, patch(
            "jmoshivis.trainer.torch.cuda.is_available", return_value=True
        ), patch(
            "jmoshivis.trainer.torch.cuda.get_rng_state_all",
            side_effect=AssertionError("must not query every visible device"),
        ) as get_all:
            _save_accelerate_state_without_global_cuda_rng(
                FakeAccelerator(), Path(directory)
            )

        self.assertEqual(observed_cuda_availability, [False])
        get_all.assert_not_called()

    def test_real_accelerate_save_path_omits_global_cuda_rng_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, _ = self._make_trainer(directory)
            self._optimizer_update(trainer)
            with patch(
                "jmoshivis.trainer.torch.cuda.is_available", return_value=True
            ), patch(
                "jmoshivis.trainer.torch.cuda.get_rng_state_all",
                side_effect=AssertionError("must not query every visible device"),
            ) as get_all:
                checkpoint = trainer.save_training_checkpoint()

            self.assertTrue((checkpoint / TRAINING_CHECKPOINT_COMPLETE).is_file())
            get_all.assert_not_called()

    def test_accelerate_load_never_restores_all_visible_cuda_rngs(self) -> None:
        class FakeAccelerator:
            def load_state(self, path):
                del path
                # Exercise the call made by Accelerate 1.14.0.  The helper
                # must intercept even a non-empty value; the strict per-rank
                # restore is tested separately below.
                torch.cuda.set_rng_state_all([torch.tensor([1], dtype=torch.uint8)])

        with patch(
            "jmoshivis.trainer.torch.cuda.is_available", return_value=True
        ), patch(
            "jmoshivis.trainer.torch.cuda.set_rng_state_all",
            side_effect=AssertionError("must not restore every visible device"),
        ) as set_all:
            _load_accelerate_state_without_global_cuda_rng(
                FakeAccelerator(), Path("unused")
            )

        set_all.assert_not_called()

    def test_rank_rng_uses_and_restores_only_assigned_cuda_device(self) -> None:
        accelerator = SimpleNamespace(
            process_index=1,
            device=torch.device("cuda:1"),
            step=7,
        )
        assigned_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as directory, patch(
            "jmoshivis.trainer.torch.cuda.is_available", return_value=True
        ), patch(
            "jmoshivis.trainer.torch.cuda.get_rng_state",
            return_value=assigned_state,
        ) as get_assigned, patch(
            "jmoshivis.trainer.torch.cuda.get_rng_state_all",
            side_effect=AssertionError("must not query every visible device"),
        ) as get_all:
            checkpoint = Path(directory)
            _save_rank_rng_state(checkpoint, accelerator)
            state = _read_rank_rng_state(checkpoint, accelerator)

            self.assertEqual(
                state["schema_version"], ASSIGNED_DEVICE_RNG_SCHEMA_VERSION
            )
            self.assertEqual(state["torch_cuda_manual_seed"], [])
            get_assigned.assert_called_once_with(torch.device("cuda:1"))
            get_all.assert_not_called()

            with patch(
                "jmoshivis.trainer.torch.cuda.set_rng_state"
            ) as set_assigned, patch(
                "jmoshivis.trainer.torch.cuda.set_rng_state_all",
                side_effect=AssertionError("must not restore every visible device"),
            ) as set_all:
                _restore_rank_rng_state(state, accelerator)

            set_assigned.assert_called_once()
            self.assertTrue(
                torch.equal(set_assigned.call_args.args[0], assigned_state)
            )
            self.assertEqual(
                set_assigned.call_args.kwargs["device"], torch.device("cuda:1")
            )
            set_all.assert_not_called()

    def test_incomplete_and_incompatible_checkpoints_are_rejected_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, _ = self._make_trainer(directory)
            self._optimizer_update(trainer)
            checkpoint = trainer.save_training_checkpoint()
            marker = checkpoint / TRAINING_CHECKPOINT_COMPLETE
            marker_contents = marker.read_bytes()
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_training_checkpoint_metadata(checkpoint)
            marker.write_bytes(marker_contents)

            rng_state = checkpoint / "random_states_0.pkl"
            rng_contents = rng_state.read_bytes()
            rng_state.unlink()
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                read_training_checkpoint_metadata(checkpoint)
            rng_state.write_bytes(rng_contents)

            optimizer_state = checkpoint / "optimizer.bin"
            optimizer_contents = optimizer_state.read_bytes()
            optimizer_state.write_bytes(optimizer_contents + b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                read_training_checkpoint_metadata(checkpoint)
            optimizer_state.write_bytes(optimizer_contents)

            original_fingerprint = trainer.run_fingerprint
            trainer.run_fingerprint = "different-run"
            before = next(trainer.model.parameters()).detach().clone()
            with self.assertRaisesRegex(ValueError, "incompatible run"):
                trainer.load_training_checkpoint(checkpoint)
            self.assertTrue(
                torch.equal(next(trainer.model.parameters()).detach(), before)
            )
            trainer.run_fingerprint = original_fingerprint

            restored_state = read_training_checkpoint_metadata(checkpoint)
            restored_state["world_size"] = 2
            with self.assertRaisesRegex(ValueError, "world_size changed"):
                trainer._validate_resume_metadata(checkpoint, restored_state)

            state_path = checkpoint / "trainer_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["global_step"] = 123
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_training_checkpoint_metadata(checkpoint)

    def test_retention_only_removes_completed_old_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, _ = self._make_trainer(directory)
            incomplete = Path(directory) / "step_0"
            incomplete.mkdir()
            for _ in range(3):
                self._optimizer_update(trainer)
                trainer.save_training_checkpoint()

            self.assertTrue(incomplete.exists())
            self.assertFalse((Path(directory) / "step_1").exists())
            self.assertTrue((Path(directory) / "step_2").exists())
            self.assertTrue((Path(directory) / "step_3").exists())

    def test_unwrapped_scheduler_advances_once_per_optimizer_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, scheduler = self._make_trainer(directory)
            initial_step = scheduler.last_epoch
            for _ in range(4):
                self._optimizer_update(trainer)
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(scheduler.last_epoch, initial_step + 4)

    def test_model_only_inference_export_remains_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, _, _ = self._make_trainer(directory)
            output = Path(directory) / "model.safetensors"
            trainer.save_checkpoint(str(output))
            tensors = load_file(output)
            self.assertIn("weight", tensors)
            self.assertIn("image_prefix.weight", tensors)
            self.assertFalse((Path(directory) / TRAINING_CHECKPOINT_COMPLETE).exists())

    def test_failed_model_only_export_removes_partial_without_touching_final(self) -> None:
        def fail_after_partial_write(_bundle, path: str) -> None:
            Path(path).write_bytes(b"partial export")
            raise OSError("injected save failure")

        failure_injections = {
            "save": patch(
                "safetensors.torch.save_model",
                side_effect=fail_after_partial_write,
            ),
            "file_fsync": patch(
                "jmoshivis.trainer.os.fsync",
                side_effect=OSError("injected fsync failure"),
            ),
            "replace": patch(
                "jmoshivis.trainer.os.replace",
                side_effect=OSError("injected replace failure"),
            ),
        }

        for stage, injected_failure in failure_injections.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                trainer, _, _ = self._make_trainer(directory)
                output = Path(directory) / "model.safetensors"
                temporary = Path(directory) / ".model.safetensors.incomplete"
                original = b"previous complete export"
                output.write_bytes(original)

                with injected_failure, self.assertRaisesRegex(
                    OSError, "injected .*failure"
                ):
                    trainer.save_checkpoint(str(output))

                self.assertEqual(output.read_bytes(), original)
                self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
