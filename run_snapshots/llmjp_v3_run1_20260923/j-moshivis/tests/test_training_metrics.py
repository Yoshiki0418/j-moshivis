from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from jmoshivis.conditioners.base import ConditionAttributes, TensorCondition
from jmoshivis.modules.transformer import Transformer
from jmoshivis.tools.writer import WandBMetricsWriter
from accelerate import Accelerator

from jmoshivis.trainer import (
    audio_forward_plan,
    compute_text_token_metrics,
    finalize_effective_batch_statistics,
    gradient_accumulation_steps,
    new_effective_batch_statistics,
    require_batched_image_conditions,
    update_effective_batch_statistics,
)


class TextTokenMetricTests(unittest.TestCase):
    def test_id_zero_id_three_and_content_are_disjoint(self) -> None:
        target = torch.tensor([[[0, 0, 3, 10, 11, 12]]])
        prediction = torch.tensor([[[0, 8, 3, 10, 8, 12]]])
        mask = torch.ones_like(target, dtype=torch.bool)

        metrics = compute_text_token_metrics(
            prediction,
            target,
            mask,
            text_padding_token_id=3,
            end_of_text_padding_id=0,
        )

        self.assertAlmostEqual(metrics["acc_global"].item(), 4 / 6)
        self.assertAlmostEqual(metrics["acc_end_of_text_padding"].item(), 1 / 2)
        self.assertAlmostEqual(metrics["acc_text_padding"].item(), 1.0)
        self.assertAlmostEqual(metrics["acc_content"].item(), 2 / 3)
        self.assertAlmostEqual(metrics["target_ratio_end_of_text_padding"].item(), 2 / 6)
        self.assertAlmostEqual(metrics["target_ratio_text_padding"].item(), 1 / 6)
        self.assertAlmostEqual(metrics["target_ratio_content"].item(), 3 / 6)
        self.assertAlmostEqual(metrics["prediction_ratio_end_of_text_padding"].item(), 1 / 6)
        self.assertAlmostEqual(metrics["prediction_ratio_text_padding"].item(), 1 / 6)
        self.assertAlmostEqual(metrics["prediction_ratio_content"].item(), 4 / 6)
        self.assertEqual(metrics["content_to_text_padding_rate"].item(), 0.0)
        self.assertEqual(metrics["content_to_end_of_text_padding_rate"].item(), 0.0)
        ratio_sum = sum(
            metrics[key].item()
            for key in (
                "target_ratio_end_of_text_padding",
                "target_ratio_text_padding",
                "target_ratio_content",
            )
        )
        self.assertAlmostEqual(ratio_sum, 1.0)

    def test_masked_positions_do_not_affect_metrics(self) -> None:
        target = torch.tensor([[[0, 3, 10, 0]]])
        prediction = torch.tensor([[[9, 3, 10, 0]]])
        mask = torch.tensor([[[False, True, True, False]]])

        metrics = compute_text_token_metrics(
            prediction,
            target,
            mask,
            text_padding_token_id=3,
            end_of_text_padding_id=0,
        )

        self.assertEqual(metrics["acc_global"].item(), 1.0)
        self.assertTrue(math.isnan(metrics["acc_end_of_text_padding"].item()))
        self.assertEqual(metrics["target_ratio_end_of_text_padding"].item(), 0.0)
        self.assertEqual(metrics["target_ratio_text_padding"].item(), 0.5)
        self.assertEqual(metrics["target_ratio_content"].item(), 0.5)

    def test_empty_valid_mask_returns_nan_not_false_accuracy(self) -> None:
        target = torch.tensor([[[0, 3, 10]]])
        prediction = target.clone()
        mask = torch.zeros_like(target, dtype=torch.bool)

        metrics = compute_text_token_metrics(
            prediction,
            target,
            mask,
            text_padding_token_id=3,
            end_of_text_padding_id=0,
        )

        self.assertTrue(all(math.isnan(value.item()) for value in metrics.values()))

    def test_rejects_ambiguous_ids_and_shape_mismatch(self) -> None:
        target = torch.zeros((1, 1, 2), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            compute_text_token_metrics(
                target,
                target,
                torch.ones_like(target, dtype=torch.bool),
                text_padding_token_id=0,
                end_of_text_padding_id=0,
            )
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_text_token_metrics(
                target[..., :1],
                target,
                torch.ones_like(target, dtype=torch.bool),
                text_padding_token_id=3,
                end_of_text_padding_id=0,
            )


class EffectiveBatchMetricTests(unittest.TestCase):
    def test_counts_are_aggregated_across_every_microbatch(self) -> None:
        stats = new_effective_batch_statistics(torch.device("cpu"))
        update_effective_batch_statistics(
            stats,
            loss=torch.tensor(2.0),
            text_loss=torch.tensor(1.0),
            audio_loss=torch.tensor(3.0),
            pred_text=torch.tensor([[[10]]]),
            text_target=torch.tensor([[[10]]]),
            text_mask=torch.tensor([[[True]]]),
            text_padding_token_id=3,
            end_of_text_padding_id=0,
            pred_audio=torch.tensor([[[9]]]),
            audio_target=torch.tensor([[[8]]]),
            audio_mask=torch.tensor([[[True]]]),
        )
        update_effective_batch_statistics(
            stats,
            loss=torch.tensor(4.0),
            text_loss=torch.tensor(3.0),
            audio_loss=torch.tensor(5.0),
            pred_text=torch.tensor([[[0, 11, 3]]]),
            text_target=torch.tensor([[[10, 11, 3]]]),
            text_mask=torch.tensor([[[True, True, True]]]),
            text_padding_token_id=3,
            end_of_text_padding_id=0,
            pred_audio=torch.tensor([[[1, 2, 0]]]),
            audio_target=torch.tensor([[[1, 9, 0]]]),
            audio_mask=torch.tensor([[[True, True, False]]]),
        )

        metrics = finalize_effective_batch_statistics(stats)

        self.assertEqual(metrics["loss"].item(), 3.0)
        self.assertEqual(metrics["text_loss"].item(), 2.0)
        self.assertEqual(metrics["audio_loss"].item(), 4.0)
        self.assertEqual(metrics["acc_global"].item(), 0.75)
        self.assertEqual(metrics["acc_content"].item(), 2 / 3)
        self.assertEqual(metrics["audio_acc_global"].item(), 1 / 3)
        self.assertEqual(metrics["audio_acc_codebook0"].item(), 1 / 3)

    def test_gate_is_averaged_over_microbatches(self) -> None:
        stats = new_effective_batch_statistics(torch.device("cpu"))
        common = dict(
            loss=torch.tensor(1.0),
            text_loss=torch.tensor(1.0),
            audio_loss=torch.tensor(0.0),
            pred_text=torch.tensor([[[1]]]),
            text_target=torch.tensor([[[1]]]),
            text_mask=torch.tensor([[[True]]]),
            text_padding_token_id=3,
            end_of_text_padding_id=0,
            pred_audio=None,
            audio_target=torch.empty((1, 0, 1), dtype=torch.long),
            audio_mask=torch.empty((1, 0, 1), dtype=torch.bool),
        )
        update_effective_batch_statistics(stats, gate=torch.tensor(0.2), **common)
        update_effective_batch_statistics(stats, gate=torch.tensor(0.6), **common)

        metrics = finalize_effective_batch_statistics(stats)
        self.assertAlmostEqual(metrics["gate_mean"].item(), 0.4)

    def test_speechless_microbatch_contributes_text_but_no_audio(self) -> None:
        stats = new_effective_batch_statistics(torch.device("cpu"))
        update_effective_batch_statistics(
            stats,
            loss=torch.tensor(1.0),
            text_loss=torch.tensor(0.5),
            audio_loss=torch.tensor(0.0),
            pred_text=torch.tensor([[[7]]]),
            text_target=torch.tensor([[[7]]]),
            text_mask=torch.tensor([[[True]]]),
            text_padding_token_id=3,
            end_of_text_padding_id=0,
            pred_audio=None,
            audio_target=torch.empty((1, 0, 1), dtype=torch.long),
            audio_mask=torch.empty((1, 0, 1), dtype=torch.bool),
        )
        metrics = finalize_effective_batch_statistics(stats)
        self.assertEqual(metrics["acc_global"].item(), 1.0)
        self.assertTrue(math.isnan(metrics["audio_acc_global"].item()))


class AccumulationTests(unittest.TestCase):
    def test_target_batch_must_be_exact(self) -> None:
        self.assertEqual(gradient_accumulation_steps(16, 1, 2), 8)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            gradient_accumulation_steps(15, 2, 2)

    def test_accelerate_backward_performs_the_only_loss_division(self) -> None:
        accelerator = Accelerator(cpu=True)
        # Match train.py: world size is known only after Accelerator creation.
        accelerator.gradient_accumulation_steps = 2
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model = accelerator.prepare(model)
        for value in (1.0, 3.0):
            with accelerator.accumulate(model):
                accelerator.backward(model(torch.tensor([[value]])).sum())
        self.assertAlmostEqual(model.weight.grad.item(), 2.0)

    def test_depformer_plan_preserves_trainable_graph(self) -> None:
        self.assertEqual(
            audio_forward_plan(
                depformer_trainable=True, speechless_batch=True, dep_q=8
            ),
            (8, True),
        )
        self.assertEqual(
            audio_forward_plan(
                depformer_trainable=False, speechless_batch=False, dep_q=8
            ),
            (2, True),
        )
        self.assertEqual(
            audio_forward_plan(
                depformer_trainable=False, speechless_batch=True, dep_q=8
            ),
            (2, False),
        )


class MandatoryImageConditionTests(unittest.TestCase):
    @staticmethod
    def _condition(value: float) -> ConditionAttributes:
        condition = ConditionAttributes()
        condition.tensor["image"] = TensorCondition(
            tensor=torch.full((1, 3, 2, 2), value),
            mask=torch.ones((1, 1), dtype=torch.bool),
        )
        return condition

    def test_complete_image_conditions_form_exact_code_batch(self) -> None:
        image_batch = require_batched_image_conditions(
            [self._condition(1.0), self._condition(2.0)],
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(image_batch.shape), (2, 3, 2, 2))
        self.assertEqual(image_batch[:, 0, 0, 0].tolist(), [1.0, 2.0])

    def test_missing_or_misaligned_image_conditions_fail_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "to be a list"):
            require_batched_image_conditions(
                None, batch_size=1, device=torch.device("cpu")
            )
        with self.assertRaisesRegex(RuntimeError, "batch length mismatch"):
            require_batched_image_conditions(
                [self._condition(1.0)],
                batch_size=2,
                device=torch.device("cpu"),
            )
        missing = ConditionAttributes()
        with self.assertRaisesRegex(RuntimeError, "missing it"):
            require_batched_image_conditions(
                [missing], batch_size=1, device=torch.device("cpu")
            )


class GateAggregationTests(unittest.TestCase):
    def test_transformer_returns_tensor_mean_for_last_ten_real_gates(self) -> None:
        class GateLayer(torch.nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(value))

            def forward(self, x, *args, **kwargs):
                del args, kwargs
                return x, self.value

        transformer = Transformer(
            d_model=2,
            num_heads=1,
            num_layers=0,
            positional_embedding="none",
        )
        transformer.layers = torch.nn.ModuleList(
            [GateLayer(float(index)) for index in range(12)]
        )
        _, gate = transformer(torch.zeros((1, 2, 2)))

        self.assertIsInstance(gate, torch.Tensor)
        self.assertFalse(gate.requires_grad)
        self.assertAlmostEqual(gate.detach().item(), 6.5)
        self.assertIsNone(transformer.layers[0].value.grad)
        self.assertIsNone(transformer.layers[1].value.grad)
        self.assertIsNone(transformer.layers[2].value.grad)


class MetricsWriterTests(unittest.TestCase):
    @patch("jmoshivis.tools.writer.wandb.log")
    def test_legacy_positional_audio_arguments_keep_their_meaning(self, log) -> None:
        writer = object.__new__(WandBMetricsWriter)
        writer.log_step(7, 1.0, 0.5, 0.5, 0.6, 0.4, 0.8, 0.7, 0.9)

        data = log.call_args.args[0]
        self.assertEqual(data["train/audio_acc_global"], 0.7)
        self.assertEqual(data["train/audio_acc_codebook0"], 0.9)
        self.assertNotIn("train/text_acc_end_of_text_padding", data)

    @patch("jmoshivis.tools.writer.wandb.log")
    def test_writer_keeps_old_keys_and_emits_explicit_categories(self, log) -> None:
        # Avoid wandb.init: log_step itself has no instance state dependency.
        writer = object.__new__(WandBMetricsWriter)
        writer.log_step(
            step=7,
            loss=1.0,
            text_loss=0.5,
            audio_loss=0.5,
            text_acc_global=0.6,
            text_acc_content=0.4,
            text_acc_pad=0.8,
            text_acc_end_of_text_padding=0.9,
            text_target_ratio_content=0.2,
            text_target_ratio_pad=0.3,
            text_target_ratio_end_of_text_padding=0.5,
            text_prediction_ratio_content=0.4,
            text_prediction_ratio_pad=0.35,
            text_prediction_ratio_end_of_text_padding=0.25,
            text_content_to_pad_rate=0.12,
            text_content_to_end_of_text_padding_rate=0.18,
            gate_mean=0.42,
        )

        data = log.call_args.args[0]
        self.assertEqual(log.call_args.kwargs["step"], 7)
        self.assertEqual(data["train/text_acc_pad"], 0.8)
        self.assertEqual(data["train/text_acc_text_padding"], 0.8)
        self.assertEqual(data["train/text_acc_end_of_text_padding"], 0.9)
        self.assertEqual(data["train/text_target_ratio_content"], 0.2)
        self.assertEqual(data["train/text_target_ratio_pad"], 0.3)
        self.assertEqual(data["train/text_target_ratio_text_padding"], 0.3)
        self.assertEqual(data["train/text_target_ratio_end_of_text_padding"], 0.5)
        self.assertEqual(data["train/text_prediction_ratio_content"], 0.4)
        self.assertEqual(data["train/gate_mean"], 0.42)
        self.assertEqual(data["train/text_prediction_ratio_pad"], 0.35)
        self.assertEqual(data["train/text_prediction_ratio_text_padding"], 0.35)
        self.assertEqual(data["train/text_prediction_ratio_end_of_text_padding"], 0.25)
        self.assertEqual(data["train/text_content_to_pad_rate"], 0.12)
        self.assertEqual(data["train/text_content_to_text_padding_rate"], 0.12)
        self.assertEqual(data["train/text_content_to_end_of_text_padding_rate"], 0.18)


if __name__ == "__main__":
    unittest.main()
