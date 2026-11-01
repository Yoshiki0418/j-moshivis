from __future__ import annotations

import unittest

import torch

from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.loss import compute_loss_with_mask


class DepformerTrainingTests(unittest.TestCase):
    @staticmethod
    def _tiny_model() -> MoshiVis:
        torch.random.default_generator.manual_seed(7)
        return MoshiVis(
            dim=8,
            num_heads=2,
            num_layers=1,
            context=16,
            cross_attention=False,
            positional_embedding="none",
            n_q=4,
            dep_q=4,
            audio_card=9,
            depformer=True,
            depformer_multi_linear=True,
            depformer_pos_emb="none",
            depformer_dim=8,
            depformer_dim_feedforward=16,
            depformer_num_layers=1,
            depformer_num_heads=2,
            depformer_weights_per_step=True,
            depformer_context=4,
            delays=[0, 0, 0, 0, 0],
            text_card=13,
            padding_token_id=3,
            dtype=torch.float32,
        ).eval()

    def test_two_codebook_prefix_matches_full_forward_exactly(self) -> None:
        model = self._tiny_model()
        sequence = torch.empty((2, 5, 3), dtype=torch.long)
        sequence[:, 0].random_(0, 13)
        sequence[:, 1:].random_(0, 9)
        transformer_out = torch.randn(2, 3, 8)

        with torch.no_grad():
            full = model.forward_depformer_training(sequence, transformer_out)
            prefix = model.forward_depformer_training(
                sequence, transformer_out, num_audio_codebooks=2
            )

        self.assertEqual(prefix.shape, (2, 2, 3, 9))
        # Different attention sequence lengths can change floating-point
        # reduction order while preserving the causal prefix mathematically.
        torch.testing.assert_close(prefix, full[:, :2], rtol=1e-6, atol=1e-6)

    def test_prefix_rejects_noncausal_depformer(self) -> None:
        model = self._tiny_model()
        for layer in model.depformer.layers:
            layer.causal = False
        sequence = torch.zeros((1, 5, 2), dtype=torch.long)
        transformer_out = torch.randn(1, 2, 8)
        with self.assertRaisesRegex(RuntimeError, "non-causal"):
            model.forward_depformer_training(
                sequence, transformer_out, num_audio_codebooks=2
            )

    def test_empty_audio_mask_keeps_zero_gradient_graph_for_ddp(self) -> None:
        logits = torch.randn(1, 2, 3, 9, requires_grad=True)
        target = torch.zeros((1, 2, 3), dtype=torch.long)
        mask = torch.zeros_like(target, dtype=torch.bool)
        loss = compute_loss_with_mask(logits, target, mask, mode="audio")
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_masked_nan_logits_never_enter_cross_entropy(self) -> None:
        logits = torch.tensor(
            [[[[2.0, -1.0], [float("nan"), float("nan")]]]],
            requires_grad=True,
        )
        target = torch.tensor([[[0, 0]]])
        mask = torch.tensor([[[True, False]]])
        loss = compute_loss_with_mask(logits, target, mask, mode="audio")
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 0, 1].abs().sum().item(), 0.0)

    def test_full_forward_marks_later_trainable_codebook_branches_used(self) -> None:
        model = self._tiny_model()
        sequence = torch.empty((1, 5, 2), dtype=torch.long)
        sequence[:, 0].random_(0, 13)
        sequence[:, 1:].random_(0, 9)
        transformer_out = torch.randn(1, 2, 8)
        full = model.forward_depformer_training(sequence, transformer_out)
        full[:, :2].sum().backward()

        late_linear_grad = model.audio_linears[-1].weight.grad
        late_embedding_grad = model.depformer_emb[-1].weight.grad
        self.assertIsNotNone(late_linear_grad)
        self.assertIsNotNone(late_embedding_grad)
        self.assertEqual(late_linear_grad.abs().sum().item(), 0.0)
        self.assertEqual(late_embedding_grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
