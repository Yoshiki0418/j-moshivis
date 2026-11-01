import unittest

import torch
from torch.nn import functional as F

from jmoshivis.loss import compute_loss_with_mask


class TextTokenWeightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [[[[3.0, 0.0, 0.0, 0.0],
               [0.0, 3.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 3.0]]]],
            dtype=torch.float32,
        )
        self.target = torch.tensor([[[0, 0, 3]]])
        self.mask = torch.ones_like(self.target, dtype=torch.bool)

    def test_text_token_weights_are_applied_independently(self) -> None:
        loss = compute_loss_with_mask(
            self.logits,
            self.target,
            self.mask,
            mode="text",
            text_token_weights={3: 0.5, 0: 1.0},
        )

        per_token = F.cross_entropy(
            self.logits.reshape(-1, self.logits.size(-1)),
            self.target.reshape(-1),
            reduction="none",
        )
        expected = (per_token[0] + per_token[1] + 0.5 * per_token[2]) / 2.5
        torch.testing.assert_close(loss, expected)

    def test_legacy_shared_padding_weight_is_unchanged(self) -> None:
        legacy = compute_loss_with_mask(
            self.logits,
            self.target,
            self.mask,
            mode="text",
            text_padding_weight=0.5,
            text_padding_ids={0, 3},
        )
        separated_but_equal = compute_loss_with_mask(
            self.logits,
            self.target,
            self.mask,
            mode="text",
            text_token_weights={0: 0.5, 3: 0.5},
        )

        torch.testing.assert_close(legacy, separated_but_equal)


if __name__ == "__main__":
    unittest.main()
