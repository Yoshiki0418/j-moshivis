from __future__ import annotations

import math
import unittest

import torch

from jmoshivis.evaluation.text_diagnostics import (
    diagnose_text_logits,
    require_finite_scalar,
    require_finite_tensor,
    summarize_text_decisions,
)


class TextDecisionDiagnosticTests(unittest.TestCase):
    def test_distinguishes_alignment_end_padding_and_lexical_tokens(self) -> None:
        logits = torch.zeros(1, 1, 1, 10)
        logits[..., 3] = 5.0
        logits[..., 8] = 2.0

        result = diagnose_text_logits(logits, frame=17)

        self.assertEqual(result["decision_frame"], 17)
        self.assertEqual(result["argmax_id"], 3)
        self.assertEqual(result["argmax_class"], "alignment_pad")
        self.assertEqual(result["best_lexical_id"], 8)
        self.assertAlmostEqual(
            result["logit_margin_vs_best_lexical"]["alignment_pad_id_3"],
            3.0,
        )
        expected_pad_probability = math.exp(5.0) / (
            math.exp(5.0) + math.exp(2.0) + 8.0
        )
        self.assertAlmostEqual(
            result["probability"]["alignment_pad_id_3"],
            expected_pad_probability,
            places=6,
        )

    def test_summarizes_response_window_and_first_lexical_delay(self) -> None:
        pad_logits = torch.zeros(1, 1, 1, 10)
        pad_logits[..., 0] = 4.0
        lexical_logits = torch.zeros(1, 1, 1, 10)
        lexical_logits[..., 9] = 4.0
        decisions = [
            diagnose_text_logits(pad_logits, frame=20),
            diagnose_text_logits(lexical_logits, frame=21),
        ]

        summary = summarize_text_decisions(
            decisions,
            question_end_frame=20,
            frame_rate=12.5,
        )

        self.assertEqual(summary["num_response_window_frames"], 2)
        self.assertEqual(summary["argmax_count"]["end_pad"], 1)
        self.assertEqual(summary["argmax_count"]["lexical"], 1)
        self.assertEqual(summary["argmax_rate"]["end_pad"], 0.5)
        self.assertEqual(summary["first_lexical_argmax_frame"], 21)
        self.assertAlmostEqual(summary["first_lexical_argmax_delay_sec"], 0.08)

    def test_rejects_unexpected_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected text logits"):
            diagnose_text_logits(torch.zeros(1, 10), frame=0)

    def test_rejects_nonfinite_logits_and_scalars(self) -> None:
        logits = torch.zeros((1, 1, 1, 12))
        logits[..., 8] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            diagnose_text_logits(logits, frame=7)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            require_finite_scalar(float("inf"), name="gate")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            require_finite_tensor(logits, name="text logits")


if __name__ == "__main__":
    unittest.main()
