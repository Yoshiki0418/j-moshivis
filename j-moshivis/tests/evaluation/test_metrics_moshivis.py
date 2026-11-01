from __future__ import annotations

import unittest
from unittest.mock import patch

from jmoshivis.evaluation.metrics_moshivis import (
    cider_score,
    normalize_ocr_vqa_answer,
    normalize_vqav2_answer,
    official_coco_cider_score,
    ocr_vqa_accuracy,
    ocr_vqa_exact_match,
    tokenize_coco_caption,
    vqav2_accuracy,
    vqav2_corpus_accuracy,
)


class OcrVqaMetricTests(unittest.TestCase):
    def test_paligemma_normalization_is_lowercase_only(self) -> None:
        self.assertEqual(normalize_ocr_vqa_answer("Red Bull!"), "red bull!")
        self.assertEqual(ocr_vqa_exact_match("SEVEN BANK", ["seven bank"]), 1.0)
        # OCR-VQA is deliberately sensitive to punctuation and answer prose.
        self.assertEqual(ocr_vqa_exact_match("seven bank.", ["seven bank"]), 0.0)
        self.assertEqual(ocr_vqa_exact_match(" seven bank ", ["seven bank"]), 0.0)
        self.assertEqual(ocr_vqa_exact_match("It says seven bank", ["seven bank"]), 0.0)

    def test_corpus_accuracy_and_validation(self) -> None:
        self.assertEqual(
            ocr_vqa_accuracy(["ABC", "wrong"], [["abc"], "answer"]),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            ocr_vqa_accuracy(["a"], [])


class VqaV2MetricTests(unittest.TestCase):
    def test_official_normalization(self) -> None:
        self.assertEqual(normalize_vqav2_answer("The two cats!"), "2 cats")
        self.assertEqual(normalize_vqav2_answer("1,000.5"), "1000.5")
        self.assertEqual(normalize_vqav2_answer("isnt"), "isn't")

    def test_leave_one_out_scores_not_count_over_three_shortcut(self) -> None:
        # The extra distinct answer activates the official normalizer.
        for number_of_matches, expected in ((1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0)):
            answers = ["cat"] * number_of_matches + ["dog"] * (10 - number_of_matches)
            self.assertAlmostEqual(vqav2_accuracy("cat", answers), expected)

    def test_digit_article_normalization_and_corpus_mean(self) -> None:
        normalized_hit = ["two"] * 4 + ["three"] * 6
        miss = ["red"] * 5 + ["blue"] * 5
        self.assertEqual(vqav2_accuracy("2", normalized_hit), 1.0)
        self.assertEqual(
            vqav2_corpus_accuracy(["2", "green"], [normalized_hit, miss]),
            0.5,
        )

    def test_unanimous_answers_preserve_official_raw_comparison(self) -> None:
        # The official script skips punctuation/case normalization when every
        # annotator supplied the identical raw answer.
        self.assertEqual(vqav2_accuracy("Cat", ["cat"] * 10), 0.0)

    def test_requires_ten_human_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            vqav2_accuracy("cat", ["cat"])


class CiderMetricTests(unittest.TestCase):
    def test_dependency_free_tokenizer(self) -> None:
        self.assertEqual(
            tokenize_coco_caption("A Dog's ball, isn't red!"),
            "a dog 's ball is n't red",
        )

    def test_identical_pretokenized_captions_reach_ten(self) -> None:
        # At least two images are needed for non-zero IDF in the official
        # scorer. Duplicate references are intentional: an exact match then
        # has cosine similarity one against every reference at all 4 orders.
        predictions = ["a red car drives fast", "two blue birds fly high"]
        references = [
            [predictions[0], predictions[0]],
            [predictions[1], predictions[1]],
        ]
        result = cider_score(predictions, references, tokenized=True)
        self.assertAlmostEqual(result.score, 10.0)
        self.assertEqual(result.per_example, (10.0, 10.0))
        self.assertAlmostEqual(result.score_x100, 1000.0)
        self.assertEqual(result.tokenization, "pretokenized_ptb")

    def test_no_shared_ngrams_scores_zero(self) -> None:
        predictions = ["red car drives quickly", "blue bird flies above"]
        references = [
            ["green frog sits quietly"],
            ["white horse walks below"],
        ]
        result = cider_score(predictions, references, tokenized=True)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.per_example, (0.0, 0.0))

    def test_corpus_is_deterministic_and_inputs_are_validated(self) -> None:
        predictions = ["a cat sits on mat", "a dog runs in park"]
        references = [
            ["a cat sits on mat", "cat resting on the mat"],
            ["a dog runs in park", "dog running through a park"],
        ]
        self.assertEqual(
            cider_score(predictions, references),
            cider_score(predictions, references),
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            cider_score(predictions, references[:1])
        with self.assertRaisesRegex(ValueError, "at least one reference"):
            cider_score(["caption"], [[]])

    def test_official_path_has_actionable_optional_dependency_error(self) -> None:
        with patch(
            "jmoshivis.evaluation.metrics_moshivis.shutil.which",
            return_value="/usr/bin/java",
        ):
            with patch.dict("sys.modules", {"pycocoevalcap": None}):
                with self.assertRaisesRegex(RuntimeError, "pycocoevalcap"):
                    official_coco_cider_score(
                        ["a generated caption"], [["a reference caption"]]
                    )

    def test_official_path_has_actionable_java_error(self) -> None:
        with patch(
            "jmoshivis.evaluation.metrics_moshivis.shutil.which",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "Java runtime"):
                official_coco_cider_score(
                    ["a generated caption"], [["a reference caption"]]
                )


if __name__ == "__main__":
    unittest.main()
