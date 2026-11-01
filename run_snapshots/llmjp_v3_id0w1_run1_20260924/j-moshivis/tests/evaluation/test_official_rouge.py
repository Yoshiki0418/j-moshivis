from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from jmoshivis.evaluation.official_rouge_ja import official_rouge_ja


HAS_OFFICIAL_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("emoji", "fugashi", "neologdn", "rouge_score")
)


@unittest.skipUnless(HAS_OFFICIAL_DEPS, "official Japanese ROUGE dependencies not installed")
class OfficialRougeTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_restores_numpy_rng(self) -> None:
        references = ["赤い車です", "猫が二匹います", "東京タワー"]
        predictions = ["赤い車です", "猫が2匹います", "東京タワーです"]

        np.random.seed(123)
        before = np.random.get_state()
        first = official_rouge_ja(references, predictions)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

        np.random.seed(999)
        second = official_rouge_ja(references, predictions)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
