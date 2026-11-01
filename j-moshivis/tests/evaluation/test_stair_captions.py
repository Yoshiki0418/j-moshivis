from __future__ import annotations

import unittest
import importlib.util

from jmoshivis.evaluation.cider_ja import japanese_cider_score


@unittest.skipUnless(
    importlib.util.find_spec("fugashi") and importlib.util.find_spec("pycocoevalcap"),
    "Japanese CIDEr dependencies are installed only in the evaluation environment",
)
class JapaneseCaptionMetricTests(unittest.TestCase):
    def test_japanese_cider_rewards_matching_scene(self) -> None:
        references = [
            ["赤い車が道路を走っている", "道路を走る赤い自動車", "車が街の道を走る", "赤い車が進んでいる", "自動車が道路にいる"],
            ["白い犬が芝生で走る", "芝生を走る白い犬", "犬が緑の芝生にいる", "白い犬が外で走る", "草の上を走る犬"],
            ["青い傘を持つ女性", "女性が青い傘を持っている", "青い傘の下に女性がいる", "女性が傘を差している", "女性と青い傘が見える"],
        ]
        matching = ["赤い車が道路を走っている", "白い犬が芝生で走る", "青い傘を持つ女性"]
        mismatched = [matching[1], matching[2], matching[0]]
        good = japanese_cider_score(matching, references)
        bad = japanese_cider_score(mismatched, references)
        self.assertGreater(good.score, bad.score)
        self.assertEqual(len(good.per_example), 3)


if __name__ == "__main__":
    unittest.main()
