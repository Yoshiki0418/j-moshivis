from __future__ import annotations

import sys
import unittest
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
if str(TTS_DIR) not in sys.path:
    sys.path.insert(0, str(TTS_DIR))

from dataset_keys import (  # noqa: E402
    is_eval_group,
    speech_sample_uid,
    split_group_key,
)


class SpeechSampleUidTest(unittest.TestCase):
    def test_normal_mode_retains_legacy_uid(self):
        self.assertEqual("image:123", speech_sample_uid({"uid": "image:123"}))
        self.assertEqual(
            "image:123",
            speech_sample_uid({"uid": "image:123", "task_mode": "normal"}),
        )

    def test_task_variants_have_distinct_storage_ids(self):
        source = {"uid": "image:123"}
        counting = speech_sample_uid({**source, "task_mode": "counting"})
        ocr = speech_sample_uid({**source, "task_mode": "ocr"})
        self.assertEqual("image:123__task_counting", counting)
        self.assertEqual("image:123__task_ocr", ocr)
        self.assertNotEqual(counting, ocr)


class SplitGroupKeyTest(unittest.TestCase):
    def test_explicit_group_key_takes_precedence(self):
        metadata = {
            "uid": "stored__task_counting",
            "source_uid": "source:123",
            "split_group_key": "dataset:original-123",
        }
        self.assertEqual("dataset:original-123", split_group_key(metadata))

    def test_source_uid_and_uid_are_backward_compatible_fallbacks(self):
        self.assertEqual(
            "source:123",
            split_group_key({"uid": "stored", "source_uid": "source:123"}),
        )
        self.assertEqual("legacy:123", split_group_key({"uid": "legacy:123"}))

    def test_all_task_variants_share_one_split_decision(self):
        seed = "test-seed"
        ratio = 0.2
        normal = {"uid": "source:123", "split_group_key": "source:123"}
        counting = {
            "uid": "source:123__task_counting",
            "source_uid": "source:123",
            "split_group_key": "source:123",
        }
        ocr = {
            "uid": "source:123__task_ocr",
            "source_uid": "source:123",
            "split_group_key": "source:123",
        }
        decisions = {
            is_eval_group(split_group_key(row), ratio, seed)
            for row in (normal, counting, ocr)
        }
        self.assertEqual(1, len(decisions))


class EvalAssignmentTest(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        args = ("source:123", 0.2, "stable-seed")
        self.assertEqual(is_eval_group(*args), is_eval_group(*args))

    def test_zero_ratio_never_selects_eval(self):
        self.assertFalse(is_eval_group("source:123", 0.0, "stable-seed"))

    def test_invalid_ratio_is_rejected(self):
        for ratio in (-0.01, 1.0, 1.01):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    is_eval_group("source:123", ratio, "stable-seed")


if __name__ == "__main__":
    unittest.main()
