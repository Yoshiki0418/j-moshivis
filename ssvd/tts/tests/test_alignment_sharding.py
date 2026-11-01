import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotate_alignments import completed_dialogues


class AlignmentShardingTests(unittest.TestCase):
    def test_live_growth_preserves_ownership_and_excludes_partial_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for uid in ("b", "d", "f"):
                p = root / uid
                p.mkdir()
                (p / "stereo_dialogue.wav").touch()
                (p / "dialogue.json").touch()
            before = [set(completed_dialogues([root], i, 6, True)) for i in range(6)]
            for uid in ("a", "c", "partial"):
                p = root / uid
                p.mkdir()
                (p / "stereo_dialogue.wav").touch()
                if uid != "partial":
                    (p / "dialogue.json").touch()
            after = [set(completed_dialogues([root], i, 6, True)) for i in range(6)]
            self.assertEqual(sum(map(len, after)), 5)
            self.assertEqual(len(set().union(*after)), 5)
            for old, new in zip(before, after):
                self.assertLessEqual(old, new)

    def test_sources_interleave_and_duplicate_roots_are_not_reprocessed(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("daily", "ocr")]
            for root in roots:
                for uid in ("a", "b"):
                    p = root / uid
                    p.mkdir(parents=True)
                    (p / "stereo_dialogue.wav").touch()
                    (p / "dialogue.json").touch()
            rows = completed_dialogues([*roots, roots[0]], 0, 1)
            self.assertEqual([p.parent.name for p in rows], ["daily", "ocr", "daily", "ocr"])

    def test_invalid_shards_fail(self):
        for shard, count in ((0, 0), (-1, 2), (2, 2)):
            with self.assertRaises(ValueError):
                completed_dialogues([], shard, count, True)
