"""Offline consistency checks for honest answers and multi-task image observations."""

import unittest

from ssvd.dialogue.regeneration_grounding import (
    contains_grounded_answer, explicit_question_kinds, grounding_problem, task_problem,
)


class AdaptiveGroundingTests(unittest.TestCase):
    def count(self, mode, n=None):
        task = dict(kind="counting", target="自転車", mode=mode)
        if mode != "exact":
            task["reason"] = "奥のほうは車輪が重なっている。"
        if n is not None:
            task["count"] = n
        return task

    def ocr(self, mode, written="A7", reading="エーなな"):
        task = dict(kind="ocr", target="看板", mode=mode)
        if mode != "exact":
            task["reason"] = "右端が柱に隠れている。"
        if mode != "unknown":
            task.update(written=written, reading=reading)
        return task

    def inspect(self, tasks):
        return grounding_problem("counting", dict(status="ok", scene="店先に自転車と看板がある。", tasks=tasks))

    def test_grounding_can_choose_either_both_or_neither(self):
        for tasks in ([], [self.ocr("partial")], [self.count("approximate", 12), self.ocr("unknown")]):
            with self.subTest(tasks=tasks):
                self.assertIsNone(self.inspect(tasks))
        self.assertIsNotNone(self.inspect([self.ocr("partial"), self.ocr("unknown")]))
        self.assertIsNotNone(self.inspect([dict(self.count("unknown"), count=7)]))
        self.assertIsNotNone(self.inspect([dict(self.ocr("unknown"), written="AB")]))
        self.assertIsNotNone(self.inspect([dict(self.count("lower_bound", 7), reason="")]))
        self.assertIsNotNone(self.inspect([self.ocr("partial", "AB", "abracadabra")]))
        self.assertFalse(contains_grounded_answer("ocr", self.ocr("unknown"), "看板の近くに人がいる。"))

    def test_count_strength_and_number_follow_observation(self):
        q = "自転車、何台見える？"
        for mode, n, answer, accepted in [
            ("exact", 1, "一台だよ。", True),
            ("exact", 105, "百五台です。", True),
            ("approximate", 12, "奥も含めると十二台くらいに見えるね。", True),
            ("approximate", 12, "ざっと見て12台かな。", True),
            ("approximate", 12, "12台です。", False),
            ("lower_bound", 4, "四台以上はありそうだよ。", True),
            ("lower_bound", 4, "少なくとも4台見える。奥までは分からないな。", True),
            ("lower_bound", 4, "少なくとも四台見える。全部で何台かは分からない。", True),
            ("lower_bound", 4, "はっきり見えるのは四台。", True),
            ("lower_bound", 4, "全部で四台です。", False),
            ("lower_bound", 4, "少なくとも四台で、全部で四台です。", False),
            ("lower_bound", 4, "少なくとも五台です。", False),
            ("unknown", None, "重なっちゃって、何台かは分かんないな。", True),
            ("unknown", None, "ちょっと数えきれません。", True),
            ("unknown", None, "はっきり分からないけど三台かな。", False),
        ]:
            with self.subTest(mode=mode, answer=answer):
                self.assertEqual(task_problem("counting", self.count(mode, n), q, answer) is None, accepted)

    def test_ocr_exact_partial_unknown(self):
        q = "看板には何て書いてある？"
        for mode, answer, accepted in [
            ("exact", "エーなな、と書いてあるよ。", True),
            ("partial", "最初はエーなな。先は柱で隠れてるね。", True),
            ("partial", "『エーなな』の部分だけ読めるよ。", True),
            ("partial", "エーななと書いてあるよ。", False),
            ("partial", "『エーななホテル』までは読めるよ。", False),
            ("unknown", "小さすぎて文字までは読めないよ。", True),
            ("unknown", "ごめん、私には分からない。", True),
            ("unknown", "よく分からないけど、『駅』って読めそう。", False),
        ]:
            with self.subTest(mode=mode, answer=answer):
                self.assertEqual(task_problem("ocr", self.ocr(mode), q, answer) is None, accepted)

    def test_question_types_for_unscheduled_turns(self):
        self.assertEqual(explicit_question_kinds("そこに何台ある？"), {"counting"})
        self.assertEqual(explicit_question_kinds("あの看板には何て書いてある？"), {"ocr"})
        self.assertEqual(explicit_question_kinds("何人か集まってるんだ。どんな場所？"), set())
        self.assertEqual(explicit_question_kinds("台数を教えて。"), {"counting"})
        self.assertEqual(explicit_question_kinds("看板の電話番号は？"), {"ocr"})

    def test_visible_only_or_unreadable_can_be_conveyed_colloquially(self):
        q = "自転車、何台見える？"
        task = self.count("lower_bound", 3)
        for answer in ("三台が見えていて、奥は数えきれません。", "見えてるだけで三台かな。"):
            self.assertIsNone(task_problem("counting", task, q, answer))
        for answer in ("三台以上はいそうにない。", "三台以上は見当たりません。", "三台以上はなさそう。"):
            self.assertIsNotNone(task_problem("counting", task, q, answer))
        self.assertIsNotNone(task_problem("counting", self.count("unknown"), q, "十数台かな。正確にはわからない。"))
        unknown = self.ocr("unknown")
        for answer in ("遠くて文字がつぶれてるよ。", "ちょっと読めそうにないね。"):
            self.assertIsNone(task_problem("ocr", unknown, "看板、読める？", answer))
        self.assertIsNotNone(task_problem("ocr", unknown, "看板、読める？", "きっと中央駅だよ。あとは読めないな。"))

    def test_partial_does_not_support_completing_a_fragment(self):
        partial = self.ocr("partial", "駅", "えき")
        self.assertFalse(contains_grounded_answer("ocr", partial, "駅前に看板があるよ。"))
        self.assertTrue(contains_grounded_answer("ocr", partial, "『駅』と書いてあるよ。"))
        self.assertIsNotNone(task_problem("ocr", partial, "看板には何て書いてある？",
                                          "駅前商店街と書いてあります。先頭だけ読めます。"))
        self.assertFalse(contains_grounded_answer("counting", self.count("exact", 3), "三番って書いてあるよ。"))
