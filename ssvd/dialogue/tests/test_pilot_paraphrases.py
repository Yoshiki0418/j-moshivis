"""Opt-in wording checks on synthetic Japanese visual conversations."""

import unittest

from ssvd.dialogue.regeneration_grounding import mentions_target, task_problem


class PilotParaphraseTests(unittest.TestCase):
    def test_particle_insertion_and_extra_modifiers(self):
        for target, utterance in [
            ("案内板中央の白い文字", "案内板の中央には白い文字があります。"),
            ("鳥の形をした像", "道端には鳥の形をした青い像がある。"),
            ("駅上の横長看板", "駅の上には横長の看板が見える。"),
            ("売場の縦長看板", "売場の中央には縦長の看板が立っています。"),
        ]:
            with self.subTest(target=target):
                self.assertTrue(mentions_target(target, utterance, allow_paraphrase=True))
        self.assertFalse(mentions_target("案内板中央の白い文字", "案内板の中央には白い文字があります。"))

    def test_colour_or_position_on_another_object_does_not_match(self):
        for wrong in ("店の前には白い看板があります。", "赤い花が咲いていて、その横に白い看板があります。",
                      "赤い車。看板があります。", "赤い丸い看板は見えません。"):
            self.assertFalse(mentions_target("赤い丸い看板", wrong, allow_paraphrase=True))
        self.assertFalse(mentions_target("左の赤い看板", "右の赤い看板が見える。", allow_paraphrase=True))

    def test_introducing_an_ocr_object_before_asking_for_an_attribute(self):
        target = "赤い丸い標識の漢字"
        utterance = "道に赤い丸い標識があります。"
        self.assertFalse(mentions_target(target, utterance, allow_paraphrase=True))
        self.assertTrue(mentions_target(target, utterance, allow_paraphrase=True, for_ocr=True))
        self.assertFalse(mentions_target(target, "道に青い丸い標識があります。", allow_paraphrase=True, for_ocr=True))
        self.assertTrue(mentions_target("壁の映画ポスターの題名", "右の壁には映画のポスターが掛かっている。",
                                        allow_paraphrase=True, for_ocr=True))
        self.assertFalse(mentions_target("壁の映画ポスターの題名", "壁には絵がある。床には映画のポスターがある。",
                                         allow_paraphrase=True, for_ocr=True))

    def test_abbreviated_single_previous_referent_keeps_quantity(self):
        task = {"kind":"counting", "mode":"exact", "target":"正面の大きな店の看板", "count":3}
        previous = "正面には大きな店の看板があるね。"
        answer = "正面に大きな看板が三つ見えます。"
        def check(q="看板はいくつ見えますか？", a=answer, p=previous):
            return task_problem("counting", task, q, a, p, allow_paraphrase=True)
        self.assertIsNone(check())
        self.assertIsNone(check(a="三つです。"))
        self.assertIsNotNone(task_problem("counting", task, "看板はいくつ見えますか？", answer, previous))
        self.assertIsNotNone(check(a=answer.replace("三つ", "四つ")))
        self.assertIsNotNone(check(a="奥の看板が三つあるよ。"))
        self.assertIsNotNone(check(a="看板の横に犬が三匹います。"))
        self.assertIsNotNone(check(a="看板の写真が三枚飾られているよ。"))
        self.assertIsNotNone(check(a="三匹、犬がいるよ。"))
        self.assertIsNotNone(check(q="右の看板はいくつ見えますか？"))
        self.assertIsNotNone(check(p=previous + "反対側にも看板があるよ。"))

    def test_combined_kanji_and_romanized_label(self):
        q = "標識には何と書いてある？"
        task = dict(kind="ocr", mode="exact", target="案内標識", written="高崎 Takasaki", reading="たかさき")
        def check(answer, evidence=task):
            return task_problem("ocr", evidence, q, answer, allow_paraphrase=True)
        answer = "「高崎」って書いてあるよ。"
        self.assertIsNone(check(answer))
        self.assertIsNotNone(task_problem("ocr", task, q, answer))
        self.assertIsNone(check("タカサキと書いてあります。"))
        for wrong in ("「高崎駅」って書いてあるよ。", "「高崎」とは書いてないよ。", "Takasakiです。"):
            self.assertIsNotNone(check(wrong))
        self.assertIsNotNone(check(answer, dict(task, written="高崎 HOTEL")))
        self.assertIsNotNone(check(answer, dict(task, mode="partial", reason="端が切れている")))


if __name__ == "__main__":
    unittest.main()
