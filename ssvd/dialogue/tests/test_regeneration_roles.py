"""Replay synthetic dialogues and inspect the information each speaker receives."""

import json
import unittest

from ssvd.dialogue.regeneration_roles_v3 import PAIR_COUNTS, next_step

IMAGE = "data:image/png;base64,AA=="  # Only the request/replay contract is tested.
COUNT = {"kind": "counting", "mode": "exact", "target": "ベンチ", "count": 3}
OCR = {"kind": "ocr", "mode": "exact", "target": "看板", "written": "AOBA 24",
       "reading": "アオバ にじゅうよん"}
QUESTIONS = {"counting": "それ、何脚あるの？", "ocr": "その看板には何て書いてあるの？"}
ANSWERS = {"counting": "三脚だよ。", "ocr": "アオバにじゅうよんって書いてあるよ。"}
SCENE = "手前にベンチが並び、奥の建物に看板が掛かっている。"


class RegenerationRoleTests(unittest.TestCase):
    def step(self, replies, *, source="ocr", pairs=4, uid="synthetic-role-example", **kwargs):
        return next_step(uid, source, IMAGE, replies, pairs=pairs, **kwargs)

    def exercise(self, tasks, *, answers=None, override=None, **settings):
        """Answer all requests, optionally substituting one utterance to exercise a failure."""
        observation = {"status": "ok", "scene": SCENE, "tasks": tasks}
        replies = [observation]
        result = self.step(replies, **settings)
        requests = []
        while result["status"] == "request":
            pair = 1 + (len(replies) - 1) // 2
            phase = result["phase"]
            positions = result["task_pairs"]
            active = next((kind for kind, pos in positions.items() if pos == pair), None)
            requests.append((pair, phase, result["request"]))
            if phase == "user":
                text = QUESTIONS[active] if active else "どんな様子？"
            elif active:
                text = (answers or ANSWERS)[active]
            else:
                text = SCENE
            if override:
                text = override(pair, phase, positions, text)
            replies.append({"text": text})
            result = self.step(replies, **settings)
        return result, replies, requests

    def test_public_length_and_reproducibility(self):
        counts = set()
        for index in range(80):
            step = self.step([], uid=str(index), pairs=None)
            counts.add(step["pair_count"])
            self.assertEqual(step, self.step([], uid=str(index), pairs=step["pair_count"]))
            self.assertEqual(step["task_pairs"], {})
            self.assertEqual(step["phase"], "grounding")
            self.assertIn(IMAGE, json.dumps(step["request"]))
        self.assertEqual(counts, set(PAIR_COUNTS))
        for value in (3, 9, True):
            with self.assertRaises(ValueError):
                self.step([], pairs=value)

    def test_observation_hold_and_malformed_replay(self):
        held = {"status": "hold", "reason": "画像全体を確認できない"}
        self.assertEqual(self.step([held])["status"], "hold")
        with self.assertRaises(ValueError):
            self.step([held, {"text": "次は？"}])
        flat = {"status": "ok", "scene": SCENE, **OCR}
        self.assertEqual(self.step([flat])["status"], "hold")

    def test_source_label_does_not_force_the_task_and_all_candidates_need_review(self):
        for tasks in ([], [OCR], [COUNT], [OCR, COUNT], [COUNT, OCR]):
            with self.subTest(tasks=[task["kind"] for task in tasks]):
                result, replies, requests = self.exercise(tasks, source="counting")
                self.assertEqual(result["status"], "pending_review", result)
                self.assertTrue(result["requires_human_review"])
                self.assertEqual(result["source_kind"], "counting")
                self.assertEqual(result["grounding"]["tasks"], tasks)
                positions = result["task_pairs"]
                self.assertEqual(list(positions), [task["kind"] for task in tasks])
                if len(tasks) == 2:
                    first, second = positions.values()
                    self.assertGreaterEqual(second - first, 2)
                self.assertEqual(result["dialogue"], [
                    {"role": phase, "text": reply["text"]}
                    for (_, phase, _), reply in zip(requests, replies[1:])
                ])
                self.assertEqual(self.step(replies, source="counting"), result)

    def test_maximum_length_is_not_truncated(self):
        result, _, _ = self.exercise([COUNT, OCR], pairs=8)
        self.assertEqual(result["status"], "pending_review", result)
        self.assertEqual(len(result["dialogue"]), 16)

    def test_both_roles_receive_source_image_every_turn_and_answers_stay_private_until_due(self):
        result, _, requests = self.exercise([COUNT, OCR], model="gpt-6.1-terra")
        self.assertEqual(result["status"], "pending_review", result)
        for pair, phase, request in requests:
            wire = json.dumps(request, ensure_ascii=False)
            self.assertEqual(request["model"], "gpt-6.1-terra")
            parts = request["input"][0]["content"]
            self.assertEqual([part["type"] for part in parts], ["input_text", "input_image"])
            self.assertEqual(parts[1], {"type": "input_image", "image_url": IMAGE, "detail": "high"})
            self.assertEqual(request["max_output_tokens"], 1024)
            due = result["task_pairs"]["ocr"]
            if pair < due or (pair == due and phase == "user"):
                self.assertNotIn(OCR["written"], wire)
                self.assertNotIn(OCR["reading"], wire)
            if pair == due and phase == "assistant":
                self.assertIn(OCR["written"], wire)
            if phase == "user":
                self.assertNotIn("画像の観察情報", wire)
                payload = json.loads(parts[0]["text"].splitlines()[1])
                self.assertNotIn("この返答の根拠", wire)
                for kind, target in ((COUNT["kind"], COUNT["target"]), (OCR["kind"], OCR["target"])):
                    if pair == result["task_pairs"][kind]:
                        self.assertEqual(payload["質問の対象"], target)
                        self.assertEqual(set(payload), {"質問の対象", "公開された会話"})
        observation = self.step([])["request"]
        self.assertEqual(observation["max_output_tokens"], 2048)
        self.assertEqual(observation["input"][0]["content"][1]["image_url"], IMAGE)

    def test_question_can_introduce_a_new_target_directly_from_the_image(self):
        for task, questions in [
            (COUNT, [("手前のベンチは何脚ある？", "pending_review"), ("自転車は何台ある？", "hold")]),
            ({**OCR, "target": "看板中央の白い文字"}, [
                ("この看板、何て書いてあるんだろう？", "pending_review"), ("何て書いてある？", "pending_review")]),
        ]:
            def override(pair, phase, positions, old):
                if phase == "assistant" and pair < positions[task["kind"]]:
                    return "手前に芝生が見えるよ。"
                if phase == "user" and pair == positions[task["kind"]]:
                    return question
                return old
            for question, expected in questions:
                with self.subTest(task=task["kind"], question=question):
                    result, _, _ = self.exercise([task], override=override)
                    self.assertEqual(result["status"], expected, result)

    def test_reference_to_the_previous_sign_does_not_license_asking_about_a_different_sign(self):
        task = {**OCR, "target": "左の看板"}
        def replace(pair, phase, positions, old):
            if phase == "assistant" and pair + 1 == positions["ocr"]:
                return "木の隣に左の看板があるよ。"
            if phase == "user" and pair == positions["ocr"]:
                return question
            return old
        for question, expected in [
            ("その看板には何て書いてある？", "pending_review"),
            ("何て書いてある？", "pending_review"),
            ("右の看板には何て書いてある？", "hold"),
        ]:
            with self.subTest(question=question):
                result, _, _ = self.exercise([task], override=replace)
                self.assertEqual(result["status"], expected, result)

    def test_embedded_evidence_is_reproducible_after_json_roundtrip_with_sorted_keys(self):
        observation = {"status": "ok", "scene": SCENE, "tasks": [OCR]}
        result, replies, _ = self.exercise([OCR])
        answer_due = 2 * result["task_pairs"]["ocr"]
        prefix = [observation, *replies[1:answer_due]]
        request = self.step(prefix)
        self.assertEqual(request["phase"], "assistant")
        self.assertEqual(request, self.step(json.loads(json.dumps(prefix, sort_keys=True))))

    def test_uncertainty_modes_can_finish_as_pending(self):
        for kind, mode, gold, answer in [
            ("counting", "approximate", {"count": 3}, "三脚くらいに見えるよ。"),
            ("counting", "lower_bound", {"count": 3}, "少なくとも三脚は見えるよ。"),
            ("counting", "unknown", {}, "重なっていて何脚かは分からないよ。"),
            ("ocr", "partial", {"written": "入口", "reading": "いりぐち"}, "入口の部分だけは読めるよ。"),
            ("ocr", "unknown", {}, "小さくて文字までは読めないよ。"),
        ]:
            task = {"kind": kind, "mode": mode, "target": COUNT["target"] if kind == "counting" else OCR["target"],
                    "reason": "奥が隠れている", **gold}
            with self.subTest(kind=kind, mode=mode):
                result, _, _ = self.exercise([task], answers={kind: answer})
                self.assertEqual(result["status"], "pending_review", result)
                self.assertEqual(result["grounding"]["tasks"][0]["mode"], mode)

    def test_unrecorded_explicit_question_is_held(self):
        for text in ("その看板に何と書いてあるの？", "それ、何台ある？"):
            result, _, _ = self.exercise([], override=lambda p, role, pos, previous: text)
            self.assertEqual(result["status"], "hold")
            self.assertEqual(result["reason"], "unrecorded_visual_task_question")

    def test_uncertainty_phrasing_varies_while_evidence_stays_the_same(self):
        task = {**COUNT, "mode": "lower_bound", "reason": "木の陰にも続いている"}
        instructions = set()
        for index in range(16):
            result, _, requests = self.exercise([task], uid=str(index),
                                                answers={"counting": "少なくとも三脚はあるね。"})
            self.assertEqual(result["status"], "pending_review", result)
            position = result["task_pairs"]["counting"]
            request = next(body for pair, phase, body in requests if pair == position and phase == "assistant")
            cue, payload, _ = request["input"][0]["content"][0]["text"].splitlines()
            self.assertEqual(json.loads(payload)["画像の観察情報"]["この返答の根拠"], task)
            instructions.add(cue)
        self.assertGreater(len(instructions), 1)

    def test_introduction_spoilers_and_wrong_answers(self):
        def substitute(locate, kind, phase, utterance):
            def change(pair, actual_phase, positions, old):
                if pair == locate(positions[kind]) and phase == actual_phase:
                    return utterance
                return old
            return change

        for override in (
            substitute(lambda position: position - 1, "counting", "assistant", "緑の芝生だよ。"),
            substitute(lambda position: position, "counting", "assistant", "ベンチは四脚だよ。"),
            substitute(lambda position: position, "counting", "assistant", "自転車は三台だよ。"),
            substitute(lambda position: position, "counting", "user", "ベンチは三脚あるの？"),
            substitute(lambda position: position - 1, "ocr", "assistant", "看板には AOBA 24 とあるよ。"),
            substitute(lambda position: position, "counting", "assistant", "三脚だよ。看板に AOBA 24 って書いてある。"),
        ):
            with self.subTest(override=override):
                result, replies, _ = self.exercise([COUNT, OCR], override=override)
                self.assertEqual(result["status"], "hold", result)
                with self.assertRaises(ValueError):
                    self.step([*replies, {"text": "先へ"}])
