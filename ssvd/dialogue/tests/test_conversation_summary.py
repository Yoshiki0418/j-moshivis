"""Keep the question-mark baseline comparable as people review talks and prompts change."""

from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest

from ssvd.dialogue import production
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore


class ConversationSummaryTests(unittest.TestCase):
    def test_completed_dialogues_are_counted_by_version_independently_of_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / DEFAULT_REVIEW_EVENTS
            events.parent.mkdir(parents=True)
            ReviewStore(events).append(fingerprint="a" * 64, decision="approved",
                                       uid="first", collection=production.pilot.COLLECTION)
            path = root / production.RUN_PATH
            path.mkdir(parents=True)
            cases = [
                ("first", "visual_dialogue_v10", ["何色？", "何個? "]),
                ("second", "visual_dialogue_v11", ["どの色？", "青って落ち着くよね。"]),
                ("third", "visual_dialogue_v11", ["青がきれい。", "隣にもあるね。"]),
            ]
            with closing(production.connect(path)) as db, db:
                db.execute("CREATE TABLE calls(uid, charged, reserved)")
                db.execute("CREATE TABLE outcomes(uid, outcome)")
                for uid, version, utterances in cases:
                    dialogue = []
                    for text in utterances:
                        dialogue.extend([{"speaker": "ユーザー", "text": text}, {"speaker": "アシスタント", "text": "そうだね。"}])
                    outcome = {"status": "pending_review", "candidate": {"prompt_version": version, "dialogue": dialogue}}
                    db.execute("INSERT INTO outcomes VALUES (?, ?)", (uid, json.dumps(outcome)))
            report = production.snapshot(root, [{"uid": c[0]} for c in cases], 0)
            self.assertEqual(report["outcomes"], {"protected": 1, "pending_review": 2})
            old, new = (report["user_turns_by_prompt"][v] for v in ("visual_dialogue_v10", "visual_dialogue_v11"))
            self.assertEqual((old["question_mark_endings"], old["user_turns"], old["all_question_mark_endings"]),
                             (2, 2, 1))
            self.assertEqual(new, {"dialogues": 2, "user_turns": 4, "question_mark_endings": 1,
                                   "all_question_mark_endings": 0, "no_question_mark_endings": 1})
