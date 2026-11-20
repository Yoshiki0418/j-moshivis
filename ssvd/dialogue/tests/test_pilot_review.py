"""The offline pilot sheet can be shared with the reviewer without revealing A/B."""
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ssvd.dialogue.pilot_review import write_review
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, review_metadata


class PilotReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        events = self.root / DEFAULT_REVIEW_EVENTS
        events.parent.mkdir(parents=True)
        events.touch()
        self.run = self.root / "pilot"
        with BytesIO() as picture:
            Image.new("RGB", (34, 20), "green").save(picture, "PNG")
            self.digest = sha256(picture.getvalue()).hexdigest()
            self.url = "data:image/png;base64," + base64.b64encode(picture.getvalue()).decode("ascii")
        self.samples = []
        self.sources = {}
        self.outcomes = {}
        self.models = ("gpt-6-luna", "gpt-6.1-terra")
        for index in range(6):
            uid = f"private-id-{index}"
            src = self.record(uid, "Qwen", f"旧対話{index}")
            self.sources[uid] = (src, "ocr", self.url, src["review_id"])
            self.samples.append(dict(uid=uid, base_fingerprint=src["review_id"], image_sha256=self.digest, kind="ocr"))
            for model in self.models:
                candidate = self.record(uid, model, f"候補文{index} <script>bad()</script>")
                self.outcomes[uid, model] = dict(candidate=candidate, status="pending_review")
        self.touched = set()
        patched = patch("ssvd.dialogue.pilot_review._load_source", side_effect=self.load)
        patched.start()
        self.addCleanup(patched.stop)

    def record(self, uid, model, text):
        record = dict(uid=uid, image_sha256=self.digest, generator=model,
                      dialogue=[{"speaker": "ユーザー", "text": text}],
                      auxiliary_task=dict(kind="ocr", recognized_text="<sign>", spoken_reading="サイン"))
        record.update(review_metadata(record))
        return record

    def load(self, root, uid):
        if uid in self.touched:
            raise ValueError("already reviewed")
        return self.sources[uid]

    def render(self):
        path = write_review(self.root, self.run, {"samples": self.samples}, self.outcomes)
        mapping = json.loads((self.run / "blind_key.json").read_text())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.run / "blind_key.json").stat().st_mode & 0o777, 0o600)
        return path.read_text(), mapping

    def test_blinded_balanced_stable_offline(self):
        html, key = self.render()
        self.assertEqual(len(key), 6)
        self.assertEqual(html.count('<a href="#img-'), 6)
        self.assertIn('id="img-6"', html)
        self.assertEqual(sum(r["models"]["A"]["model"] == self.models[0] for r in key.values()), 3)
        self.samples.reverse()
        self.assertEqual(self.render()[1], key)
        for private in (*self.models, "private-id-", "<script>bad()", 'src="https:', "blind_key.json"):
            self.assertNotIn(private, html)
        for expected in ("旧対話0", "候補文5", "&lt;sign&gt;", "サイン", 'name="edit_seconds"'):
            self.assertIn(expected, html)
        encoded = re.search(r'<img [^>]*src="data:image/png;base64,([^"]+)', html).group(1)
        with Image.open(BytesIO(base64.b64decode(encoded))) as resized:
            self.assertEqual(resized.size, (448, 448))

    def test_missing_decision_history_stops_export(self):
        (self.root / DEFAULT_REVIEW_EVENTS).unlink()
        with self.assertRaisesRegex(RuntimeError, "decision history"):
            self.render()
        self.assertFalse((self.run / "review.html").exists())

    def test_review_warnings_link_to_the_candidate_and_mark_the_exact_utterance(self):
        uid = self.samples[0]["uid"]
        model = self.models[0]
        candidate = self.outcomes[uid, model]["candidate"]
        candidate["dialogue"].append({"speaker": "アシスタント", "text": "テスト回答"})
        candidate["auxiliary_task"] = {"kind": "counting", "answer_turn_index": 1,
                                       "counted_target": "箱の中のもの", "minimum_count": 3}
        candidate["review_warnings"] = [
            {"code": "topic_not_covered", "turn": None},
            {"code": "spoken_form", "turn": 2},
            {"code": "<script>future_warning()</script>", "turn": 2},
        ]
        candidate.update(review_metadata(candidate))
        html, key = self.render()
        self.assertIn("注意付きの候補 1 件", html)
        row = next(r for r in key.values() if r["uid"] == uid)
        side = next(s for s, m in row["models"].items() if m["model"] == model)
        self.assertEqual(row["models"][side]["status"], "pending_review")
        self.assertIn(f'href="#img-1-{side}"', html)
        section = html.split(f'<section id="img-1-{side}">', 1)[1].split("</section>", 1)[0]
        heading, turns = section.split("<ol>", 1)
        self.assertIn("個数・文字の記録なし", heading)
        question, answer = turns.split("</li>", 1)
        self.assertNotIn('class="review-warning"', question)
        self.assertIn("自然に読み上げられる表記", answer)
        self.assertIn("画像・会話の流れと照合", answer)
        self.assertIn("箱の中のもの", answer)
        self.assertIn("確認できる最小の数</dt><dd>3", answer)
        self.assertNotIn("箱の中のもの", question)
        self.assertNotIn("future_warning", html)
        self.assertIn('name="decision"', section)

        candidate["review_warnings"].clear()
        html, _ = self.render()
        self.assertIn("照合エラー", html)
        self.assertNotIn("注意付きの候補 1 件", html)

    def test_stale_or_protected_sources_removed_on_refresh(self):
        self.render()
        self.touched.add(self.samples[0]["uid"])
        self.samples[1]["base_fingerprint"] = "stale"
        self.samples[2]["image_sha256"] = "another-image"
        self.outcomes[self.samples[3]["uid"], self.models[0]] = {"status": "protected"}
        uid = self.samples[4]["uid"]
        self.outcomes[uid, self.models[0]] = dict(status="hold", reason="private upstream error")
        self.outcomes[uid, self.models[1]]["candidate"]["dialogue"][0]["text"] = "modified!"
        del self.outcomes[self.samples[5]["uid"], self.models[0]]
        html, key = self.render()
        self.assertEqual(len(key), 2)
        self.assertNotIn('<a href="#img-', html)
        self.assertIn("保留", html)
        self.assertIn("照合エラー", html)
        self.assertIn("未生成", html)
        self.assertNotIn("modified!", html)
        self.assertNotIn("private upstream error", html)
        for index in range(4):
            self.assertNotIn(f"旧対話{index}", html)
            self.assertNotIn(f"候補文{index}", html)


if __name__ == "__main__":
    unittest.main()
