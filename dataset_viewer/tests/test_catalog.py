from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dataset_viewer.catalog import DatasetCatalog
from ssvd.review_store import ReviewConflictError, image_file_sha256


def _write_jsonl(path: Path, records: list[dict], partial: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
        if partial is not None:
            stream.write(json.dumps(partial, ensure_ascii=False).encode("utf-8"))


class DatasetCatalogTest(unittest.TestCase):
    def _required_ai_suggestion(self):
        source = self.data / "dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl"
        record = {"uid": "count:one", "task_mode": "normal", "generator": "test-vlm",
                  "image": str(self.counting_one_image), "image_sha256": image_file_sha256(self.counting_one_image),
                  "verification_status": "unverified_vlm_visual",
                  "auxiliary_task": {"kind": "counting", "grounding_source": "vlm_visual",
                                     "question_turn_index": 0, "answer_turn_index": 1,
                                     "counted_target": "リンゴ", "answer_count": 1},
                  "dialogue": [{"speaker": "ユーザー", "text": "リンゴは何個？"},
                               {"speaker": "アシスタント", "text": "1個です。"}]}
        _write_jsonl(source, [record])
        self.catalog.refresh()
        detail = self.catalog.get_item(self.catalog.list_items(
            collection="counting_v4", review_required=True)["items"][0]["token"])
        suggestion = self.catalog.ai_suggestion_store.put_suggestion(
            collection=detail["collection"], uid=detail["uid"],
            base_fingerprint=detail["content_sha256"], image_sha256=record["image_sha256"],
            dialogue=[record["dialogue"][0], {"speaker": "アシスタント", "text": "2個あります。"}],
            auxiliary_task={**record["auxiliary_task"], "answer_count": 2}, model="gpt-6-luna",
            changes=[{"turn_index": 1, "reason": "画像の個数に合わせて訂正"}],
            concerns=[], verdict="corrected")
        return source, record, detail, suggestion

    def test_ai_suggestion_is_pending_overlay_until_human_approval(self):
        source, record, original, suggestion = self._required_ai_suggestion()
        original_bytes = source.read_bytes()
        shown = self.catalog.get_item(original["token"])
        self.assertEqual(shown["dialogue"][1]["text"], "2個あります。")
        self.assertEqual(shown["original_dialogue"][1]["text"], "1個です。")
        self.assertEqual(shown["auxiliary_task"]["answer_count"], 2)
        self.assertEqual(shown["original_auxiliary_task"]["answer_count"], 1)
        self.assertEqual(shown["ai_suggestion"]["suggestion_id"], suggestion["suggestion_id"])
        self.assertEqual(shown["review_status"], "pending")
        self.assertFalse(shown["dataset_usable"])
        self.assertEqual(shown["content_sha256"], original["content_sha256"])
        self.assertEqual(self.catalog.review_store.latest_decisions(), {})
        self.assertEqual(self.catalog.review_store.resolve_record(record), record)
        self.assertEqual(source.read_bytes(), original_bytes)
        approved = self.catalog.edit_dialogue(original["token"],
            fingerprint=shown["content_sha256"], dialogue=shown["dialogue"],
            auxiliary_task=shown["auxiliary_task"], ai_suggestion_id=suggestion["suggestion_id"])["item"]
        self.assertEqual(approved["review_status"], "approved")
        self.assertEqual(approved["dialogue"][1]["text"], "2個あります。")
        self.assertEqual(approved["auxiliary_task"]["answer_count"], 2)
        self.assertNotEqual(approved["content_sha256"], original["content_sha256"])
        self.assertIsNone(approved["ai_suggestion"])
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual(self.catalog.review_store.resolve_record(record)["dialogue"][1]["text"], "2個あります。")

    def test_ai_suggestion_cannot_replace_human_decisions_including_reset(self):
        _, _, original, suggestion = self._required_ai_suggestion()
        for decision in ("approved", "rejected", "pending"):
            with self.subTest(decision=decision):
                self.catalog.review_item(original["token"], fingerprint=original["content_sha256"], decision=decision)
                current = self.catalog.get_item(original["token"])
                self.assertIsNone(current["ai_suggestion"])
                self.assertEqual(self.catalog.list_items(ai_status="ready")["total"], 0)
                self.assertEqual(current["dialogue"][1]["text"], "1個です。")
                events_before = self.catalog.review_store.path.read_bytes()
                with self.assertRaises(ReviewConflictError):
                    self.catalog.edit_dialogue(original["token"], fingerprint=original["content_sha256"],
                        dialogue=suggestion["dialogue"], ai_suggestion_id=suggestion["suggestion_id"])
                self.assertEqual(self.catalog.review_store.path.read_bytes(), events_before)

    def test_ai_suggestion_is_hidden_after_image_or_source_change(self):
        source, record, original, suggestion = self._required_ai_suggestion()
        self.counting_one_image.write_bytes(b"new source image")
        current = self.catalog.get_item(original["token"])
        self.assertIsNone(current["ai_suggestion"])
        self.assertEqual(self.catalog.list_items(ai_status="ready")["total"], 0)
        with self.assertRaises(ReviewConflictError):
            self.catalog.edit_dialogue(original["token"], fingerprint=original["content_sha256"],
                dialogue=suggestion["dialogue"], ai_suggestion_id=suggestion["suggestion_id"])
        self.counting_one_image.write_bytes(b"one count")
        record["dialogue"][1]["text"] = "生成元が更新されました。"
        _write_jsonl(source, [record])
        self.catalog.refresh()
        current = self.catalog.get_item(original["token"])
        self.assertIsNone(current["ai_suggestion"])
        self.assertEqual(current["dialogue"][1]["text"], "生成元が更新されました。")
        self.assertEqual(self.catalog.review_store.latest_decisions(), {})

    def test_ai_suggestion_rejects_wrong_id_and_preserves_manual_changes(self):
        _, _, original, suggestion = self._required_ai_suggestion()
        with self.assertRaises(ReviewConflictError):
            self.catalog.edit_dialogue(original["token"], fingerprint=original["content_sha256"],
                dialogue=suggestion["dialogue"], ai_suggestion_id="unseen-draft")
        human = self.catalog.edit_dialogue(original["token"], fingerprint=original["content_sha256"],
            dialogue=[{"text": "質問も修正"}, {"text": "人手で直した本文"}])["item"]
        self.assertIsNone(human["ai_suggestion"])
        self.assertEqual(human["dialogue"][1]["text"], "人手で直した本文")
        self.catalog.review_item(original["token"], fingerprint=human["content_sha256"], decision="pending")
        reset = self.catalog.get_item(original["token"])
        self.assertEqual(reset["dialogue"][1]["text"], "人手で直した本文")
        self.assertIsNone(reset["ai_suggestion"])

    def test_ai_suggestion_excludes_prior_human_activity_on_another_fingerprint(self):
        _, _, original, _ = self._required_ai_suggestion()
        self.catalog.review_store.append(fingerprint="f" * 64, decision="rejected",
            uid=original["uid"], collection=original["collection"])
        self.catalog.review_store.append(fingerprint="f" * 64, decision="approved",
            uid=original["uid"], collection="another_collection")
        shown = self.catalog.get_item(original["token"])
        self.assertEqual(shown["review_status"], "pending")
        self.assertIsNone(shown["ai_suggestion"])
        self.assertEqual(shown["dialogue"][1]["text"], "1個です。")
        self.assertEqual(self.catalog.list_items(ai_status="ready")["total"], 0)

    def test_ai_ready_filter_matches_exact_untouched_content_and_navigation(self):
        source, record, original, suggestion = self._required_ai_suggestion()
        attention_image = self.data / "images/counting/two.jpg"
        attention_record = {**record, "uid": "count:two", "image": str(attention_image),
                            "image_sha256": image_file_sha256(attention_image)}
        unsuggested_record = {**record, "uid": "count:three"}
        _write_jsonl(source, [record, attention_record, unsuggested_record])
        self.catalog.refresh()
        candidates = self.catalog.list_items(collection="counting_v4", review_required=True)["items"]
        attention = next(item for item in candidates if item["uid"] == "count:two")
        self.catalog.ai_suggestion_store.put_suggestion(
            collection=attention["collection"], uid=attention["uid"],
            base_fingerprint=attention["content_sha256"], image_sha256=attention_record["image_sha256"],
            dialogue=suggestion["dialogue"], auxiliary_task=suggestion["auxiliary_task"],
            model="gpt-6-luna", changes=[], concerns=["個数を確認してください"], verdict="needs_attention")
        # List queries join compact keys; they must never reread all sources.
        with patch.object(self.catalog, "_read_locator", side_effect=AssertionError("source read during list")):
            ready = self.catalog.list_items(ai_status="ready")
            concerns = self.catalog.list_items(ai_status="needs_attention")
        self.assertEqual([item["uid"] for item in ready["items"]], ["count:one", "count:two"])
        self.assertEqual(ready["total"], 2)
        self.assertEqual(concerns["total"], 1)
        self.assertEqual(concerns["items"][0]["uid"], "count:two")
        navigation = self.catalog.get_item(original["token"], filters={"ai_status": "ready"})["navigation"]
        self.assertEqual(navigation["next"], attention["token"])
        self.assertEqual(navigation["total"], 2)
        attention_navigation = self.catalog.get_item(attention["token"], filters={"ai_status": "needs_attention"})["navigation"]
        self.assertIsNone(attention_navigation["previous"])
        self.assertIsNone(attention_navigation["next"])
        self.assertEqual(attention_navigation["total"], 1)
        self.assertEqual(self.catalog.status()["ai_suggestion_count"], 2)
        self.assertEqual(self.catalog.list_items(ai_status="ready", review_status="approved")["total"], 0)

    def test_ai_ready_filter_without_store_is_empty_and_does_not_create_store(self):
        self.assertFalse(self.catalog.ai_suggestion_store.path.exists())
        self.assertEqual(self.catalog.list_items(ai_status="ready")["total"], 0)
        self.assertFalse(self.catalog.ai_suggestion_store.path.exists())
        with self.assertRaises(ValueError):
            self.catalog.list_items(ai_status="unknown")

    def test_ai_ready_filter_does_not_select_an_old_image_or_fingerprint(self):
        _, _, original, _ = self._required_ai_suggestion()
        for key in ("review_id", "current_image_sha256"):
            with self.subTest(key=key), self.catalog._connect() as db:
                if key == "review_id":
                    db.execute("UPDATE items SET review_id=? WHERE collection=? AND uid=?",
                               ("e" * 64, original["collection"], original["uid"]))
                else:
                    db.execute("UPDATE items SET review_id=?,summary_json=json_set(summary_json,'$.current_image_sha256',?) WHERE collection=? AND uid=?",
                               (original["content_sha256"], "e" * 64, original["collection"], original["uid"]))
            self.assertEqual(self.catalog.list_items(ai_status="ready")["total"], 0)

    def _save_during_refresh(self, *, edit=False, sql_scan=False):
        token = self.catalog.list_items(collection="visual_v3")["items"][0]["token"]
        detail = self.catalog.get_item(token)
        before = self.catalog.status()
        entered, release = threading.Event(), threading.Event()

        def slow_scan(db):
            # Simulate a refresh that has already modified its SQLite snapshot.
            # The next annotation must see the previous complete catalogue,
            # never these partially scanned rows or metadata.
            db.execute("INSERT INTO meta(key,value) VALUES('interrupted_test','uncommitted')")
            db.execute("UPDATE items SET review_status='rejected',dataset_usable=0 WHERE collection='visual_v3'")
            entered.set()
            if sql_scan:
                db.execute("""WITH RECURSIVE sequence(x) AS
                    (VALUES(0) UNION ALL SELECT x+1 FROM sequence WHERE x<20000000)
                    SELECT SUM(x) FROM sequence""").fetchone()
            else:
                while not release.wait(0.005):
                    self.catalog._check_refresh_interruption()
            return {}

        save = (lambda: self.catalog.edit_dialogue(token, fingerprint=detail["content_sha256"],
                    dialogue=[{"text": "写真には何がありますか？"}, {"text": "猫が写っています。"}])) if edit else (
                lambda: self.catalog.review_item(token, fingerprint=detail["content_sha256"], decision="approved"))
        with patch.object(self.catalog, "_refresh_targeted_speech_roots", side_effect=slow_scan):
            with ThreadPoolExecutor(max_workers=2) as executor:
                refresh = executor.submit(self.catalog.refresh)
                try:
                    self.assertTrue(entered.wait(1))
                    started = time.monotonic()
                    saved = executor.submit(save).result(timeout=2)
                    elapsed = time.monotonic() - started
                    interrupted = refresh.result(timeout=2)
                finally:
                    release.set()
                    # This also releases a SQL test promptly if an assertion
                    # fails before the review worker has announced itself.
                    if not refresh.done():
                        self.catalog.request_interactive_pause(1)
        self.assertTrue(saved["ok"])
        self.assertLess(elapsed, 2)
        self.assertTrue(interrupted["refresh_interrupted"])
        with self.catalog._connect() as db:
            self.assertIsNone(db.execute("SELECT value FROM meta WHERE key='interrupted_test'").fetchone())
        self.assertEqual(self.catalog.status()["last_refresh"], before["last_refresh"])
        after = self.catalog.get_item(token)
        self.assertTrue(after["human_certified"])
        self.assertTrue(after["dataset_usable"])
        self.assertFalse(self.catalog._interactive_pending.is_set())
        if edit:
            self.assertEqual(after["dialogue"][-1]["text"], "猫が写っています。")
        # The rolled-back scan can run again and must retain the durable save.
        self.assertNotIn("refresh_interrupted", self.catalog.refresh())
        self.assertTrue(self.catalog.get_item(token)["human_certified"])
        return elapsed

    def test_approval_interrupts_background_scan_and_rolls_back_partial_catalog(self):
        self._save_during_refresh()

    def test_edit_approval_interrupts_background_scan_and_keeps_revised_text(self):
        self._save_during_refresh(edit=True)

    def test_approval_interrupts_long_sql_phase_of_background_refresh(self):
        self._save_during_refresh(sql_scan=True)

    def test_interactive_pause_postpones_refresh_without_changing_catalogue(self):
        before = self.catalog.status()
        self.catalog.request_interactive_pause(30)
        with patch.object(self.catalog, "_refresh_targeted_speech_roots") as scan:
            result = self.catalog.refresh()
        self.assertTrue(result["refresh_interrupted"])
        self.assertEqual(result["catalog_revision"], before["catalog_revision"])
        self.assertEqual(result["last_refresh"], before["last_refresh"])
        scan.assert_not_called()

    def test_manual_refresh_resumes_pause_but_preserves_pending_writer_priority(self):
        self.catalog.request_interactive_pause(30)
        self.assertTrue(self.catalog.refresh()["refresh_interrupted"])
        self.catalog.resume_background_refresh()
        self.assertNotIn("refresh_interrupted", self.catalog.refresh())

        entered, release = threading.Event(), threading.Event()
        def queued_annotation():
            with self.catalog._interactive_write():
                entered.set()
                release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            writer = executor.submit(queued_annotation)
            try:
                self.assertTrue(entered.wait(1))
                self.catalog.request_interactive_pause(30)
                self.catalog.resume_background_refresh()
                self.assertTrue(self.catalog._interactive_pending.is_set())
                self.assertTrue(self.catalog._refresh_should_yield())
            finally:
                release.set()
            writer.result(timeout=1)
        self.assertFalse(self.catalog._refresh_should_yield())

    def test_unchanged_detail_does_not_wait_for_background_refresh_lock(self):
        token = self.catalog.list_items(collection="visual_v3")["items"][0]["token"]
        before = self.catalog.get_item(token)
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.catalog._refresh_lock.acquire()
            try:
                after = executor.submit(self.catalog.get_item, token).result(timeout=1)
            finally:
                self.catalog._refresh_lock.release()
        self.assertEqual(before["content_sha256"], after["content_sha256"])
        self.assertEqual(before["dialogue"], after["dialogue"])

    def test_saving_unchanged_text_only_certifies_without_regeneration(self):
        token = self.catalog.list_items(collection="visual_v3")["items"][0]["token"]
        detail = self.catalog.get_item(token)
        result = self.catalog.edit_dialogue(token, fingerprint=detail["content_sha256"],
                                           dialogue=[{"text": turn["text"]} for turn in detail["dialogue"]])
        self.assertFalse(result["content_changed"])
        self.assertTrue(result["item"]["human_certified"])
        self.assertTrue(result["item"]["availability"]["audio"])
        self.assertIsNone(result["item"]["human_revision"])
        self.assertEqual(self.catalog.dashboard()["review"]["regeneration_pending"], 0)

    def test_optional_approval_edit_and_regeneration_dashboard(self):
        from ssvd.tts.dataset_keys import speech_sample_uid
        original = self.catalog.list_items(collection="visual_v3")["items"][0]
        token = original["token"]
        detail = self.catalog.get_item(token)
        self.assertFalse(detail["review_required"])
        self.assertTrue(detail["dataset_usable"])
        self.assertFalse(detail["human_certified"])
        self.catalog.review_item(token, fingerprint=detail["content_sha256"], decision="approved")
        approved = self.catalog.get_item(token)
        self.assertTrue(approved["human_certified"])
        self.assertFalse(approved["regeneration_pending"])
        self.assertTrue(approved["availability"]["audio"])
        edited = self.catalog.edit_dialogue(token, fingerprint=approved["content_sha256"],
            dialogue=[{"text": "写真には何が見えますか？"}, {"text": "猫が写っています。"}])["item"]
        self.assertFalse(edited["review_required"])
        self.assertTrue(edited["regeneration_pending"])
        self.assertEqual(self.catalog.dashboard()["review"]["certified_total"], 1)
        self.assertEqual(self.catalog.dashboard()["review"]["regeneration_pending"], 1)
        self.assertEqual(self.catalog.list_items(stage="regeneration_pending")["total"], 1)
        self.catalog.refresh(force=True)
        self.assertTrue(self.catalog.get_item(token)["regeneration_pending"])
        source = self.data / "dialogues_merged/dialogues_final_v3.jsonl"
        record = json.loads(source.read_text().strip())
        effective = self.catalog.review_store.resolve_record(record, image_sha256=image_file_sha256(self.visual_image))
        speech = self.data / "speech_v3/data_stereo" / speech_sample_uid(effective)
        speech.mkdir()
        (speech / "stereo_dialogue.wav").write_bytes(b"revised audio")
        self.catalog.refresh()
        stats = self.catalog.dashboard()["review"]
        self.assertEqual(stats["regeneration_pending"], 1)
        self.assertEqual(stats["regeneration_audio_pending"], 0)
        self.assertEqual(stats["regeneration_alignment_pending"], 1)
        (speech / "assistant_dialogue.refined.json").write_text('{"alignments": []}')
        self.catalog.refresh()
        self.assertEqual(self.catalog.dashboard()["review"]["regeneration_pending"], 0)
        self.assertEqual(self.catalog.dashboard()["review"]["certified_complete"], 1)
        self.assertEqual(self.catalog.list_items(stage="regeneration_pending")["total"], 0)
        current = self.catalog.get_item(token)
        self.catalog.review_item(token, fingerprint=current["content_sha256"], decision="pending")
        reverted = self.catalog.get_item(token)
        self.assertFalse(reverted["human_certified"])
        self.assertTrue(reverted["dataset_usable"])
        self.catalog.review_item(token, fingerprint=current["content_sha256"], decision="rejected")
        self.assertFalse(self.catalog.get_item(token)["dataset_usable"])

    def test_required_pending_filter_excludes_certified_and_rejected(self):
        source = self.data / "dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl"
        record = {"uid": "count:one", "image": str(self.counting_one_image), "task_mode": "normal",
                  "review_required": True, "generator": "test-vlm",
                  "auxiliary_task": {"kind": "counting", "grounding_source": "vlm_visual",
                                     "question_turn_index": 0, "answer_turn_index": 1,
                                     "counted_target": "リンゴ", "answer_count": 1},
                  "dialogue": [{"speaker": "ユーザー", "text": "何個？"}, {"speaker": "アシスタント", "text": "1個です。"}]}
        _write_jsonl(source, [record])
        self.catalog.refresh()
        pending = self.catalog.list_items(review_required=True, review_status="pending")
        self.assertEqual(pending["total"], 1)
        detail = self.catalog.get_item(pending["items"][0]["token"])
        stats = self.catalog.dashboard()["review"]
        self.assertEqual((stats["required_total"], stats["required_pending"]), (1, 1))
        self.catalog.review_item(detail["token"], fingerprint=detail["content_sha256"], decision="approved")
        self.assertEqual(self.catalog.list_items(review_required=True, review_status="pending")["total"], 0)
        self.assertEqual(self.catalog.dashboard()["review"]["required_total"], 1)
        self.assertEqual(self.catalog.dashboard()["review"]["required_pending"], 0)
        self.catalog.review_item(detail["token"], fingerprint=detail["content_sha256"], decision="rejected")
        self.assertEqual(self.catalog.list_items(review_required=True, review_status="pending")["total"], 0)
        self.assertEqual(self.catalog.dashboard()["review"]["rejected_total"], 1)

    def test_optional_nonvisual_can_be_edited_and_certified(self):
        item = self.catalog.list_items(collection="nonvisual_v1")["items"][0]
        detail = self.catalog.get_item(item["token"])
        turns = [{"text": turn["text"]} for turn in detail["dialogue"]]
        turns[-1]["text"] = "日本語の応答を人間が修正しました。"
        result = self.catalog.edit_dialogue(item["token"], fingerprint=detail["content_sha256"], dialogue=turns)["item"]
        self.assertTrue(result["human_certified"])
        self.assertFalse(result["review_required"])
        self.assertEqual(result["dialogue"][-1]["text"], turns[-1]["text"])
        self.catalog.refresh(force=True)
        self.assertTrue(self.catalog.get_item(item["token"])["human_certified"])

    def test_edit_counting_invalidates_old_speech_and_shows_revised_assets(self):
        from ssvd.tts.dataset_keys import speech_sample_uid
        from ssvd.review_store import content_fingerprint
        source = self.data / "dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl"
        record = {"uid": "count:one", "task_mode": "normal", "generator": "test-vlm",
                  "image": str(self.counting_one_image), "image_sha256": image_file_sha256(self.counting_one_image),
                  "verification_status": "unverified_vlm_visual",
                  "auxiliary_task": {"kind": "counting", "grounding_source": "vlm_visual",
                                     "question_turn_index": 0, "answer_turn_index": 1,
                                     "counted_target": "リンゴ", "answer_count": 1},
                  "dialogue": [{"speaker": "ユーザー", "text": "リンゴは何個？"},
                               {"speaker": "アシスタント", "text": "1個です。"}]}
        _write_jsonl(source, [record])
        original_bytes = source.read_bytes()
        speech_root = self.data / "speech_counting_review/data_stereo"
        original = speech_root / record["uid"]
        original.mkdir(parents=True)
        (original / "stereo_dialogue.wav").write_bytes(b"original audio")
        (original / "assistant_dialogue.refined.json").write_text('{"alignments": []}')
        (original / "dialogue.json").write_text(json.dumps({"uid": record["uid"], "source_uid": record["uid"],
            "task_mode": "normal", "review_id": content_fingerprint(record), "duration": 2.0}))
        self.catalog.refresh()
        detail = self.catalog.get_item(self.catalog.list_items(collection="counting_v4", review_required=True)["items"][0]["token"])
        token = detail["token"]
        self.assertTrue(detail["availability"]["audio"])
        old_id = detail["content_sha256"]
        edited = self.catalog.edit_dialogue(token, fingerprint=old_id,
            dialogue=[{"text": "リンゴは何個？"}, {"text": "2個あります。"}],
            auxiliary_task={"answer_count": 2, "counted_target": "リンゴ"})["item"]
        self.assertEqual(edited["summary"]["auxiliary_task"]["answer_count"], 2)
        self.assertFalse(edited["availability"]["audio"])
        self.assertFalse(edited["availability"]["alignment"])
        with self.assertRaises(FileNotFoundError):
            self.catalog.resolve_media(token, "audio")
        self.catalog.refresh(force=True)
        self.assertFalse(self.catalog.get_item(token)["availability"]["audio"])
        effective = self.catalog.review_store.resolve_record(record)
        revised_dir = speech_root / speech_sample_uid(effective)
        revised_dir.mkdir()
        (revised_dir / "stereo_dialogue.wav").write_bytes(b"revised audio")
        (revised_dir / "assistant_dialogue.refined.json").write_text('{"alignments": []}')
        (revised_dir / "dialogue.json").write_text(json.dumps({"uid": revised_dir.name, "source_uid": record["uid"],
            "task_mode": "normal", "review_id": content_fingerprint(effective), "duration": 3.0}))
        self.catalog.refresh()
        finished = self.catalog.get_item(token)
        self.assertTrue(finished["availability"]["audio"])
        self.assertTrue(finished["availability"]["alignment"])
        self.assertEqual(self.catalog.resolve_media(token, "audio"), revised_dir / "stereo_dialogue.wav")
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual((original / "stereo_dialogue.wav").read_bytes(), b"original audio")
        with self.assertRaises(ValueError):
            self.catalog.edit_dialogue(token, fingerprint=finished["content_sha256"], dialogue=[{"text": ""}, {"text": "2個"}])

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.data = self.repo / "data"

        image = self.data / "images/coco/train/a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"fake image")
        self.visual_image = image
        visual_uid = "visual-a"
        speech = self.data / f"speech_v3/data_stereo/{visual_uid}"
        speech.mkdir(parents=True)
        (speech / "stereo_dialogue.wav").write_bytes(b"RIFFfake")
        (speech / "image.jpg").write_bytes(b"speech copy")
        (speech / "assistant_dialogue.refined.json").write_text(
            json.dumps(
                {
                    "format": "spm_pieces_v2",
                    "alignments": [["▁猫", [1.0, 1.2], "ASSISTANT"]],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (speech / "dialogue.json").write_text(
            json.dumps({"uid": visual_uid, "duration": 2.0}), encoding="utf-8"
        )
        _write_jsonl(
            self.data / "dialogues_merged/dialogues_final_v3.jsonl",
            [
                {
                    "uid": visual_uid,
                    "image": str(image),
                    "dialogue": [
                        {"speaker": "ユーザー", "text": "何が見える？"},
                        {"speaker": "アシスタント", "text": "猫です"},
                    ],
                    "generator": "test",
                }
            ],
        )
        _write_jsonl(
            self.data / "speech_v3/asset_manifest.v1.jsonl",
            [
                {"type": "header"},
                {
                    "type": "sample",
                    "sample_id": visual_uid,
                    "split": "train",
                    "assets": [
                        {"role": "image", "path": str(speech / "image.jpg")},
                        {"role": "audio", "path": str(speech / "stereo_dialogue.wav")},
                        {
                            "role": "alignment",
                            "path": str(speech / "assistant_dialogue.refined.json"),
                            "entries": 1,
                        },
                    ],
                },
            ],
        )

        counting_image = self.data / "images/counting/two.jpg"
        counting_image.parent.mkdir(parents=True)
        counting_image.write_bytes(b"count")
        self.counting_one_image = self.data / "images/counting/one.jpg"
        self.counting_one_image.write_bytes(b"one count")
        _write_jsonl(
            self.data / "images/targeted_v4/image_pool_v1/counting_images.jsonl",
            [
                {
                    "candidate_uid": "count:one",
                    "path": "data/images/counting/one.jpg",
                    "capability": "counting",
                    "reference_annotations": [{"answer_count": 1}],
                },
                {
                    "candidate_uid": "count:two",
                    "path": "data/images/counting/two.jpg",
                    "capability": "counting",
                    "reference_annotations": [{"answer_count": 2}],
                },
            ],
        )
        _write_jsonl(
            self.data / "dialogues_counting/full/qwen3vl_30b_counting_seed5.jsonl",
            [
                {
                    "uid": "count:two",
                    "image": "data/images/counting/two.jpg",
                    "dialogue": [
                        {"speaker": "ユーザー", "text": "いくつ？"},
                        {"speaker": "アシスタント", "text": "2つです"},
                    ],
                    "grounding": {"answer_count": 2},
                }
            ],
            # A writer has not emitted the final newline yet; this must not leak.
            partial={
                "uid": "count:partial",
                "dialogue": [{"speaker": "ユーザー", "text": "途中"}],
            },
        )

        nonvisual_uid = "nonvisual-a"
        nonvisual_speech = self.data / f"speech_nonvisual_v1/data_stereo/{nonvisual_uid}"
        nonvisual_speech.mkdir(parents=True)
        (nonvisual_speech / "stereo_dialogue.wav").write_bytes(b"RIFFnonvisual")
        (nonvisual_speech / "assistant_dialogue.refined.json").write_text(
            json.dumps(
                {
                    "format": "spm_pieces_v2",
                    "alignments": [["▁そうですね", [1.0, 1.5], "ASSISTANT"]],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (nonvisual_speech / "dialogue.json").write_text(
            json.dumps({"uid": nonvisual_uid, "duration": 2.0}), encoding="utf-8"
        )
        _write_jsonl(
            self.data / "dialogues_nonvisual/claude_sonnet5_nonvisual_tts_v1.jsonl",
            [
                {
                    "uid": nonvisual_uid,
                    "kind": "nonvisual",
                    "topic": "日常会話",
                    "dialogue": [
                        {"speaker": "ユーザー", "text": "今日は何をしよう？"},
                        {"speaker": "アシスタント", "text": "散歩はいかがですか？"},
                    ],
                    "generator": "test",
                }
            ],
        )
        self.catalog = DatasetCatalog(self.repo, self.repo / ".cache/catalog.sqlite3")
        self.catalog.refresh()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_dashboard_stages_and_eligibility(self) -> None:
        dashboard = self.catalog.dashboard()
        self.assertEqual(dashboard["totals"]["items"], 4)
        self.assertEqual(dashboard["totals"]["images"], 3)
        self.assertEqual(dashboard["totals"]["dialogues"], 3)
        self.assertEqual(dashboard["totals"]["audio"], 2)
        self.assertEqual(dashboard["totals"]["alignments"], 2)
        self.assertEqual(dashboard["totals"]["audio_duration_seconds"], 4.0)
        self.assertEqual(dashboard["totals"]["audio_duration_known"], 2)

        visual = dashboard["groups"]["visual"]
        self.assertEqual(visual["items"], 3)
        self.assertEqual(visual["images"], 3)
        self.assertEqual(visual["dialogues"], 2)
        self.assertEqual(visual["audio"], 1)
        self.assertEqual(visual["alignments"], 1)
        self.assertEqual(visual["audio_duration_seconds"], 2.0)
        self.assertEqual(visual["audio_duration_known"], 1)

        nonvisual = dashboard["groups"]["nonvisual"]
        self.assertEqual(nonvisual["items"], 1)
        self.assertEqual(nonvisual["images"], 0)
        self.assertEqual(nonvisual["dialogues"], 1)
        self.assertEqual(nonvisual["audio"], 1)
        self.assertEqual(nonvisual["alignments"], 1)
        self.assertEqual(nonvisual["audio_duration_seconds"], 2.0)
        self.assertEqual(nonvisual["audio_duration_known"], 1)

        nonvisual_only = self.catalog.dashboard(collection="nonvisual_v1")
        self.assertEqual(nonvisual_only["totals"]["items"], 1)
        self.assertEqual(nonvisual_only["groups"]["visual"]["items"], 0)
        self.assertEqual(nonvisual_only["groups"]["nonvisual"]["items"], 1)
        self.assertEqual(nonvisual_only["groups"]["nonvisual"]["images"], 0)

        counting = next(row for row in dashboard["collections"] if row["collection"] == "counting_v4")
        self.assertEqual(counting["eligible"], 1)
        self.assertEqual(counting["planned"], 1)

    def test_filters_and_partial_jsonl(self) -> None:
        dialogues = self.catalog.list_items(collection="counting_v4", stage="dialogue")
        self.assertEqual(dialogues["total"], 1)
        self.assertEqual(dialogues["items"][0]["uid"], "count:two")
        self.assertEqual(
            self.catalog.list_items(collection="counting_v4", stage="missing_dialogue")[
                "total"
            ],
            1,
        )
        self.assertEqual(
            self.catalog.list_items(
                collection="counting_v4", stage="planned_missing_dialogue"
            )["total"],
            0,
        )
        self.assertEqual(self.catalog.list_items(query="partial")["total"], 0)
        eligible = [item for item in self.catalog.list_items(collection="counting_v4")["items"] if item["eligible"]]
        self.assertEqual([item["uid"] for item in eligible], ["count:two"])
        visual_directory = self.catalog.list_items(collection="visual_v3")["items"][0][
            "directory"
        ]
        self.assertEqual(visual_directory, "data/images/coco")
        self.assertEqual(
            self.catalog.list_items(directory=visual_directory)["total"], 1
        )

    def test_ocr_eligibility_matches_short_text_rule(self) -> None:
        _write_jsonl(
            self.data / "images/targeted_v4/image_pool_v1/ocr_images.jsonl",
            [
                {
                    "candidate_uid": "ocr:short",
                    "path": "data/images/ocr/short.jpg",
                    "capability": "japanese_ocr",
                    "reference_annotations": [
                        {"kind": "exact_transcription", "text": " 営業 中 "}
                    ],
                },
                {
                    "candidate_uid": "ocr:long",
                    "path": "data/images/ocr/long.jpg",
                    "capability": "japanese_ocr",
                    "reference_annotations": [
                        {"kind": "exact_transcription", "text": "長" * 25}
                    ],
                },
            ],
        )
        self.catalog.refresh()
        items = self.catalog.list_items(collection="ocr_v4", limit=10)["items"]
        self.assertEqual(len(items), 2)
        self.assertEqual([item["uid"] for item in items if item["eligible"]], ["ocr:short"])

    def test_detail_alignment_and_navigation(self) -> None:
        visual = self.catalog.list_items(collection="visual_v3")["items"][0]
        detail = self.catalog.get_item(visual["token"], {"collection": "visual_v3"})
        self.assertEqual(detail["dialogue"][1]["speaker"], "assistant")
        self.assertEqual(detail["alignment"]["items"][0]["text"], "▁猫")
        self.assertTrue(detail["availability"]["audio"])
        self.assertEqual(detail["navigation"]["index"], 1)
        self.assertEqual(detail["navigation"]["total"], 1)

        counting = self.catalog.list_items(collection="counting_v4", limit=10)["items"]
        first = self.catalog.get_item(counting[0]["token"], {"collection": "counting_v4"})
        second = self.catalog.get_item(first["navigation"]["next"], {"collection": "counting_v4"})
        self.assertEqual(second["navigation"]["previous"], counting[0]["token"])

    def test_incremental_refresh_completes_partial_line(self) -> None:
        path = self.data / "dialogues_counting/full/qwen3vl_30b_counting_seed5.jsonl"
        with path.open("ab") as stream:
            stream.write(b"\n")
        self.catalog.refresh()
        self.assertEqual(self.catalog.list_items(query="count:partial")["total"], 1)

    def test_japanese_daily_image_bank_is_indexed_incrementally(self) -> None:
        image_root = self.data / "images/japanese_daily_v1/images"
        image_root.mkdir(parents=True)
        records: list[dict] = []
        for index, bucket in enumerate(
            ("japanese_text_rich", "japanese_everyday", "shops_and_streets"),
            start=1,
        ):
            image = image_root / f"daily-{index}.jpg"
            image.write_bytes(f"daily image {index}".encode())
            records.append(
                {
                    "uid": f"daily-{index}",
                    "source_uid": f"commons-{index}",
                    "path": str(image),
                    "source": "wikimedia_commons",
                    "assigned_bucket": bucket,
                    "collection_buckets": [bucket],
                    "task_routing": "vlm_adaptive",
                    "title": f"日本の日常 {index}",
                    "source_url": f"https://commons.wikimedia.org/wiki/File:daily-{index}",
                    "license": "CC BY-SA 4.0",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
                    "artist": "test author",
                    "download": {"decoded_width": 1280, "decoded_height": 960},
                    "fingerprints": {"file_sha256": str(index) * 64},
                }
            )

        manifest = self.data / "images/japanese_daily_v1/accepted_images.jsonl"
        _write_jsonl(manifest, [records[0]])
        self.catalog.refresh()

        first = self.catalog.list_items(collection="japanese_daily_v1")
        self.assertEqual(first["total"], 1)
        item = first["items"][0]
        self.assertEqual(item["category"], "japanese_text_rich")
        self.assertTrue(item["eligible"])
        self.assertEqual(item["stage"], "image")
        self.assertEqual(
            item["availability"],
            {"image": True, "dialogue": False, "audio": False, "alignment": False},
        )
        detail = self.catalog.get_item(item["token"])
        self.assertEqual(detail["metadata"]["license"], "CC BY-SA 4.0")
        self.assertEqual(detail["metadata"]["artist"], "test author")
        self.assertFalse(detail["summary"]["training_ready"])

        with manifest.open("ab") as stream:
            stream.write(
                json.dumps(records[1], ensure_ascii=False).encode("utf-8") + b"\n"
            )
            stream.write(json.dumps(records[2], ensure_ascii=False).encode("utf-8"))
        self.catalog.refresh()
        self.assertEqual(
            self.catalog.list_items(collection="japanese_daily_v1")["total"], 2
        )
        source = next(
            row
            for row in self.catalog.status()["sources"]
            if row["source_id"] == "japanese_daily_v1.images"
        )
        self.assertEqual(source["row_count"], 2)
        self.assertEqual(source["ignored_partial"], 1)
        self.assertLess(source["indexed_bytes"], source["size_bytes"])

        with manifest.open("ab") as stream:
            stream.write(b"\n")
        self.catalog.refresh()
        self.assertEqual(
            self.catalog.list_items(collection="japanese_daily_v1")["total"], 3
        )
        source = next(
            row
            for row in self.catalog.status()["sources"]
            if row["source_id"] == "japanese_daily_v1.images"
        )
        self.assertEqual(source["row_count"], 3)
        self.assertEqual(source["ignored_partial"], 0)
        self.assertEqual(source["indexed_bytes"], source["size_bytes"])

        dashboard = self.catalog.dashboard()
        daily = next(
            row
            for row in dashboard["collections"]
            if row["collection"] == "japanese_daily_v1"
        )
        self.assertEqual(daily["images"], 3)
        self.assertEqual(daily["dialogues"], 0)
        self.assertEqual(daily["eligible"], 3)

    def test_refresh_discovers_nonvisual_media_without_source_change(self) -> None:
        uid = "nonvisual-pending"
        source = (
            self.data / "dialogues_nonvisual/claude_sonnet5_nonvisual_tts_v1.jsonl"
        )
        with source.open("ab") as stream:
            stream.write(
                json.dumps(
                    {
                        "uid": uid,
                        "kind": "nonvisual",
                        "topic": "日常会話",
                        "dialogue": [
                            {"speaker": "ユーザー", "text": "元気ですか？"},
                            {"speaker": "アシスタント", "text": "元気です。"},
                        ],
                        "generator": "test",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
        self.catalog.refresh()

        source_stat = source.stat()
        before_dashboard = self.catalog.dashboard()
        before_status = self.catalog.status()

        speech = self.data / f"speech_nonvisual_v1/data_stereo/{uid}"
        speech.mkdir(parents=True)
        (speech / "stereo_dialogue.wav").write_bytes(b"RIFFpending")
        (speech / "assistant_dialogue.refined.json").write_text(
            json.dumps(
                {
                    "format": "spm_pieces_v2",
                    "alignments": [["▁元気です", [1.0, 1.5], "ASSISTANT"]],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.assertEqual(source.stat(), source_stat)

        self.catalog.refresh()
        after_dashboard = self.catalog.dashboard()
        after_status = self.catalog.status()
        self.assertEqual(
            after_dashboard["totals"]["audio"],
            before_dashboard["totals"]["audio"] + 1,
        )
        self.assertEqual(
            after_dashboard["totals"]["alignments"],
            before_dashboard["totals"]["alignments"] + 1,
        )
        self.assertEqual(
            after_status["catalog_revision"],
            before_status["catalog_revision"] + 1,
        )

        self.catalog.refresh()
        unchanged_status = self.catalog.status()
        self.assertEqual(
            unchanged_status["catalog_revision"], after_status["catalog_revision"]
        )
        self.assertEqual(unchanged_status["last_changed"], after_status["last_changed"])

    def test_targeted_speech_root_joins_by_source_uid(self) -> None:
        storage_uid = "count:two__task_counting"
        sample = self.data / "speech_counting_v4/data_stereo" / storage_uid
        sample.mkdir(parents=True)
        (sample / "stereo_dialogue.wav").write_bytes(b"RIFFtargeted")
        (sample / "assistant_dialogue.refined.json").write_text(
            json.dumps(
                {"alignments": [["2つです", [1.0, 1.4], "ASSISTANT"]]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (sample / "dialogue.json").write_text(
            json.dumps(
                {
                    "uid": storage_uid,
                    "source_uid": "count:two",
                    "task_mode": "counting",
                    "duration": 2.0,
                    "segments": [
                        {"speaker": "USER", "text": "いくつ？"},
                        {"speaker": "ASSISTANT", "text": "2つです"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.catalog.refresh()
        item = self.catalog.list_items(collection="counting_v4", query="count:two")[
            "items"
        ][0]
        detail = self.catalog.get_item(item["token"])
        self.assertTrue(detail["availability"]["audio"])
        self.assertTrue(detail["availability"]["alignment"])
        self.assertEqual(detail["alignment"]["count"], 1)
        self.assertEqual(
            self.catalog.resolve_media(item["token"], "audio"),
            (sample / "stereo_dialogue.wav").resolve(),
        )
        # Unchanged media must not trigger a full collection rebuild each tick.
        with patch.object(self.catalog, "_rebuild_collection", wraps=self.catalog._rebuild_collection) as full, \
             patch.object(self.catalog, "_rebuild_items_for_uids", wraps=self.catalog._rebuild_items_for_uids) as partial:
            self.catalog.refresh()
            full.assert_not_called()
            partial.assert_not_called()
            # Changes inside a sample (without changing the parent directory)
            # still refresh that UID, including removal of existing media.
            (sample / "assistant_dialogue.refined.json").unlink()
            self.catalog.refresh()
            full.assert_not_called()
            partial.assert_called_once()
            self.assertEqual(partial.call_args.args[1], "counting_v4")
            self.assertEqual(set(partial.call_args.args[2]), {"count:two"})
        self.assertFalse(self.catalog.get_item(item["token"])["availability"]["alignment"])

    def test_normal_speech_uses_its_current_counting_root(self) -> None:
        sample = self.data / "speech_counting_normal/data_stereo/count:one"
        sample.mkdir(parents=True)
        (sample / "stereo_dialogue.wav").write_bytes(b"RIFFnormal")
        (sample / "dialogue.json").write_text(
            json.dumps(
                {
                    "uid": "count:one",
                    "source_uid": "count:one",
                    "task_mode": "normal",
                    "duration": 1.5,
                    "segments": [
                        {"speaker": "USER", "text": "何が見えますか？"},
                        {"speaker": "ASSISTANT", "text": "リンゴです。"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # This later unrelated speech root made the old implementation reuse
        # the wrong loop variable and silently discard the normal sample.
        (self.data / "speech_z_unrelated").mkdir()

        self.catalog.refresh()
        item = self.catalog.list_items(
            collection="counting_v4", query="count:one"
        )["items"][0]
        self.assertTrue(item["availability"]["audio"])
        self.assertEqual(item["stage"], "audio")

    def test_ocr_document_speech_is_visible_in_ocr_collection(self):
        sample = self.data / "speech_ocr_v1/data_stereo/ocr:doc__task_ocr_document"
        sample.mkdir(parents=True)
        (sample / "stereo_dialogue.wav").write_bytes(b"RIFFocr")
        (sample / "dialogue.json").write_text(json.dumps({
            "uid": sample.name, "source_uid": "ocr:doc", "task_mode": "ocr_document",
            "duration": 2, "segments": [{"speaker": "USER", "text": "何と書いてありますか？"},
                                       {"speaker": "ASSISTANT", "text": "営業時間です。"}]
        }), encoding="utf-8")
        self.catalog.refresh()
        item = self.catalog.list_items(collection="ocr_v4", query="ocr:doc")["items"][0]
        self.assertTrue(item["availability"]["audio"])
        self.assertEqual(item["uid"], "ocr:doc")

    def test_unverified_adaptive_dialogue_requires_exact_review(self) -> None:
        source = (
            self.data
            / "dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl"
        )
        record = {
            "uid": "count:one",
            "image": "data/images/counting/one.jpg",
            "image_sha256": image_file_sha256(self.counting_one_image),
            "task_mode": "normal",
            "normal_strategy": "vlm_adaptive",
            "dialogue": [
                {"speaker": "ユーザー", "text": "写真には何がありますか？"},
                {"speaker": "アシスタント", "text": "リンゴがあります。"},
                {"speaker": "ユーザー", "text": "リンゴはいくつありますか？"},
                {"speaker": "アシスタント", "text": "1個あります。"},
            ],
            "auxiliary_task": {
                "kind": "counting",
                "grounding_source": "vlm_visual",
                "question_turn_index": 2,
                "answer_turn_index": 3,
                "counted_target": "リンゴ",
                "answer_count": 1,
            },
            "verification_status": "unverified_vlm_visual",
            "generator": "test-vlm",
        }
        _write_jsonl(source, [record])
        self.catalog.refresh()

        pending = self.catalog.list_items(
            collection="counting_v4",
            review_required=True,
            review_status="pending",
            auxiliary_kind="counting",
        )
        self.assertEqual(pending["total"], 1)
        item = pending["items"][0]
        self.assertEqual(item["uid"], "count:one")
        self.assertTrue(item["review_required"])
        self.assertFalse(item["dataset_usable"])
        self.assertEqual(item["verification_status"], "unverified_vlm_visual")
        self.assertEqual(item["auxiliary_task"]["answer_count"], 1)
        fingerprint = item["content_sha256"]
        self.assertEqual(len(fingerprint), 64)

        detail = self.catalog.get_item(
            item["token"],
            {
                "collection": "counting_v4",
                "review_status": "pending",
                "review_required": True,
                "auxiliary_kind": "counting",
            },
        )
        self.assertEqual(detail["navigation"]["total"], 1)
        approved = self.catalog.review_item(
            item["token"],
            fingerprint=fingerprint,
            decision="approved",
            note="画像と回答を確認",
        )
        self.assertTrue(approved["dataset_usable"])
        reviewed_detail = self.catalog.get_item(item["token"])
        self.assertEqual(reviewed_detail["review"]["status"], "approved")
        self.assertEqual(reviewed_detail["review"]["note"], "画像と回答を確認")
        self.assertIsNotNone(reviewed_detail["review"]["reviewed_at"])
        self.assertEqual(
            self.catalog.list_items(review_status="approved")["total"], 1
        )

        self.counting_one_image.write_bytes(b"changed image bytes")
        self.catalog.refresh()
        image_changed = self.catalog.list_items(
            collection="counting_v4", review_required=True
        )["items"][0]
        self.assertNotEqual(image_changed["content_sha256"], fingerprint)
        self.assertEqual(image_changed["review_status"], "pending")
        self.assertFalse(image_changed["dataset_usable"])
        with self.assertRaises(ReviewConflictError):
            self.catalog.review_item(
                image_changed["token"],
                fingerprint=fingerprint,
                decision="approved",
            )
        fingerprint = image_changed["content_sha256"]
        self.catalog.review_item(
            image_changed["token"],
            fingerprint=fingerprint,
            decision="approved",
            note="差替え後の画像も再確認",
        )
        # PUT itself re-hashes; this must fail closed even before periodic
        # catalog refresh notices the second replacement.
        self.counting_one_image.write_bytes(b"changed again without refresh")
        with self.assertRaises(ReviewConflictError):
            self.catalog.review_item(
                image_changed["token"],
                fingerprint=fingerprint,
                decision="approved",
            )
        direct_change = self.catalog.list_items(
            collection="counting_v4", review_required=True
        )["items"][0]
        self.assertNotEqual(direct_change["content_sha256"], fingerprint)
        self.assertEqual(direct_change["review_status"], "pending")
        fingerprint = direct_change["content_sha256"]
        self.catalog.review_item(
            direct_change["token"],
            fingerprint=fingerprint,
            decision="approved",
            note="承認直前hash検証後に再確認",
        )

        stale_speech = (
            self.data
            / "speech_counting_review/data_stereo/count:one__task_normal"
        )
        stale_speech.mkdir(parents=True)
        (stale_speech / "dialogue.json").write_text(
            json.dumps(
                {
                    "uid": "count:one__task_normal",
                    "source_uid": "count:one",
                    "task_mode": "normal",
                    "segments": [
                        {"speaker": "USER", "text": turn["text"]}
                        if turn["speaker"] == "ユーザー"
                        else {"speaker": "ASSISTANT", "text": turn["text"]}
                        for turn in record["dialogue"]
                    ],
                    "review_required": True,
                    # Deliberately stale: synthesized before source-image
                    # replacement and retained after a new approval.
                    "review_id": item["content_sha256"],
                    "content_sha256": item["content_sha256"],
                    "verification_status": "unverified_vlm_visual",
                    "auxiliary_task": record["auxiliary_task"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.catalog.refresh()
        self.assertEqual(
            self.catalog.list_items(review_status="approved")["total"], 1
        )

        # A regenerated dialogue has a new content hash.  The old approval is
        # retained in the ledger but cannot approve the replacement, even when
        # an old synthesized-speech directory still carries the former hash.
        record["dialogue"][-1]["text"] = "リンゴは一つです。"
        _write_jsonl(source, [record])
        self.catalog.refresh()
        regenerated = self.catalog.list_items(
            collection="counting_v4", review_required=True
        )["items"][0]
        self.assertNotEqual(regenerated["content_sha256"], fingerprint)
        self.assertEqual(regenerated["review_status"], "pending")
        self.assertFalse(regenerated["dataset_usable"])
        with self.assertRaises(ReviewConflictError):
            self.catalog.review_item(
                regenerated["token"],
                fingerprint=fingerprint,
                decision="approved",
            )

    def test_review_candidate_convention_is_allowlisted(self) -> None:
        source = (
            self.data
            / "dialogues_review_candidates/ocr_v4/future.jsonl"
        )
        _write_jsonl(
            source,
            [
                {
                    "uid": "ocr:future",
                    "image": "data/images/ocr/future.jpg",
                    "task_mode": "normal",
                    "dialogue": [
                        {"speaker": "ユーザー", "text": "何が見えますか？"},
                        {"speaker": "アシスタント", "text": "看板です。"},
                    ],
                    "auxiliary_task": {"kind": "none", "grounding_source": "none"},
                    "generator": "future-vlm",
                }
            ],
        )
        self.catalog.refresh()
        item = self.catalog.list_items(collection="ocr_v4", query="ocr:future")[
            "items"
        ][0]
        self.assertEqual(item["category"], "normal")
        self.assertFalse(item["review_required"])
        self.assertTrue(item["dataset_usable"])

    def test_media_path_cannot_escape_data_root(self) -> None:
        visual = self.catalog.list_items(collection="visual_v3")["items"][0]
        # The original source image must win over the speech directory's copy.
        self.assertEqual(
            self.catalog.resolve_media(visual["token"], "image"),
            self.visual_image.resolve(),
        )
        with self.catalog._connect() as db:  # deliberate corruption simulates hostile metadata
            db.execute(
                "UPDATE items SET image_path='../outside.jpg' WHERE collection='visual_v3'"
            )
        with self.assertRaises(ValueError):
            self.catalog.resolve_media(visual["token"], "image")
        with self.assertRaises(ValueError):
            self.catalog.get_item("not-a-valid-token")


if __name__ == "__main__":
    unittest.main()
