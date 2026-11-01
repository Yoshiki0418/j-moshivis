from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_viewer import app_daily
from dataset_viewer import catalog as catalog_module
from dataset_viewer.catalog import DatasetCatalog
from ssvd.review_store import ReviewConflictError, image_file_sha256


class DailyDialogueViewerTest(unittest.TestCase):
    def test_live_dialogues_join_images_and_preserve_review_gate(self) -> None:
        original_specs = catalog_module._SOURCE_SPECS
        original_component = DatasetCatalog._component_from_record
        self.addCleanup(setattr, catalog_module, "_SOURCE_SPECS", original_specs)
        self.addCleanup(setattr, DatasetCatalog, "_component_from_record", original_component)
        app_daily._install_daily_support()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "data/images/japanese_daily_v1/images"
            image_root.mkdir(parents=True)
            images = []
            for number in (1, 2):
                image = image_root / f"daily-{number}.jpg"
                image.write_bytes(f"image {number}".encode())
                images.append(image)

            manifest = root / "data/images/japanese_daily_v1/accepted_images.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                for number, image in enumerate(images, 1):
                    handle.write(json.dumps({
                        "uid": f"daily-{number}",
                        "path": str(image),
                        "assigned_bucket": "japanese_text_rich",
                    }) + "\n")

            output = root / app_daily.DAILY_DIALOGUE_PATH
            output.parent.mkdir(parents=True)
            first = {
                "uid": "daily-1",
                "image": str(images[0]),
                "dialogue": [
                    {"speaker": "ユーザー", "text": "何が見えますか？"},
                    {"speaker": "アシスタント", "text": "お店です。"},
                ],
                "generator": "test-vlm",
                "task_mode": "normal",
                "normal_strategy": "vlm_adaptive",
                "auxiliary_task": {
                    "kind": "none", "grounding_source": "none",
                    "question_turn_index": None, "answer_turn_index": None,
                },
            }
            output.write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")

            catalog = DatasetCatalog(root, root / "data/.dataset_viewer/test.sqlite3")
            catalog.refresh()
            daily = next(row for row in catalog.dashboard()["collections"]
                         if row["collection"] == "japanese_daily_v1")
            self.assertEqual((daily["images"], daily["dialogues"]), (2, 1))
            listed = catalog.list_items(collection="japanese_daily_v1", query="daily-1")
            item = listed["items"][0]
            self.assertEqual(item["category"], "japanese_text_rich")
            self.assertEqual(item["stage"], "dialogue")
            detail = catalog.get_item(item["token"])
            self.assertEqual(detail["summary"]["generator"], "test-vlm")
            self.assertEqual(detail["summary"]["normal_strategy"], "vlm_adaptive")

            second = {
                **first,
                "uid": "daily-2",
                "image": str(images[1]),
                "image_sha256": image_file_sha256(images[1]),
                "dialogue": [
                    {"speaker": "ユーザー", "text": "看板には何と書いてある？"},
                    {"speaker": "アシスタント", "text": "青空商店です。"},
                ],
                "verification_status": "unverified_vlm_visual",
                "review_required": True,
                "auxiliary_task": {
                    "kind": "ocr", "grounding_source": "vlm_visual",
                    "question_turn_index": 0, "answer_turn_index": 1,
                    "recognized_text": "青空商店",
                },
            }
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second, ensure_ascii=False) + "\n")
            catalog.refresh()
            daily = next(row for row in catalog.dashboard()["collections"]
                         if row["collection"] == "japanese_daily_v1")
            self.assertEqual((daily["images"], daily["dialogues"]), (2, 2))
            listed = catalog.list_items(collection="japanese_daily_v1", query="daily-2")
            self.assertEqual(listed["items"][0]["review_status"], "pending")
            detail = catalog.get_item(listed["items"][0]["token"])
            self.assertEqual(detail["summary"]["verification_status"],
                             "unverified_vlm_visual")
            original_bytes = output.read_bytes()
            token = detail["token"]
            old_id = detail["review"]["content_sha256"]
            changed_turns = [{"text": "店の看板は何と書かれていますか？"},
                             {"text": "看板には青葉商店と書かれています。"}]
            result = catalog.edit_dialogue(token, fingerprint=old_id, dialogue=changed_turns,
                auxiliary_task={"recognized_text": "青葉商店"}, note="画像を見て文字を修正")
            edited = result["item"]
            new_id = edited["review"]["content_sha256"]
            self.assertNotEqual(new_id, old_id)
            self.assertEqual(edited["review"]["status"], "approved")
            self.assertEqual(edited["dialogue"][1]["text"], changed_turns[1]["text"])
            self.assertEqual(edited["summary"]["auxiliary_task"]["recognized_text"], "青葉商店")
            self.assertEqual(output.read_bytes(), original_bytes)
            catalog.refresh(force=True)
            reopened = DatasetCatalog(root, catalog.db_path).get_item(token)
            self.assertEqual(reopened["dialogue"], edited["dialogue"])
            self.assertEqual(reopened["review"]["status"], "approved")
            with self.assertRaises(ReviewConflictError):
                catalog.edit_dialogue(token, fingerprint=old_id, dialogue=changed_turns)
            catalog.review_item(token, fingerprint=new_id, decision="rejected", note="後で一括再生成")
            self.assertEqual(catalog.get_item(token)["dialogue"], edited["dialogue"])
            self.assertEqual(catalog.list_items(collection="japanese_daily_v1", review_status="rejected")["total"], 1)
            changed_turns[1]["text"] = "青葉商店というお店です。"
            second_edit = catalog.edit_dialogue(token, fingerprint=new_id, dialogue=changed_turns)["item"]
            self.assertEqual(second_edit["review"]["status"], "approved")
            self.assertNotEqual(second_edit["review"]["content_sha256"], new_id)
            images[1].write_bytes(b"replaced image")
            catalog.refresh()
            after_change = catalog.get_item(token)
            self.assertEqual(after_change["review"]["status"], "pending")
            self.assertEqual(after_change["dialogue"][1]["text"], second["dialogue"][1]["text"])
            self.assertIsNone(after_change["human_revision"])


if __name__ == "__main__":
    unittest.main()
