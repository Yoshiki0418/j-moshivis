# pylint: disable=C0413,C0411
import os
import json
import sqlite3
from hashlib import sha256
from typing import Literal, Optional
from io import BytesIO
import random

import requests
import datasets
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from multiturn_instruct import MTCInstruct


# === モデルロード ===
def load_vlm(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )
    return processor, model


# === 画像取得ユーティリティ ===
def fetch_image(sample, dataset_name: str):
    if dataset_name in {"pixmo", "pixelprose"}:
        url = sample.get("image_url")
        if not url:
            return None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            return None
    if dataset_name == "docci":
        img = sample["image"]
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            return Image.fromarray(img).convert("RGB")
    return None


# === 1ターン生成関数 ===
@torch.inference_mode()
def vlm_generate_turn(processor, model, pil_image, prompt_text: str, chat_history: str = ""):
    """画像＋テキストプロンプトで単一応答を生成"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": chat_history + "\n" + prompt_text},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[pil_image], return_tensors="pt").to(
        model.device, dtype=model.dtype
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# === メインロジック ===
class LauncherVLM:
    """画像入力VLMを使って質問者・回答者を交互に動作させるマルチターン対話生成"""

    def run(
        self,
        task: str = "comb",
        dataset: Literal["docci", "pixelprose", "pixmo"] = "pixelprose",
        split: str = "train",
        out_dir: str = "./synthetic_visual_dialogues_vlm_mt",
        max_samples: Optional[int] = 1000,
        convo_length: int = 8,
        overwrite: Literal["yes", "no"] = "no",
        output_format: Literal["jsonl", "db", "both"] = "jsonl",
    ):
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, f"{task}_{dataset}_{split}.jsonl")
        db_path = os.path.join(out_dir, f"{dataset}_ssvd.db")

        # --- データ読み込み ---
        print("📦 Loading dataset...")
        if dataset == "pixmo":
            ds = datasets.load_dataset("allenai/pixmo-cap", split=split)
            ds = ds.add_column("uid", [sha256(x.encode()).hexdigest() for x in ds["image_url"]])
            ds = ds.select_columns(["uid", "image_url"])
        elif dataset == "pixelprose":
            ds = datasets.load_dataset("tomg-group-umd/pixelprose", split=split)
            ds = ds.filter(lambda x: x.get("url") is not None)
            ds = ds.rename_column("url", "image_url")
            if "uid" not in ds.column_names:
                ds = ds.add_column("uid", [sha256(x.encode()).hexdigest() for x in ds["image_url"]])
            ds = ds.select_columns(["uid", "image_url"])
        elif dataset == "docci":
            ds = datasets.load_dataset("google/docci", split=split)
            ds = ds.rename_column("example_id", "uid")
            ds = ds.select_columns(["uid", "image"])
        else:
            raise NotImplementedError(f"Unsupported dataset {dataset}")

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"[info] Loaded {len(ds)} samples")

        # --- モデルロード ---
        print("🧠 Loading Qwen2.5-VL-7B-Instruct ...")
        processor, model = load_vlm()
        print("✅ Model ready.")

        # --- instruct設定 ---
        mtc = MTCInstruct(task)
        setting_func = mtc.get_method()
        system_template, system_1, system_2, start_conv = setting_func()

        # --- 出力初期化 ---
        if output_format in {"db", "both"}:
            db = sqlite3.connect(db_path, timeout=60, isolation_level=None)
            cur = db.cursor()
            table_name = f"{split}_{task}"
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    uid TEXT,
                    turn INTEGER,
                    speaker TEXT,
                    text TEXT,
                    PRIMARY KEY(uid, turn, speaker, text)
                )
                """
            )
        else:
            db, cur, table_name = None, None, None

        # --- メインループ ---
        for sample in tqdm(ds):
            uid = sample["uid"]
            pil_img = fetch_image(sample, dataset)
            if pil_img is None:
                continue

            dialogue = []
            chat_history = ""

            # 最初の質問
            user_prompt = start_conv + "\n" + system_1.format(caption="")
            question = vlm_generate_turn(processor, model, pil_img, user_prompt, chat_history)
            dialogue.append({"speaker": "ユーザー", "text": question})
            chat_history += f"ユーザー: {question}\n"

            # 以降のターン
            for turn in range(1, convo_length):
                if turn % 2 == 1:  # アシスタント応答
                    asst_prompt = system_2.format(caption="")
                    answer = vlm_generate_turn(processor, model, pil_img, asst_prompt, chat_history)
                    dialogue.append({"speaker": "アシスタント", "text": answer})
                    chat_history += f"アシスタント: {answer}\n"
                else:  # 再びユーザー質問
                    user_prompt = system_1.format(caption="")
                    question = vlm_generate_turn(processor, model, pil_img, user_prompt, chat_history)
                    dialogue.append({"speaker": "ユーザー", "text": question})
                    chat_history += f"ユーザー: {question}\n"

            # --- JSON出力 ---
            if output_format in {"jsonl", "both"}:
                with open(jsonl_path, "a", encoding="utf-8") as fout:
                    json.dump({"uid": uid, "dialogue": dialogue}, fout, ensure_ascii=False)
                    fout.write("\n")

            # --- DB出力 ---
            if output_format in {"db", "both"}:
                for i, turn in enumerate(dialogue):
                    try:
                        cur.execute(
                            f"INSERT OR REPLACE INTO {table_name} VALUES(?, ?, ?, ?)",
                            (uid, i, turn["speaker"], turn["text"]),
                        )
                    except sqlite3.Error:
                        continue

        # --- 後処理 ---
        if output_format in {"db", "both"}:
            db.commit()
            cur.close()
            db.close()

        print("🎉 Done! Multiturn visual dialogues generated successfully.")

if __name__ == "__main__":
    import fire
    fire.Fire(LauncherVLM)