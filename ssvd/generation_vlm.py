import os
import json
import sqlite3
from hashlib import sha256
from typing import Literal, Optional
from io import BytesIO
import gc

import requests
import datasets
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


# === 対話生成プロンプト ===
DIALOGUE_PROMPT = """あなたは画像を見て人間と自然に会話する日本語のAIアシスタントです。
次の画像を見て、ユーザーとアシスタントの自然な会話を「5〜8ターン」生成してください。

【会話のルール】
- 出力は会話のみで構成し、各行を「ユーザー:」「アシスタント:」で始めてください。
- ユーザーとアシスタントの発話が交互に続く、完全な会話にしてください。
- 各発話は1〜2文程度を目安に短く簡潔にしてください。
- 会話は自然な口語体で、説明文や物語調にはしないでください。
- ユーザーは画像の内容に興味を持って質問します。
- アシスタントは画像から分かる範囲で答え、根拠のない推測は避けます。
- 会話の流れを意識し、少しずつ話題が広がるようにしてください。
- 最後のターンでは、自然に会話を締めくくってください。

それでは次の画像について、上記のように5〜8ターンの会話を生成してください。
"""


# === Qwen2.5-VL モデルロード ===
def load_vlm(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    return processor, model


# === 画像取得ユーティリティ ===
def fetch_image(sample, dataset_name: str):
    """Return PIL.Image for this sample, or None on failure."""
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


# === VLMでの対話生成 ===
@torch.inference_mode()
def generate_dialogue_from_image(
    processor,
    model,
    pil_image,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    # Qwen2.5-VL 系では chat形式で画像を指定する必要がある
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": DIALOGUE_PROMPT},
            ],
        }
    ]

    # processor.apply_chat_template が正式対応
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # 入力を準備
    inputs = processor(
        text=[text_prompt],
        images=[pil_image],
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # 生成
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text



# === メインクラス ===
class LauncherVLM:
    """Generate Japanese multi-turn dialogues directly from images using Qwen2.5-VL-14B-Instruct"""

    def run(
        self,
        dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
        split: str = "train",
        out_dir: str = "./synthetic_visual_dialogues_vlm",
        max_samples: Optional[int] = 100000,
        overwrite: Literal["yes", "no"] = "no",
        output_format: Literal["jsonl", "db", "both"] = "jsonl",
    ):
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, f"{dataset}_{split}.jsonl")
        db_path = os.path.join(out_dir, f"{dataset}_ssvd.db")

        # --- 既存uidを読み込み（重複防止） ---
        existing_uids = set()
        if output_format in {"jsonl", "both"} and overwrite == "no" and os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                        existing_uids.add(obj["uid"])
                    except Exception:
                        continue
            print(f"[info] Found {len(existing_uids)} existing samples in {jsonl_path}")

        # --- データ読み込み ---
        print("📦 Loading dataset...")
        if dataset == "pixmo":
            ds = datasets.load_dataset("allenai/pixmo-cap", split=split)
            ds = ds.add_column("uid", [sha256(x.encode()).hexdigest() for x in ds["image_url"]])
            ds = ds.select_columns(["uid", "image_url"])

        elif dataset == "docci":
            ds = datasets.load_dataset("google/docci", split=split)
            ds = ds.rename_column("example_id", "uid")
            ds = ds.select_columns(["uid", "image"])

        elif dataset == "pixelprose":
            ds = datasets.load_dataset("tomg-group-umd/pixelprose", split=split)
            ds = ds.filter(lambda x: x.get("url") is not None)
            ds = ds.rename_column("url", "image_url")
            if "uid" not in ds.column_names:
                ds = ds.add_column("uid", [sha256(x.encode()).hexdigest() for x in ds["image_url"]])
            ds = ds.select_columns(["uid", "image_url"])
        else:
            raise NotImplementedError(f"Unsupported dataset: {dataset}")

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"[info] Loaded {len(ds)} candidates")

        # --- DB初期化 ---
        if output_format in {"db", "both"}:
            annotations_db = sqlite3.connect(db_path, timeout=60, isolation_level=None)
            cursor = annotations_db.cursor()
            table_name = f"{split}_vlm"
            cursor.execute(
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
            annotations_db, cursor, table_name = None, None, None

        # --- モデルロード ---
        print("🧠 Loading Qwen2.5-VL-14B-Instruct ...")
        processor, model = load_vlm()
        print("✅ Model ready.")

        success, skipped = 0, 0

        # --- メインループ ---
        for sample in tqdm(ds):
            uid = sample["uid"]
            if uid in existing_uids:
                continue

            pil_img = fetch_image(sample, dataset_name=dataset)
            if pil_img is None:
                skipped += 1
                continue

            dialogue_text = generate_dialogue_from_image(processor, model, pil_img).strip()

            # --- 対話構造化 ---
            turns = []
            for line in dialogue_text.splitlines():
                line = line.strip()
                if line.startswith("ユーザー:"):
                    turns.append({"speaker": "ユーザー", "text": line[len("ユーザー:"):].strip()})
                elif line.startswith("アシスタント:"):
                    turns.append({"speaker": "アシスタント", "text": line[len("アシスタント:"):].strip()})
            if not turns:
                turns = [{"speaker": "アシスタント", "text": dialogue_text}]

            # --- JSONL出力 ---
            if output_format in {"jsonl", "both"}:
                try:
                    with open(jsonl_path, "a", encoding="utf-8") as fout:
                        json.dump({"uid": uid, "dialogue": turns}, fout, ensure_ascii=False)
                        fout.write("\n")
                except Exception as e:
                    print(f"❌ JSON write error for {uid}: {e}")

            # --- DB出力 ---
            if output_format in {"db", "both"}:
                try:
                    for turn_idx, t in enumerate(turns):
                        cursor.execute(
                            f"INSERT OR REPLACE INTO {table_name} VALUES(?, ?, ?, ?)",
                            (uid, turn_idx, t["speaker"], t["text"]),
                        )
                except sqlite3.Error as e:
                    print(f"❌ DB write error for {uid}: {e}")

            success += 1

            # === 💡 キャッシュ解放セクション ===
            if success % 50 == 0:
                del pil_img, dialogue_text, turns, sample
                gc.collect()
                torch.cuda.empty_cache()

        # --- 後処理 ---
        if output_format in {"db", "both"}:
            annotations_db.commit()
            cursor.close()
            annotations_db.close()

        print(f"🎉 Done. Saved {success} dialogues, skipped {skipped} (invalid images).")


if __name__ == "__main__":
    import fire
    fire.Fire(LauncherVLM)
