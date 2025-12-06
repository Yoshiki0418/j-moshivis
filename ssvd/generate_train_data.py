import json
import soundfile as sf
from pathlib import Path

# ========= 設定 =========
DATA_ROOT = Path("/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo")
OUTPUT_PATH = Path("/workspace/data/speech/train_data_refined.jsonl")

# ========= 初期化 =========
created = 0
skipped = 0
missing = 0
missing_dirs = []

image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

# ========= 探索開始 =========
for uid_dir in sorted(DATA_ROOT.iterdir()):
    if not uid_dir.is_dir():
        continue

    uid = uid_dir.name
    wav_path = uid_dir / f"stereo_dialogue.wav"
    align_path = uid_dir / f"stereo_dialogue.refined.json"
    dialogue_path = uid_dir / "dialogue.json"

    # --- 画像ファイル探索 ---
    image_path = None
    for ext in image_extensions:
        candidate = uid_dir / f"image{ext}"
        if candidate.exists():
            image_path = candidate
            break

    # --- 必要ファイル確認 ---
    if not (wav_path.exists() and align_path.exists() and dialogue_path.exists() and image_path):
        missing += 1
        missing_dirs.append({
            "uid": uid,
            "wav": wav_path.exists(),
            "align": align_path.exists(),
            "dialogue": dialogue_path.exists(),
            "image": bool(image_path)
        })
        continue

    # --- duration計算 ---
    try:
        info = sf.info(str(wav_path))
        duration = info.duration
    except Exception as e:
        print(f"⚠️ Failed to read {wav_path.name}: {e}")
        continue

    # --- JSONロード ---
    try:
        with open(align_path, "r", encoding="utf-8") as f:
            align_data = json.load(f)
        alignments = align_data.get("tokens", [])

        with open(dialogue_path, "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)
        dialogue_texts = [d["text"] for d in dialogue_data.get("dialogue", [])]
        text_joined = " ".join(dialogue_texts)
    except Exception as e:
        print(f"⚠️ Failed to load JSON in {uid}: {e}")
        continue

    # --- エントリ作成 ---
    entry = {
        "path": str(wav_path),
        "duration": duration,
        "text": text_joined,
        "alignments": alignments,
        "image": str(image_path)  # ← 画像パスを追加
    }

    # --- JSONLに書き出し ---
    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    created += 1

# ========= 結果出力 =========
print(f"✅ Created entries: {created}")
print(f"⚠️ Skipped / Missing: {missing}")

if missing_dirs:
    print("\n--- Missing file summary ---")
    for m in missing_dirs[:10]:  # 最初の10件だけ表示
        print(
            f"UID: {m['uid']} | wav: {m['wav']} | align: {m['align']} "
            f"| dialogue: {m['dialogue']} | image: {m['image']}"
        )
    if len(missing_dirs) > 10:
        print(f"... and {len(missing_dirs) - 10} more.")
