import json
from pathlib import Path

# ======== 入力・出力パス設定 ========
INPUT_JSONL = Path("/workspace/data/filtered_dialogue/filtered.jsonl")
OUTPUT_ROOT = Path("/workspace/data/speech/data_stereo")

# ======== 出力ディレクトリ確認 ========
assert INPUT_JSONL.exists(), f"❌ Not found: {INPUT_JSONL}"
assert OUTPUT_ROOT.exists(), f"❌ Not found: {OUTPUT_ROOT}"

# ======== 各UIDのdialogue.jsonを生成 ========
created = 0
skipped = 0

with INPUT_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        uid = item["uid"]
        dialogue = item["dialogue"]

        out_dir = OUTPUT_ROOT / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dialogue.json"

        # 既に存在する場合はスキップ（必要に応じて上書き可）
        if out_path.exists():
            skipped += 1
            continue

        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump({"dialogue": dialogue}, out_f, ensure_ascii=False, indent=2)

        created += 1

print(f"✅ Created {created} dialogue.json files.")
if skipped:
    print(f"⚠️ Skipped {skipped} (already exist).")

print("🎯 Done.")
