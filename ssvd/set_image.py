import os
import json
import requests
from tqdm import tqdm
from datasets import load_dataset

# === 設定 ===
BASE_DIR = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo"
META_PATH = "./download_log.jsonl"  # ダウンロード記録を保存
MAX_SAMPLES = None  # 例: 5000 にすると PixelProse の上位5000件を使用

# === 画像ダウンロード関数 ===
def download_image(url, path):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def main(max_samples=None):
    print("📦 Loading PixelProse dataset (metadata only)...")
    ds = load_dataset("tomg-group-umd/pixelprose", split="train")

    if max_samples:
        ds = ds.select(range(max_samples))

    # UIDディレクトリ一覧を先に取得
    uid_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    print(f"📂 Found {len(uid_dirs)} UID directories in {BASE_DIR}")

    # PixelProse内のUID→URL, captionマッピングを先に辞書化
    print("🔍 Indexing PixelProse UID→URL mapping...")
    uid_to_info = {}
    for x in tqdm(ds, desc="Indexing"):
        uid = x.get("uid")
        if uid:
            uid_to_info[uid] = {"url": x.get("url"), "caption": x.get("vlm_caption")}

    print(f"📇 Indexed {len(uid_to_info)} total UIDs from PixelProse.")

    # 自分のUIDの中でPixelProseに存在するものだけ抽出
    matched_uids = [uid for uid in uid_dirs if uid in uid_to_info]
    print(f"🎯 Found {len(matched_uids)} matching UIDs in both datasets.")

    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    fout = open(META_PATH, "w", encoding="utf-8")

    success, skipped = 0, 0

    # 一致するUIDだけを対象にダウンロード
    for uid in tqdm(matched_uids, desc="Downloading matched images"):
        info = uid_to_info[uid]
        url, caption = info["url"], info["caption"]

        if not url or not caption:
            skipped += 1
            continue

        uid_dir = os.path.join(BASE_DIR, uid)
        img_path = os.path.join(uid_dir, "image.jpg")

        # 既に存在すればスキップ
        if os.path.exists(img_path):
            skipped += 1
            continue

        ok = download_image(url, img_path)
        if not ok:
            skipped += 1
            continue

        json.dump(
            {"uid": uid, "image_path": img_path, "url": url, "caption": caption},
            fout,
            ensure_ascii=False,
        )
        fout.write("\n")
        success += 1

    fout.close()
    print(f"✅ Done: downloaded {success} images, skipped {skipped}.")
    print(f"📄 Log saved at: {META_PATH}")


if __name__ == "__main__":
    main(MAX_SAMPLES)
