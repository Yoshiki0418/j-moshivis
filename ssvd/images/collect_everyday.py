# COCO / DOCCI から日常シーン画像を収集するスクリプト
#
# 使い方:
#   python3 collect_everyday.py --source coco --split val2017 --max_samples 500
#   python3 collect_everyday.py --source docci --max_samples 500
#
# - COCO: 公式zipをダウンロード・展開して metadata.jsonl を作成
#   (val2017=5千枚/約1GB, train2017=11.8万枚/約19GB)
# - DOCCI: 公式配布 (storage.googleapis.com/docci) から tar.gz を取得・展開。
#   1人の撮影者による日常写真＋高精細キャプション付き (CC-BY 4.0)
#   ※ HuggingFace の google/docci は旧式スクリプト形式のため使用不可
# - どちらも VLM で対話生成するためキャプションは必須ではないが、
#   DOCCI のキャプションは品質フィルタや対話生成の補助情報として保存する

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

from tqdm import tqdm

DATA_ROOT = Path("/gpu-server/yamamoto/j-moshivis/data/images")


def collect_coco(split: str, max_samples: int | None):
    out_dir = DATA_ROOT / "coco"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{split}.zip"
    img_dir = out_dir / split

    if not img_dir.exists():
        if not zip_path.exists():
            url = f"http://images.cocodataset.org/zips/{split}.zip"
            print(f"⬇️ Downloading {url} ...")
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(zip_path), url],
                           check=True)
        print("📦 Extracting ...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(out_dir)], check=True)
        zip_path.unlink()  # 展開後はzipを削除

    meta_path = out_dir / f"metadata_{split}.jsonl"
    files = sorted(img_dir.glob("*.jpg"))
    if max_samples:
        files = files[:max_samples]

    with meta_path.open("w", encoding="utf-8") as fout:
        for p in tqdm(files, desc="Indexing COCO"):
            uid = sha256(f"coco_{p.stem}".encode()).hexdigest()
            fout.write(json.dumps({
                "uid": uid,
                "path": str(p),
                "source": f"coco_{split}",
                "license": "CC (Flickr, per-image)",
            }, ensure_ascii=False) + "\n")
    print(f"✅ {len(files)} images indexed -> {meta_path}")


DOCCI_IMAGES_URL = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
DOCCI_DESC_URL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"


def collect_docci(max_samples: int | None):
    out_dir = DATA_ROOT / "docci"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    tar_path = out_dir / "docci_images.tar.gz"
    desc_path = out_dir / "docci_descriptions.jsonlines"
    meta_path = out_dir / "metadata.jsonl"

    # --- ダウンロード & 展開 ---
    if not desc_path.exists():
        print("⬇️ Downloading DOCCI descriptions ...")
        subprocess.run(["wget", "-q", "-O", str(desc_path), DOCCI_DESC_URL], check=True)
    if not any(img_dir.glob("*.jpg")) if img_dir.exists() else True:
        if not tar_path.exists():
            print("⬇️ Downloading DOCCI images (数GBあります) ...")
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(tar_path),
                            DOCCI_IMAGES_URL], check=True)
        print("📦 Extracting ...")
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(out_dir)], check=True)
        # 展開ディレクトリ名は images/ (tar内構造に依存するため確認)
        if not img_dir.exists():
            cands = [d for d in out_dir.iterdir() if d.is_dir() and d.name != "images"]
            if len(cands) == 1:
                cands[0].rename(img_dir)
        tar_path.unlink(missing_ok=True)

    # --- description を読み込み、画像とマッチング ---
    descs = {}
    with desc_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            descs[d["image_file"]] = d

    files = sorted(img_dir.glob("*.jpg"))
    if max_samples:
        files = files[:max_samples]

    with meta_path.open("w", encoding="utf-8") as fout:
        count = 0
        for p in tqdm(files, desc="Indexing DOCCI"):
            d = descs.get(p.name)
            uid = sha256(f"docci_{p.name}".encode()).hexdigest()
            fout.write(json.dumps({
                "uid": uid,
                "path": str(p),
                "source": "docci",
                "example_id": d["example_id"] if d else None,
                "caption": d["description"] if d else None,
                "license": "CC-BY 4.0",
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"✅ {count} images indexed -> {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["coco", "docci"], required=True)
    parser.add_argument("--split", default="val2017", help="COCO split (val2017/train2017)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.source == "coco":
        collect_coco(args.split, args.max_samples)
    else:
        collect_docci(args.max_samples)


if __name__ == "__main__":
    main()
