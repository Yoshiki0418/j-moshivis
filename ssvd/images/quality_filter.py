# 収集画像の品質フィルタ・重複除去スクリプト
#
# 使い方:
#   python3 quality_filter.py \
#       --meta /gpu-server/yamamoto/j-moshivis/data/images/wikimedia/metadata.jsonl \
#              /gpu-server/yamamoto/j-moshivis/data/images/docci/metadata.jsonl \
#       --output /gpu-server/yamamoto/j-moshivis/data/images/filtered_images.jsonl
#
# フィルタ内容:
#   1. 破損画像・読み込み不能画像の除外
#   2. 短辺 448px 未満、極端なアスペクト比 (>3:1) の除外
#   3. perceptual hash (dHash) による近似重複の除去
#   4. 単色・低情報量画像（ヒストグラムエントロピーが低い）の除外

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

MIN_SHORT_SIDE = 448
MAX_ASPECT = 3.0
MIN_ENTROPY = 3.5  # 8bitグレースケールヒストグラムのエントロピー下限


def dhash(img: Image.Image, size: int = 8) -> int:
    """difference hash: 近似重複検出用"""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(g, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def entropy(img: Image.Image) -> float:
    hist = np.asarray(img.convert("L").histogram(), dtype=np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", nargs="+", required=True,
                        help="入力 metadata.jsonl (複数指定可)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dup_threshold", type=int, default=4,
                        help="dHashのハミング距離がこの値以下なら重複とみなす")
    args = parser.parse_args()

    entries = []
    for meta in args.meta:
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    print(f"[info] {len(entries)} entries loaded")

    stats = {"kept": 0, "broken": 0, "small": 0, "aspect": 0,
             "low_entropy": 0, "duplicate": 0}
    hashes = []  # (hash, uid)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for e in tqdm(entries, desc="Filtering"):
            try:
                img = Image.open(e["path"])
                img.load()
                img = img.convert("RGB")
            except Exception:
                stats["broken"] += 1
                continue

            w, h = img.size
            if min(w, h) < MIN_SHORT_SIDE:
                stats["small"] += 1
                continue
            if max(w, h) / min(w, h) > MAX_ASPECT:
                stats["aspect"] += 1
                continue
            if entropy(img) < MIN_ENTROPY:
                stats["low_entropy"] += 1
                continue

            hval = dhash(img)
            if any(hamming(hval, prev) <= args.dup_threshold for prev, _ in hashes):
                stats["duplicate"] += 1
                continue
            hashes.append((hval, e["uid"]))

            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print("\n===== Summary =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
