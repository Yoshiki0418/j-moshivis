# Openverse API から日本関連のCC画像を収集するスクリプト
#
# 使い方:
#   python3 collect_openverse.py --out_dir /gpu-server/yamamoto/j-moshivis/data/images/openverse
#
# - Openverse は Flickr 等の CC ライセンス画像を横断検索できる (キー不要)
# - 匿名利用の制限: 20件/リクエスト, 20リクエスト/分, 200リクエスト/日
#   → 1日あたり最大約4,000件のメタデータ取得。画像本体は提供元から直接DLするため制限外
# - ライセンス: CC BY / CC BY-SA / CC0 のみ
# - metadata.jsonl に出典・作者・ライセンスを記録（アトリビューション用）

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "https://api.openverse.org/v1/images/"
HEADERS = {
    "User-Agent": "J-MoshiVis-DatasetBuilder/1.0 (research; yoshikiyamamoto06@gmail.com)"
}

# 日本の日常・観光地を狙った検索キーワード
KEYWORDS = [
    "japan street",
    "japan convenience store",
    "japanese food restaurant",
    "ramen shop",
    "izakaya",
    "tokyo cafe",
    "japan supermarket",
    "japan train station",
    "japan train interior",
    "kyoto street",
    "osaka dotonbori",
    "japan vending machine",
    "japanese breakfast",
    "bento box",
    "japan festival matsuri",
    "cherry blossom japan",
    "shibuya crossing",
    "japan shopping street",
    "japanese house interior",
]

PAGE_SIZE = 20          # 匿名の上限
SLEEP_API = 3.5         # 20req/分制限に対する安全マージン
MIN_SHORT_SIDE = 448


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="/gpu-server/yamamoto/j-moshivis/data/images/openverse")
    parser.add_argument("--pages_per_keyword", type=int, default=10)
    parser.add_argument("--max_api_requests", type=int, default=190,
                        help="1日の匿名上限(200)以下に抑える")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.jsonl"

    seen = set()
    if meta_path.exists():
        with meta_path.open() as f:
            seen = {json.loads(l)["uid"] for l in f if l.strip()}
    print(f"[info] {len(seen)} images already collected")

    api_used = 0
    stats = {"downloaded": 0, "skipped_dup": 0, "skipped_size": 0, "failed": 0}

    fout = meta_path.open("a", encoding="utf-8")
    try:
        for kw in KEYWORDS:
            if api_used >= args.max_api_requests:
                print("[info] API request budget exhausted")
                break
            print(f"\n🔍 {kw}")
            for page in range(1, args.pages_per_keyword + 1):
                if api_used >= args.max_api_requests:
                    break
                try:
                    r = requests.get(API_URL, params={
                        "q": kw,
                        "license": "by,by-sa,cc0",
                        "page_size": PAGE_SIZE,
                        "page": page,
                    }, headers=HEADERS, timeout=30)
                    api_used += 1
                    if r.status_code == 429:
                        print("⚠️ Rate limited, stopping")
                        api_used = args.max_api_requests
                        break
                    r.raise_for_status()
                    results = r.json().get("results", [])
                except Exception as e:
                    print(f"⚠️ API error: {e}")
                    break
                if not results:
                    break

                for item in results:
                    uid = sha256(item["id"].encode()).hexdigest()
                    if uid in seen:
                        stats["skipped_dup"] += 1
                        continue
                    w, h = item.get("width") or 0, item.get("height") or 0
                    if w and h and min(w, h) < MIN_SHORT_SIDE:
                        stats["skipped_size"] += 1
                        continue
                    url = item.get("url")
                    if not url:
                        continue
                    ext = ".png" if url.lower().endswith(".png") else ".jpg"
                    img_path = img_dir / f"{uid}{ext}"
                    try:
                        ir = requests.get(url, headers=HEADERS, timeout=30)
                        ir.raise_for_status()
                        img_path.write_bytes(ir.content)
                    except Exception:
                        stats["failed"] += 1
                        continue

                    fout.write(json.dumps({
                        "uid": uid,
                        "path": str(img_path),
                        "source": f"openverse_{item.get('provider', '')}",
                        "keyword": kw,
                        "title": item.get("title", ""),
                        "foreign_landing_url": item.get("foreign_landing_url", ""),
                        "license": f"CC {item.get('license', '').upper()} {item.get('license_version', '')}",
                        "creator": item.get("creator", ""),
                        "width": w,
                        "height": h,
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                    seen.add(uid)
                    stats["downloaded"] += 1

                time.sleep(SLEEP_API)
    finally:
        fout.close()

    print("\n===== Summary =====")
    print(f"api_requests_used: {api_used}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
