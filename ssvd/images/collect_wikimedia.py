# Wikimedia Commons から日本の日常シーン画像を収集するスクリプト
#
# 使い方:
#   python3 collect_wikimedia.py --out_dir /gpu-server/yamamoto/j-moshivis/data/images/wikimedia \
#       --per_category 100 --depth 1
#
# - カテゴリごとに CC ライセンス (CC0 / CC BY / CC BY-SA / PD) の画像のみ取得
# - 元画像の短辺 >= MIN_SHORT_SIDE のものだけを対象に、幅 THUMB_WIDTH のサムネイルを保存
# - メタデータ (ライセンス・作者・出典URL) は metadata.jsonl に記録（アトリビューション用）
# - 再実行時は既存ファイルをスキップ（レジューム可能）

import argparse
import json
import re
import time
from hashlib import sha256
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "J-MoshiVis-DatasetBuilder/1.0 (research; yoshikiyamamoto06@gmail.com)"
}

# === 収集対象カテゴリ（日本の日常生活を中心に選定） ===
DEFAULT_CATEGORIES = [
    # 食
    "Category:Cuisine of Japan",
    "Category:Ramen",
    "Category:Sushi",
    "Category:Bento",
    "Category:Interiors of restaurants in Japan",
    "Category:Izakaya",
    # 街並み・買い物
    "Category:Shopping streets in Japan",
    "Category:Streets in Tokyo",
    "Category:Streets in Osaka",
    "Category:Streets in Kyoto",
    "Category:Convenience stores in Japan",
    "Category:Supermarkets in Japan",
    "Category:Vending machines in Japan",
    # 交通
    "Category:Railway stations in Japan",
    "Category:Interiors of trains of Japan",
    "Category:Buses in Japan",
    # 住まい・日常
    "Category:Interiors of houses in Japan",
    "Category:Tatami",
    "Category:Kotatsu",
    "Category:Japanese gardens in Japan",
    "Category:Parks in Japan",
    # 文化・季節
    "Category:Cherry blossoms in Japan",
    "Category:Autumn in Japan",
    "Category:Matsuri",
    "Category:Shinto shrines in Japan",
    "Category:Buddhist temples in Japan",
    "Category:Torii",
    "Category:Onsen",
    "Category:Kimonos being worn",
    # 学校・仕事
    "Category:Schools in Japan",
    "Category:Classrooms in Japan",
]

# スポーツ・動物カテゴリ（--set sports_animals で使用）
SPORTS_ANIMALS_CATEGORIES = [
    # スポーツ（日本らしさ重視）
    "Category:Sumo",
    "Category:Baseball in Japan",
    "Category:Baseball venues in Japan",
    "Category:Association football in Japan",
    "Category:Kendo",
    "Category:Judo",
    "Category:Kyūdō",
    "Category:Marathons in Japan",
    "Category:Undōkai",
    "Category:Skiing in Japan",
    "Category:Surfing in Japan",
    "Category:Basketball in Japan",
    "Category:Volleyball in Japan",
    "Category:Table tennis",
    "Category:Swimming pools in Japan",
    # 動物（身近な動物・日本らしい動物）
    "Category:Cats in Japan",
    "Category:Dogs of Japan",
    "Category:Shiba Inu",
    "Category:Akita Inu",
    "Category:Sika deer in Nara Prefecture",
    "Category:Japanese macaque",
    "Category:Cat cafés",
    "Category:Ueno Zoo",
    "Category:Zoos in Japan",
    "Category:Aquaria in Japan",
    "Category:Koi",
    "Category:Birds of Japan",
    "Category:Insects of Japan",
    "Category:Squirrels",
    "Category:Rabbits",
    "Category:Hamsters",
]

# 植物カテゴリ（--set plants で使用）
PLANTS_CATEGORIES = [
    # 日本らしい花・樹木
    "Category:Prunus mume",              # 梅
    "Category:Hydrangea in Japan",       # 紫陽花
    "Category:Wisteria in Japan",        # 藤
    "Category:Chrysanthemums",           # 菊
    "Category:Lycoris radiata",          # 彼岸花
    "Category:Nelumbo nucifera",         # ハス
    "Category:Camellia japonica",        # 椿
    "Category:Autumn leaf color in Japan",  # 紅葉
    "Category:Ginkgo trees in Japan",    # イチョウ
    "Category:Bamboo forests in Japan",  # 竹林
    "Category:Pinus in Japan",           # 松
    # 栽培・園芸・農
    "Category:Bonsai",                   # 盆栽
    "Category:Moss gardens",             # 苔庭
    "Category:Rice fields in Japan",     # 稲田
    "Category:Tea fields in Japan",      # 茶畑
    "Category:Vegetable gardens",        # 家庭菜園
    "Category:Greenhouses in Japan",     # 温室
    "Category:Flower fields in Japan",   # 花畑
    "Category:Ikebana",                  # 生け花
    "Category:Houseplants",              # 観葉植物
    # 一般的な花
    "Category:Sunflowers",               # ひまわり
    "Category:Tulips",                   # チューリップ
    "Category:Roses",                    # バラ
    "Category:Cosmos (plant)",           # コスモス
]

# 主要観光地カテゴリ（--set tourist で使用）
TOURIST_CATEGORIES = [
    "Category:Shibuya Crossing",
    "Category:Takeshita Street",
    "Category:Sensō-ji",
    "Category:Tokyo Skytree",
    "Category:Tokyo Tower",
    "Category:Akihabara",
    "Category:Shinjuku Golden Gai",
    "Category:Fushimi Inari-taisha",
    "Category:Kinkaku-ji",
    "Category:Arashiyama",
    "Category:Kiyomizu-dera",
    "Category:Dōtonbori",
    "Category:Osaka Castle",
    "Category:Nara Park",
    "Category:Itsukushima Shrine",
    "Category:Mount Fuji",
    "Category:Kamakura, Kanagawa",
    "Category:Enoshima",
    "Category:Otaru Canal",
    "Category:Kenroku-en",
    "Category:Shirakawa-gō",
    "Category:Hakone",
    "Category:Beppu Onsen",
    "Category:Shurijō",
]

# 許可するライセンス（extmetadata の LicenseShortName に対する正規表現）
LICENSE_ALLOW = re.compile(
    r"^(CC0|CC BY(-SA)?( \d\.\d)?|Public domain|PD)", re.IGNORECASE
)

MIN_SHORT_SIDE = 448   # 元画像の短辺の最小ピクセル数
THUMB_WIDTH = 1024     # 保存するサムネイルの幅
VALID_MIME = {"image/jpeg", "image/png"}
SLEEP_SEC = 0.3        # APIへの負荷を抑えるためのウェイト


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_subcategories(category: str) -> list:
    """カテゴリ直下のサブカテゴリ一覧を取得"""
    subs = []
    cont = {}
    while True:
        data = api_get({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "subcat",
            "cmlimit": 500,
            **cont,
        })
        subs += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(SLEEP_SEC)
    return subs


def iter_category_files(category: str, limit: int):
    """カテゴリ内のファイルを imageinfo 付きで列挙"""
    count = 0
    cont = {}
    while count < limit:
        data = api_get({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": THUMB_WIDTH,
            **cont,
        })
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo")
            if not infos:
                continue
            yield page["title"], infos[0]
            count += 1
            if count >= limit:
                break
        if "continue" not in data or count >= limit:
            break
        cont = data["continue"]
        time.sleep(SLEEP_SEC)


def check_license(info: dict):
    """ライセンスが許可対象なら (license, artist, credit_url) を返す"""
    meta = info.get("extmetadata", {})
    lic = meta.get("LicenseShortName", {}).get("value", "")
    if not LICENSE_ALLOW.match(lic):
        return None
    artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
    return lic, artist


def get_capture_year(info: dict):
    """extmetadata から撮影年を取得（不明なら None）"""
    meta = info.get("extmetadata", {})
    dt = meta.get("DateTimeOriginal", {}).get("value", "") or meta.get("DateTime", {}).get("value", "")
    m = re.search(r"(19|20)\d{2}", dt)
    return int(m.group(0)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="/gpu-server/yamamoto/j-moshivis/data/images/wikimedia")
    parser.add_argument("--per_category", type=int, default=100,
                        help="1カテゴリあたりの最大取得枚数")
    parser.add_argument("--depth", type=int, default=0,
                        help="サブカテゴリを辿る深さ (0=直下のファイルのみ)")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="対象カテゴリを上書き指定")
    parser.add_argument("--set", choices=["daily", "tourist", "sports_animals", "plants"],
                        default="daily",
                        help="カテゴリセット: daily=日常生活, tourist=主要観光地, "
                             "sports_animals=スポーツ・動物, plants=植物")
    parser.add_argument("--min_year", type=int, default=None,
                        help="この年以降に撮影された画像のみ取得 (例: 2018)。撮影日不明は除外")
    parser.add_argument("--scan_multiplier", type=int, default=3,
                        help="per_category の何倍の候補をスキャンするか (フィルタが厳しい時は大きく)")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.jsonl"

    # 既存UIDを読み込んでスキップ対象にする（レジューム）
    seen_uids = set()
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen_uids.add(json.loads(line)["uid"])
                except Exception:
                    continue
    print(f"[info] {len(seen_uids)} images already collected (resume mode)")

    category_sets = {
        "daily": DEFAULT_CATEGORIES,
        "tourist": TOURIST_CATEGORIES,
        "sports_animals": SPORTS_ANIMALS_CATEGORIES,
        "plants": PLANTS_CATEGORIES,
    }
    categories = args.categories or category_sets[args.set]

    # サブカテゴリ展開
    expanded = []
    for cat in categories:
        expanded.append(cat)
        if args.depth > 0:
            try:
                expanded += get_subcategories(cat)[:20]  # 爆発防止で上限20
            except Exception as e:
                print(f"⚠️ Failed to expand {cat}: {e}")
    print(f"[info] {len(expanded)} categories to crawl")

    stats = {"downloaded": 0, "skipped_license": 0, "skipped_size": 0,
             "skipped_mime": 0, "skipped_dup": 0, "skipped_year": 0, "failed": 0}

    fout = meta_path.open("a", encoding="utf-8")
    try:
        for cat in expanded:
            cat_slug = re.sub(r"[^\w]+", "_", cat.replace("Category:", ""))[:60]
            img_dir = out_root / cat_slug
            print(f"\n📂 {cat}")
            try:
                files = list(iter_category_files(cat, args.per_category * args.scan_multiplier))
            except Exception as e:
                print(f"⚠️ Failed to list {cat}: {e}")
                continue

            got = 0
            for title, info in tqdm(files, desc=cat_slug):
                if got >= args.per_category:
                    break
                uid = sha256(title.encode()).hexdigest()
                if uid in seen_uids:
                    stats["skipped_dup"] += 1
                    continue
                if info.get("mime") not in VALID_MIME:
                    stats["skipped_mime"] += 1
                    continue
                w, h = info.get("width", 0), info.get("height", 0)
                if min(w, h) < MIN_SHORT_SIDE:
                    stats["skipped_size"] += 1
                    continue
                lic_info = check_license(info)
                if lic_info is None:
                    stats["skipped_license"] += 1
                    continue
                lic, artist = lic_info
                year = get_capture_year(info)
                if args.min_year and (year is None or year < args.min_year):
                    stats["skipped_year"] += 1
                    continue

                thumb_url = info.get("thumburl") or info.get("url")
                ext = ".png" if info["mime"] == "image/png" else ".jpg"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"{uid}{ext}"
                ok = False
                for attempt in range(3):  # 429等に備えてリトライ
                    try:
                        r = requests.get(thumb_url, headers=HEADERS, timeout=30)
                        if r.status_code == 429:
                            time.sleep(5 * (attempt + 1))
                            continue
                        r.raise_for_status()
                        img_path.write_bytes(r.content)
                        ok = True
                        break
                    except Exception:
                        time.sleep(2 * (attempt + 1))
                if not ok:
                    stats["failed"] += 1
                    continue

                fout.write(json.dumps({
                    "uid": uid,
                    "path": str(img_path),
                    "title": title,
                    "category": cat,
                    "source_url": info.get("descriptionurl", ""),
                    "license": lic,
                    "artist": artist,
                    "year": year,
                    "width": w,
                    "height": h,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                seen_uids.add(uid)
                got += 1
                stats["downloaded"] += 1
                time.sleep(SLEEP_SEC)
    finally:
        fout.close()

    print("\n===== Summary =====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
