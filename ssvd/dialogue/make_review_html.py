# 複数モデルの対話生成結果を画像付きで並べて比較するHTMLレビューページを生成
#
# 使い方:
#   .venv-data/bin/python ssvd/dialogue/make_review_html.py \
#       --inputs "Claude Sonnet 5=data/dialogues/claude_sonnet5_pilot.jsonl" \
#                "Qwen3-VL-30B-A3B=data/dialogues/qwen3vl_30b_pilot.jsonl" \
#       --out data/dialogues/review.html

import argparse
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def img_to_b64(path: str, max_side: int = 720) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="'ラベル=path.jsonl' の形式")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels, by_uid = [], defaultdict(dict)
    image_of = {}
    for spec in args.inputs:
        label, path = spec.split("=", 1)
        labels.append(label)
        if not Path(path).exists():
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            by_uid[r["uid"]][label] = r
            image_of[r["uid"]] = r["image"]

    css = """
    body{font-family:sans-serif;margin:20px;background:#f6f6f6;color:#222}
    .sample{background:#fff;border-radius:10px;padding:16px;margin-bottom:28px;box-shadow:0 1px 4px #0002}
    .sample img{max-width:520px;max-height:420px;border-radius:6px;display:block;margin-bottom:12px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
    .col{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fafafa}
    .col h3{margin:0 0 8px;font-size:15px;border-bottom:2px solid #4a7;padding-bottom:4px}
    .turn{margin:4px 0;font-size:13.5px;line-height:1.5}
    .u{color:#1a5fb4}.a{color:#26a269}
    .meta{font-size:12px;color:#777;margin-bottom:6px}
    .missing{color:#c01c28;font-style:italic}
    """
    parts = [f"<html><head><meta charset='utf-8'><title>対話生成モデル比較</title><style>{css}</style></head><body>",
             f"<h1>対話生成モデル比較 ({len(by_uid)}画像 × {len(labels)}モデル)</h1>"]
    for i, (uid, results) in enumerate(by_uid.items(), 1):
        parts.append(f"<div class='sample'><h2>サンプル{i} <span class='meta'>{uid[:16]}</span></h2>")
        parts.append(f"<img src='{img_to_b64(image_of[uid])}'>")
        parts.append("<div class='grid'>")
        for label in labels:
            parts.append(f"<div class='col'><h3>{html.escape(label)}</h3>")
            r = results.get(label)
            if r is None:
                parts.append("<div class='missing'>生成失敗 / 未生成</div>")
            else:
                cost = f" / ${r['cost_usd']:.3f}" if r.get("cost_usd") else ""
                parts.append(f"<div class='meta'>{len(r['dialogue'])}ターン{cost}</div>")
                grounding = r.get("grounding")
                if isinstance(grounding, dict):
                    question = html.escape(str(grounding.get("question_en", "")))
                    answer = html.escape(str(
                        grounding.get("answer_count", grounding.get("text", ""))
                    ))
                    tier = grounding.get("question_quality_tier")
                    tier_text = f" / tier {tier}" if tier is not None else ""
                    parts.append(
                        f"<div class='meta'><b>正解条件:</b> {question} "
                        f"→ <b>{answer}</b>{tier_text}</div>"
                    )
                for t in r["dialogue"]:
                    cls = "u" if t["speaker"] == "ユーザー" else "a"
                    parts.append(f"<div class='turn'><b class='{cls}'>{t['speaker']}</b>: {html.escape(t['text'])}</div>")
            parts.append("</div>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"✅ {args.out} ({len(by_uid)} samples, {len(labels)} models)")


if __name__ == "__main__":
    main()
