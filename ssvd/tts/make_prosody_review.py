# ほのか 抑揚パラメータ比較ページ生成
import base64, html, io, json
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
D = ROOT / "data/tts_samples/honoka_prosody"
OUT = ROOT / "data/tts_samples/honoka_prosody_review.html"
SENTS = json.loads((ROOT / "data/tts_samples/sentences.json").read_text(encoding="utf-8"))
VARIANTS = json.loads((D / "variants.json").read_text(encoding="utf-8"))

def b64(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr > 24000:
        g = gcd(24000, sr); a = resample_poly(a, 24000 // g, sr // g).astype(np.float32); sr = 24000
    a = a / (float(np.abs(a).max()) or 1.0) * 0.89
    buf = io.BytesIO(); sf.write(buf, a, sr, format="OGG", subtype="VORBIS")
    return "data:audio/ogg;base64," + base64.b64encode(buf.getvalue()).decode()

rows = []
for v in VARIANTS:
    cells = []
    for s in SENTS:
        p = D / f"{v['id']}__{s['id']}.wav"
        cells.append(f'<td><audio controls preload="none" src="{b64(p)}"></audio><div class="dur">{sf.info(p).duration:.1f}s</div></td>' if p.exists() else '<td class="na">—</td>')
    params = html.escape(json.dumps(v["params"], ensure_ascii=False))
    rows.append(f'<tr><th><div class="t">{html.escape(v["id"])}</div><div class="d">{html.escape(v["desc"])}</div><div class="l">{params}</div></th>{"".join(cells)}</tr>')
heads = "".join(f'<th class="sent"><span class="sid">{s["id"]}</span> {html.escape(s["text"])}</th>' for s in SENTS)
OUT.write_text(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>ほのか 抑揚パラメータ比較</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}h1{{font-size:20px}}p.note{{color:#555;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}}
th.sent{{font-weight:normal;font-size:13px;background:#f3f6fa;min-width:220px}}.sid{{display:inline-block;background:#3b6;color:#fff;border-radius:4px;padding:0 6px;font-size:11px;margin-right:4px}}
tr>th:first-child{{width:300px;background:#fcfcfc}}.t{{font-weight:600}}.d{{font-size:12px;color:#555;margin-top:2px}}.l{{font-size:10px;color:#888;margin-top:2px;word-break:break-all}}
audio{{width:220px}}.dur{{font-size:11px;color:#999}}td.na{{color:#bbb;text-align:center}}</style></head><body>
<h1>ほのか (AivisHub) 抑揚パラメータ比較 ({len(rows)} 設定 × {len(SENTS)} 文)</h1>
<p class="note">同一モデル・同一台詞で、スタイル / style_weight / intonation_scale / sdp_ratio / assist_text / pitch_scale を変えたもの。音量はピーク正規化済み。</p>
<table><thead><tr><th>設定</th>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table></body></html>""", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(rows)} variants)")
