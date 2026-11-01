# TTS候補の聴き比べHTMLを生成 (音声はbase64で埋め込み、単一ファイルで配布可能)
# 使い方: python3 ssvd/tts/make_tts_review.py
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
SAMPLES = ROOT / "data/tts_samples"
OUT = SAMPLES / "tts_review.html"
SENTS = json.loads((SAMPLES / "sentences.json").read_text(encoding="utf-8"))

# 表示名, 説明, ライセンス/利用条件
VOICES = {
    "aivis_honoka":       ("ほのか (AivisHub / リアボVC公式)", "実在する20代女性の声から権利購入して構築。SBV2 JP-Extra", "独自: クレジット表記「声は(x.com/dopoiro)のAI」必須"),
    "aivis_karin":        ("かりん (AivisHub / リアボVC公式)", "同上シリーズ", "同上"),
    "aivis_wakana":       ("わかな (AivisHub / リアボVC公式)", "同上シリーズ", "同上"),
    "aivis_kohaku":       ("コハク (AivisSpeech デフォルト)", "AivisSpeech同梱の公式モデル", "ACML (商用可・要クレジット等、要確認)"),
    "aivis_mao":          ("まお (AivisSpeech デフォルト)", "AivisHub最多DLの公式モデル", "ACML (要確認)"),
    "sbv2_jvnv-F1":       ("jvnv-F1 (Style-Bert-VITS2 JP-Extra)", "JVNVコーパス女性話者1", "CC BY-SA 4.0 (JVNV)"),
    "sbv2_jvnv-F2":       ("jvnv-F2 (Style-Bert-VITS2 JP-Extra)", "JVNVコーパス女性話者2", "CC BY-SA 4.0 (JVNV)"),
    "sbv2_koharune-ami":  ("小春音アミ (Style-Bert-VITS2)", "あみたろの声素材工房", "独自規約 (要確認)"),
    "sbv2_amitaro":       ("あみたろ (Style-Bert-VITS2)", "あみたろの声素材工房", "独自規約 (要確認)"),
    "irodori_600M_v3":    ("Irodori-TTS 600M v3 VoiceDesign", "日本語特化・キャプションで声を指定 (参照音声なし)", "MIT"),
    "irodori_v4.1_small": ("Irodori-TTS v4.1 Small", "同上・最新版", "MIT"),
    "qwen3tts_OnoAnna":   ("Qwen3-TTS 1.7B CustomVoice: Ono_Anna", "日本語女性プリセット話者", "Apache-2.0"),
    "qwen3tts_design":    ("Qwen3-TTS 1.7B VoiceDesign", "自然言語で声を指定", "Apache-2.0"),
    "voxcpm2_design":     ("VoxCPM2 Voice Design", "自然言語で声を指定", "Apache-2.0"),
    # ---- ユーザー側候補 (男性) ----
    "sbv2_jvnv-M1":       ("【男性】jvnv-M1 (Style-Bert-VITS2 JP-Extra)", "JVNVコーパス男性話者1", "CC BY-SA 4.0 (JVNV)"),
    "sbv2_jvnv-M2":       ("【男性】jvnv-M2 (Style-Bert-VITS2 JP-Extra)", "JVNVコーパス男性話者2", "CC BY-SA 4.0 (JVNV)"),
    "aivis_aida_shigeru": ("【男性】阿井田 茂 (AivisHub)", "バリトンボイスの朗らかなおじさん (CV: 古山キリヲ)", "ACML (要確認)"),
    "aivis_shinjou":      ("【男性】Shinjou Tomoharu (AivisHub)", "落ち着いた中年男性", "ACML (要確認)"),
    "aivis_fumifumi":     ("【男性】fumifumi (AivisHub)", "落ち着いた青年", "ACML (要確認)"),
    "aivis_tategami_kei": ("【男性】立神ケイ (AivisHub)", "低い声・ややボソついた声", "ACML (要確認)"),
    "aivis_rotejin":      ("【男性】ろてじん 長老ボイス (AivisHub)", "老人の声を意識した演技", "ACML (要確認)"),
    "aivis_morioki":      ("【女性】morioki (AivisHub)", "一般的な女性の声", "ACML (要確認)"),
}


def wav_b64(path: Path) -> str:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr > 24000:  # 埋め込みサイズ削減 (試聴用途なので24kHzで十分)
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(24000, sr)
        audio = resample_poly(audio, 24000 // g, sr // g).astype(np.float32)
        sr = 24000
    # モデル間の音量差を揃える (ピーク -1 dBFS に正規化)
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak * 0.89
    buf = io.BytesIO()
    try:  # OGG Vorbis で軽量化 (libsndfile対応時)。不可ならWAV
        sf.write(buf, audio, sr, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"
    except Exception:
        buf = io.BytesIO()
        sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
        mime = "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


rows = []
for key, (title, desc, lic) in VOICES.items():
    cells = []
    have = False
    for s in SENTS:
        p = SAMPLES / f"{key}__{s['id']}.wav"
        if p.exists():
            have = True
            dur = sf.info(p).duration
            cells.append(f'<td><audio controls preload="none" src="{wav_b64(p)}"></audio>'
                         f'<div class="dur">{dur:.1f}s</div></td>')
        else:
            cells.append('<td class="na">—</td>')
    if not have:
        continue
    rows.append(f'<tr><th><div class="t">{html.escape(title)}</div><div class="d">{html.escape(desc)}</div>'
                f'<div class="l">{html.escape(lic)}</div></th>{"".join(cells)}</tr>')

heads = "".join(f'<th class="sent"><span class="sid">{s["id"]}</span> {html.escape(s["text"])}</th>' for s in SENTS)
page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>アシスタント音声 TTS候補 聴き比べ</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:20px}} p.note{{color:#555;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}}
th.sent{{font-weight:normal;font-size:13px;background:#f3f6fa;min-width:220px}}
.sid{{display:inline-block;background:#3b6;color:#fff;border-radius:4px;padding:0 6px;font-size:11px;margin-right:4px}}
tr>th:first-child{{width:260px;background:#fcfcfc}}
.t{{font-weight:600}} .d{{font-size:12px;color:#555;margin-top:2px}} .l{{font-size:11px;color:#888;margin-top:2px}}
audio{{width:220px}} .dur{{font-size:11px;color:#999}} td.na{{color:#bbb;text-align:center}}
</style></head><body>
<h1>アシスタント音声 TTS候補 聴き比べ ({len(rows)} 候補 × {len(SENTS)} 文)</h1>
<p class="note">同じ台詞(実際に生成したアシスタント発話)を各モデルで合成。目的: LLM-jp-Moshi-v1に近い「人間らしい女性の声」の選定。</p>
<table><thead><tr><th>モデル / 話者</th>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table>
</body></html>"""
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(rows)} voices)")
