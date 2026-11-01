# ほのか (AivisHub) の抑揚パラメータ比較サンプル
# 使い方: /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/sample_honoka_prosody.py
import json
from pathlib import Path

import soundfile as sf
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
OUT = ROOT / "data/tts_samples/honoka_prosody"
OUT.mkdir(parents=True, exist_ok=True)
SENTS = json.loads((ROOT / "data/tts_samples/sentences.json").read_text(encoding="utf-8"))
W = ROOT / "models/tts/aivis/honoka"

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm").float()
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
model = TTSModel(model_path=str(W / "honoka.safetensors"), config_path=str(W / "config.json"),
                 style_vec_path=str(W / "style_vectors.npy"), device="cpu")

# (variant名, 説明, infer kwargs)
VARIANTS = [
    ("00_base",            "現状 (ノーマル, intonation 1.0, sdp 0.2)", dict(style="ノーマル")),
    ("01_intonation1.3",   "intonation_scale 1.3 (ピッチの抑揚を1.3倍)", dict(style="ノーマル", intonation_scale=1.3)),
    ("02_intonation1.6",   "intonation_scale 1.6", dict(style="ノーマル", intonation_scale=1.6)),
    ("03_sdp0.6",          "sdp_ratio 0.6 (リズム・間の揺らぎ増)", dict(style="ノーマル", sdp_ratio=0.6)),
    ("04_uresii_w1",       "スタイル: うれしい (weight 1.0)", dict(style="uresii_uresii", style_weight=1.0)),
    ("05_uresii_w2",       "スタイル: うれしい (weight 2.0)", dict(style="uresii_uresii", style_weight=2.0)),
    ("06_odoroki_w1",      "スタイル: おどろき (weight 1.0)", dict(style="odoroki_odoroki", style_weight=1.0)),
    ("07_hutuu_w1",        "スタイル: ふつう (weight 1.0)", dict(style="hutuu_hutuu", style_weight=1.0)),
    ("08_assist_text",     "assist_text: 「明るく楽しそうに、少し驚きを込めて話す」", dict(style="ノーマル", use_assist_text=True, assist_text="わあ、すごい!とても楽しそうに、明るく驚きを込めて話しています。", assist_text_weight=1.0)),
    ("09_combo",           "うれしい w1.0 + intonation 1.3 + sdp 0.5", dict(style="uresii_uresii", style_weight=1.0, intonation_scale=1.3, sdp_ratio=0.5)),
    ("10_combo_pitch",     "同上 + pitch_scale 1.05", dict(style="uresii_uresii", style_weight=1.0, intonation_scale=1.3, sdp_ratio=0.5, pitch_scale=1.05)),
]

meta = []
for vid, desc, kw in VARIANTS:
    for s in SENTS:
        sr, audio = model.infer(text=s["text"], language=Languages.JP, **kw)
        sf.write(OUT / f"{vid}__{s['id']}.wav", audio, sr)
    meta.append({"id": vid, "desc": desc, "params": kw})
    print(f"✅ {vid}", flush=True)
(OUT / "variants.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
