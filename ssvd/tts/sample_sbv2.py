# Style-Bert-VITS2 (JP-Extra) 女性話者のサンプル合成
# 使い方: /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/sample_sbv2.py
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
OUT = ROOT / "data/tts_samples"
ASSETS = ROOT / "models/tts/sbv2"
SENTS = json.loads((OUT / "sentences.json").read_text(encoding="utf-8"))

# (表示名, repo_id, モデルディレクトリ名, safetensorsファイル名)
VOICES = [
    ("sbv2_jvnv-F1", "litagin/style_bert_vits2_jvnv", "jvnv-F1-jp", "jvnv-F1-jp_e160_s14000.safetensors"),
    ("sbv2_jvnv-F2", "litagin/style_bert_vits2_jvnv", "jvnv-F2-jp", "jvnv-F2_e166_s20000.safetensors"),
    ("sbv2_koharune-ami", "litagin/sbv2_koharune_ami", "koharune-ami", "koharune-ami.safetensors"),
    ("sbv2_amitaro", "litagin/sbv2_amitaro", "amitaro", "amitaro.safetensors"),
    ("sbv2_jvnv-M1", "litagin/style_bert_vits2_jvnv", "jvnv-M1-jp", "jvnv-M1-jp_e158_s14000.safetensors"),
    ("sbv2_jvnv-M2", "litagin/style_bert_vits2_jvnv", "jvnv-M2-jp", "jvnv-M2-jp_e159_s17000.safetensors"),
]

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm").float()  # transformers新版はfp16 checkpointをfp16でロードするためfp32へ
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
device = "cpu"  # torch2.13+cu130でHalf/float不整合が出るためCPU推論 (SBV2はCPUで十分高速)

for name, repo, mdir, mfile in VOICES:
    if all((OUT / f"{name}__{s['id']}.wav").exists() for s in SENTS):
        print(f"⏭️ {name} (exists)", flush=True)
        continue
    try:
        files = {}
        for f in ("config.json", "style_vectors.npy", mfile):
            files[f] = hf_hub_download(repo, f"{mdir}/{f}", local_dir=ASSETS)
        model = TTSModel(
            model_path=files[mfile], config_path=files["config.json"],
            style_vec_path=files["style_vectors.npy"], device=device,
        )
        for s in SENTS:
            sr, audio = model.infer(text=s["text"], language=Languages.JP)
            sf.write(OUT / f"{name}__{s['id']}.wav", audio, sr)
        print(f"✅ {name}", flush=True)
    except Exception as e:
        print(f"⚠️ {name} failed: {e}", file=sys.stderr, flush=True)
