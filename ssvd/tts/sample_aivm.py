# AivisHub の AIVM モデル (Style-Bert-VITS2 JP-Extra 形式) をSBV2ライブラリで直接推論
# 使い方: CUDA_VISIBLE_DEVICES=1 /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/sample_aivm.py
import json
import sys
from pathlib import Path

import aivmlib
import soundfile as sf
import torch
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
OUT = ROOT / "data/tts_samples"
AIVM_DIR = ROOT / "models/tts/aivis"
SENTS = json.loads((OUT / "sentences.json").read_text(encoding="utf-8"))

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm").float()  # transformers新版はfp16 checkpointをfp16でロードするためfp32へ
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
device = "cpu"  # torch2.13+cu130でHalf/float不整合が出るためCPU推論 (SBV2はCPUで十分高速)

for aivm in sorted(AIVM_DIR.glob("*.aivm")):
    name = f"aivis_{aivm.stem}"
    if all((OUT / f"{name}__{s['id']}.wav").exists() for s in SENTS):
        print(f"⏭️ {name} (exists)", flush=True)
        continue
    try:
        with aivm.open("rb") as f:
            meta = aivmlib.read_aivm_metadata(f)
        work = AIVM_DIR / aivm.stem
        work.mkdir(exist_ok=True)
        (work / "config.json").write_text(
            meta.hyper_parameters.model_dump_json(indent=2), encoding="utf-8")
        assert meta.style_vectors is not None, "style_vectors missing"
        (work / "style_vectors.npy").write_bytes(meta.style_vectors)
        # SBV2は拡張子で形式判定するため .safetensors としてシンボリックリンク (AIVMはsafetensors互換)
        link = work / f"{aivm.stem}.safetensors"
        if not link.exists():
            link.symlink_to(aivm.resolve())
        print(f"[info] {name}: speakers={[s.name for s in meta.manifest.speakers]} "
              f"styles={[st.name for st in meta.manifest.speakers[0].styles]}", flush=True)
        model = TTSModel(
            model_path=str(link), config_path=str(work / "config.json"),
            style_vec_path=str(work / "style_vectors.npy"), device=device,
        )
        style = meta.manifest.speakers[0].styles[0].name  # AIVMのスタイル名はモデル固有 (例: ノーマル)
        for s in SENTS:
            sr, audio = model.infer(text=s["text"], language=Languages.JP, style=style)
            sf.write(OUT / f"{name}__{s['id']}.wav", audio, sr)
        print(f"✅ {name}", flush=True)
    except Exception as e:
        print(f"⚠️ {name} failed: {e}", file=sys.stderr, flush=True)
