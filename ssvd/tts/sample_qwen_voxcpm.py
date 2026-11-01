# Qwen3-TTS (CustomVoice: Ono_Anna / VoiceDesign) と VoxCPM2 (Voice Design) のサンプル合成
# 使い方: CUDA_VISIBLE_DEVICES=1 /gpu-server/yamamoto/miniforge3/envs/tts/bin/python ssvd/tts/sample_qwen_voxcpm.py [qwen|voxcpm|all]
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path("/gpu-server/yamamoto/j-moshivis")
OUT = ROOT / "data/tts_samples"
SENTS = json.loads((OUT / "sentences.json").read_text(encoding="utf-8"))
which = sys.argv[1] if len(sys.argv) > 1 else "all"

FEMALE_DESIGN_JA = "落ち着いた自然な話し方の、二十代後半の日本人女性。明るすぎず、聞き取りやすい柔らかい声で、会話するように親しみを込めて話す"


def run_qwen():
    from qwen_tts import Qwen3TTSModel
    # CustomVoice: 日本語女性プリセット Ono_Anna
    m = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16)
    for s in SENTS:
        wavs, sr = m.generate_custom_voice(text=s["text"], language="Japanese", speaker="Ono_Anna")
        sf.write(OUT / f"qwen3tts_OnoAnna__{s['id']}.wav", wavs[0], sr)
    print("✅ qwen3tts_OnoAnna", flush=True)
    del m; torch.cuda.empty_cache()
    # VoiceDesign: 自然言語で声を指定
    m = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)
    for s in SENTS:
        wavs, sr = m.generate_voice_design(text=s["text"], language="Japanese", instruct=FEMALE_DESIGN_JA)
        sf.write(OUT / f"qwen3tts_design__{s['id']}.wav", wavs[0], sr)
    print("✅ qwen3tts_design", flush=True)
    del m; torch.cuda.empty_cache()


def run_voxcpm():
    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    for s in SENTS:
        wav = m.generate(text=f"({FEMALE_DESIGN_JA}){s['text']}", cfg_value=2.0)
        sf.write(OUT / f"voxcpm2_design__{s['id']}.wav", wav, m.tts_model.sample_rate)
    print("✅ voxcpm2_design", flush=True)


if which in ("qwen", "all"):
    try:
        run_qwen()
    except Exception as e:
        print(f"⚠️ qwen failed: {e}", file=sys.stderr, flush=True)
if which in ("voxcpm", "all"):
    try:
        run_voxcpm()
    except Exception as e:
        print(f"⚠️ voxcpm failed: {e}", file=sys.stderr, flush=True)
