import os, math, argparse
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

# Moshi 用のモデルクラス（CausalLM ではなく ConditionalGeneration）
from transformers import MoshiForConditionalGeneration

MODEL_ID = "kyutai/moshiko-pytorch-bf16"

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def to_mono_24k(wav, sr):
    import torchaudio, torch
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav_t = torch.tensor(wav, dtype=torch.float32)
    if sr != 24000:
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=24000)
    return wav_t.numpy(), 24000

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default="moshi_reply.wav")
    ap.add_argument("--max_new_audio_tokens", type=int, default=25)  # ≈2秒
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

    print(f"Loading Moshi base model: {MODEL_ID}")
    model = MoshiForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()

    # 1) ユーザ音声を読み込み → 24kHz mono に揃える
    wav, sr = sf.read(args.wav)
    try:
        wav, sr = to_mono_24k(wav, sr)
    except Exception:
        # torchaudio が無い環境向けの簡易フォールバック（線形補間）
        import numpy as np
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 24000:
            t_old = np.linspace(0, 1, num=len(wav), endpoint=False)
            t_new = np.linspace(0, 1, num=int(round(len(wav) * 24000 / sr)), endpoint=False)
            wav = np.interp(t_new, t_old, wav).astype(np.float32)
            sr = 24000

    user_input_values = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)  # (1,1,T)

    # 2) ★ここがポイント：他は渡さない（モデルに自動初期化させる）
    with torch.no_grad():
        out = model.generate(
            user_input_values=user_input_values,
            do_sample=False,                        # まずはgreedyでNaN回避
            max_new_tokens=args.max_new_audio_tokens,
            use_cache=True,
        )

    audio_wave = out.audio_sequences[0, 0].detach().cpu().float().numpy()
    sf.write(args.out, audio_wave, 24000)
    print(f"✅ Saved: {args.out}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    main()