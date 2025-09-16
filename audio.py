import soundfile as sf
import torch
import torchaudio

in_path = "/workspace/src/data/mvis/coco-en-jsonl/sample.wav"
out_path = "/workspace/src/data/mvis/coco-en-jsonl/sample_16k.wav"

# wav 読み込み
waveform, sr = sf.read(in_path)
waveform = torch.tensor(waveform, dtype=torch.float32)

# ステレオ → モノラル化
if waveform.ndim > 1:
    waveform = waveform.mean(dim=1)

# サンプリングレート変換
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

# 保存
sf.write(out_path, waveform.numpy(), 16000)
print(f"✅ 変換完了: {out_path}")
