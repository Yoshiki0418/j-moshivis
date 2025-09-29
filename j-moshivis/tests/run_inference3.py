# 音声からMoshiVisで推論を行い、wavファイルとして保存するサンプルコード
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import soundfile as sf

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from moshi.models.loaders import get_mimi
from jmoshivis.models.moshivis import MoshiVisGen
from jmoshivis.models.loaders import get_moshi_vis

TARGET_SR = 24000

# ===== 1. Configと重みのロード =====
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")

device="cuda:1" if torch.cuda.is_available() else "cpu"

weights_path = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="model.safetensors"
)
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    # cache_dir="/workspace/cache/huggingface"  # 任意
)

moshi_vis, image_embedder = get_moshi_vis(
    config,
    moshi_weight=weights_path,
    device=device,
    dtype=torch.bfloat16,
)

# ===== 2. サンプル音声の読み込み =====
print("Loading sample audio...")
mimi = get_mimi(mimi_weight, device)
wav_path = "/workspace/data/sample2.wav"
waveform, sr = torchaudio.load(wav_path)  # waveform: (channels, time)
waveform = waveform.mean(dim=0, keepdim=True)  # モノラル化

if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    sr = TARGET_SR

# ===== 無音の追加 =====
silence_duration = 2.0  # 秒
num_silence_samples = int(silence_duration * TARGET_SR)
silence = torch.zeros(1, num_silence_samples)  


# 音声の後ろに無音を連結
waveform = torch.cat([waveform, silence], dim=1)

print(f"waveform with silence: {waveform.shape}, sr={sr}")

# ===== 3. Mimi codecでトークン化 =====
waveform = waveform.unsqueeze(0)  # batch次元を追加 (1, channels, time)
waveform = waveform.to(device, dtype=torch.float32)  # Mimiはfloat32を想定
codes = mimi.encode(waveform)
print("Encoded tokens shape:", codes.shape)  # (T,)å

# ===== 4. MoshiVisに順次入力（decodeはまとめて最後にやる） =====
all_audio_tokens = []

with moshi_vis.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]  # [1, 8, 1]
        out, gate = moshi_vis.step(chunk)

        if out is not None:
            # 出力トークンを溜めるだけ
            audio_tokens = out[:, 1:, :]  # [1, 8, 1]
            all_audio_tokens.append(audio_tokens.cpu())

# ===== 5. 一括デコード =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)  # [1, 8, total_T]
    generated_audio = mimi.decode(all_audio_tokens.to(device))  # [1, 1, T]

    # wavファイルとして保存
    sf.write(
        "generated.wav",
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )
    print("✅ generated.wav として保存しました！")
else:
    print("⚠️ 有効な生成が得られませんでした")