# 音声からMoshiVisで推論を行うサンプルコード（形状確認まで）
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import soundfile as sf

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from moshi.models.loaders import get_mimi
from jmoshivis.models.moshivis import MoshiVisGen
from jmoshivis.utils import audio_codec

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
state_dict = load_file(weights_path)

lm_gen = MoshiVisGen.from_config(
    config,
    moshi_weight=state_dict,
    device=device,
    dtype=torch.bfloat16,
).to(dtype=torch.bfloat16)

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
print("Encoded tokens shape:", codes.shape)  # (T,)

# ===== 4. MoshiVisに順次入力 =====
generated_audio = []
with lm_gen.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]  # [1, 8, 1]
        out, gate = lm_gen.step(chunk)

        if out is None:
            continue

        # 出力トークンのオーディオ部分を切り出し
        audio_tokens = out[:, 1:, :]  # shape [1, 8, 1]

        # Mimiでデコード（波形化）
        audio_waveform = mimi.decode(audio_tokens)  # [B, 1, T]

        # CPUに持ってきて保存用にappend
        generated_audio.append(audio_waveform.cpu())

# ===== 連結して1本の波形にする =====
if len(generated_audio) > 0:
    generated_audio = torch.cat(generated_audio, dim=-1)  # [1, 1, total_time]

    # wavファイルとして保存
    sf.write(
        "generated.wav",
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )

    print("✅ generated.wav として保存しました！")
else:
    print("⚠️ 有効な生成が得られませんでした")
