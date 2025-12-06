import torch
import torchaudio
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import soundfile as sf
from pathlib import Path

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis
import sentencepiece as spm

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===== 1. Configロード =====
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# ===== 2. Tokenizer =====
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")


# weights_path = hf_hub_download(
#     repo_id="kyutai/moshika-vis-pytorch-bf16",
#     filename="model.safetensors"
# )
# ===== 3. Mimi / MoshiVis ロード =====
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)

moshi_vis, image_embedder = get_moshi_vis(
    kyuteye_config,
    moshi_weight=Path("/workspace/j-moshivis/checkpoints/step_5000.safetensors"),
    # moshi_weight=weights_path,
    device=device,
    dtype=torch.bfloat16,
)

mimi = get_mimi(mimi_weight, device)

# ===== 4. 画像 Encoding =====
# img_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
img_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1c99faed050be74fb35a7a6e05f35d93742b98923aab56ae9e2a8dcba43fa640/image.jpg"

print("Encoding image...")
image_processor = ImageProcessor()
image_tensor = image_processor(img_path).unsqueeze(0)

with torch.no_grad():
    img_out = image_embedder(image_tensor)
    ca_src = img_out["cross_attention_src"]
print("Encoded image:", ca_src.shape)

# ===== 5. 音声読み込み =====
wav_path = "/workspace/data/sample6.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)

if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

# 無音追加
silence = torch.zeros(1, int(6.0 * TARGET_SR))
waveform = torch.cat([silence, waveform, silence], dim=1).unsqueeze(0).to(device)

# ===== 6. Mimi codec でトークン化 =====
codes = mimi.encode(waveform)
print("Audio tokens:", codes.shape)

# ===== 7. Streaming (音声 + テキスト + gate) =====
generated_text = ""
all_audio_tokens = []
all_text_tokens = []
gate_values = []

print("Starting streaming generation...")

with moshi_vis.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]

        tokens, gate = moshi_vis.step(
            chunk,
            ca_src=ca_src,  # ← ここを画像にしたいなら ca_src に変更
        )
        print(ca_src.mean(), ca_src.std())

        # ==== (A) gate を記録 ====
        gate_values.append(gate)
        print(f"GATE[{t}]:", float(gate))

        # ===== (B) Audio tokens =====
        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]   # skip text stream
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]     # text stream
            all_text_tokens.append(text_tokens.cpu())

        # ===== (C) Text logits from internal streaming state =====
        state = moshi_vis._streaming_state
        if state is not None and hasattr(state, "text_logits") and state.text_logits is not None:
            logits = state.text_logits      # [B,1,1,vocab]
            token_id = logits.argmax(dim=-1).item()

            if token_id not in {
                moshi_vis.lm_model.text_padding_token_id,
                moshi_vis.lm_model.end_of_text_padding_id,
            }:
                generated_text += tokenizer.decode([token_id])
                print("TEXT:", generated_text)


# ===== 8. オーディオデコード =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio = mimi.decode(all_audio_tokens.to(device))

    sf.write(
        "generated_audio_with_image.wav",
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )

print("🎉 Saved: generated_audio_with_image.wav")
print("Generated TEXT:", generated_text)
print("Collected TEXT tokens:", all_text_tokens)
print("Collected GATE values:", gate_values)
