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

# =================================================================
# 0. Device Strategy
# =================================================================
# Automatically select CUDA device to avoid OOM or conflicts (prefer cuda:1 if available)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# =================================================================
# 1. Load Configs
# =================================================================
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# =================================================================
# 2. Tokenizer
# =================================================================
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

# =================================================================
# 3. Load Weights (Mimi & MoshiVis)
# =================================================================
print("📂 Loading Mimi weights...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)

print("📂 Loading MoshiVis weights...")
# CRITICAL FIX: Use step_6000 checkpoint instead of older ones or step_2000
moshi_weight_path = Path("/workspace/j-moshivis/checkpoints/step_30000.safetensors")

moshi_vis, image_embedder = get_moshi_vis(
    kyuteye_config,
    moshi_weight=moshi_weight_path,
    device=device,
    dtype=torch.bfloat16,
)

# CRITICAL FIX: Set models to EVAL mode to disable Dropout/BatchNormalization noise
moshi_vis.eval()
image_embedder.eval()
print("✅ Models set to EVAL mode.")

mimi = get_mimi(mimi_weight, device)
mimi.eval()

# =================================================================
# 4. Image Encoding
# =================================================================
img_path = "/workspace/test.jpg"
print(f"🖼️  Encoding image: {img_path}")

image_processor = ImageProcessor()
image_tensor = image_processor(img_path).unsqueeze(0)

with torch.no_grad():
    img_out = image_embedder(image_tensor)
    ca_src = img_out["cross_attention_src"]

print(f"   Encoded Image Shape: {ca_src.shape}")
print(f"   Image Stats: Mean={ca_src.mean().item():.5f}, Std={ca_src.std().item():.5f}")

# =================================================================
# 5. Audio Loading & Preprocessing
# =================================================================
# wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1c99faed050be74fb35a7a6e05f35d93742b98923aab56ae9e2a8dcba43fa640/user_dialogue.wav"
wav_path = "/workspace/data/sample3.wav"
print(f"🎤 Loading audio: {wav_path}")

waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True) # Stereo -> Mono

if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

# Add silence padding
silence_s = torch.zeros(1, int(5.0 * TARGET_SR))
silence_e = torch.zeros(1, int(10.0 * TARGET_SR))
waveform = torch.cat([waveform, silence_e], dim=1).unsqueeze(0).to(device)

# =================================================================
# 6. Encode Audio to Tokens
# =================================================================
codes = mimi.encode(waveform)
print(f"   Audio tokens: {codes.shape}")

# =================================================================
# 7. Streaming Generation
# =================================================================
print("🌊 Starting streaming generation...")

all_audio_tokens = []
all_text_tokens = []
gate_values = []

with moshi_vis.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]

        # CRITICAL FIX: Pass ca_src (image features) to step()
        tokens, gate = moshi_vis.step(
            chunk,
            ca_src=ca_src, 
        )
        
        # Log progress periodically
        if t % 50 == 0:
             print(f"   Step {t}/{codes.shape[-1]}: Gate={float(gate):.5f}")

        gate_values.append(gate)

        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]   # skip text stream
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]     # text stream
            all_text_tokens.append(text_tokens.cpu())

        # Inspect generated text periodically
        if t % 50 == 0:
             token_ids = [tk.item() for tk in all_text_tokens]
             decoded_text_preview = tokenizer.decode(token_ids)
             # print(f"      Decoded so far: {decoded_text_preview[-50:]}") # Show last 50 chars

# =================================================================
# 8. Decode & Save Audio
# =================================================================
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio = mimi.decode(all_audio_tokens.to(device))

    output_filename = "generated_audio_with_image.wav"
    sf.write(
        output_filename,
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )
    print(f"💾 Saved generated audio to: {output_filename}")

# Final Text Decode
final_token_ids = [t.item() for t in all_text_tokens]
final_text = tokenizer.decode(final_token_ids)
print("\n📝 Generated Full Text:")
print(final_text)

print("\n📈 Gate Statistics:")
gate_tensor = torch.tensor([float(g) for g in gate_values])
print(f"   Mean: {gate_tensor.mean():.5f}, Max: {gate_tensor.max():.5f}, Min: {gate_tensor.min():.5f}")

print("🎉 Inference Complete.")
