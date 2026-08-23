"""
Test inference with ORIGINAL MoshiVis weights (English, not finetuned).
This establishes a baseline to compare against the finetuned J-MoshiVis model.
"""
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import soundfile as sf
from pathlib import Path

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis
import sentencepiece as spm

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ===== 1. Load Configs =====
# Use original MoshiVis config
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")

# ===== 2. Tokenizer =====
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

# ===== 3. Load ORIGINAL Weights (not finetuned) =====
print("📂 Loading Mimi weights...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)

print("📂 Loading ORIGINAL MoshiVis weights (local file)...")
# Use local model instead of downloading
original_weight = "/workspace/j-moshivis/model.safetensors"

moshi_vis, image_embedder = get_moshi_vis(
    kyuteye_config,
    moshi_weight=Path(original_weight),
    device=device,
    dtype=torch.bfloat16,
)
moshi_vis.eval()
image_embedder.eval()
print("✅ Models set to EVAL mode.")

mimi = get_mimi(mimi_weight, device)
mimi.eval()

# ===== 4. NO IMAGE =====
print("⚠️  TESTING ORIGINAL MODEL WITHOUT IMAGE (ca_src=None)")
ca_src = None

# ===== 5. Audio Loading =====
wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
print(f"🎤 Loading audio: {wav_path}")

waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)

if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

silence_e = torch.zeros(1, int(5.0 * TARGET_SR))
waveform = torch.cat([waveform, silence_e], dim=1).unsqueeze(0).to(device)

# ===== 6. Encode Audio =====
codes = mimi.encode(waveform)
print(f"   Audio tokens: {codes.shape}")

# ===== 7. Streaming Generation =====
print("🌊 Starting streaming generation with ORIGINAL model...")

all_audio_tokens = []
all_text_tokens = []

with moshi_vis.streaming():
    for t in range(min(codes.shape[-1], 500)):  # Limit to 500 steps for speed
        chunk = codes[:, :, t:t+1]

        tokens, gate = moshi_vis.step(
            chunk,
            ca_src=None,
        )
        
        if t % 100 == 0:
             print(f"   Step {t}: Gate={float(gate):.5f}")

        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]
            all_text_tokens.append(text_tokens.cpu())

# ===== 8. Decode & Save =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio = mimi.decode(all_audio_tokens.to(device))

    output_filename = "generated_audio_ORIGINAL_MODEL.wav"
    sf.write(
        output_filename,
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )
    print(f"💾 Saved: {output_filename}")

# Final Text
final_token_ids = [t.item() for t in all_text_tokens]
final_text = tokenizer.decode(final_token_ids)
print("\n📝 Generated Text (ORIGINAL model):")
print(final_text[:500])

print("\n🎉 Baseline Test Complete.")
print("   Compare this with the finetuned J-MoshiVis output.")
