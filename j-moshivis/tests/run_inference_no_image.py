"""
Test inference WITHOUT image conditioning.
This helps determine if the base audio model works correctly.
If this produces clean audio, the problem is with image integration.
If this also produces broken audio, the base model itself has issues.
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
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# ===== 2. Tokenizer =====
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

# ===== 3. Load Weights =====
print("📂 Loading Mimi weights...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)

print("📂 Loading MoshiVis weights...")
moshi_weight_path = Path("/workspace/j-moshivis/checkpoints/step_6000.safetensors")

moshi_vis, image_embedder = get_moshi_vis(
    kyuteye_config,
    moshi_weight=moshi_weight_path,
    device=device,
    dtype=torch.bfloat16,
)
moshi_vis.eval()
image_embedder.eval()
print("✅ Models set to EVAL mode.")

mimi = get_mimi(mimi_weight, device)
mimi.eval()

# ===== 4. NO IMAGE - Skip image encoding =====
print("⚠️  TESTING WITHOUT IMAGE CONDITIONING (ca_src=None)")
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

# ===== 7. Streaming Generation (NO IMAGE) =====
print("🌊 Starting streaming generation WITHOUT image...")

all_audio_tokens = []
all_text_tokens = []

with moshi_vis.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]

        # NO IMAGE: ca_src=None
        tokens, gate = moshi_vis.step(
            chunk,
            ca_src=None,
        )
        
        if t % 100 == 0:
             print(f"   Step {t}/{codes.shape[-1]}: Gate={float(gate):.5f}")

        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]
            all_text_tokens.append(text_tokens.cpu())

# ===== 8. Decode & Save =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio = mimi.decode(all_audio_tokens.to(device))

    output_filename = "generated_audio_NO_IMAGE.wav"
    sf.write(
        output_filename,
        generated_audio.squeeze().detach().cpu().numpy(),
        TARGET_SR
    )
    print(f"💾 Saved: {output_filename}")

# Final Text
final_token_ids = [t.item() for t in all_text_tokens]
final_text = tokenizer.decode(final_token_ids)
print("\n📝 Generated Text (without image):")
print(final_text[:500])

print("\n🎉 Test Complete. Compare this audio with the image-conditioned version.")
print("   If this is CLEAN: Problem is with image integration")
print("   If this is BROKEN: Problem is with base model/training")
