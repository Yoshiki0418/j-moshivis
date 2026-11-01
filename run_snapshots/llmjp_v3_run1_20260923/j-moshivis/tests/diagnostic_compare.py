"""
Diagnostic script: Compare model outputs between:
1. Original MoshiVis checkpoint (model_merged_bf16.safetensors)
2. J-MoshiVis finetuned checkpoint (step_6000.safetensors)

WITHOUT cross-attention (ca_src=None) to isolate the issue.
"""
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import soundfile as sf
from pathlib import Path

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from moshi.models.loaders import get_mimi
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from safetensors.torch import load_file

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

# ===== Load Mimi =====
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)
mimi = get_mimi(mimi_weight, device)
mimi.eval()

# ===== Load J-MoshiVis Config =====
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# ===== Load Audio =====
wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
silence_e = torch.zeros(1, int(5.0 * TARGET_SR))
waveform = torch.cat([waveform, silence_e], dim=1).unsqueeze(0).to(device)
codes = mimi.encode(waveform)
print(f"📊 Audio tokens: {codes.shape}")

def run_inference(model, name, codes, mimi, max_steps=300):
    """Run streaming inference and return generated audio"""
    print(f"\n{'='*50}")
    print(f"🔍 Testing: {name}")
    print(f"{'='*50}")
    
    all_audio_tokens = []
    
    with model.streaming():
        for t in range(min(codes.shape[-1], max_steps)):
            chunk = codes[:, :, t:t+1]
            tokens, gate = model.step(chunk, ca_src=None)
            
            if t == 0:
                print(f"   First gate value: {float(gate):.5f}")
            
            if tokens is not None:
                audio_tokens = tokens[:, 1:, :]
                all_audio_tokens.append(audio_tokens.cpu())
    
    if len(all_audio_tokens) > 0:
        all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
        generated_audio = mimi.decode(all_audio_tokens.to(device))
        
        output_file = f"diagnostic_{name.replace(' ', '_').lower()}.wav"
        sf.write(output_file, generated_audio.squeeze().detach().cpu().numpy(), TARGET_SR)
        print(f"   💾 Saved: {output_file}")
        print(f"   Audio shape: {generated_audio.shape}")
        return generated_audio
    return None

# ===== Test 1: Original merged checkpoint =====
print("\n📂 Loading ORIGINAL model (model_merged_bf16.safetensors)...")
original_state = load_file("/workspace/j-moshivis/model_merged_bf16.safetensors", device="cpu")

# Check keys
print(f"   Total keys: {len(original_state)}")
ca_keys = [k for k in original_state if "cross_attention" in k]
print(f"   Cross-attention keys: {len(ca_keys)}")

# Build model
moshivis_orig = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=torch.bfloat16)
missing, unexpected = moshivis_orig.load_state_dict(original_state, strict=False)

# Filter expected missing (shared cross-attention)
missing_filtered = [k for k in missing if "cross_attention.mha" not in k or "layers.0" in k]
print(f"   Missing (filtered): {len(missing_filtered)}")
print(f"   Unexpected: {len(unexpected)}")

model_orig = MoshiVisGen(moshi_vis=moshivis_orig.eval().to(device).to(torch.bfloat16))
audio_orig = run_inference(model_orig, "Original Merged", codes, mimi)

# ===== Test 2: J-MoshiVis finetuned checkpoint =====
print("\n📂 Loading J-MOSHIVIS finetuned (step_6000.safetensors)...")
finetuned_state = load_file("/workspace/j-moshivis/checkpoints/step_6000.safetensors", device="cpu")

# Clean keys
model_state = {}
for key, v in finetuned_state.items():
    if key.startswith("image_prefix."):
        continue  # Skip image embedder weights
    model_state[key] = v

print(f"   Total keys: {len(model_state)}")
ca_keys_ft = [k for k in model_state if "cross_attention" in k]
print(f"   Cross-attention keys: {len(ca_keys_ft)}")

# Build model
moshivis_ft = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=torch.bfloat16)
missing_ft, unexpected_ft = moshivis_ft.load_state_dict(model_state, strict=False)

# Filter expected missing (shared cross-attention)
missing_ft_filtered = [k for k in missing_ft if "cross_attention.mha" not in k or "layers.0" in k]
print(f"   Missing (filtered): {len(missing_ft_filtered)}")
if missing_ft_filtered:
    print(f"   Missing samples: {missing_ft_filtered[:5]}")
print(f"   Unexpected: {len(unexpected_ft)}")

model_ft = MoshiVisGen(moshi_vis=moshivis_ft.eval().to(device).to(torch.bfloat16))
audio_ft = run_inference(model_ft, "J-MoshiVis Finetuned", codes, mimi)

# ===== Compare =====
print("\n" + "="*50)
print("📊 COMPARISON SUMMARY")
print("="*50)
if audio_orig is not None and audio_ft is not None:
    diff = (audio_orig - audio_ft).abs().mean()
    print(f"   Mean audio difference: {diff.item():.6f}")
    if diff.item() < 0.01:
        print("   ✅ Audio outputs are SIMILAR - backbone weights preserved")
    else:
        print("   ❌ Audio outputs are DIFFERENT - something changed in backbone")

print("\n🎉 Diagnostic complete. Listen to the generated files.")
