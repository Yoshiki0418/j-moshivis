"""
Non-Streaming Batch Inference Script for J-MoshiVis

This script uses forward_speech() instead of streaming step() to verify
that the issue is isolated to streaming mode inference.

Uses DataLoader to get proper 17-codebook input format (1 text + 16 audio).
"""
import torch
import soundfile as sf
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sentencepiece as spm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from jmoshivis.datasets.data_loader import build_data_loader
from jmoshivis.datasets.args import DataArgs
from jmoshivis.datasets.interleaver import InterleavedTokenizer
from moshi.models.loaders import get_mimi

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
print(f"🚀 Device: {device}")

# ===== 1. Paths =====
config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_1000.safetensors"
tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
data_path = "/workspace/data/speech/train_data_refined_a.jsonl"
target_len = 125

# ===== 2. Load Configs =====
kyuteye_config = KyuteyeConfig.from_yml(config_path)

# ===== 3. Tokenizer =====
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)

# ===== 4. Load Mimi =====
print("📂 Loading Mimi...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)
mimi = get_mimi(mimi_weight, device)
mimi.eval()

# ===== 5. Load Checkpoint =====
print(f"📂 Loading checkpoint: {checkpoint_path}")
loaded_weights = load_file(checkpoint_path, device="cpu")

# Separate image projection and model weights
image_proj_state = {}
model_state = {}

for key, v in loaded_weights.items():
    clean_key = key
    is_image_proj = False
    
    if clean_key.startswith("image_prefix."):
        clean_key = clean_key[len("image_prefix."):]
        is_image_proj = True
    if clean_key.startswith("module."):
        clean_key = clean_key[len("module."):]
    if clean_key.startswith("_orig_mod."):
        clean_key = clean_key[len("_orig_mod."):]
    
    if is_image_proj:
        image_proj_state[clean_key] = v
    else:
        model_state[clean_key] = v

print(f"✅ Loaded weights: {len(model_state)} model, {len(image_proj_state)} image_proj")

# Check cross-attention keys
ca_keys = [k for k in model_state.keys() if "cross_attention" in k]
print(f"📊 Cross-attention keys in checkpoint: {len(ca_keys)}")
if not ca_keys:
    print("   ⚠️ WARNING: No cross-attention keys found in checkpoint!")

# ===== 6. Build MoshiVis Model =====
print("📂 Building MoshiVis model...")
moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
missing, unexpected = moshi_vis.load_state_dict(model_state, strict=False)

# Filter expected missing (shared cross-attention for layers > 0)
missing_filtered = [k for k in missing if "cross_attention.mha" not in k or "layers.0" in k]
print(f"Missing keys (filtered): {len(missing_filtered)}")
print(f"Unexpected keys: {len(unexpected)}")

moshi_vis = moshi_vis.eval().to(device).to(dtype)

# ===== 7. Load Image Embedder =====
print("📂 Loading Image Embedder...")
image_embedder = ImageProjection.from_config(
    kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
)
image_embedder = image_embedder.eval().to(dtype)
print("✅ Models loaded and set to EVAL mode.")

# ===== 8. Build Interleaved Tokenizer =====
print("📂 Building Interleaved Tokenizer...")
from jmoshivis.datasets.interleaver import Interleaver

interleaver = Interleaver(
    tokenizer,
    mimi.frame_rate,
    moshi_vis.text_padding_token_id,
    moshi_vis.end_of_text_padding_id,
    moshi_vis.zero_token_id,
    keep_main_only=True,
    device=device,
)
instruct_tokenizer = InterleavedTokenizer(
    mimi=mimi,
    interleaver=interleaver,
    duration_sec=10.0,
)

# ===== 9. Build DataLoader =====
print("📂 Building DataLoader...")
data_args = DataArgs(
    train_data=data_path,
    eval_data="",
    shuffle=False,
)
data_loader = build_data_loader(
    instruct_tokenizer=instruct_tokenizer,
    args=data_args,
    batch_size=1,
    seed=42,
    rank=0,
    world_size=1,
    is_eval=False,
    image_root=None,
    image_embedder=None,
    device=torch.device(device),
    mode="speech",
    text_tokenizer=tokenizer,
    target_len=target_len,
)

# ===== 10. Get one batch =====
print("📦 Fetching batch...")
batch = next(iter(data_loader))
codes = batch.codes.to(device)  # Shape: [B, 17, T]
print(f"✅ Input codes shape: {codes.shape}")

# ===== 11. Get Image Embedding =====
image_input = None
if isinstance(batch.condition_attributes, list):
    tensors = []
    for ca in batch.condition_attributes:
        if hasattr(ca, "tensor") and "image" in ca.tensor:
            tensors.append(ca.tensor["image"].tensor.to(device).to(dtype))
    if tensors:
        image_input = torch.cat(tensors, dim=0)

if image_input is not None:
    print(f"🖼️ Image tensor shape: {image_input.shape}")
    with torch.no_grad():
        embedder_out = image_embedder(image_input)
        ca_src = embedder_out["cross_attention_src"]
    print(f"Image features: {ca_src.shape}")
    print(f"Stats: mean={ca_src.mean().item():.4f}, std={ca_src.std().item():.4f}")
else:
    ca_src = None
    print("⚠️ No image found in batch!")

# ===== 12. Non-Streaming Batch Inference =====
print("\n" + "="*60)
print("🔥 Running NON-STREAMING batch inference with forward_speech()...")
print("="*60)

with torch.no_grad():
    outputs = moshi_vis.forward_speech(
        input_ids=codes,
        cross_attention_src=ca_src,
    )
    
    text_logits = outputs["text_logits"]
    audio_logits = outputs["audio_logits"]
    gate = outputs.get("gate", None)

print(f"📊 Output shapes:")
print(f"   Text logits: {text_logits.shape}")
print(f"   Audio logits: {audio_logits.shape}")
if gate is not None:
    print(f"   Gate: {gate:.5f}")

# Get predicted tokens
pred_audio_codes = torch.argmax(audio_logits, dim=-1)  # [B, dep_q, T]
print(f"Predicted audio codes: {pred_audio_codes.shape}")

# Check for valid audio codes
valid_mask = (pred_audio_codes >= 0) & (pred_audio_codes < 2048)
invalid_count = (~valid_mask).sum().item()
if invalid_count > 0:
    print(f"⚠️ Found {invalid_count} invalid audio codes, replacing with 0")
    pred_audio_codes = pred_audio_codes.clone()
    pred_audio_codes[~valid_mask] = 0

# Decode audio
print("🔊 Decoding predicted audio...")
with torch.no_grad():
    generated_audio = mimi.decode(pred_audio_codes.to(device))

output_file = "generated_audio_batch_inference.wav"
sf.write(output_file, generated_audio.squeeze().detach().cpu().numpy(), TARGET_SR)
print(f"💾 Saved: {output_file}")

# Decode text
pred_text_ids = torch.argmax(text_logits, dim=-1)
valid_text_ids = pred_text_ids[0, 0].cpu().tolist()
valid_text_ids = [t for t in valid_text_ids if t >= 0 and t != 3]
decoded_text = tokenizer.decode(valid_text_ids[:100]) if valid_text_ids else "(empty)"
print(f"\n📝 Generated Text (first 100 tokens): {decoded_text[:500]}")

# ===== 13. Streaming Inference for Comparison =====
print("\n" + "="*60)
print("🔄 Now running STREAMING inference for comparison...")
print("="*60)

# Extract user audio stream (last 8 codebooks: indices 9:17)
# Structure: codes[B, 17, T] = [text(1), moshi_audio(8), user_audio(8)]
# The step() function expects only user_audio (8 codebooks)
user_audio_codes = codes[:, 9:, :]  # [B, 8, T] - user's 8 audio codebooks
print(f"User audio codes for streaming: {user_audio_codes.shape}")

moshi_vis_gen = MoshiVisGen(moshi_vis=moshi_vis)

all_audio_tokens = []
all_text_tokens = []
gate_values = []

with moshi_vis_gen.streaming():
    for t in range(user_audio_codes.shape[-1]):
        # Streaming mode uses only user audio codes (8 codebooks)
        chunk = user_audio_codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen.step(chunk, ca_src=ca_src)

        if t % 50 == 0:
            gate_str = f"{float(gate):.5f}" if gate else "N/A"
            print(f"Step {t}/{user_audio_codes.shape[-1]}: Gate={gate_str}")
        if gate:
            gate_values.append(float(gate))

        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]
            all_text_tokens.append(text_tokens.cpu())

if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio_stream = mimi.decode(all_audio_tokens.to(device))

    output_file_stream = "generated_audio_streaming_inference.wav"
    sf.write(output_file_stream, generated_audio_stream.squeeze().detach().cpu().numpy(), TARGET_SR)
    print(f"💾 Saved streaming result: {output_file_stream}")

    # Text decoding for streaming
    final_token_ids = [int(t.squeeze().item()) for t in all_text_tokens]
    final_text = tokenizer.decode(final_token_ids)
    print(f"\n📝 Streaming Generated Text: {final_text[:500]}")

    # Gate stats for streaming
    if gate_values:
        print(f"\n📊 Streaming Gate Statistics:")
        print(f"Mean: {sum(gate_values)/len(gate_values):.5f}")
        print(f"Max: {max(gate_values):.5f}")
        print(f"Min: {min(gate_values):.5f}")

print("\n🎉 Inference comparison complete!")
print("Compare the two output files:")
print(f"  1. Batch inference: generated_audio_batch_inference.wav")
print(f"  2. Streaming inference: generated_audio_streaming_inference.wav")
