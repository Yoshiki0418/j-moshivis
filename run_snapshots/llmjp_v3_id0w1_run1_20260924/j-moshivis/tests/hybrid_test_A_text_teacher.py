"""
Hybrid Inference Test A: Text from Teacher, Audio Self-Generated

This test isolates whether AUDIO autoregression is causing streaming failure.
- Text tokens: Teacher-forced (from ground truth)
- Audio tokens: Self-generated (autoregressive)

If this test fails → Audio autoregression is the problem
If this test passes → Audio autoregression is NOT the problem
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
from jmoshivis.datasets.interleaver import InterleavedTokenizer, Interleaver
from moshi.models.loaders import get_mimi

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
print(f"🚀 Device: {device}")
print("=" * 60)
print("🧪 HYBRID TEST A: Text=Teacher, Audio=Self-Generated")
print("=" * 60)

# ===== Paths =====
config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_1000.safetensors"
tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
data_path = "/workspace/data/speech/train_data_refined_a.jsonl"
target_len = 125

# ===== Load Models =====
kyuteye_config = KyuteyeConfig.from_yml(config_path)
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)

print("📂 Loading Mimi...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)
mimi = get_mimi(mimi_weight, device)
mimi.eval()

print(f"📂 Loading checkpoint: {checkpoint_path}")
loaded_weights = load_file(checkpoint_path, device="cpu")

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

print("📂 Building MoshiVis model...")
moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
moshi_vis.load_state_dict(model_state, strict=False)

# Check if Depformer weights were loaded
depformer_keys = [k for k in model_state.keys() if "depformer" in k]
if depformer_keys:
    print(f"✅ Loaded {len(depformer_keys)} Depformer parameters from checkpoint.")
else:
    print("⚠️ WARNING: No Depformer parameters found in checkpoint! Model may use random initialization for Audio.")
moshi_vis = moshi_vis.eval().to(device).to(dtype)

print("📂 Loading Image Embedder...")
image_embedder = ImageProjection.from_config(
    kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
)
image_embedder = image_embedder.eval().to(dtype)

# ===== Build DataLoader =====
interleaver = Interleaver(
    tokenizer, mimi.frame_rate,
    moshi_vis.text_padding_token_id,
    moshi_vis.end_of_text_padding_id,
    moshi_vis.zero_token_id,
    keep_main_only=True, device=device,
)
instruct_tokenizer = InterleavedTokenizer(mimi=mimi, interleaver=interleaver, duration_sec=10.0)

data_args = DataArgs(train_data=data_path, eval_data="", shuffle=False)
data_loader = build_data_loader(
    instruct_tokenizer=instruct_tokenizer, args=data_args,
    batch_size=1, seed=42, rank=0, world_size=1, is_eval=False,
    image_root=None, image_embedder=None, device=torch.device(device),
    mode="speech", text_tokenizer=tokenizer, target_len=target_len,
)

# ===== Get batch =====
batch = next(iter(data_loader))
codes = batch.codes.to(device)  # [B, 17, T]
print(f"✅ Input codes shape: {codes.shape}")

# Structure: codes[B, 17, T] = [text(1), moshi_audio(8), user_audio(8)]
teacher_text = codes[:, 0:1, :]           # [B, 1, T] - Ground truth text
teacher_moshi_audio = codes[:, 1:9, :]    # [B, 8, T] - Ground truth moshi audio
user_audio = codes[:, 9:17, :]            # [B, 8, T] - User's audio

# ===== Get Image Embedding =====
image_input = None
if isinstance(batch.condition_attributes, list):
    for ca in batch.condition_attributes:
        if hasattr(ca, "tensor") and "image" in ca.tensor:
            image_input = ca.tensor["image"].tensor.to(device).to(dtype)
            break

if image_input is not None:
    with torch.no_grad():
        embedder_out = image_embedder(image_input)
        ca_src = embedder_out["cross_attention_src"]
    print(f"🖼️ Image features: {ca_src.shape}")
else:
    ca_src = None
    print("⚠️ No image found!")

# ===== Hybrid Inference: Text=Teacher, Audio=Self-Generated =====
print("\n" + "=" * 60)
print("🔥 HYBRID: Text from Teacher, Audio Self-Generated")
print("=" * 60)

moshi_vis_gen = MoshiVisGen(
    moshi_vis=moshi_vis,
    use_sampling=True,
    temp=0.8,       # Back to default for early training analysis
    temp_text=0.7,  # Back to default
    top_k=250,      # Back to default
    top_k_text=25,  # Back to default
)
T = codes.shape[-1]

all_audio_tokens = []
all_text_tokens = []
gate_values = []

print("🎤 Generating audio step-by-step...")
with moshi_vis_gen.streaming():
    for t in range(T):
        # Use teacher text for this step (from ground truth)
        # But we need to feed the streaming state with user audio
        user_chunk = user_audio[:, :, t:t+1]  # [B, 8, 1]
        
        # Call step - this generates moshi's response
        tokens, gate = moshi_vis_gen.step(user_chunk, ca_src=ca_src)
        
        if tokens is not None:
            if t < 5 or (20 < t < 25): # Print some tokens
                print(f"   Step {t} Audio: {tokens[0, 1:, :].squeeze().tolist()}")
        
        if t % 25 == 0:
            gate_str = f"{float(gate):.5f}" if gate else "N/A"
            print(f"Step {t}/{T}: Gate={gate_str}")
        
        if gate:
            gate_values.append(float(gate))
        
        if tokens is not None:
            # tokens: [B, 9, 1] = [text(1), audio(8)]
            # Replace generated text with teacher text
            generated_audio = tokens[:, 1:, :]  # [B, 8, 1]
            teacher_text_t = teacher_text[:, :, t:t+1]  # [B, 1, 1]
            
            # For audio, we use the SELF-GENERATED tokens
            all_audio_tokens.append(generated_audio.cpu())
            # For text, we record what was generated (but model used teacher internally)
            all_text_tokens.append(tokens[:, :1, :].cpu())

# ===== Decode Results =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    print(f"Generated audio tokens: {all_audio_tokens.shape}")
    
    # Validate
    valid_mask = (all_audio_tokens >= 0) & (all_audio_tokens < 2048)
    invalid_count = (~valid_mask).sum().item()
    if invalid_count > 0:
        print(f"⚠️ Found {invalid_count} invalid codes, replacing with 0")
        all_audio_tokens[~valid_mask] = 0
    
    generated_audio = mimi.decode(all_audio_tokens.to(device))
    output_file = "hybrid_test_A_text_teacher_audio_autoregressive.wav"
    sf.write(output_file, generated_audio.squeeze().detach().cpu().numpy(), TARGET_SR)
    print(f"💾 Saved: {output_file}")
    
    # Text decoding
    if all_text_tokens:
        final_text_ids = [int(t.squeeze().item()) for t in all_text_tokens]
        decoded_text = tokenizer.decode([t for t in final_text_ids if t > 0 and t != 3])
        print(f"\n📝 Generated Text: {decoded_text[:500]}")
    
    # Gate stats
    if gate_values:
        print(f"\n📊 Gate Statistics:")
        print(f"   Mean: {sum(gate_values)/len(gate_values):.5f}")
        print(f"   Max: {max(gate_values):.5f}")
        print(f"   Min: {min(gate_values):.5f}")

print("\n" + "=" * 60)
print("🎉 HYBRID TEST A COMPLETE!")
print("=" * 60)
print("Result: Check hybrid_test_A_text_teacher_audio_autoregressive.wav")
print("If audio is broken → Audio autoregression is the issue")
print("If audio is clean → Audio autoregression is NOT the issue")
