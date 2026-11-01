"""
Blindfold Hypothesis Verification

Test whether cross-attention was effectively disabled during training
by comparing xa_scope masks and actual attention outputs between
batch and streaming modes.
"""
import torch
import torchaudio
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sentencepiece as spm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection, ImageProcessor
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from jmoshivis.datasets.data_loader import build_data_loader
from jmoshivis.datasets.args import DataArgs
from jmoshivis.datasets.interleaver import Interleaver, InterleavedTokenizer
from moshi.models.loaders import get_mimi

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print("="*70)
print("🎭 BLINDFOLD HYPOTHESIS: Was Cross-Attention masked during training?")
print("="*70)

# Setup
config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"
tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
data_path = "/workspace/data/speech/train_data_refined_a.jsonl"

kyuteye_config = KyuteyeConfig.from_yml(config_path)
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)

print("📂 Loading models...")
mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)
mimi = get_mimi(mimi_weight, device)
mimi.eval()

# Load checkpoint
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

moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
moshi_vis.load_state_dict(model_state, strict=False)
moshi_vis = moshi_vis.eval().to(device).to(dtype)

image_embedder = ImageProjection.from_config(
    kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
)
image_embedder = image_embedder.eval().to(dtype)

# ===== CHECK 1: get_xa_scope behavior =====
print("\n" + "="*70)
print("🔍 CHECK 1: get_xa_scope() mask generation")
print("="*70)

layer0_ca = moshi_vis.llm.transformer.layers[0].cross_attention

print(f"\nCross-Attention config:")
print(f"  xa_start: {layer0_ca.xa_start}")
print(f"  xa_end: {layer0_ca.xa_end}")
print(f"  xa_step: {layer0_ca.xa_step}")

# Test get_xa_scope with different input shapes
# Batch mode: input shape [B, T, D]
test_batch_input = torch.randn(1, 100, 4096, device=device, dtype=dtype)  # 100 timesteps
test_stream_input = torch.randn(1, 1, 4096, device=device, dtype=dtype)    # 1 timestep

print(f"\n📊 Testing get_xa_scope:")

# Check if get_xa_scope exists and what it returns
if hasattr(layer0_ca, 'get_xa_scope'):
    try:
        # Batch mode (not streaming)
        scope_batch = layer0_ca.get_xa_scope(test_batch_input, image_tokens_mask=None)
        print(f"\n  Batch mode (T=100):")
        print(f"    Scope shape: {scope_batch.shape if scope_batch is not None else None}")
        if scope_batch is not None:
            print(f"    Scope values: min={scope_batch.min().item():.1f}, max={scope_batch.max().item():.1f}")
            print(f"    All ones (full attention): {torch.all(scope_batch == 1).item()}")
            print(f"    All zeros (no attention): {torch.all(scope_batch == 0).item()}")
    except Exception as e:
        print(f"  Error calling get_xa_scope: {e}")
else:
    print("  get_xa_scope method not found")

# ===== CHECK 2: Compare cross-attention outputs in batch vs streaming =====
print("\n" + "="*70)
print("🔍 CHECK 2: Cross-Attention output magnitude comparison")
print("="*70)

# Load image
image_processor = ImageProcessor(size=(448, 448), device=device)
image_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
image_tensor = image_processor(image_path).unsqueeze(0).to(dtype)

with torch.no_grad():
    ca_src = image_embedder(image_tensor)["cross_attention_src"]

print(f"\nImage features (ca_src): {ca_src.shape}")

# Track cross-attention outputs in both modes
ca_outputs_batch = []
ca_outputs_streaming = []

original_ca_forward = layer0_ca.forward

def tracking_forward(x, key=None, value=None, cross_attention_mask=None, image_tokens_mask=None):
    result = original_ca_forward(x, key, value, cross_attention_mask, image_tokens_mask)
    out, gate = result
    return result, {
        'x_shape': x.shape,
        'key_shape': key.shape if key is not None else None,
        'out_shape': out.shape,
        'out_mean': out.mean().item(),
        'out_std': out.std().item(),
        'out_max': out.abs().max().item(),
        'gate': gate.mean().item() if hasattr(gate, 'mean') else gate,
    }

# ===== Batch forward_speech =====
print("\n🔄 Testing BATCH mode (forward_speech)...")

# Build data loader to get proper input
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
    mimi=mimi, interleaver=interleaver, duration_sec=10.0
)
data_args = DataArgs(train_data=data_path, eval_data="", shuffle=False)
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
    target_len=50,  # Short for testing
)

batch = next(iter(data_loader))
codes = batch.codes.to(device)
print(f"Input codes shape: {codes.shape}")

# Get image from batch
image_input = None
if isinstance(batch.condition_attributes, list):
    for ca in batch.condition_attributes:
        if hasattr(ca, "tensor") and "image" in ca.tensor:
            image_input = ca.tensor["image"].tensor.to(device).to(dtype)
            break

if image_input is not None:
    with torch.no_grad():
        ca_src_batch = image_embedder(image_input)["cross_attention_src"]
    print(f"Batch ca_src: {ca_src_batch.shape}")
else:
    ca_src_batch = ca_src
    print("Using default ca_src")

# Run batch forward
with torch.no_grad():
    outputs = moshi_vis.forward_speech(
        input_ids=codes,
        cross_attention_src=ca_src_batch,
    )
    
print(f"Batch forward completed")
print(f"  Gate from output: {outputs.get('gate', 'N/A')}")

# ===== Streaming mode =====
print("\n🔄 Testing STREAMING mode (step)...")

# Load audio for streaming
wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
waveform = waveform[:, :2 * TARGET_SR].unsqueeze(0).to(device)
stream_codes = mimi.encode(waveform)

moshi_vis_gen = MoshiVisGen(moshi_vis=moshi_vis)
streaming_gates = []

with moshi_vis_gen.streaming():
    for t in range(min(10, stream_codes.shape[-1])):
        chunk = stream_codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen.step(chunk, ca_src=ca_src)
        streaming_gates.append(float(gate) if gate else 0.0)

print(f"Streaming gates: {streaming_gates[:5]}")

# ===== CHECK 3: Compare gate magnitudes =====
print("\n" + "="*70)
print("🔍 CHECK 3: Gate magnitude analysis")
print("="*70)

batch_gate = outputs.get('gate', 0)
if hasattr(batch_gate, 'item'):
    batch_gate = batch_gate.item()

stream_gate_mean = sum(streaming_gates) / len(streaming_gates) if streaming_gates else 0

print(f"\n📊 Gate comparison:")
print(f"  Batch mode gate: {batch_gate:.5f}")
print(f"  Streaming mode gate (mean): {stream_gate_mean:.5f}")

if abs(batch_gate - stream_gate_mean) > 0.01:
    print(f"  ⚠️ Significant difference in gate values!")
else:
    print(f"  ✅ Gate values are similar")

# ===== CHECK 4: Verify cross-attention weights are not random =====
print("\n" + "="*70)
print("🔍 CHECK 4: Are Cross-Attention weights learned or random?")
print("="*70)

# 🚨 CRITICAL: Clear SharedCrossAttention singleton cache before creating fresh model
# Otherwise fresh_model and moshi_vis share the SAME MHA instance!
from jmoshivis.modules.cross_attention import SharedCrossAttention
if hasattr(SharedCrossAttention, '_instances'):
    SharedCrossAttention._instances = {}
    print("   (Cleared SharedCrossAttention singleton cache)")

# Compare checkpoint weights with a freshly initialized model
fresh_model = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
fresh_ca_weights = fresh_model.llm.transformer.layers[0].cross_attention.mha.in_proj_weight
loaded_ca_weights = moshi_vis.llm.transformer.layers[0].cross_attention.mha.in_proj_weight

# Verify they are different objects
print(f"\n   Fresh MHA id: {id(fresh_ca_weights)}")
print(f"   Loaded MHA id: {id(loaded_ca_weights)}")
if id(fresh_ca_weights) == id(loaded_ca_weights):
    print("   ❌ ERROR: Fresh and Loaded are the SAME object (singleton issue)!")
else:
    print("   ✅ Fresh and Loaded are different objects")

print(f"\n📊 Cross-Attention MHA in_proj_weight comparison:")
print(f"  Fresh (random): mean={fresh_ca_weights.mean().item():.6f}, std={fresh_ca_weights.std().item():.6f}")
print(f"  Loaded (trained): mean={loaded_ca_weights.mean().item():.6f}, std={loaded_ca_weights.std().item():.6f}")

# Check if they're significantly different
weights_diff = (fresh_ca_weights.cpu().to(dtype) - loaded_ca_weights.cpu()).abs().mean().item()
print(f"  Mean absolute difference: {weights_diff:.6f}")

if weights_diff < 0.001:
    print(f"  ❌ WARNING: Weights are nearly identical to random init!")
    print(f"     This suggests cross-attention was NOT trained!")
else:
    print(f"  ✅ Weights differ from random init (training occurred)")

# ===== CHECK 5: Compare gate weights with fresh init =====
print("\n🔍 Gate weight comparison:")
fresh_gate_w = fresh_model.llm.transformer.layers[0].cross_attention.gate.alpha[0].weight
loaded_gate_w = moshi_vis.llm.transformer.layers[0].cross_attention.gate.alpha[0].weight

print(f"  Fresh gate[0]: mean={fresh_gate_w.mean().item():.6f}, std={fresh_gate_w.std().item():.6f}")
print(f"  Loaded gate[0]: mean={loaded_gate_w.mean().item():.6f}, std={loaded_gate_w.std().item():.6f}")

gate_diff = (fresh_gate_w.cpu().to(dtype) - loaded_gate_w.cpu()).abs().mean().item()
print(f"  Mean absolute difference: {gate_diff:.6f}")

if gate_diff < 0.001:
    print(f"  ❌ WARNING: Gate weights nearly identical to random init!")
else:
    print(f"  ✅ Gate weights differ from random init")

print("\n" + "="*70)
print("📋 CONCLUSION")
print("="*70)
