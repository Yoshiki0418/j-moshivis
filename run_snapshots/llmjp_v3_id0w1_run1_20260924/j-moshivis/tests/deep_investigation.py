"""
Deep investigation of streaming inference issues.

Hypotheses to verify:
1. Visual feature normalization mismatch between batch/streaming
2. Visual tokens disappearing in step 1+
3. Cross-attention mask bugs in streaming mode
"""
import torch
import torchaudio
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sentencepiece as spm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection, ImageProcessor
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from moshi.models.loaders import get_mimi

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print("="*70)
print("🔬 DEEP INVESTIGATION: Streaming Inference Issues")
print("="*70)

# Setup
config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"
tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"

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

# Load test data
image_processor = ImageProcessor(size=(448, 448), device=device)
image_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"

# ===== HYPOTHESIS 1: Visual Feature Normalization =====
print("\n" + "="*70)
print("🔍 HYPOTHESIS 1: Visual Feature Normalization")
print("="*70)

# Raw image tensor from processor
image_tensor_raw = image_processor(image_path)
print(f"\n📷 Raw image tensor from ImageProcessor:")
print(f"   Shape: {image_tensor_raw.shape}")
print(f"   Mean: {image_tensor_raw.mean().item():.4f}")
print(f"   Std: {image_tensor_raw.std().item():.4f}")
print(f"   Min: {image_tensor_raw.min().item():.4f}")
print(f"   Max: {image_tensor_raw.max().item():.4f}")

# Check if values are normalized (should be around 0 mean, 1 std for vision models)
if image_tensor_raw.max() > 10:
    print("   ⚠️ WARNING: Max > 10, image might not be normalized!")
elif abs(image_tensor_raw.mean()) > 1:
    print("   ⚠️ WARNING: Mean is not near 0, check normalization!")
else:
    print("   ✅ Values look normalized")

# Image features from embedder
image_tensor = image_tensor_raw.unsqueeze(0).to(dtype)
with torch.no_grad():
    embedder_out = image_embedder(image_tensor)
    ca_src = embedder_out["cross_attention_src"]

print(f"\n🖼️ Image features (ca_src) from embedder:")
print(f"   Shape: {ca_src.shape}")
print(f"   Mean: {ca_src.mean().item():.6f}")
print(f"   Std: {ca_src.std().item():.6f}")
print(f"   Min: {ca_src.min().item():.6f}")
print(f"   Max: {ca_src.max().item():.6f}")

# Check for NaN or Inf
has_nan = torch.isnan(ca_src).any().item()
has_inf = torch.isinf(ca_src).any().item()
print(f"   Contains NaN: {has_nan}")
print(f"   Contains Inf: {has_inf}")

if has_nan or has_inf:
    print("   ❌ CRITICAL: NaN or Inf in visual features!")
elif abs(ca_src.max().item()) > 100:
    print("   ⚠️ Large values in visual features, might cause issues")
else:
    print("   ✅ Visual features look reasonable")

# ===== HYPOTHESIS 2: Visual Token Persistence in Streaming =====
print("\n" + "="*70)
print("🔍 HYPOTHESIS 2: Visual Token Persistence in Streaming")
print("="*70)

# Load audio
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
waveform = waveform[:, :2 * TARGET_SR].unsqueeze(0).to(device)  # 2 seconds
codes = mimi.encode(waveform)
print(f"Audio codes: {codes.shape}")

# Track what ca_src is at each step
print("\n🔄 Tracking ca_src through streaming steps...")

# Patch the forward_text method to track inputs
original_forward_text = moshi_vis.forward_text
ca_src_per_step = []

def tracked_forward_text(input_ids, cross_attention_src=None, **kwargs):
    # Record what ca_src is being passed
    if cross_attention_src is not None:
        ca_src_per_step.append({
            'step': len(ca_src_per_step),
            'ca_src_shape': cross_attention_src.shape,
            'ca_src_mean': cross_attention_src.mean().item(),
            'ca_src_ptr': cross_attention_src.data_ptr(),
        })
    else:
        ca_src_per_step.append({
            'step': len(ca_src_per_step),
            'ca_src_shape': None,
            'ca_src_mean': None,
            'ca_src_ptr': None,
        })
    return original_forward_text(input_ids, cross_attention_src=cross_attention_src, **kwargs)

moshi_vis.forward_text = tracked_forward_text

# Run streaming
moshi_vis_gen = MoshiVisGen(moshi_vis=moshi_vis)

with moshi_vis_gen.streaming():
    for t in range(min(10, codes.shape[-1])):  # First 10 steps
        chunk = codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen.step(chunk, ca_src=ca_src)

# Restore original method
moshi_vis.forward_text = original_forward_text

# Analyze results
print(f"\n📊 ca_src tracking results (first 10 steps):")
first_ptr = ca_src_per_step[0]['ca_src_ptr'] if ca_src_per_step else None

for info in ca_src_per_step[:10]:
    step = info['step']
    if info['ca_src_shape'] is None:
        print(f"   Step {step}: ca_src = None ❌")
    else:
        same_ptr = info['ca_src_ptr'] == first_ptr
        ptr_status = "✅ same" if same_ptr else "⚠️ DIFFERENT"
        print(f"   Step {step}: shape={info['ca_src_shape']}, mean={info['ca_src_mean']:.4f}, ptr={ptr_status}")

# Check if ca_src is consistent
if ca_src_per_step:
    all_have_ca = all(info['ca_src_shape'] is not None for info in ca_src_per_step)
    all_same_ptr = all(info['ca_src_ptr'] == first_ptr for info in ca_src_per_step if info['ca_src_ptr'])
    
    if not all_have_ca:
        print("\n   ❌ CRITICAL: Some steps have ca_src=None!")
    elif not all_same_ptr:
        print("\n   ⚠️ WARNING: ca_src pointer changes between steps!")
    else:
        print("\n   ✅ ca_src is consistently passed to all steps")

# ===== HYPOTHESIS 3: Cross-Attention Mask in Streaming =====
print("\n" + "="*70)
print("🔍 HYPOTHESIS 3: Cross-Attention Mask/Processing")
print("="*70)

# Check GatedCrossAttention's is_active logic
layer0_ca = moshi_vis.llm.transformer.layers[0].cross_attention

print(f"\n📋 Cross-Attention Config:")
print(f"   Type: {type(layer0_ca)}")
if hasattr(layer0_ca, 'xa_start'):
    print(f"   xa_start: {layer0_ca.xa_start}")
if hasattr(layer0_ca, 'xa_end'):
    print(f"   xa_end: {layer0_ca.xa_end}")
if hasattr(layer0_ca, 'xa_step'):
    print(f"   xa_step: {layer0_ca.xa_step}")

# Check the forward behavior in streaming context
print("\n🧪 Testing Cross-Attention in streaming context...")

# Create a fresh MoshiVisGen and check during streaming
moshi_vis_gen2 = MoshiVisGen(moshi_vis=moshi_vis)

ca_outputs_in_streaming = []

# Patch to capture cross-attention outputs
original_ca_forward = layer0_ca.forward

def tracked_ca_forward(x, key=None, value=None, cross_attention_mask=None, image_tokens_mask=None):
    result = original_ca_forward(x, key, value, cross_attention_mask, image_tokens_mask)
    out, gate = result
    ca_outputs_in_streaming.append({
        'step': len(ca_outputs_in_streaming),
        'input_mean': x.mean().item(),
        'key_mean': key.mean().item() if key is not None else None,
        'output_mean': out.mean().item(),
        'gate': gate,
        'is_zero_output': torch.allclose(out, torch.zeros_like(out), atol=1e-6),
    })
    return result

layer0_ca.forward = tracked_ca_forward

with moshi_vis_gen2.streaming():
    for t in range(min(5, codes.shape[-1])):
        chunk = codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen2.step(chunk, ca_src=ca_src)

layer0_ca.forward = original_ca_forward

print(f"\n📊 Cross-Attention Layer 0 outputs during streaming:")
for info in ca_outputs_in_streaming[:5]:
    key_status = f"key_mean={info['key_mean']:.4f}" if info['key_mean'] is not None else "key=None ❌"
    zero_status = "ZERO ❌" if info['is_zero_output'] else "non-zero ✅"
    gate_val = info['gate']
    if gate_val is None:
        gate_str = "None"
    elif isinstance(gate_val, (int, float)):
        gate_str = f"{gate_val:.5f}"
    elif hasattr(gate_val, 'numel') and gate_val.numel() == 1:
        gate_str = f"{gate_val.item():.5f}"
    elif hasattr(gate_val, 'mean'):
        gate_str = f"mean={gate_val.mean().item():.5f}"
    else:
        gate_str = str(type(gate_val))
    print(f"   Step {info['step']}: {key_status}, out={zero_status}, gate={gate_str}")

# Check if any outputs are zero (would indicate cross-attention not working)
any_zero = any(info['is_zero_output'] for info in ca_outputs_in_streaming)
any_no_key = any(info['key_mean'] is None for info in ca_outputs_in_streaming)

if any_no_key:
    print("\n   ❌ CRITICAL: Some steps have no key (ca_src not reaching CA layer)!")
elif any_zero:
    print("\n   ⚠️ WARNING: Some CA outputs are zero (is_active might be False)")
else:
    print("\n   ✅ Cross-attention is producing non-zero outputs in all steps")

print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)
