"""
Debug script to compare Cross-Attention behavior between 
batch inference (forward_speech) and streaming inference (step)
"""
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sentencepiece as spm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection, ImageProcessor
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from moshi.models.loaders import get_mimi
import torchaudio

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
print(f"🚀 Device: {device}")

# ===== 1. Load everything =====
config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"
tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"

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

print(f"✅ Loaded weights: {len(model_state)} model, {len(image_proj_state)} image_proj")

# Check cross-attention keys
ca_keys = [k for k in model_state.keys() if "cross_attention" in k]
print(f"📊 Cross-attention keys in checkpoint: {len(ca_keys)}")
for k in ca_keys[:10]:
    print(f"   {k}")

# Build model
print("📂 Building MoshiVis model...")
moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
missing, unexpected = moshi_vis.load_state_dict(model_state, strict=False)

# Show missing cross-attention keys
missing_ca = [k for k in missing if "cross_attention" in k]
print(f"\n❓ Missing cross-attention keys: {len(missing_ca)}")
for k in missing_ca[:10]:
    print(f"   {k}")

moshi_vis = moshi_vis.eval().to(device).to(dtype)

# Load Image Embedder
image_embedder = ImageProjection.from_config(
    kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
)
image_embedder = image_embedder.eval().to(dtype)

# ===== 2. Load test data =====
image_processor = ImageProcessor(size=(448, 448), device=device)

image_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
print(f"🖼️ Loading image: {image_path}")
image_tensor = image_processor(image_path).unsqueeze(0).to(dtype)

with torch.no_grad():
    embedder_out = image_embedder(image_tensor)
    ca_src = embedder_out["cross_attention_src"]
print(f"Image features: {ca_src.shape}, mean={ca_src.mean().item():.4f}")

wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
print(f"🎤 Loading audio: {wav_path}")

waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

# Just first 2 seconds for debugging
waveform = waveform[:, :2 * TARGET_SR]
waveform = waveform.unsqueeze(0).to(device)
codes = mimi.encode(waveform)  # [1, 8, T]
print(f"Audio tokens: {codes.shape}")

# ===== 3. Test forward_text batch mode =====
print("\n" + "="*60)
print("🔍 Testing BATCH forward_text (single step for debugging)...")
print("="*60)

# Create proper input (text + all audio codebooks)
# For forward_text we need [B, 17, T] input
num_codebooks = moshi_vis.num_codebooks  # should be 17
print(f"Model expects {num_codebooks} codebooks")

# Create dummy full input for single-step debugging
batch_input = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
batch_input[:, 0, :] = 3  # text padding token
# Fill user audio at positions 9:17
batch_input[:, 9:, :] = codes[:, :, :1]  # first step's audio

print(f"Batch input shape: {batch_input.shape}")

with torch.no_grad():
    # Single step batch forward
    transformer_out_batch, text_logits_batch, gate_batch = moshi_vis.forward_text(
        batch_input,
        cross_attention_src=ca_src,
    )
    
print(f"Batch output: transformer_out={transformer_out_batch.shape}, text_logits={text_logits_batch.shape}, gate={gate_batch}")

# ===== 4. Test streaming step mode =====
print("\n" + "="*60)
print("🔍 Testing STREAMING step (single step for debugging)...")
print("="*60)

moshi_vis_gen = MoshiVisGen(moshi_vis=moshi_vis)

with moshi_vis_gen.streaming():
    chunk = codes[:, :, 0:1]  # First step
    tokens, gate_stream = moshi_vis_gen.step(chunk, ca_src=ca_src)
    print(f"Streaming output: tokens={tokens}, gate={gate_stream}")
    
    # Step 1
    if codes.shape[-1] > 1:
        chunk = codes[:, :, 1:2]
        tokens, gate_stream = moshi_vis_gen.step(chunk, ca_src=ca_src)
        print(f"Streaming step 1: tokens={tokens}, gate={gate_stream}")

# ===== 5. Check cross-attention weights on layer 0 =====
print("\n" + "="*60)
print("🔍 Checking Cross-Attention weights...")
print("="*60)

layer0 = moshi_vis.llm.transformer.layers[0]
if hasattr(layer0, 'cross_attention') and layer0.cross_attention is not None:
    ca = layer0.cross_attention
    print(f"✅ Cross-attention found on layer 0")
    print(f"   Type: {type(ca)}")
    
    # Check gate
    if hasattr(ca, 'gate') and ca.gate is not None:
        if hasattr(ca.gate, 'alpha'):
            alpha = ca.gate.alpha
            if isinstance(alpha, torch.nn.Parameter):
                print(f"   Gate alpha: {alpha.item():.5f}")
            else:
                print(f"   Gate alpha type: {type(alpha)}")
    
    # Check mha weights
    if hasattr(ca, 'mha'):
        mha = ca.mha
        print(f"   MHA type: {type(mha)}")
        if hasattr(mha, 'in_proj_weight'):
            print(f"   in_proj_weight: {mha.in_proj_weight.shape}, mean={mha.in_proj_weight.mean().item():.5f}")
else:
    print("❌ No cross-attention on layer 0!")

# Check other layers
for i in [1, 5, 10, 20, 31]:
    if i < len(moshi_vis.llm.transformer.layers):
        layer = moshi_vis.llm.transformer.layers[i]
        has_ca = hasattr(layer, 'cross_attention') and layer.cross_attention is not None
        print(f"Layer {i}: cross_attention={'Yes' if has_ca else 'No'}")

print("\n🎉 Debug complete!")
