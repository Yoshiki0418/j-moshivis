"""
Compare streaming generation WITH vs WITHOUT image conditioning
to verify cross-attention is actually affecting generation.
"""
import torch
import torchaudio
import soundfile as sf
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
print(f"🚀 Device: {device}")

# ===== Setup =====
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

# ===== Check Cross-Attention Weights in Checkpoint =====
print("\n" + "="*60)
print("🔍 Cross-Attention Weights in Checkpoint:")
print("="*60)

ca_keys = [k for k in model_state.keys() if "cross_attention" in k]
print(f"Total CA keys: {len(ca_keys)}")

if ca_keys:
    for k in ca_keys[:5]:
        w = model_state[k]
        print(f"  {k}: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")
else:
    print("  ❌ NO cross-attention keys in checkpoint!")
    print("  This means CA weights are randomly initialized!")

# Check gate weights
gate_keys = [k for k in model_state.keys() if "gate" in k.lower()]
print(f"\nGate-related keys: {len(gate_keys)}")
for k in gate_keys[:5]:
    w = model_state[k]
    if w.numel() < 10:
        print(f"  {k}: value={w.squeeze().tolist()}")
    else:
        print(f"  {k}: shape={w.shape}, mean={w.mean().item():.6f}")

# ===== Build Model =====
print("\n📂 Building model...")
moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
missing, _ = moshi_vis.load_state_dict(model_state, strict=False)
missing_ca = [k for k in missing if "cross_attention" in k]
print(f"Missing CA keys after load: {len(missing_ca)}")

moshi_vis = moshi_vis.eval().to(device).to(dtype)

# Image embedder
image_embedder = ImageProjection.from_config(
    kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
)
image_embedder = image_embedder.eval().to(dtype)

# ===== Check Model Weights After Loading =====
print("\n" + "="*60)
print("🔍 Model CA Weights After Loading:")
print("="*60)

layer0 = moshi_vis.llm.transformer.layers[0]
if layer0.cross_attention is not None:
    ca = layer0.cross_attention
    if hasattr(ca, 'mha'):
        mha = ca.mha
        if hasattr(mha, 'in_proj_weight'):
            w = mha.in_proj_weight
            print(f"Layer 0 MHA in_proj_weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
        if hasattr(mha, 'out_proj'):
            w = mha.out_proj.weight
            print(f"Layer 0 MHA out_proj: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
    if hasattr(ca, 'gate') and ca.gate is not None:
        gate = ca.gate
        if hasattr(gate, 'alpha'):
            if isinstance(gate.alpha, torch.nn.Parameter):
                print(f"Gate alpha (scalar): {gate.alpha.item():.6f}")
            else:
                # It's a Sequential network for conditional gating
                for name, param in gate.alpha.named_parameters():
                    print(f"Gate {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

# ===== Load Test Data =====
image_processor = ImageProcessor(size=(448, 448), device=device)
image_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
image_tensor = image_processor(image_path).unsqueeze(0).to(dtype)

with torch.no_grad():
    ca_src = image_embedder(image_tensor)["cross_attention_src"]
print(f"\nImage features: {ca_src.shape}")

wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

# Use 5 seconds of audio
waveform = waveform[:, :5 * TARGET_SR]
silence = torch.zeros(1, 5 * TARGET_SR)
waveform = torch.cat([waveform, silence], dim=1).unsqueeze(0).to(device)
codes = mimi.encode(waveform)
print(f"Audio tokens: {codes.shape}")

# ===== Streaming Generation: WITH Image =====
print("\n" + "="*60)
print("🖼️ Streaming WITH image conditioning...")
print("="*60)

moshi_vis_gen = MoshiVisGen(moshi_vis=moshi_vis)
all_tokens_with_image = []
gates_with_image = []

with moshi_vis_gen.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen.step(chunk, ca_src=ca_src)
        
        if gate:
            gates_with_image.append(float(gate))
        if tokens is not None:
            all_tokens_with_image.append(tokens[:, 1:, :].cpu())  # audio only

if all_tokens_with_image:
    audio_with_image = torch.cat(all_tokens_with_image, dim=-1)
    generated = mimi.decode(audio_with_image.to(device))
    sf.write("streaming_WITH_image.wav", generated.squeeze().detach().cpu().numpy(), TARGET_SR)
    print(f"✅ Saved: streaming_WITH_image.wav")
    print(f"   Gate stats: mean={sum(gates_with_image)/len(gates_with_image):.5f}, max={max(gates_with_image):.5f}")

# ===== Streaming Generation: WITHOUT Image =====
print("\n" + "="*60)
print("🔇 Streaming WITHOUT image conditioning (ca_src=None)...")
print("="*60)

moshi_vis_gen2 = MoshiVisGen(moshi_vis=moshi_vis)
all_tokens_no_image = []
gates_no_image = []

with moshi_vis_gen2.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]
        tokens, gate = moshi_vis_gen2.step(chunk, ca_src=None)  # NO IMAGE
        
        if gate:
            gates_no_image.append(float(gate))
        if tokens is not None:
            all_tokens_no_image.append(tokens[:, 1:, :].cpu())

if all_tokens_no_image:
    audio_no_image = torch.cat(all_tokens_no_image, dim=-1)
    generated = mimi.decode(audio_no_image.to(device))
    sf.write("streaming_NO_image.wav", generated.squeeze().detach().cpu().numpy(), TARGET_SR)
    print(f"✅ Saved: streaming_NO_image.wav")
    if gates_no_image:
        print(f"   Gate stats: mean={sum(gates_no_image)/len(gates_no_image):.5f}, max={max(gates_no_image):.5f}")
    else:
        print("   No gate values (expected when no image)")

# ===== Compare Token Outputs =====
print("\n" + "="*60)
print("📊 Comparing outputs WITH vs WITHOUT image:")
print("="*60)

if all_tokens_with_image and all_tokens_no_image:
    tokens_with = torch.cat(all_tokens_with_image, dim=-1)
    tokens_no = torch.cat(all_tokens_no_image, dim=-1)
    
    # Check if they're different
    diff = (tokens_with != tokens_no).float().mean().item()
    print(f"Token difference rate: {diff*100:.2f}%")
    
    if diff < 0.01:
        print("   ⚠️ WARNING: Outputs are almost IDENTICAL!")
        print("   This suggests cross-attention is NOT affecting generation!")
    elif diff > 0.5:
        print("   ✅ Outputs are significantly DIFFERENT")
        print("   Cross-attention IS affecting generation")
    else:
        print("   Outputs have some differences")

print("\n🎉 Comparison complete!")
print("Listen to both audio files and compare:")
print("  - streaming_WITH_image.wav")
print("  - streaming_NO_image.wav")
