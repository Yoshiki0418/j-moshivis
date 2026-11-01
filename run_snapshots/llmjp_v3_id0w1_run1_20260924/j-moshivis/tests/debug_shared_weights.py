"""
Diagnostic: Check if SharedCrossAttention weights are properly shared across layers.
"""
import torch
from safetensors.torch import load_file
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.moshivis import MoshiVis

device = "cpu"
dtype = torch.bfloat16

# Build model
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")
moshivis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

# Load checkpoint
finetuned_state = load_file("/workspace/j-moshivis/checkpoints/step_1000.safetensors", device="cpu")
model_state = {k: v for k, v in finetuned_state.items() if not k.startswith("image_prefix.")}
missing, unexpected = moshivis.load_state_dict(model_state, strict=False)

print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# Check which layers have cross_attention
print("\n🔍 Checking Cross-Attention layers...")
ca_layers = []
for i, layer in enumerate(moshivis.llm.transformer.layers):
    if hasattr(layer, 'cross_attention') and layer.cross_attention is not None:
        ca_layers.append(i)
print(f"   Layers with cross_attention: {ca_layers}")

# Check if MHA weights are shared (same tensor object)
print("\n🔍 Checking if MHA weights are SHARED (same memory)...")
if len(ca_layers) >= 2:
    layer0 = moshivis.llm.transformer.layers[ca_layers[0]]
    layer1 = moshivis.llm.transformer.layers[ca_layers[1]]
    
    w0 = layer0.cross_attention.mha.in_proj_weight
    w1 = layer1.cross_attention.mha.in_proj_weight
    
    print(f"   Layer {ca_layers[0]} mha.in_proj_weight id: {id(w0)}")
    print(f"   Layer {ca_layers[1]} mha.in_proj_weight id: {id(w1)}")
    
    if id(w0) == id(w1):
        print("   ✅ Same tensor object! Weights ARE shared.")
    else:
        print("   ❌ DIFFERENT tensor objects! Weights are NOT shared!")
        # Check if values are the same
        if torch.allclose(w0, w1):
            print("   ⚠️  Values are identical but objects are different.")
        else:
            print("   🚨 Values are DIFFERENT! This is the bug!")
            print(f"      Layer {ca_layers[0]} weight stats: mean={w0.float().mean():.6f}, std={w0.float().std():.6f}")
            print(f"      Layer {ca_layers[1]} weight stats: mean={w1.float().mean():.6f}, std={w1.float().std():.6f}")

# Check actual weight values for each layer
print("\n🔍 Cross-Attention MHA weight statistics per layer:")
for i in ca_layers[:4]:  # First 4 CA layers
    layer = moshivis.llm.transformer.layers[i]
    w = layer.cross_attention.mha.in_proj_weight.float()
    print(f"   Layer {i}: mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.4f}, max={w.max():.4f}")

print("\n✅ Diagnostic complete.")
