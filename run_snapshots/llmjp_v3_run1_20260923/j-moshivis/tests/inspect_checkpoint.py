"""
Check what's actually saved in the checkpoint
"""
from safetensors.torch import load_file
import torch

checkpoint_path = "/workspace/j-moshivis/checkpoints/step_1.safetensors"

print("="*70)
print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
print("="*70)

weights = load_file(checkpoint_path, device="cpu")

# Find MHA keys
mha_keys = [k for k in weights.keys() if "cross_attention.mha" in k]
gate_keys = [k for k in weights.keys() if "gate" in k and "cross_attention" in k]

print(f"\n📊 Cross-Attention keys in checkpoint:")
print(f"   MHA keys: {len(mha_keys)}")
print(f"   Gate keys: {len(gate_keys)}")

if mha_keys:
    print(f"\n🔍 MHA keys found:")
    for k in mha_keys[:10]:
        w = weights[k]
        print(f"   {k}: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")

# Check if MHA weights match expected initialization (std=0.02 or 0.019897)
if mha_keys:
    first_mha = weights[mha_keys[0]]
    print(f"\n📊 First MHA weight analysis:")
    print(f"   std={first_mha.std().item():.6f}")
    
    # Check if this is near 0.02 (our init) or 0.019897 (default kaiming)
    if abs(first_mha.std().item() - 0.02) < 0.001:
        print("   ⚠️ std ≈ 0.02 (our manual initialization)")
    elif abs(first_mha.std().item() - 0.019897) < 0.001:
        print("   ⚠️ std ≈ 0.019897 (default initialization)")
    
    # Check if values are all very small (near zero mean)
    if abs(first_mha.mean().item()) < 0.0001:
        print("   ⚠️ mean ≈ 0 (initialization-like)")

# Also check layers other than 0
print(f"\n🔍 Checking if only layer 0 has MHA:")
layer_mha_keys = {}
for k in mha_keys:
    for i in range(32):
        if f"layers.{i}." in k:
            if i not in layer_mha_keys:
                layer_mha_keys[i] = []
            layer_mha_keys[i].append(k)
            break

print(f"   Layers with MHA keys: {list(layer_mha_keys.keys())}")
