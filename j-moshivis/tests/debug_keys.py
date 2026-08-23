"""
Debug: Check what keys are saved vs what's needed
"""
import torch
from safetensors.torch import load_file

# Load both checkpoints
original = load_file("/workspace/j-moshivis/model_merged_bf16.safetensors", device="cpu")
finetuned = load_file("/workspace/j-moshivis/checkpoints/step_6000.safetensors", device="cpu")

orig_keys = set(original.keys())
ft_keys = set(k for k in finetuned.keys() if not k.startswith("image_prefix."))

print(f"Original model keys: {len(orig_keys)}")
print(f"Finetuned model keys (excluding image_prefix): {len(ft_keys)}")

# Find keys in original but not in finetuned
missing_in_ft = orig_keys - ft_keys
print(f"\n🔍 Keys in ORIGINAL but NOT in FINETUNED: {len(missing_in_ft)}")

# Sample some
missing_samples = list(missing_in_ft)[:20]
print("Samples:")
for k in sorted(missing_samples):
    print(f"  - {k}")

# Check if backbone keys are missing
backbone_missing = [k for k in missing_in_ft if "llm" in k and "cross_attention" not in k]
print(f"\n❌ LLM backbone keys missing (exc. cross-attn): {len(backbone_missing)}")
print("Samples:")
for k in sorted(backbone_missing)[:10]:
    print(f"  - {k}")
