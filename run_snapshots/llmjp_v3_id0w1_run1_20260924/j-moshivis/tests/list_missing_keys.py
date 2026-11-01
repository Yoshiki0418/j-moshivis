"""
Script to list all keys missing from the J-MoshiVis finetuned checkpoint
compared to the original MoshiVis checkpoint.
"""
from safetensors.torch import load_file
import sys

def analyze_missing_keys():
    print("📂 Loading original backbone (model_merged_bf16.safetensors)...")
    pretrained_state = load_file("/workspace/j-moshivis/model_merged_bf16.safetensors", device="cpu")
    pretrained_keys = set(pretrained_state.keys())

    print("📂 Loading finetuned adapters (step_1000.safetensors)...")
    try:
        finetuned_state = load_file("/workspace/j-moshivis/checkpoints/step_1000.safetensors", device="cpu")
        # Remove image_prefix from finetuned keys for fair comparison
        finetuned_keys = set()
        for k in finetuned_state.keys():
            if k.startswith("image_prefix."):
                continue
            finetuned_keys.add(k)
    except Exception as e:
        print(f"Error loading finetuned checkpoint: {e}")
        return

    # Find missing keys
    missing_keys = sorted(list(pretrained_keys - finetuned_keys))
    
    print(f"\n📊 Statistics:")
    print(f"   Original keys: {len(pretrained_keys)}")
    print(f"   Finetuned keys (matched): {len(finetuned_keys)}")
    print(f"   MISSING keys: {len(missing_keys)}")
    
    print(f"\n✅ Missing Keys Dump:")
    for key in missing_keys:
        print(f"MISSING: {key}")
            
    # Analyze patterns
    print("\n🔍 Missing Key Patterns (Categorized):")
    patterns = {}
    for key in missing_keys:
        prefix = ".".join(key.split(".")[:4]) # Group by layer/component
        patterns[prefix] = patterns.get(prefix, 0) + 1
        
    for p, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"   {p}: {count} keys")

if __name__ == "__main__":
    analyze_missing_keys()
