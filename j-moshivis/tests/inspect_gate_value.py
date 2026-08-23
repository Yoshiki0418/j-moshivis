
import torch
import sys
from safetensors.torch import load_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_gate_value.py <checkpoint_path>")
        return

    path = sys.argv[1]
    tensors = load_file(path)
    
    print(f"Loading {path}...")
    
    gate_params = []
    for k, v in tensors.items():
        if "gate" in k and "alpha" in k:
            gate_params.append((k, v))
            
    if not gate_params:
        print("No gate params found!")
        return

    print(f"Found {len(gate_params)} gate params.")
    
    changed_count = 0
    zero_count = 0
    
    for name, value in gate_params:
        if value.numel() == 1:
            val = value.item()
            if val == 0.0:
                zero_count += 1
            else:
                changed_count += 1
                print(f"{name}: {val}")
        else:
            # Handle non-scalar gate parameters (e.g. weights)
            is_zero = (value == 0.0).all().item()
            if is_zero:
                zero_count += 1
                print(f"{name}: shape={value.shape} (ALL ZEROS)")
            else:
                changed_count += 1
                mean_val = value.float().mean().item()
                std_val = value.float().std().item()
                print(f"{name}: shape={value.shape}, mean={mean_val:.6f}, std={std_val:.6f}")

    print("-" * 40)
    print(f"Total frozen (exactly 0.0): {zero_count}")
    print(f"Total changed: {changed_count}")
    
    if changed_count == 0:
        print("⚠️  GATE IS COMPLETELY FROZEN (or hasn't learned anything).")
    else:
        print("✅ Gate is learning (values have moved from 0.0).")

if __name__ == "__main__":
    main()
