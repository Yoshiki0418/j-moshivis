# check_weights.py
import torch
from safetensors.torch import load_file

# Step 2000のファイルを指定
ckpt_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors" 
state_dict = load_file(ckpt_path)

print("--- Gate Parameters Check ---")
zero_count = 0
total_count = 0
for k, v in state_dict.items():
    if "gate" in k or "alpha" in k:
        # Gateパラメータ（スカラーまたは小規模テンソル）
        mean_val = v.float().mean().item()
        print(f"{k}: {mean_val}")
        if abs(mean_val) < 1e-9:
            zero_count += 1
        total_count += 1

print(f"\nResult: {zero_count} / {total_count} gates are still EXACTLY 0.0")