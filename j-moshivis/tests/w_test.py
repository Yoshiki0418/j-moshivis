import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import numpy as np

# 設定
repo_id = "kyutai/moshika-vis-pytorch-bf16"
filename = "model.safetensors"

print(f"📥 Downloading/Loading {filename} from {repo_id}...")
weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

# ローカル
model_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"

state_dict = load_file(model_path)

print("\n🔍 Analyzing Cross-Attention Weights...")

# チェックしたいキー（Cross-Attentionの射影層）
target_keys = [
    "llm.transformer.layers.0.cross_attention.mha.in_proj_weight",
    "llm.transformer.layers.0.cross_attention.mha.out_proj.weight"
]

for key in target_keys:
    if key in state_dict:
        weight = state_dict[key].float() # 統計計算のためにfloatに
        
        print(f"\nTarget: {key}")
        print(f"  Shape: {weight.shape}")
        print(f"  Min: {weight.min().item():.6f}")
        print(f"  Max: {weight.max().item():.6f}")
        print(f"  Mean: {weight.mean().item():.6f}")
        print(f"  Std:  {weight.std().item():.6f}") # ★ここが最重要
        
        # さらに詳細：重みの分布を見てみる
        abs_mean = weight.abs().mean().item()
        print(f"  Mean(Abs): {abs_mean:.6f}")

    else:
        print(f"\n⚠️ Key not found: {key}")

# 参考：もしGateのパラメータがあればそれも確認
print("\n🔍 Checking Gate Initialization...")
gate_keys = [k for k in state_dict.keys() if "gate" in k and "weight" in k][:3] # 最初の3つだけ
for key in gate_keys:
    w = state_dict[key].float()
    print(f"Gate: {key} -> Std: {w.std().item():.6f}, Mean: {w.mean().item():.6f}")