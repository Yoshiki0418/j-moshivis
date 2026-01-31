from safetensors.torch import load_file
import re

ckpt_path = "/workspace/j-moshivis/checkpoints/step_100.safetensors"
weights = load_file(ckpt_path)

# "layers.数字" の部分を抽出して集計
layer_indices = set()
for key in weights.keys():
    if "cross_attention" in key:
        # 正規表現で数字を抜き出す
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

print(f"🔍 Checkpoint contains Cross-Attention at layers: {sorted(list(layer_indices))}")