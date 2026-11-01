import torch
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 設定
device = "cuda"
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")
ckpt_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors" # 崩壊したCheckpoint

hf_path = hf_hub_download(repo_id="kyutai/moshika-vis-pytorch-bf16", filename="model.safetensors")

# モデルロード
embedder = ImageProjection(config, 4096).to(device)
state_dict = load_file(hf_path)
# image_prefix. を取り除いてロード
embedder_dict = {k.replace("image_prefix.", ""): v for k, v in state_dict.items() if "image_prefix." in k}

print("total keys:", len(state_dict))
keys = list(state_dict.keys())
print("first 40 keys:")
for k in keys[:40]:
    print(" ", k)

# 画像prefixっぽいキーをサンプル表示
cand = [k for k in keys if "image" in k.lower() or "prefix" in k.lower()]
print("\nkeys containing 'image' or 'prefix' (first 60):")
for k in cand[:60]:
    print(" ", k)

# ImageProjectionが要求するキーの例
print("\nembedder expected keys (first 40):")
for k in list(embedder.state_dict().keys())[:40]:
    print(" ", k)
embedder.load_state_dict(embedder_dict)

# ダミー入力でテスト
dummy_img = torch.randn(1, 3, 448, 448).to(device)
with torch.no_grad():
    out = embedder(dummy_img)
    src = out["cross_attention_src"]

print(f"Max Value: {src.max().item()}")
print(f"Min Value: {src.min().item()}")
print(f"Std Value: {src.std().item()}")
print(f"Mean Value: {src.mean().item()}")
print(src)