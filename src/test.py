from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Hugging Face からダウンロード
path = hf_hub_download(
    repo_id="kyutai/moshiko-pytorch-bf16",
    filename="model.safetensors"
)

# safetensors を開く
with safe_open(path, framework="pt", device="cpu") as f:
    print("✅ Keys example:", list(f.keys())[:20])   # 最初の20個のキー
    print("✅ Metadata:", f.metadata())              # モデル設定がある場合はここに出る
