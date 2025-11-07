from huggingface_hub import hf_hub_download

repo_id = "kyutai/moshika-vis-pytorch-bf16"
filename = "model.safetensors"

local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="./models")
print("Saved to:", local_path)
