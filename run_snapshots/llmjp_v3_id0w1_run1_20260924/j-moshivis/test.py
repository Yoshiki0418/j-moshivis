from huggingface_hub import list_repo_files

repo_id = "kyutai/moshika-vis-pytorch-bf16"
files = list_repo_files(repo_id)
print(files)