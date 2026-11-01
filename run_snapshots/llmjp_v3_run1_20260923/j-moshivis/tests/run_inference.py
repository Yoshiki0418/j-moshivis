#　ダミー推論コード
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.moshivis import MoshiVisGen

# YAML設定をロード
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")

# safetensors を Hugging Face から取得
weights_path = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="model.safetensors"
)

# state_dict をロード
state_dict = load_file(weights_path)

# モデルを構築
lm_gen = MoshiVisGen.from_config(
    config,
    moshi_weight=state_dict,
    device="cuda:1" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16,
)
lm_gen = lm_gen.to(dtype=torch.bfloat16)

# 推論テスト
with lm_gen.streaming():
    user_tokens = torch.arange(8, dtype=torch.long, device=lm_gen.lm_model.device)
    user_tokens = user_tokens.view(1, 8, 1) 
    for i in range(20):
        out, gate = lm_gen.step(user_tokens)
        print(f"step {i}:", "out=None" if out is None else out.shape, "gate:", gate)
