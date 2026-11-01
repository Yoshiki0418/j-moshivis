import torch
from safetensors.torch import save_model
from jmoshivis.models.loaders import get_moshi_vis_train
from jmoshivis.config.kyuteye_config import KyuteyeConfig

# 1. 読み込む checkpoint
pt_path = "/workspace/j-moshivis/checkpoints/step_2500.pt"

# 2. YAML から config を読み込む
kyuteye_config = KyuteyeConfig.from_yml(
    "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. MoshiVis を構築（moshivis_weight を使わない）
moshi_vis, image_embedder = get_moshi_vis_train(
    kyuteye_config=kyuteye_config,
    moshivis_weight=None,     # ← これ重要
    device=device,
    dtype=torch.bfloat16,
    strict=False,
    freeze_backbone=False,    # load_state_dict のため False にする
)

# 4. checkpoint(state_dict) をロード
state_dict = torch.load(pt_path, map_location="cpu")

# 5. モデルに読み込み
missing, unexpected = moshi_vis.load_state_dict(
    state_dict,
    strict=False,
)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 6. safetensors として保存
save_model(moshi_vis, "model_step2500.safetensors")

print("Saved safetensors successfully!")
