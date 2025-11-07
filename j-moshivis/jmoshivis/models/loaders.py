# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Load moshi-vis neccessary components."""

from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.models.moshivis import MoshiVisGen, MoshiVis


def get_moshi_vis(
    kyuteye_config: KyuteyeConfig,
    moshi_weight: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MoshiVisGen, ImageProjection]:
    """Return main Moshi model"""
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}

    if moshi_weight is not None:
        from safetensors.torch import load_file

        for key, v in load_file(moshi_weight, device=device).items():  # type: ignore
            if key.startswith("image_prefix."):
                image_proj_state[key[13:]] = v
            else:
                model_state[key] = v

    moshi_vis = MoshiVisGen.from_config(
        kyuteye_config, model_state, device, dtype, **(gen_kwargs or {})
    )
    image_embedder = ImageProjection.from_config(
        kyuteye_config, moshi_vis.model_dim, image_proj_state, device
    )

    return moshi_vis.to(dtype), image_embedder.to(dtype)


def get_moshi_vis_train(
    kyuteye_config: KyuteyeConfig,
    moshivis_weight: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    strict: bool = False,
    freeze_backbone: bool = True
) -> Tuple[MoshiVis, ImageProjection]:
    """
    学習用に MoshiVis モデルと ImageProjection を構築する関数。

    Args:
        kyuteye_config (KyuteyeConfig): MoshiVis の設定オブジェクト。
        moshi_weight (Optional[str]): safetensors 形式の重みファイルパス。
        device (str | torch.device): 使用するデバイス。
        dtype (torch.dtype): データ型。学習時は float32 推奨。
        strict (bool): load_state_dict の strict モード。

    Returns:
        Tuple[MoshiVis, ImageProjection]: モデル本体と画像埋め込みモジュール。
    """

    # --- ステート分離用ディクショナリ ---
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}

    if moshivis_weight is not None:
        print(f"🔹 Loading pretrained weights from {moshivis_weight}")
        weights = load_file(moshivis_weight, device="cpu")

        for key, v in weights.items():
            if key.startswith("image_prefix."):
                image_proj_state[key[len("image_prefix."):]] = v
            else:
                model_state[key] = v

    # --- モデル構築 ---
    moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

    # --- 重みロード ---
    if model_state:
        missing, unexpected = moshi_vis.load_state_dict(model_state, strict=strict)
        print(f"✅ MoshiVis loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if image_proj_state:
        image_embedder = ImageProjection.from_config(
            kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
        )

    if freeze_backbone:
        print("🔒 Applying MoshiVis paper-style freezing (train only cross-attn & gating modules).")
        for name, param in moshi_vis.named_parameters():
            # Cross-AttentionとGating部分のみ学習対象に
            if (
                name.startswith("llm.transformer.layers") and
                ("cross_attention" in name or "gating" in name)
            ):
                param.requires_grad = True
                param.data = param.data.to(device)
            else:
                param.requires_grad = False
                param.data = param.data.to("cpu")
        
        torch.cuda.empty_cache()

        # ImageEmbedder も凍結
        for p in image_embedder.parameters():
            p.requires_grad = False

        print("✅ Trainable: cross_attention.*, gating.*")
        print("🚫 Frozen: vision_encoder, self_attn, norm*, text_emb, text_linear, out_norm")

    else:
        print("🟢 Backbone trainable: full fine-tune mode")

    # --- モード設定 ---
    moshi_vis.train()
    image_embedder.eval()

    return moshi_vis.to(dtype=dtype), image_embedder.to(dtype=dtype)
