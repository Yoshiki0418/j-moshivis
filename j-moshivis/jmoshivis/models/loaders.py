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
                image_proj_state[key[len("image_prefix."):]] = v
            else:
                model_state[key] = v

    print("🔍 Num image_prefix params:", len(image_proj_state))
    print("🔍 Example keys:", list(image_proj_state.keys())[:10])

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
                # =========================================================
                # ★ 修正ポイント: Cross-Attention/Gate の重みをロードから除外
                # =========================================================
                # "cross_attention" がキーに含まれる場合（Gateもこれに含まれる構成が一般的）
                # 辞書に追加せずスキップすることで、これらのパラメータはロードされず、
                # モデル初期化時の値（ランダム or ゼロ初期化）が維持されます。
                if "cross_attention" in key:
                    # デバッグ用に最初の数個だけログに出しても良い
                    # print(f"Skipping init for: {key}") 
                    continue

                model_state[key] = v

    print("🔍 Num image_prefix params:", len(image_proj_state))
    print(f"🔍 Model params to load: {len(model_state)} (Cross-Attention excluded)")

    # --- モデル構築 ---
    # ここで __init__ が走り、Cross-AttentionやGateはランダム(または0)で初期化される
    moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

    # --- 重みロード ---
    if model_state:
        # Cross-Attentionのキーが model_state に無いため、missing_keys に含まれることになる
        # strict=False なのでエラーにはならない
        missing, unexpected = moshi_vis.load_state_dict(model_state, strict=False)

        # 期待通り cross_attention が missing になっているか確認
        ca_missing = [k for k in missing if "cross_attention" in k]
        print("✅ MoshiVis loaded.")
        print(f"   - Total Missing: {len(missing)}")
        print(f"   - Cross-Attention Missing (As Expected): {len(ca_missing)}")
        print(f"   - Unexpected: {len(unexpected)}")

    if image_proj_state:
        image_embedder = ImageProjection.from_config(
            kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
        )

    # ----------------------------------------------------
    # 3. Freeze / unfreeze strategy
    # ----------------------------------------------------
    if freeze_backbone:
        print("🔒 Applying selective fine-tune: cross-attn only.")

        trainable_count = 0
        for name, param in moshi_vis.named_parameters():
            if "llm.transformer.layers" in name and (
                "cross_attention" in name or
                "norm_cross" in name or
                "gate" in name
            ):
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False

        # ImageProjection fully frozen
        image_embedder.train()

        embedder_trainable_count = 0
        for name, p in image_embedder.named_parameters():
            # "enc" (SigLIPなどのバックボーン) は凍結
            if "enc." in name:
                p.requires_grad = False
            # それ以外 (proj_xa, norm_xa 等) は学習させる
            else:
                p.requires_grad = True
                embedder_trainable_count += 1

        print(f"🔥 Trainable params count: Moshi(CA)={trainable_count}, Embedder(Proj)={embedder_trainable_count}")

        # =========================================================
        # Gateパラメータのゼロ初期化 (Zero Initialization)
        # =========================================================
        print("🧹 Initializing Gate parameters with small weights...")
        for name, p in moshi_vis.named_parameters():
            if "gate" in name and p.requires_grad:
                # 重み(weight)は少し値を持たせる
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(p, gain=0.01) 
                    # または torch.nn.init.normal_(p, mean=0.0, std=0.01)
                # バイアス(bias)は閉じる方向に設定（元のロジックを維持）
                elif "bias" in name:
                    # XAGateの実装が x - 4 としているなら 0.0 でOK
                    # 実装に依存しますが、今のままでOKな可能性が高い
                    torch.nn.init.constant_(p, 0.0)

    else:
        # freeze_backbone=False の場合は全学習
        print("🟢 Full fine-tuning enabled (all params trainable).")
        moshi_vis.train()
        image_embedder.train()

    # --- モード設定 ---
    moshi_vis.train()
    image_embedder.eval()

    return moshi_vis.to(dtype=dtype), image_embedder.to(dtype=dtype)