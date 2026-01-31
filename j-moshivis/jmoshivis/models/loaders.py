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
        loaded_weights = load_file(moshi_weight, device="cpu")

        # =========================================================
        # 🔍 デバッグ: ファイル内のキー名を確認 (最初の50個)
        # =========================================================
        print("\n🔍 [DEBUG] File Keys Preview (First 50):")
        all_file_keys = list(loaded_weights.keys())
        for k in all_file_keys[:50]:
            print(f"  - {k}")
        
        # 特にCross-Attention関連のキーを抽出して表示
        print("\n🔍 [DEBUG] Cross-Attention Keys in File:")
        ca_keys = [k for k in all_file_keys if "cross_attention" in k]
        if ca_keys:
            for k in ca_keys[:20]: # 長いので最初の20個
                print(f"  - {k}")
        else:
            print("  ❌ No cross_attention keys found!")
        print("=========================================================\n")

        for key, v in load_file(moshi_weight, device=device).items():  # type: ignore
            clean_key = key
            is_image_proj = False

            # 1. "image_prefix." の処理
            if clean_key.startswith("image_prefix."):
                clean_key = clean_key[len("image_prefix."):]
                is_image_proj = True
            
            # 2. "module." の強制削除 (これが今回の肝)
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            # 3. "_orig_mod." (torch.compile由来) も念のため削除
            if clean_key.startswith("_orig_mod."):
                clean_key = clean_key[len("_orig_mod."):]

            # 4. 振り分け
            if is_image_proj:
                image_proj_state[clean_key] = v
            else:
                model_state[clean_key] = v

    print("🔍 Num image_prefix params:", len(image_proj_state))
    print("🔍 Example keys:", list(image_proj_state.keys())[:10])

    moshi_vis = MoshiVisGen.from_config(
        kyuteye_config, model_state, device, dtype, **(gen_kwargs or {})
    )
    image_embedder = ImageProjection.from_config(
        kyuteye_config, moshi_vis.model_dim, image_proj_state, device
    )

    # --- 2. 厳密なロード確認 (Verification) ---
    print("\n🔍 --- Weight Loading Verification ---")

    # (A) Image Embedder の Projection層
    # SigLIP (enc.model...) だけでなく、学習させた projection (proj, linearなど) があるか？
    embedder_keys = set(image_embedder.state_dict().keys())
    loaded_embedder_keys = set(image_proj_state.keys())
    
    # 必須チェック: "proj" または "linear" を含む層（実装依存だが通常存在するはず）
    proj_params = [k for k in embedder_keys if "proj" in k or "linear" in k]
    missing_proj = [k for k in proj_params if k not in loaded_embedder_keys]

    if missing_proj:
        print(f"❌ [CRITICAL] Image Projection weights MISSING! ({len(missing_proj)} params)")
        print(f"   Example missing: {missing_proj[:3]}")
        print("   -> 画像の特徴量がMoshiの次元に変換されず、推論が壊れます。")
    else:
        print(f"✅ Image Projection weights loaded ({len(proj_params)} params).")

    # (B) MoshiVis の Cross-Attention と Gate
    # モデルの全パラメータ名を取得
    moshi_keys = set(moshi_vis.lm_model.state_dict().keys())
    loaded_moshi_keys = set(model_state.keys())

    # チェック対象: Cross-Attention と Gate
    target_keywords = ["cross_attention", "gate", "xa"]
    important_params = [k for k in moshi_keys if any(x in k for x in target_keywords)]
    
    missing_important = [k for k in important_params if k not in loaded_moshi_keys]

    if missing_important:
        print(f"❌ [CRITICAL] Cross-Attention/Gate weights MISSING! ({len(missing_important)} params)")
        print(f"   Example missing: {missing_important[:-1]}")
        print("   -> モデルは画像情報を無視してテキストのみで生成します。")
    else:
        print(f"✅ Cross-Attention & Gate weights loaded ({len(important_params)} params).")

    print("---------------------------------------\n")

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

        print("😎 Initializing Cross-Attention weights to handle large inputs...")
        for name, p in moshi_vis.named_parameters():
            if "cross_attention" in name and "weight" in name and p.requires_grad:
                if "in_proj" in name or "out_proj" in name or "linear" in name:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)
            if "cross_attention" in name and "bias" in name and p.requires_grad:
                torch.nn.init.constant_(p, 0.0)

        # =========================================================
        # Gateパラメータのゼロ初期化 (Zero Initialization)
        # =========================================================
        print("🧹 Initializing Gate parameters with small weights...")
        for name, p in moshi_vis.named_parameters():
            if "gate" in name and p.requires_grad:
                # 重み(weight)は少し値を持たせる
                if "weight" in name:
                    torch.nn.init.normal_(p, mean=0.0, std=0.01)

    else:
        # freeze_backbone=False の場合は全学習
        print("🟢 Full fine-tuning enabled (all params trainable).")
        moshi_vis.train()
        image_embedder.train()

    # --- モード設定 ---
    moshi_vis.train()
    image_embedder.eval()

    return moshi_vis.to(dtype=dtype), image_embedder.to(dtype=dtype)