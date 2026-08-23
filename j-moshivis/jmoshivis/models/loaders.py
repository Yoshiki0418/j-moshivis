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
    freeze_backbone: bool = True,
    depformer_config: Optional[dict] = None,
    gate_config: Optional[dict] = None,
    cross_attention_config: Optional[dict] = None,
) -> Tuple[MoshiVis, ImageProjection]:
    """
    学習用に MoshiVis モデルと ImageProjection を構築する関数。

    Args:
        kyuteye_config (KyuteyeConfig): MoshiVis の設定オブジェクト。
        moshi_weight (Optional[str]): safetensors 形式の重みファイルパス。
        device (str | torch.device): 使用するデバイス。
        dtype (torch.dtype): データ型。学習時は float32 推奨。
        strict (bool): load_state_dict の strict モード。
        freeze_backbone (bool): バックボーンを凍結するか。
        depformer_config (dict): Depformer学習設定
            - train: bool, Depformerを学習するか
            - learning_rate: float, 学習率
            - layers: str, "all" または "0,1,2" で層を指定
            - train_embedding: bool, embeddingも学習するか
            - train_linear: bool, audio_linearsも学習するか
        gate_config (dict): Gate設定
            - freeze: bool, Gateを凍結するか
            - init_alpha: float, α初期値
        cross_attention_config (dict): Cross-Attention初期化設定
            - inherit_from_pretrained: bool
              False=ランダム初期化で学習開始 (現状の挙動),
              True =MoshiVisのCross-Attention/Gateを継承して学習開始

    Returns:
        Tuple[MoshiVis, ImageProjection]: モデル本体と画像埋め込みモジュール。
    """

    # --- ステート分離用ディクショナリ ---
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}

    # Cross-Attention/Gate の初期化方針 (config駆動)
    #   False (デフォルト): ロードから除外し、後段でランダム初期化 → 現状と完全に同じ挙動
    #   True             : MoshiVis の Cross-Attention/Gate を継承して学習開始
    inherit_ca = bool(
        cross_attention_config
        and cross_attention_config.get("inherit_from_pretrained", False)
    )

    if moshivis_weight is not None:
        print(f"🔹 Loading pretrained weights from {moshivis_weight}")
        weights = load_file(moshivis_weight, device="cpu")

        # チェックポイント内に Cross-Attention の重みが存在するか確認
        ca_keys_in_file = [k for k in weights.keys() if "cross_attention" in k]
        if inherit_ca and len(ca_keys_in_file) == 0:
            print(
                "⚠️ inherit_from_pretrained=True ですが、チェックポイントに "
                "Cross-Attention の重みがありません。ランダム初期化にフォールバックします。"
            )
            inherit_ca = False

        if inherit_ca:
            print(
                f"🧬 Cross-Attention: MoshiVis から継承して学習開始 "
                f"({len(ca_keys_in_file)} keys)"
            )
        else:
            print("🎲 Cross-Attention: ランダム初期化で学習開始 (現状と同じ挙動)")

        for key, v in weights.items():
            if key.startswith("image_prefix."):
                image_proj_state[key[len("image_prefix."):]] = v
            else:
                # Cross-Attention/Gate の重み:
                #   inherit_ca=False → ロードから除外 (後段でランダム初期化)
                #   inherit_ca=True  → ロードして MoshiVis の重みを継承
                if "cross_attention" in key and not inherit_ca:
                    continue

                model_state[key] = v

    print("🔍 Num image_prefix params:", len(image_proj_state))
    print(
        f"🔍 Model params to load: {len(model_state)} "
        f"(Cross-Attention {'included' if inherit_ca else 'excluded'})"
    )

    # --- モデル構築 ---
    # ここで __init__ が走り、Cross-AttentionやGateはランダム(または0)で初期化される
    moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

    # --- 重みロード ---
    if model_state:
        # Cross-Attentionのキーが model_state に無いため、missing_keys に含まれることになる
        # strict=False なのでエラーにはならない
        missing, unexpected = moshi_vis.load_state_dict(model_state, strict=False)

        # Cross-Attention の missing 状況を確認
        #   inherit_ca=False → cross_attention は意図的に missing になる
        #   inherit_ca=True  → cross_attention も継承されるため missing は ~0
        ca_missing = [k for k in missing if "cross_attention" in k]
        print("✅ MoshiVis loaded.")
        print(f"   - Total Missing: {len(missing)}")
        print(
            f"   - Cross-Attention Missing: {len(ca_missing)} "
            f"({'inherited' if inherit_ca else 'random-init予定'})"
        )
        print(f"   - Unexpected: {len(unexpected)}")

    if image_proj_state:
        image_embedder = ImageProjection.from_config(
            kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
        )

    # ----------------------------------------------------
    # 3. Freeze / unfreeze strategy
    # ----------------------------------------------------
    # デフォルト設定
    if depformer_config is None:
        depformer_config = {"train": False, "layers": "all", "train_embedding": False, "train_linear": False}
    if gate_config is None:
        gate_config = {"freeze": False, "init_alpha": 0.0}  # デフォルトは学習可能

    if freeze_backbone:
        print("🔒 Applying selective fine-tune: cross-attn + configurable components")

        trainable_count = 0
        depformer_trainable_count = 0
        
        # Depformer層の解析
        depformer_layers_to_train = set()
        if depformer_config.get("train", False):
            layers_str = depformer_config.get("layers", "all")
            if layers_str == "all":
                depformer_layers_to_train = None  # Noneは全層
            else:
                depformer_layers_to_train = set(int(x.strip()) for x in layers_str.split(","))
            print(f"📊 Depformer training enabled: layers={layers_str}")
        
        for name, param in moshi_vis.named_parameters():
            should_train = False
            
            # Cross-Attention関連 (常に学習)
            if "llm.transformer.layers" in name and (
                "cross_attention" in name or
                "norm_cross" in name or
                "gate" in name
            ):
                should_train = True
                trainable_count += 1
            
            # Depformer本体
            if depformer_config.get("train", False):
                # depformer transformer layers
                if "depformer" in name and "layers" in name:
                    # 層番号を抽出 (e.g., "depformer.layers.0.self_attn...")
                    import re
                    match = re.search(r"depformer\.layers\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        if depformer_layers_to_train is None or layer_idx in depformer_layers_to_train:
                            should_train = True
                            depformer_trainable_count += 1
                
                # depformer_in (projection)
                if "depformer_in" in name:
                    should_train = True
                    depformer_trainable_count += 1
                
                # depformer_emb / depformer_text_emb
                if depformer_config.get("train_embedding", False):
                    if "depformer_emb" in name or "depformer_text_emb" in name:
                        should_train = True
                        depformer_trainable_count += 1
                
                # audio_linears
                if depformer_config.get("train_linear", False):
                    if "audio_linears" in name:
                        should_train = True
                        depformer_trainable_count += 1
            
            param.requires_grad = should_train

        # ImageProjection
        image_embedder.train()
        embedder_trainable_count = 0
        for name, p in image_embedder.named_parameters():
            if "enc." in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
                embedder_trainable_count += 1

        print(f"🔥 Trainable params count: Moshi(CA)={trainable_count}, Depformer={depformer_trainable_count}, Embedder(Proj)={embedder_trainable_count}")

        # Cross-Attention初期化
        #   inherit_ca=True  → MoshiVis から継承済みのため再初期化しない
        #   inherit_ca=False → ランダム初期化 (現状の挙動)
        if inherit_ca:
            print("🧬 Cross-Attention weights inherited from MoshiVis "
                  "(skip random init).")
        else:
            print("😎 Initializing Cross-Attention weights (random)...")
            for name, p in moshi_vis.named_parameters():
                if "cross_attention" in name and "weight" in name and p.requires_grad:
                    if "in_proj" in name or "out_proj" in name or "linear" in name:
                        torch.nn.init.normal_(p, mean=0.0, std=0.02)
                if "cross_attention" in name and "bias" in name and p.requires_grad:
                    torch.nn.init.constant_(p, 0.0)

        # =========================================================
        # Gate設定 (Config-driven)
        # =========================================================
        gate_freeze = gate_config.get("freeze", False)
        gate_init_alpha = gate_config.get("init_alpha", 0.0)
        
        if gate_freeze:
            print(f"🔓 Gate: Freezing at alpha={gate_init_alpha} (gate ≈ {1/(1+2.718**(-(gate_init_alpha-5.3))):.1%})")
        else:
            print(f"� Gate: Trainable, init_alpha={gate_init_alpha}")
        
        for name, p in moshi_vis.named_parameters():
            if "gate" in name and "alpha" in name:
                # inherit_ca=True のときは MoshiVis から継承した Gate 重みを保持し、
                # 再初期化しない (requires_grad / freeze 設定のみ適用)
                if not inherit_ca:
                    if p.ndim > 0:
                        # For Conditional Gating (MLP), we must NOT set weights to 0.0 (kills gradient).
                        # Initialize with small noise to keep output small (~0) but allow gradient flow.
                        torch.nn.init.normal_(p, mean=0.0, std=0.01)
                    else:
                        # For Scalar Gating, set to the desired initial alpha value.
                        torch.nn.init.constant_(p, gate_init_alpha)
                if gate_freeze:
                    p.requires_grad = False
                    print(f"   Frozen: {name}")
                else:
                    p.requires_grad = True
                    print(f"   Trainable: {name}")

    else:
        # freeze_backbone=False の場合は全学習
        print("🟢 Full fine-tuning enabled (all params trainable).")
        moshi_vis.train()
        image_embedder.train()

    # --- モード設定 ---
    moshi_vis.train()
    image_embedder.eval()

    return moshi_vis.to(dtype=dtype), image_embedder.to(dtype=dtype)