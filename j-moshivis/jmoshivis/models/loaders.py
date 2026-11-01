# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Load MoshiVis components with auditable visual-adapter initialization."""

import hashlib
import json
import re
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from safetensors.torch import load_file

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.models.moshivis import MoshiVisGen, MoshiVis


VisualAdapterInit = Literal[
    "inherit_visual_adapter",
    "random_visual_adapter",
]
VISUAL_ADAPTER_INIT_MODES = {
    "inherit_visual_adapter",
    "random_visual_adapter",
}


def _clean_checkpoint_key(key: str) -> str:
    """Remove wrappers introduced by DDP/torch.compile from a state-dict key."""

    while key.startswith(("module.", "_orig_mod.")):
        key = key.split(".", 1)[1]
    return key


def _is_model_visual_adapter_key(key: str) -> bool:
    """Model-side visual adapter = cross-attention, gate, and its input norm."""

    return "cross_attention" in key or "norm_cross" in key


def _is_allowed_shared_mha_alias_missing(key: str) -> bool:
    """Return whether ``key`` is an expected alias of the shared layer-0 MHA.

    ``safetensors.save_model`` stores a shared tensor once.  In the checked-in
    MoshiVis configuration, layer 0 is the canonical cross-attention MHA owner
    and layers 1..N expose aliases to those same tensors.  Only those aliases
    may be absent when an inherited visual adapter is loaded; a missing gate,
    ``norm_cross``, or layer-0 MHA tensor means the adapter is incomplete.
    """

    match = re.search(r"(?:^|\.)layers\.(\d+)\.cross_attention\.mha(?:\.|$)", key)
    return match is not None and int(match.group(1)) > 0


def _resolve_visual_adapter_init_mode(
    cross_attention_config: Optional[dict],
) -> VisualAdapterInit:
    """Resolve the explicit mode, with the old boolean accepted for compatibility."""

    config = cross_attention_config or {}
    explicit = config.get("visual_adapter_init")
    legacy = config.get("inherit_from_pretrained")
    if explicit is None:
        # Preserve the loader's historical default when neither option exists.
        return "inherit_visual_adapter" if bool(legacy) else "random_visual_adapter"
    if explicit not in VISUAL_ADAPTER_INIT_MODES:
        choices = ", ".join(sorted(VISUAL_ADAPTER_INIT_MODES))
        raise ValueError(f"visual_adapter_init must be one of: {choices}; got {explicit!r}")
    if legacy is not None:
        legacy_mode = (
            "inherit_visual_adapter" if bool(legacy) else "random_visual_adapter"
        )
        if explicit != legacy_mode:
            raise ValueError(
                "Conflicting visual-adapter initialization settings: "
                f"visual_adapter_init={explicit!r}, "
                f"inherit_from_pretrained={legacy!r}"
            )
    return explicit


def _partition_training_checkpoint(
    weights: Dict[str, torch.Tensor],
    *,
    inherit_visual_adapter: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, list[str]]]:
    """Split image/model state and exclude the *whole* adapter in random mode."""

    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}
    source_group_keys: Dict[str, list[str]] = {
        "moshi_backbone": [],
        "cross_attention_and_gate": [],
        "norm_cross": [],
        "image_encoder": [],
        "image_projection_and_norm": [],
    }
    for raw_key, value in weights.items():
        key = _clean_checkpoint_key(raw_key)
        if key.startswith("image_prefix."):
            image_key = key[len("image_prefix."):]
            image_proj_state[image_key] = value
            group = (
                "image_encoder"
                if image_key.startswith("enc.")
                else "image_projection_and_norm"
            )
            source_group_keys[group].append(key)
            continue

        if "norm_cross" in key:
            source_group_keys["norm_cross"].append(key)
        elif "cross_attention" in key:
            source_group_keys["cross_attention_and_gate"].append(key)
        else:
            source_group_keys["moshi_backbone"].append(key)

        if _is_model_visual_adapter_key(key) and not inherit_visual_adapter:
            continue
        model_state[key] = value

    return image_proj_state, model_state, source_group_keys


def _reset_image_projection_adapter(image_embedder: ImageProjection) -> None:
    """Reset learned image projection/norms while preserving the frozen encoder."""

    for module_name in ("proj_extra", "proj_xa"):
        module = getattr(image_embedder, module_name, None)
        if module is not None and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for module_name in ("norm_extra", "norm_xa"):
        module = getattr(image_embedder, module_name, None)
        if module is None:
            continue
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        else:
            # The local RMSNorm has no reset_parameters method and is initialized
            # with an all-ones scale.
            for parameter in module.parameters():
                torch.nn.init.ones_(parameter)


def _sampled_parameter_fingerprint(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    """Cheap, deterministic signature for startup audits (not a full weight hash)."""

    digest = hashlib.sha256()
    seen: set[int] = set()
    for name, parameter in sorted(named_parameters):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        tensor = parameter.detach().reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        if tensor.numel():
            indices = sorted({0, tensor.numel() // 2, tensor.numel() - 1})
            sample = tensor[indices].float().cpu().numpy()
            digest.update(sample.tobytes())
    return digest.hexdigest()


def _unique_parameter_summary(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> Dict[str, Any]:
    seen: set[int] = set()
    tensors = trainable_tensors = numel = trainable_numel = 0
    for _, parameter in named_parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        tensors += 1
        numel += parameter.numel()
        if parameter.requires_grad:
            trainable_tensors += 1
            trainable_numel += parameter.numel()
    return {
        "parameter_tensors": tensors,
        "parameters": numel,
        "trainable_parameter_tensors": trainable_tensors,
        "trainable_parameters": trainable_numel,
    }


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

        # Reuse the CPU checkpoint already read above. from_config copies it
        # into the destination modules; loading a second full copy onto CUDA
        # can exhaust shared GPUs before the image encoder is constructed.
        for key, v in loaded_weights.items():
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

    # Cross-attention MHA is one shared module referenced from every layer.
    # safetensors stores the shared tensor once (layer 0), while state_dict()
    # enumerates all alias paths.  Do not misreport those aliases as missing.
    missing_important = [
        k for k in important_params
        if k not in loaded_moshi_keys
        and not (
            "cross_attention.mha" in k
            and ".layers.0.cross_attention.mha" not in k
        )
    ]

    if missing_important:
        print(f"❌ [CRITICAL] Cross-Attention/Gate weights MISSING! ({len(missing_important)} params)")
        print(f"   Example missing: {missing_important[:3]}")
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
            - visual_adapter_init: str
              ``inherit_visual_adapter`` = projection / norm_cross /
              cross-attention / gateをMoshiVisから継承
              ``random_visual_adapter`` = 上記全体をランダム初期化
              （どちらも日本語Moshi backboneと画像encoderは継承）
            - inherit_from_pretrained: bool (deprecated compatibility alias)

    Returns:
        Tuple[MoshiVis, ImageProjection]: モデル本体と画像埋め込みモジュール。
    """

    # --- ステート分離用ディクショナリ ---
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}
    init_mode = _resolve_visual_adapter_init_mode(cross_attention_config)
    inherit_visual_adapter = init_mode == "inherit_visual_adapter"
    model_load_report: Dict[str, Any] = {
        "missing_visual_adapter_keys": 0,
        "missing_non_visual_keys": 0,
        "unexpected_checkpoint_keys": 0,
    }
    source_group_keys: Dict[str, list[str]] = {}

    if moshivis_weight is not None:
        print(f"🔹 Loading pretrained weights from {moshivis_weight}")
        weights = load_file(moshivis_weight, device="cpu")

        image_proj_state, model_state, source_group_keys = (
            _partition_training_checkpoint(
                weights,
                inherit_visual_adapter=inherit_visual_adapter,
            )
        )

        if inherit_visual_adapter:
            missing_source_groups = [
                group
                for group in (
                    "cross_attention_and_gate",
                    "norm_cross",
                    "image_projection_and_norm",
                )
                if not source_group_keys[group]
            ]
            if missing_source_groups:
                raise RuntimeError(
                    "inherit_visual_adapter requires all adapter groups in the "
                    "source checkpoint; missing: " + ", ".join(missing_source_groups)
                )
    elif inherit_visual_adapter:
        raise ValueError("inherit_visual_adapter requires moshivis_weight")
    else:
        source_group_keys = {
            "moshi_backbone": [],
            "cross_attention_and_gate": [],
            "norm_cross": [],
            "image_encoder": [],
            "image_projection_and_norm": [],
        }

    print(f"🧪 Visual adapter initialization: {init_mode}")
    print("🔍 Num image_prefix params:", len(image_proj_state))
    print(
        f"🔍 Model params to load: {len(model_state)} "
        f"(visual adapter {'included' if inherit_visual_adapter else 'excluded'})"
    )

    # --- モデル構築 ---
    # ここで __init__ が走り、Cross-AttentionやGateはランダム(または0)で初期化される
    moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

    # --- 重みロード ---
    if model_state:
        missing, unexpected = moshi_vis.load_state_dict(model_state, strict=strict)

        adapter_missing = [k for k in missing if _is_model_visual_adapter_key(k)]
        non_adapter_missing = [k for k in missing if not _is_model_visual_adapter_key(k)]
        print("✅ MoshiVis loaded.")
        print(f"   - Total Missing: {len(missing)}")
        print(
            f"   - Visual-adapter Missing: {len(adapter_missing)} "
            f"({'inherited/shared aliases' if inherit_visual_adapter else 'intentionally random'})"
        )
        print(f"   - Unexpected non-adapter Missing: {len(non_adapter_missing)}")
        print(f"   - Unexpected: {len(unexpected)}")
        model_load_report = {
            "missing_visual_adapter_keys": len(adapter_missing),
            "missing_non_visual_keys": len(non_adapter_missing),
            "unexpected_checkpoint_keys": len(unexpected),
        }
        if non_adapter_missing:
            raise RuntimeError(
                "Checkpoint/model mismatch outside the visual adapter: "
                f"missing={non_adapter_missing[:5]}"
            )
        if inherit_visual_adapter:
            incomplete_adapter = [
                key
                for key in adapter_missing
                if not _is_allowed_shared_mha_alias_missing(key)
            ]
            if incomplete_adapter:
                raise RuntimeError(
                    "Inherited visual adapter is incomplete; only non-layer-0 "
                    "aliases of the shared cross-attention MHA may be missing: "
                    f"missing={incomplete_adapter[:5]}"
                )

    if image_proj_state:
        image_embedder = ImageProjection.from_config(
            kyuteye_config, moshi_vis.llm.dim, image_proj_state, device
        )
    else:
        image_embedder = ImageProjection.from_config(
            kyuteye_config, moshi_vis.llm.dim, None, device
        )

    if not inherit_visual_adapter:
        _reset_image_projection_adapter(image_embedder)

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

        # Model-side visual adapter initialization. norm_cross stays at the
        # constructor default in random mode because it was excluded from load.
        if inherit_visual_adapter:
            print("🧬 Visual adapter inherited from MoshiVis (skip random init).")
        else:
            print("🎲 Initializing cross-attention and gates from scratch...")
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
        gate_shift = float(kyuteye_config.xa_sigmoid_shift)
        
        if gate_freeze:
            if inherit_visual_adapter:
                print("🔒 Gate: inherited weights frozen")
            else:
                initial_gate = torch.sigmoid(
                    torch.tensor(gate_init_alpha - gate_shift)
                ).item()
                print(
                    f"🔒 Gate: random-init scalar alpha={gate_init_alpha} "
                    f"(sigmoid gate ≈ {initial_gate:.1%}); conditional MLP uses "
                    "small random weights"
                )
        else:
            source = "inherited" if inherit_visual_adapter else "random-init"
            print(f"🔧 Gate: trainable ({source}, sigmoid_shift={gate_shift})")
        
        for name, p in moshi_vis.named_parameters():
            if "gate" in name and "alpha" in name:
                # inherit_visual_adapter のときは継承した Gate 重みを保持し、
                # 再初期化しない (requires_grad / freeze 設定のみ適用)
                if not inherit_visual_adapter:
                    if name.endswith("gate.alpha"):
                        # Scalar gating parameter has shape [1, 1, 1].
                        torch.nn.init.constant_(p, gate_init_alpha)
                    else:
                        # For Conditional Gating (MLP), we must NOT set weights to 0.0 (kills gradient).
                        # Initialize with small noise to keep output small (~0) but allow gradient flow.
                        torch.nn.init.normal_(p, mean=0.0, std=0.01)
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

    model_adapter_parameters = [
        (name, parameter)
        for name, parameter in moshi_vis.named_parameters()
        if _is_model_visual_adapter_key(name)
    ]
    image_adapter_parameters = [
        (f"image_prefix.{name}", parameter)
        for name, parameter in image_embedder.named_parameters()
        if not name.startswith("enc.")
    ]
    visual_adapter_parameters = model_adapter_parameters + image_adapter_parameters
    audit: Dict[str, Any] = {
        "schema_version": "jmoshivis.visual-adapter-init.v1",
        "mode": init_mode,
        "source_checkpoint": str(moshivis_weight) if moshivis_weight else None,
        "xa_sigmoid_shift": float(kyuteye_config.xa_sigmoid_shift),
        "image_preprocessing": {
            "size": int(kyuteye_config.image_size),
            "interpolation": kyuteye_config.interpolation,
        },
        "source_checkpoint_key_counts": {
            group: len(keys) for group, keys in source_group_keys.items()
        },
        "model_load_report": model_load_report,
        "effective_group_source": {
            "moshi_backbone": "checkpoint" if moshivis_weight else "constructor",
            "image_encoder": "checkpoint" if image_proj_state else "pretrained_encoder",
            "image_projection_and_norm": (
                "checkpoint" if inherit_visual_adapter else "random"
            ),
            "cross_attention_and_gate": (
                "checkpoint" if inherit_visual_adapter else "random"
            ),
            "norm_cross": "checkpoint" if inherit_visual_adapter else "random",
        },
        "model_visual_adapter": _unique_parameter_summary(model_adapter_parameters),
        "image_visual_adapter": _unique_parameter_summary(image_adapter_parameters),
        "visual_adapter_sampled_sha256": _sampled_parameter_fingerprint(
            visual_adapter_parameters
        ),
    }
    # Plain dict attributes are intentionally not part of state_dict, but allow
    # the trainer/checkpoint layer to persist the exact startup audit as metadata.
    moshi_vis.visual_adapter_init_audit = audit
    image_embedder.visual_adapter_init_audit = audit
    print("\n=== Visual Adapter Startup Audit ===")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2))
    print("====================================\n")

    return moshi_vis.to(dtype=dtype), image_embedder.to(dtype=dtype)
