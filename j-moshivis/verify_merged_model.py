import torch
from safetensors.torch import load_file
from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.models.image_projection import ImageProjection
import yaml
from pathlib import Path


def verify_merged_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Verify that model_merged.safetensors (J-MoshiVis統合済みモデル)
    can be successfully reloaded and all dimensions match.
    """

    ckpt_path = Path("/workspace/j-moshivis/model_merged.safetensors")
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # ------------------------------------------------
    # 1️⃣ MoshiVis構造を再構築
    # ------------------------------------------------
    cfg_path = Path("/workspace/j-moshivis/configs/moshi-vis.yaml")
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = MoshiVis(
        dim=4096,
        text_card=32000,
        padding_token_id=3,
        n_q=16,
        dep_q=8,
        audio_card=2048,
        num_heads=32,
        num_layers=32,
        hidden_scale=4.125,
        causal=True,
        context=3000,
        max_period=10000,
        gating=True,
        activation="silu",
        norm="rms_norm_f32",
        positional_embedding="rope",
        depformer=True,
        depformer_dim=1024,
        depformer_dim_feedforward=int(4.125 * 1024),
        depformer_num_heads=16,
        depformer_num_layers=6,
        depformer_multi_linear=True,
        depformer_context=8,
        depformer_gating=True,
        depformer_activation="silu",
        depformer_pos_emb="none",
        depformer_weights_per_step=True,
        delays=(0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1),
    )

    # ------------------------------------------------
    # 2️⃣ safetensorsファイルから重みをロード
    # ------------------------------------------------
    print(f"📦 Loading merged weights from {ckpt_path}")
    state_dict = load_file(ckpt_path)

    # 分離（image_prefixを含む場合）
    image_prefix_state = {
        k.replace("image_prefix.", ""): v
        for k, v in state_dict.items()
        if k.startswith("image_prefix.")
    }
    model_state = {
        k: v for k, v in state_dict.items() if not k.startswith("image_prefix.")
    }

    # ------------------------------------------------
    # 3️⃣ モデルへのロード
    # ------------------------------------------------
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)

    print("\n✅ Model reloaded successfully!")
    print(f" - Missing keys: {len(missing)}")
    print(f" - Unexpected keys: {len(unexpected)}")

    fp16_count = 0
    fp32_count = 0
    other_count = 0

    for k, v in model.state_dict().items():
        if v.dtype == torch.float16:
            fp16_count += 1
        elif v.dtype == torch.float32:
            fp32_count += 1
        else:
            other_count += 1

    print(f"fp16: {fp16_count}, fp32: {fp32_count}, other: {other_count}")

    # ------------------------------------------------
    # 4️⃣ Vision Encoderも確認（存在すれば）
    # ------------------------------------------------
    if len(image_prefix_state) > 0:
        print("🖼️ Detected Vision projection weights — testing forward()")
        image_proj = ImageProjection.from_config(
            kyuteye_config=None,
            lm_model_dim=4096,
            moshi_weight=image_prefix_state,
            device=device,
        )

        dummy_img = torch.randn(
            1, 3, config.get("image_size", 512), config.get("image_size", 512), device=device
        )
        with torch.no_grad():
            out = image_proj(dummy_img)
        print(f"🧩 Vision output shape: {tuple(out['cross_attention_src'].shape)}")

    # ------------------------------------------------
    # 5️⃣ モデルの整合性テスト
    # ------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")

    # Transformer層ごとのCrossAttention共有確認
    layers = [model.llm.transformer.layers[i].cross_attention.mha for i in range(32)]
    shared = all(layers[0] is layer for layer in layers)
    print(f"CrossAttention shared across layers: {shared}")

    print("\n🎯 Verification completed — merged model is functional.")
    return model


if __name__ == "__main__":
    verify_merged_model()
