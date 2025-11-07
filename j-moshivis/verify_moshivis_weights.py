import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.models.image_projection import ImageProjection
import yaml
from pathlib import Path
from safetensors.torch import save_file


def save_merged_model(
    model,
    image_proj=None,
    save_path="/workspace/j-moshivis/model_merged_fp16.safetensors"
):
    print(f"💾 Saving merged model (fp16) to {save_path} ...")

    # MoshiVis本体
    state_dict = model.state_dict()

    # ImageProjection (Vision encoder 部分) をマージ
    if image_proj is not None:
        image_prefix_state = {
            f"image_prefix.{k}": v.detach().cpu()
            for k, v in image_proj.state_dict().items()
        }
        state_dict.update(image_prefix_state)

    # fp16に変換
    converted = {}
    skipped = []
    for k, v in state_dict.items():
        try:
            converted[k] = v.detach().to(torch.float16).cpu()
        except Exception as e:
            skipped.append((k, str(e)))

    # ✅ fp16に変換済みの辞書を保存
    save_file(converted, save_path)

    print(f"✅ Saved {len(converted)} tensors in fp16 to {save_path}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters (saved model): {total_params:,}")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} tensors:")
        for k, e in skipped[:5]:
            print(f"  {k}: {e}")

# ------------------------------------------------
# ✅ Kyutai構成互換の簡易コンフィグラッパ
# ------------------------------------------------
class DummyKyuteyeConfig:
    def __init__(self, cfg: dict):
        # image系設定
        self.image = type("ImageCfg", (), {})()
        self.image.encoder_name = cfg.get("encoder_name", "siglip_gemma2_448")
        self.image.norm_xa = cfg.get("norm_xa", "rms_norm")
        self.image.norm_extra = cfg.get("norm_extra", None)
        # fuse系設定
        self.fuse = type("FuseCfg", (), {})()
        self.fuse.num_extra_tokens = cfg.get("num_extra_tokens", 0)
        self.fuse.num_crossattended_tokens = cfg.get("num_crossattended_tokens", -1)
        # xa_dimなど
        self.xa_dim = cfg.get("xa_dim", None)


def verify_moshivis_weights(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Verify MoshiVis weights (Depformer + Image encoder ENABLED, Kyutai spec-compatible)"""

    repo_id = "kyutai/moshika-vis-pytorch-bf16"
    filename = "model.safetensors"

    print(f"🔽 Downloading weights from: https://huggingface.co/{repo_id}")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True)

    moshivis_weight = Path("/workspace/j-moshivis/model.safetensors")
    weights = load_file(moshivis_weight, device=str(device))

    cfg_path = Path("/workspace/j-moshivis/configs/moshi-vis.yaml")
    assert cfg_path.exists(), f"Config file not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.cuda.empty_cache()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🧠 Using dtype: {dtype}")

    # ================================================
    # 1️⃣ MoshiVis (LLM + Depformer)
    # ================================================
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

    # print(f"📂 Loading checkpoint from {ckpt_path}")
    # state_dict = load_file(ckpt_path)

    # 分離: MoshiVis本体とImagePrefix部
    image_prefix_state = {
        k.replace("image_prefix.", ""): v
        for k, v in weights.items()
        if k.startswith("image_prefix.")
    }
    model_state = {
        k: v
        for k, v in weights.items()
        if not k.startswith("image_prefix.")
    }

    # ================================================
    # 2️⃣ ImageProjection (Vision encoder)
    # ================================================
    has_img_prefix = len(image_prefix_state) > 0
    if has_img_prefix:
        print("🖼️ Detected Vision Encoder weights (SigLIP).")

        kyuteye_cfg = DummyKyuteyeConfig(config)
        image_proj = ImageProjection.from_config(
            kyuteye_config=kyuteye_cfg,
            lm_model_dim=4096,
            moshi_weight=image_prefix_state,
            device=device,
        )
    else:
        print("⚠️ No Vision Encoder weights found (image_prefix missing).")
        image_proj = None

    # ================================================
    # 3️⃣ 各モジュールへロード
    # ================================================
    missing_model, unexpected_model = model.load_state_dict(model_state, strict=False)
    model = model.to(device=device, dtype=dtype)


    print("\n✅ Weight loading completed (Depformer + Vision ENABLED).")
    print(f" - Missing keys (LLM): {len(missing_model)}")
    print(f" - Unexpected keys (LLM): {len(unexpected_model)}")

    if missing_model:
        print("\n🟠 Missing keys (first 10):")
        for k in missing_model[:100]:
            print("  ", k)

    if unexpected_model:
        print("\n🔵 Unexpected keys (first 10):")
        for k in unexpected_model[:100]:
            print("  ", k)

    # ================================================
    # 4️⃣ Vision Encoder 確認
    # ================================================
    if image_proj:
        print("\n🔍 Testing Vision Encoder forward()...")
        dummy_img = torch.randn(1, 3, config.get("image_size", 512), config.get("image_size", 512), device=device)
        with torch.no_grad():
            out = image_proj(dummy_img)
        print(f"🧩 cross_attention_src: {tuple(out['cross_attention_src'].shape)}")

    print("\n🎯 Verification complete (Depformer + Vision active).")
    print(f"📦 Device: {next(model.parameters()).device}")
    print(f"📊 Total parameters (MoshiVis): {sum(p.numel() for p in model.parameters()):,}")

    layers = [model.llm.transformer.layers[i].cross_attention.mha for i in range(32)]
    shared = all(layers[0] is layer for layer in layers)
    print(f"CrossAttention shared across all layers: {shared}")

    return model, image_proj


if __name__ == "__main__":
    model, image_proj = verify_moshivis_weights()
    save_merged_model(model, image_proj, "/workspace/j-moshivis/moshivis_bf16.safetensors")
