import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from verify_moshivis_weights import verify_moshivis_weights

from safetensors.torch import save_file


# MoshiVis + J-Moshi の統合後モデルを保存
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


def load_jmoshi_weights_into_moshivis(device="cuda" if torch.cuda.is_available() else "cpu"):
    # =====================================
    # 1️⃣ MoshiVis (Kyutai版) のロード
    # =====================================
    model, image_proj = verify_moshivis_weights(device=device)
    print("\n✅ MoshiVis base model loaded")

    # =====================================
    # 2️⃣ J-Moshi checkpoint のダウンロード
    # =====================================
    repo_id = "nu-dialogue/j-moshi"
    filename = "model.safetensors"
    print(f"🔽 Downloading J-Moshi weights from Hugging Face: {repo_id}")
    jm_ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True)

    jmoshi_state = load_file(jm_ckpt_path)
    print(f"📦 Loaded J-Moshi checkpoint with {len(jmoshi_state)} tensors")

    # =====================================
    # 3️⃣ キーマッピング規則
    # =====================================
    mapping_rules = [
        ("transformer.layers", "llm.transformer.layers"),  # main LLM
        ("text_emb.weight", "llm.text_emb.weight"),
        ("text_linear.weight", "llm.text_linear.weight"),
        ("depformer.layers", "depformer.layers"),
        ("depformer_emb", "depformer_emb"),
        ("depformer_in", "depformer_in"),
        ("depformer_text_emb.weight", "depformer_text_emb.weight"),
        ("out_norm.alpha", "llm.out_norm.alpha"),
        ("emb", "audio_emb"),
        ("linears", "audio_linears"),
    ]

    # =====================================
    # 4️⃣ マッピング適用
    # =====================================
    state_to_update = {}
    for k, v in jmoshi_state.items():
        for old_prefix, new_prefix in mapping_rules:
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix)
                state_to_update[new_k] = v
                break  # 最初に一致したルールでマッピング

    print(f"🧩 Mapped {len(state_to_update)} keys from J-Moshi → MoshiVis")

    # =====================================
    # 5️⃣ モデルに上書きロード
    # =====================================
    missing, unexpected = model.load_state_dict(state_to_update, strict=False)
    model = model.to(device=device)

    print(f"\n✅ J-Moshi weights merged successfully.")
    print(f" - Missing keys: {len(missing)}")
    print(f" - Unexpected keys: {len(unexpected)}")

    if missing:
        print("\n🟠 Missing example keys:")
        for k in missing[:200]:
            print("  ", k)

    if unexpected:
        print("\n🔵 Unexpected example keys:")
        for k in unexpected[:20]:
            print("  ", k)

    print(f"📊 Total parameters (MoshiVis): {sum(p.numel() for p in model.parameters()):,}")

    return model, image_proj


if __name__ == "__main__":
    model, image_proj = load_jmoshi_weights_into_moshivis()
    save_merged_model(model, image_proj, "/workspace/j-moshivis/model_merged_bf16.safetensors")
