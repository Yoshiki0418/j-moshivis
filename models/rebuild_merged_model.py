# model_merged_bf16.safetensors の再構築スクリプト（新サーバー用）
#
# 旧サーバーの load_jmoshi.py と同じマージを、モデルをインスタンス化せずに
# state_dict レベルで行う（GPU不要・CPUのみで動作）。
#
#   ベース: kyutai/moshika-vis-pytorch-bf16 (MoshiVis + SigLIP vision)
#   上書き: nu-dialogue/j-moshi (日本語Moshi) を キーマッピングして LLM/Depformer 部に適用
#
# 使い方:
#   .venv-train/bin/python models/rebuild_merged_model.py                # J-Moshi版
#   .venv-train/bin/python models/rebuild_merged_model.py \
#       --japanese_weights models/llm-jp-moshi-v1/model.safetensors \
#       --out models/model_merged_llmjp_bf16.safetensors               # LLM-jp-Moshi版

import argparse

import torch
from safetensors.torch import load_file, save_file

MOSHIVIS_PATH = "/gpu-server/yamamoto/j-moshivis/models/moshika-vis-pytorch-bf16/model.safetensors"
JMOSHI_PATH = "/gpu-server/yamamoto/j-moshivis/models/j-moshi/model.safetensors"
OUT_PATH = "/gpu-server/yamamoto/j-moshivis/models/model_merged_bf16.safetensors"

# load_jmoshi.py と同一のマッピング規則
MAPPING_RULES = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshivis_weights", default=MOSHIVIS_PATH,
                        help="ベースのMoshiVis重み (視覚部分の供給元)")
    parser.add_argument("--japanese_weights", default=JMOSHI_PATH,
                        help="日本語Moshi重み (LLM/Depformer部分の供給元)")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    print("📦 Loading MoshiVis base weights ...")
    base = load_file(args.moshivis_weights)
    print(f"  {len(base)} tensors")

    print(f"📦 Loading Japanese Moshi weights: {args.japanese_weights}")
    jmoshi = load_file(args.japanese_weights)
    print(f"  {len(jmoshi)} tensors")

    # キーマッピング適用
    mapped = {}
    unmapped = []
    for k, v in jmoshi.items():
        for old_prefix, new_prefix in MAPPING_RULES:
            if k.startswith(old_prefix):
                mapped[k.replace(old_prefix, new_prefix, 1)] = v
                break
        else:
            unmapped.append(k)
    print(f"🧩 Mapped {len(mapped)} keys from J-Moshi (unmapped: {len(unmapped)})")
    if unmapped:
        for k in unmapped[:10]:
            print("  [unmapped]", k)

    # ベース側に存在するキーだけ上書き（形状もチェック）
    overwritten, shape_mismatch, not_in_base = 0, [], []
    for k, v in mapped.items():
        if k in base:
            if base[k].shape != v.shape:
                shape_mismatch.append((k, tuple(base[k].shape), tuple(v.shape)))
                continue
            base[k] = v
            overwritten += 1
        else:
            not_in_base.append(k)

    print(f"✅ Overwrote {overwritten}/{len(mapped)} tensors in base model")
    if shape_mismatch:
        print(f"⚠️ Shape mismatch: {len(shape_mismatch)}")
        for k, bs, js in shape_mismatch[:10]:
            print(f"  {k}: base{bs} vs jmoshi{js}")
    if not_in_base:
        print(f"⚠️ Keys not in base: {len(not_in_base)}")
        for k in not_in_base[:10]:
            print("  ", k)

    print("💾 Converting to bf16 and saving ...")
    out = {k: v.to(torch.bfloat16) for k, v in base.items()}
    save_file(out, args.out)
    print(f"✅ Saved {len(out)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
