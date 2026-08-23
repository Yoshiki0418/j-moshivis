"""
FIXED Inference Script for J-MoshiVis

The key fix: Load pretrained backbone FIRST, then overlay finetuned adapters.
"""
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import soundfile as sf
from pathlib import Path
from safetensors.torch import load_file
from PIL import Image

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen
from moshi.models.loaders import get_mimi
import sentencepiece as spm

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

# ===== 1. Load Configs =====
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# ===== 2. Tokenizer =====
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

# ===== 3. Load Mimi =====
print("📂 Loading Mimi...")
mimi_weight = hf_hub_download(
 repo_id="kyutai/moshika-vis-pytorch-bf16",
 filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)
mimi = get_mimi(mimi_weight, device)
mimi.eval()

# ===== 4. FIXED: Load weights with MERGE strategy =====
print("📂 Loading pretrained backbone (model_merged_bf16.safetensors)...")
pretrained_state = load_file("/workspace/j-moshivis/model_merged_bf16.safetensors", device="cpu")

print("📂 Loading finetuned adapters (step_2000.safetensors)...")
finetuned_state = load_file("/workspace/j-moshivis/checkpoints/step_2000.safetensors", device="cpu")

# Merge: start with pretrained, overlay finetuned
merged_state = pretrained_state.copy()
image_proj_state = {}

for key, value in finetuned_state.items():
    if key.startswith("image_prefix."):
        # Image projection goes to separate dict
        clean_key = key[len("image_prefix."):]
        image_proj_state[clean_key] = value
    else:
        # Override pretrained with finetuned
        merged_state[key] = value

print(f"✅ Merged: {len(pretrained_state)} pretrained + {len(finetuned_state)} finetuned keys")

# ===== 修正用コードブロック =====
print("🔄 Expanding shared Cross-Attention weights to all layers...")

# Moshiのレイヤー数（通常は32ですが、configから取得するのが安全です）
num_layers = 32

# 共有されているCross-Attention関連のキーのサフィックス
# エラーログに出ていたキーを元にリストアップ
shared_suffixes = [
    "cross_attention.mha.in_proj_weight",
    "cross_attention.mha.out_proj.weight",
    # もしバイアスも学習/保存しているなら以下も必要（エラーログには無かったので不要かもですが念のため）
    # "cross_attention.mha.in_proj_bias",
    # "cross_attention.mha.out_proj.bias",
    # "cross_attention.norm.weight", # LayerNormも共有なら必要
    # "cross_attention.norm.bias",
]

# Layer 0 (共有元) の重みを、Layer 1 ~ (N-1) にコピー
for i in range(1, num_layers):
    for suffix in shared_suffixes:
        src_key = f"llm.transformer.layers.0.{suffix}"
        tgt_key = f"llm.transformer.layers.{i}..{suffix}" # 注意: 実装によっては .layers.{i} となる
        
        # 実際のキー構造に合わせるための調整
        # merged_stateのキーが "layers.0..." なのか "llm.transformer.layers.0..." なのか確認して調整してください
        # ここでは汎用的に search & copy します
        
        # merged_state内のキーを探す
        valid_src_key = None
        for key in merged_state.keys():
            if key.endswith(f"layers.0.{suffix}"):
                valid_src_key = key
                break
        
        if valid_src_key:
            # 対応するターゲットキーを作成 (例: layers.0... -> layers.1...)
            valid_tgt_key = valid_src_key.replace("layers.0.", f"layers.{i}.")
            
            # コピー (参照渡しではなく複製するために .clone() 推奨ですが、推論なら参照でも可)
            merged_state[valid_tgt_key] = merged_state[valid_src_key]

print(f"   ✅ Expanded shared weights to {num_layers - 1} layers.")
# ==============================


# ===== 5. Build model and load merged state =====
print("📂 Building MoshiVis model...")
moshivis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=torch.bfloat16)
missing, unexpected = moshivis.load_state_dict(merged_state, strict=False)

# Filter expected missing (shared cross-attention for layers > 0)
missing_filtered = [k for k in missing if "cross_attention.mha" not in k or "layers.0" in k]
print(f"Missing (filtered): {len(missing_filtered)}")
if missing_filtered:
    print(f"Samples: {missing_filtered[:5]}")
print(f"Unexpected: {len(unexpected)}")

moshi_vis = MoshiVisGen(moshi_vis=moshivis.eval().to(device).to(torch.bfloat16))

# ===== 6. Load Image Embedder =====
print("📂 Loading Image Embedder...")
image_embedder = ImageProjection.from_config(
 kyuteye_config, moshi_vis.model_dim, image_proj_state, device
)
image_embedder.eval()
print("✅ Models loaded and set to EVAL mode.")

# ===== 7. Load and encode image =====
from jmoshivis.models.image_projection import ImageProcessor
image_processor = ImageProcessor(size=(448, 448), device=device)

image_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"
print(f"🖼️Loading image: {image_path}")
image_tensor = image_processor(image_path).unsqueeze(0).to(torch.bfloat16)

with torch.no_grad():
    embedder_output = image_embedder(image_tensor)
    ca_src = embedder_output["cross_attention_src"].to(torch.bfloat16)
print(f"Image features: {ca_src.shape}")
print(f"Stats: mean={ca_src.mean().item():.4f}, std={ca_src.std().item():.4f}")

# ===== 8. Load and encode audio =====
wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/user_dialogue.wav"
print(f"🎤 Loading audio: {wav_path}")

waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)
if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

silence_e = torch.zeros(1, int(15.0 * TARGET_SR))
waveform = torch.cat([waveform, silence_e], dim=1).unsqueeze(0).to(device)
codes = mimi.encode(waveform)
print(f"Audio tokens: {codes.shape}")

# ===== 9. Streaming Generation =====
print("🌊 Starting streaming generation (with image)...")

all_audio_tokens = []
all_text_tokens = []
gate_values = []

with moshi_vis.streaming():
    for t in range(codes.shape[-1]):
        chunk = codes[:, :, t:t+1]
        tokens, gate = moshi_vis.step(chunk, ca_src=ca_src)

        if t % 100 == 0:
            print(f"Step {t}/{codes.shape[-1]}: Gate={float(gate):.5f}")
        gate_values.append(float(gate))

        if tokens is not None:
            audio_tokens = tokens[:, 1:, :]
            all_audio_tokens.append(audio_tokens.cpu())
            text_tokens = tokens[:, :1, :]
            all_text_tokens.append(text_tokens.cpu())

# ===== 10. Decode and save =====
if len(all_audio_tokens) > 0:
    all_audio_tokens = torch.cat(all_audio_tokens, dim=-1)
    generated_audio = mimi.decode(all_audio_tokens.to(device))

    output_file = "generated_audio_FIXED.wav"
    sf.write(output_file, generated_audio.squeeze().detach().cpu().numpy(), TARGET_SR)
    print(f"💾 Saved: {output_file}")

# Text decoding
final_token_ids = [int(t.squeeze().item()) for t in all_text_tokens]
final_text = tokenizer.decode(final_token_ids)
print("\n📝 Generated Text:")
print(final_text[:500])

# Gate stats
print(f"\n📊 Gate Statistics:")
print(f"Mean: {sum(gate_values)/len(gate_values):.5f}")
print(f"Max: {max(gate_values):.5f}")
print(f"Min: {min(gate_values):.5f}")

print("\n🎉 Inference complete!")