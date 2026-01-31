import os
import glob
import random
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy
from huggingface_hub import hf_hub_download

# J-MoshiVis 関連インポート
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis

# ===== 設定 =====
NUM_SAMPLES = 20        # 解析するランダムサンプルの数
MAX_DURATION_SEC = 10   # 1サンプルあたりの推論秒数（長すぎると時間がかかるため）
TARGET_SR = 24000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# パス設定
DATA_ROOT = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo" # データセットのルート
CHECKPOINT_PATH = "/workspace/j-moshivis/checkpoints/step_4000.safetensors"
TOKENIZER_PATH = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
CONFIG_PATH = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"

def load_models():
    print("Loading models...")
    kyuteye_config = KyuteyeConfig.from_yml(CONFIG_PATH)
    
    # Mimi
    mimi_weight = hf_hub_download(repo_id="kyutai/moshika-vis-pytorch-bf16", filename="tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = get_mimi(mimi_weight, DEVICE)
    mimi.eval()

    # MoshiVis
    moshi_vis, image_embedder = get_moshi_vis(
        kyuteye_config,
        moshi_weight=Path(CHECKPOINT_PATH),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    moshi_vis.eval()
    
    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    
    return mimi, moshi_vis, image_embedder, tokenizer

def get_random_samples(root_dir, n=10):
    # image.jpg と stereo_dialogue.wav が両方あるディレクトリを探す
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    valid_dirs = []
    for d in all_dirs:
        if os.path.exists(os.path.join(d, "image.jpg")) and os.path.exists(os.path.join(d, "stereo_dialogue.wav")):
            valid_dirs.append(d)
    
    if len(valid_dirs) < n:
        print(f"Warning: Requested {n} samples, but only found {len(valid_dirs)} valid samples.")
        return valid_dirs
    
    return random.sample(valid_dirs, n)

def run_inference(sample_dir, mimi, moshi_vis, image_embedder, max_frames):
    img_path = os.path.join(sample_dir, "image.jpg")
    wav_path = os.path.join(sample_dir, "stereo_dialogue.wav")

    # 1. Image Encoding
    image_processor = ImageProcessor()
    try:
        image_tensor = image_processor(img_path).unsqueeze(0)
        with torch.no_grad():
            img_out = image_embedder(image_tensor.to(DEVICE))
            ca_src = img_out["cross_attention_src"]
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return []

    # 2. Audio Loading & Encoding
    try:
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True) # Mono化
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        
        # 長さ制限
        target_len = int(MAX_DURATION_SEC * TARGET_SR)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        
        waveform = waveform.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            codes = mimi.encode(waveform)
    except Exception as e:
        print(f"Error processing audio {wav_path}: {e}")
        return []

    # 3. Streaming Inference
    generated_tokens = []
    
    # 実際の音声長に合わせてループ (max_framesでキャップ)
    steps = min(codes.shape[-1], max_frames)
    
    with torch.no_grad(), moshi_vis.streaming():
        for t in range(steps):
            chunk = codes[:, :, t:t+1]
            tokens, gate = moshi_vis.step(chunk, ca_src=ca_src)
            
            if tokens is not None:
                # tokens shape: [B, 1 + 8, 1] -> Text is index 0
                text_token = tokens[0, 0, 0].item()
                generated_tokens.append(text_token)
                
    return generated_tokens

def analyze_and_plot(all_tokens, tokenizer, pad_id, zero_id):
    total_tokens = len(all_tokens)
    if total_tokens == 0:
        print("No tokens generated.")
        return

    counts = Counter(all_tokens)
    
    # --- 1. 定量指標 ---
    # エントロピー計算 (多様性の指標)
    probs = np.array(list(counts.values())) / total_tokens
    ent = entropy(probs)
    
    # モード崩壊の判定基準 (支配的なトークン)
    most_common_token, most_common_count = counts.most_common(1)[0]
    dominance_ratio = most_common_count / total_tokens

    # 特殊トークンの割合
    pad_ratio = counts.get(pad_id, 0) / total_tokens
    zero_ratio = counts.get(zero_id, 0) / total_tokens

    print("\n" + "="*50)
    print("📊 Quantitative Analysis of Mode Collapse")
    print("="*50)
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Unique Tokens Used    : {len(counts)}")
    print(f"Entropy (Diversity)   : {ent:.4f} (Higher is better, < 1.0 indicates collapse)")
    print(f"Dominance Ratio (Top1): {dominance_ratio:.2%} (Lower is better, > 80% indicates collapse)")
    print(f"PAD Token Ratio       : {pad_ratio:.2%}")
    print(f"ZERO Token Ratio      : {zero_ratio:.2%}")
    print("-" * 50)

    # --- 2. Top-20 分布の可視化 ---
    top_k = 20
    top_items = counts.most_common(top_k)
    
    labels = []
    values = []
    
    print(f"\nTop {top_k} Frequent Tokens:")
    for tid, count in top_items:
        # IDをデコードしてラベルにする
        if tid == pad_id:
            label = "[PAD]"
        elif tid == zero_id:
            label = "[ZERO]" # ある場合
        else:
            try:
                label = tokenizer.id_to_piece(tid)
                # 表示できない文字などの処理
                if label.startswith("<") and label.endswith(">"):
                    pass 
                elif not label.isprintable():
                    label = f"ID:{tid}"
            except:
                label = f"ID:{tid}"
        
        ratio = count / total_tokens
        print(f"  {label:<15} (ID: {tid:<5}): {count:>5} ({ratio:.2%})")
        
        labels.append(f"{label}\n({tid})")
        values.append(count)

    # グラフ描画
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.title(f"Top {top_k} Token Distribution (Entropy: {ent:.2f})")
    plt.xlabel("Token (Decoded / ID)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("token_distribution_analysis.png")
    print("\n🎉 Analysis chart saved to 'token_distribution_analysis.png'")

def main():
    mimi, moshi_vis, image_embedder, tokenizer = load_models()
    
    # ID取得
    pad_id = moshi_vis.lm_model.text_padding_token_id
    zero_id = getattr(moshi_vis, 'zero_token_id', -1) # 属性がない場合のフォールバック
    if zero_id == -1 and hasattr(moshi_vis, 'lm_model'):
         zero_id = getattr(moshi_vis.lm_model, 'zero_token_id', -1)

    print(f"PAD ID: {pad_id}, ZERO ID: {zero_id}")
    
    samples = get_random_samples(DATA_ROOT, NUM_SAMPLES)
    print(f"Target Samples: {len(samples)}")

    all_tokens = []
    
    # Mimiのフレームレート取得 (例: 12.5Hz)
    frame_rate = mimi.frame_rate
    max_frames = int(MAX_DURATION_SEC * frame_rate)

    for i, sample_dir in enumerate(tqdm(samples, desc="Inferencing")):
        tokens = run_inference(sample_dir, mimi, moshi_vis, image_embedder, max_frames)
        all_tokens.extend(tokens)

    analyze_and_plot(all_tokens, tokenizer, pad_id, zero_id)

if __name__ == "__main__":
    main()