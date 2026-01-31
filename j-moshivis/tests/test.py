import os
import glob
import random
import torch
import torchaudio
import numpy as np
import sentencepiece as spm
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy
from huggingface_hub import hf_hub_download
import traceback
from PIL import Image

# Transformers
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

# J-MoshiVis 関連インポート
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis

# ===== 設定 =====
NUM_SAMPLES = 20         # 解析サンプル数
MAX_DURATION_SEC = 100   # 推論時間
TARGET_SR = 24000

# デバイス設定
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DTYPE = torch.bfloat16
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print("⚠️ No GPU detected. Running on CPU.")

# パス設定
DATA_ROOT = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo"
CHECKPOINT_PATH = "/workspace/j-moshivis/xa_layers_26_27/step_1000.safetensors"
TOKENIZER_PATH = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
CONFIG_PATH = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"

# 一般的応答・質問返しのパターンリスト
GENERIC_PHRASES = [
    "何ですか", "教えてください", "わかりません", "知りたいです", 
    "この画像", "写真には", "写っています", "見えます",
    "綺麗な", "美しい", "素敵な", "素晴らしい",
    "？", "?", "。"
]

def load_models():
    print(f"Loading Moshi models on {DEVICE}...")
    kyuteye_config = KyuteyeConfig.from_yml(CONFIG_PATH)
    
    mimi_weight = hf_hub_download(repo_id="kyutai/moshika-vis-pytorch-bf16", filename="tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = get_mimi(mimi_weight, DEVICE)
    mimi.eval()

    moshi_vis, image_embedder = get_moshi_vis(
        kyuteye_config,
        moshi_weight=Path(CHECKPOINT_PATH),
        device=DEVICE,
        dtype=DTYPE,
    )
    moshi_vis.eval()
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    
    return mimi, moshi_vis, image_embedder, tokenizer

def load_clip_model():
    # Stability AI の正しいモデルID (vit-l-16)
    target_model_name = "stabilityai/japanese-stable-clip-vit-l-16"
    print(f"Loading CLIP model: {target_model_name} on {DEVICE}...")
    
    try:
        # trust_remote_code=True は必須
        model = AutoModel.from_pretrained(target_model_name, trust_remote_code=True).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(target_model_name, trust_remote_code=True)

        print(f"✅ Successfully loaded {target_model_name}")
        return model, tokenizer, image_processor
        
    except Exception as e:
        print(f"\n❌ Error loading {target_model_name}: {e}")
        print("Fallback to OpenAI CLIP (English) just to prevent crash...")
        
        fallback_name = "openai/clip-vit-base-patch32"
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(fallback_name).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(fallback_name)
        return model, processor.tokenizer, processor.image_processor

def get_random_samples(root_dir, n=10):
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    valid_dirs = [d for d in all_dirs if os.path.exists(os.path.join(d, "image.jpg")) and os.path.exists(os.path.join(d, "stereo_dialogue.wav"))]
    if len(valid_dirs) < n: return valid_dirs
    return random.sample(valid_dirs, n)

def run_inference_and_decode(sample_dir, mimi, moshi_vis, image_embedder, tokenizer, max_frames):
    img_path = os.path.join(sample_dir, "image.jpg")
    wav_path = os.path.join(sample_dir, "stereo_dialogue.wav")

    image_processor = ImageProcessor()
    try:
        image_tensor = image_processor(img_path).unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            img_out = image_embedder(image_tensor)
            ca_src = img_out["cross_attention_src"]
    except Exception:
        return [], ""

    try:
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR: waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        target_len = int(MAX_DURATION_SEC * TARGET_SR)
        if waveform.shape[1] > target_len: waveform = waveform[:, :target_len]
        waveform = waveform.unsqueeze(0).to(DEVICE)
        with torch.no_grad(): codes = mimi.encode(waveform)
    except Exception:
        return [], ""

    generated_token_ids = []
    steps = min(codes.shape[-1], max_frames)
    
    with torch.no_grad(), moshi_vis.streaming():
        for t in range(steps):
            chunk = codes[:, :, t:t+1]
            try:
                tokens, gate = moshi_vis.step(chunk, ca_src=ca_src)
                if tokens is not None:
                    text_token = tokens[0, 0, 0].item()
                    generated_token_ids.append(text_token)
            except Exception:
                break
    
    decoded_text = tokenizer.decode(generated_token_ids)
    return generated_token_ids, decoded_text

def calculate_clip_score(model, tokenizer, image_processor, img_path, text):
    if not text or len(text) < 2: return 0.0
    
    try:
        image = Image.open(img_path).convert("RGB")
        
        # 1. 画像の前処理 (辞書型で返ってくる)
        inputs = image_processor(images=image, return_tensors="pt")
        # デバイス転送 (辞書の中身を一つずつ送る)
        image_inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 2. テキストの前処理
        # Stability AIモデルは max_length=77 が標準
        inputs_text = tokenizer(text=[text], padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {k: v.to(DEVICE) for k, v in inputs_text.items()}
        
        with torch.no_grad():
            # Stability AIのモデルは get_image_features / get_text_features を持つ
            if hasattr(model, "get_image_features"):
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)
            else:
                # フォールバック (OpenAI CLIP等)
                outputs = model(pixel_values=image_inputs['pixel_values'], input_ids=text_inputs['input_ids'])
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

            # 正規化
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # コサイン類似度
            score = (image_features @ text_features.T).item()
            
        return score

    except Exception as e:
        print(f"\n🚨 CLIP CALC ERROR: {e}")
        # traceback.print_exc() 
        return 0.0

def clean_text_for_clip(text):
    if not text:
        return ""
    text = text.replace("⁇", "")
    text = text.strip()
    return text

def analyze_results(all_tokens, all_texts, all_clip_scores):
    total_tokens = len(all_tokens)
    if total_tokens == 0: return

    counts = Counter(all_tokens)
    probs = np.array(list(counts.values())) / total_tokens
    ent = entropy(probs)
    unique_ratio = len(counts) / total_tokens

    total_samples = len(all_texts)
    generic_count = 0
    
    print("\n--- Generated Texts Inspection & CLIP Scores ---")
    for i, text in enumerate(all_texts):
        is_generic = False
        for phrase in GENERIC_PHRASES:
            if phrase in text:
                is_generic = True
                break
        if len(text) < 10: is_generic = True
        
        if is_generic:
            generic_count += 1
            type_label = "[Generic]"
        else:
            type_label = "[Specific?]"
            
        score = all_clip_scores[i]
        display_text = text[:50].replace('\n', ' ')
        print(f" {type_label} (CLIP: {score:.3f}) {display_text}...")

    generic_ratio = generic_count / total_samples
    
    avg_clip = np.mean(all_clip_scores) if all_clip_scores else 0.0
    max_clip = np.max(all_clip_scores) if all_clip_scores else 0.0

    print("\n" + "="*60)
    print("📊 Comprehensive Analysis: Fluency, Grounding, and Relevance")
    print("="*60)
    print(f"1. Fluency (Language Quality)")
    print(f"   - Entropy            : {ent:.4f} (Normal: 4.0 - 6.0)")
    print(f"   - Unique Token Ratio : {unique_ratio:.2%} (Normal: ~10-20%)")
    
    print(f"\n2. Grounding (Generic Responses)")
    print(f"   - Generic Ratio      : {generic_ratio:.2%}")

    print(f"\n3. Relevance (CLIP Score)")
    print(f"   - Average Score      : {avg_clip:.4f} (Target: > 0.25)")
    print(f"   - Max Score          : {max_clip:.4f}")
    
    if ent < 2.0:
        conclusion = "Model Collapse (Repetitive Loops or Silence)"
    elif generic_ratio > 0.6:
        conclusion = "Shortcut Learning (Ignoring Image, Safe Answers)"
    elif avg_clip < 0.2:
        conclusion = "Hallucination or Weak Grounding"
    else:
        conclusion = "Successful Generation"
        
    print(f"\n📢 Conclusion: {conclusion}")
    print("-" * 60)

def main():
    mimi, moshi_vis, image_embedder, tokenizer = load_models()
    clip_model, clip_tokenizer, clip_image_processor = load_clip_model()
    
    samples = get_random_samples(DATA_ROOT, NUM_SAMPLES)
    
    all_tokens_flat = []
    all_texts = []
    all_clip_scores = []
    
    frame_rate = mimi.frame_rate
    max_frames = int(MAX_DURATION_SEC * frame_rate)

    for i, sample_dir in enumerate(tqdm(samples, desc="Inferencing")):
        tokens, text = run_inference_and_decode(sample_dir, mimi, moshi_vis, image_embedder, tokenizer, max_frames)
        
        if tokens:
            text = clean_text_for_clip(text)
            all_tokens_flat.extend(tokens)
            all_texts.append(text)
            
            img_path = os.path.join(sample_dir, "image.jpg")
            score = calculate_clip_score(clip_model, clip_tokenizer, clip_image_processor, img_path, text)
            all_clip_scores.append(score)

    analyze_results(all_tokens_flat, all_texts, all_clip_scores)

if __name__ == "__main__":
    main()