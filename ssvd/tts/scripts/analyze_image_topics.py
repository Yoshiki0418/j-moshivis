import json
import argparse
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import random

# 分類したいカテゴリ（予稿に合わせて調整してください）
LABELS = [
    "人物のポートレート (A portrait of a person)", 
    "人々のグループ (A group of people)",
    "室内の風景 (Indoor scene)", 
    "屋外の風景 (Outdoor scenery)",
    "食べ物や料理 (Food or dish)", 
    "乗り物 (Vehicle or car)",
    "動物 (Animal)",
    "スクリーンショットや文字 (Screenshot or text)",
    "イラストやアート (Illustration or art)"
]

# 英語ラベルの方がCLIPの精度が良い場合が多いので、英語も併記してモデルに入力します
TEXT_INPUTS = [l.split("(")[-1].replace(")", "") for l in LABELS] 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl")
    parser.add_argument("--sample-size", type=int, default=500000, help="Number of images to sample")
    parser.add_argument("--src-prefix", default="/gpu-server/user/yoshiki/j-moshivis")
    parser.add_argument("--dst-prefix", default="/workspace")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading CLIP model on {device}...")
    
    # 軽量なCLIPモデルを使用 (OpenAI公式など)
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # JSONLから画像パスを収集
    image_paths = []
    print("📂 Collecting image paths...")
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # ランダムサンプリング
    if len(lines) > args.sample_size:
        lines = random.sample(lines, args.sample_size)
        
    for line in lines:
        try:
            data = json.loads(line)
            if "image" in data: # imageキーがある場合
                img_path = data["image"]
            elif "path" in data: # pathから推測する場合
                img_path = data["path"].replace("stereo_dialogue.wav", "image.jpg") # 仮定
            else:
                continue

            # パス置換
            if not os.path.exists(img_path):
                if args.src_prefix in img_path:
                    img_path = img_path.replace(args.src_prefix, args.dst_prefix, 1)
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
        except:
            continue

    print(f"🔍 Analyzing {len(image_paths)} images...")
    
    label_counts = {label: 0 for label in LABELS}
    
    # バッチ処理はせず1枚ずつシンプルに処理（件数が少なければこれで十分）
    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path)
            
            # CLIP推論
            inputs = processor(text=TEXT_INPUTS, images=image, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # probabilities
                
            # 最も確率の高いラベルを取得
            pred_idx = probs.argmax().item()
            predicted_label = LABELS[pred_idx]
            label_counts[predicted_label] += 1
            
        except Exception as e:
            # print(f"Error processing {img_path}: {e}")
            continue

    print("\n" + "="*50)
    print(f"🖼️ 画像トピック分布 (Sample size: {len(image_paths)})")
    print("="*50)
    
    # 多い順にソートして表示
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_counts:
        ratio = (count / len(image_paths)) * 100
        print(f"  - {label.split('(')[0]:<15}: {count:,} ({ratio:.1f}%)")

if __name__ == "__main__":
    main()