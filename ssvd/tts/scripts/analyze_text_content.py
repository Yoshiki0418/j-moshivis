import json
import argparse
import os
from collections import Counter, defaultdict
from tqdm import tqdm

def classify_utterance(text):
    """テキストの内容を簡易的に分類するルールベース関数"""
    categories = []
    
    # 1. 質問 (Question)
    if any(k in text for k in ["ですか", "ますか", "？", "?", "教えて", "何", "どこ", "だれ"]):
        categories.append("Question")
        
    # 2. 推測 (Guessing)
    if any(k in text for k in ["たぶん", "おそらく", "かも", "思う", "見えます", "ようです", "可能性"]):
        categories.append("Guessing")
        
    # 3. 指示 (Instruction)
    if any(k in text for k in ["見て", "説明して", "要約して", "教えてください", "挙げてください"]):
        categories.append("Instruction")
        
    # 4. 雑談・リアクション (Chit-chat)
    if any(k in text for k in ["へえ", "なるほど", "すごい", "きれい", "素敵", "ありがとう", "こんにちは"]):
        categories.append("Chit-chat")
        
    # 5. 説明・描写 (Description) - 上記以外で長めのもの
    if not categories and len(text) > 20:
        categories.append("Description")
        
    return categories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl", help="Path to JSONL")
    parser.add_argument("--src-prefix", default="/gpu-server/user/yoshiki/j-moshivis", help="Replace src path")
    parser.add_argument("--dst-prefix", default="/workspace", help="Replace dst path")
    args = parser.parse_args()

    # 集計用
    topic_counter = Counter()
    speaker_act_counter = defaultdict(Counter) # speakerごとの行為分布
    total_files = 0

    print(f"📂 Analyzing text content from: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        try:
            data = json.loads(line)
            if "path" not in data: continue
            
            # パス解決
            json_path = data["path"].replace("stereo_dialogue.wav", "dialogue.json")
            if not os.path.exists(json_path):
                if args.src_prefix in json_path:
                    json_path = json_path.replace(args.src_prefix, args.dst_prefix, 1)
            
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as df:
                    diag_data = json.load(df)
                    dialogue = diag_data.get("dialogue", [])
                    
                    if not dialogue: continue
                    total_files += 1

                    for turn in dialogue:
                        speaker = turn.get("speaker", "Unknown")
                        text = turn.get("text", "")
                        
                        # カテゴリ判定
                        cats = classify_utterance(text)
                        
                        for cat in cats:
                            topic_counter[cat] += 1
                            speaker_act_counter[speaker][cat] += 1

        except json.JSONDecodeError:
            continue

    print("\n" + "="*50)
    print("📊 対話タイプ分布 (Dialogue Types)")
    print("="*50)
    total_acts = sum(topic_counter.values())
    for cat, count in topic_counter.most_common():
        print(f"  - {cat:<12}: {count:,} ({count/total_acts*100:.1f}%)")

    print("\n" + "="*50)
    print("🗣️ 話者ごとの行為比率 (Speech Acts by Speaker)")
    print("="*50)
    for speaker, counts in speaker_act_counter.items():
        s_total = sum(counts.values())
        if s_total == 0: continue
        print(f"👤 {speaker}:")
        for cat, count in counts.most_common():
            print(f"    - {cat:<12}: {count/s_total*100:.1f}%")

if __name__ == "__main__":
    main()