"""
Script: analyze_alignment_quality_v3.py
Purpose:
    JSONLファイル内のWhisperアライメントと、
    元データ('dialogue.json')の「アシスタント」発話を比較し、
    CERを算出する。（日本語ラベル "アシスタント" 対応版）
"""

import json
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def levenshtein_distance(s1, s2):
    """レーベンシュタイン距離（編集距離）"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """CER (Character Error Rate)"""
    # 空白・全角スペースを除去して正規化
    ref = reference.replace(" ", "").replace("　", "")
    hyp = hypothesis.replace(" ", "").replace("　", "")
    
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    
    dist = levenshtein_distance(ref, hyp)
    return dist / len(ref)

def reconstruct_text_from_alignments(alignments):
    """
    alignmentsリストからテキストを全て結合して復元する
    """
    if not alignments:
        return ""
    
    text_parts = []
    for item in alignments:
        token = item.get("token", "")
        # 特殊トークンの除去
        token = token.replace("▁", "")
        text_parts.append(token)
    
    return "".join(text_parts)

def reconstruct_assistant_text_from_dialogue(dialogue_data):
    """
    dialogue.jsonから 'アシスタント' の発話のみを結合して取得する
    """
    if not dialogue_data or "dialogue" not in dialogue_data:
        return ""
    
    text_parts = []
    for turn in dialogue_data["dialogue"]:
        speaker = turn.get("speaker", "")
        # ★修正: 日本語の「アシスタント」に対応
        if speaker in ["アシスタント", "assistant", "system", "model", "ai", "gpt"]:
            text = turn.get("text", "")
            text_parts.append(text)
    
    return "".join(text_parts)

def main():
    parser = argparse.ArgumentParser(description="Analyze Alignment Quality (CER/WER) - V3")
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl", help="Path to JSONL file.")
    parser.add_argument("--src-prefix", default="/gpu-server/user/yoshiki/j-moshivis", help="Source path prefix to replace.")
    parser.add_argument("--dst-prefix", default="/workspace", help="Destination path prefix.")
    parser.add_argument("--output-img", default="cer_distribution_v3.png", help="Output image filename.")
    
    args = parser.parse_args()

    print(f"📂 ファイル読み込み中: {args.input} ...")
    
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found at {args.input}")
        return

    cer_scores = []
    error_samples = []
    valid_count = 0
    missing_files = 0
    no_assistant_text = 0
    
    HIGH_CER_THRESHOLD = 0.3 

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"🔍 Analyzing {len(lines)} samples (Target: 'アシスタント')...")

    for line in tqdm(lines):
        try:
            data = json.loads(line)
            
            # 1. Alignments (Hypothesis)
            if "alignments" not in data:
                continue
            hyp_text = reconstruct_text_from_alignments(data["alignments"])
            
            # 2. Dialogue.json (Reference)
            if "path" not in data:
                continue
            
            audio_path = data["path"]
            json_path = audio_path.replace("stereo_dialogue.wav", "dialogue.json")
            
            # パス解決
            final_json_path = json_path
            if not os.path.exists(final_json_path):
                if args.src_prefix in json_path:
                    replaced_path = json_path.replace(args.src_prefix, args.dst_prefix, 1)
                    if os.path.exists(replaced_path):
                        final_json_path = replaced_path
            
            if not os.path.exists(final_json_path):
                missing_files += 1
                continue

            with open(final_json_path, "r", encoding="utf-8") as df:
                dialogue_data = json.load(df)
                ref_text = reconstruct_assistant_text_from_dialogue(dialogue_data)

            if not ref_text:
                no_assistant_text += 1
                continue

            # 3. CER計算
            cer = calculate_cer(ref_text, hyp_text)
            cer_scores.append(cer)
            valid_count += 1

            if cer > HIGH_CER_THRESHOLD:
                error_samples.append({
                    "path": final_json_path,
                    "cer": cer,
                    "ref_len": len(ref_text),
                    "hyp_len": len(hyp_text),
                    "ref_sample": ref_text[:50] + "...",
                    "hyp_sample": hyp_text[:50] + "..."
                })

        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    if valid_count == 0:
        print("⚠️ 有効な比較データが見つかりませんでした。")
        print("   (ヒント: dialogue.json の speaker が 'アシスタント' 以外の可能性があります)")
        return

    # --- 統計分析 ---
    avg_cer = np.mean(cer_scores)
    median_cer = np.median(cer_scores)
    max_cer = np.max(cer_scores)
    std_cer = np.std(cer_scores)
    
    perfect = sum(1 for c in cer_scores if c == 0.0)
    excellent = sum(1 for c in cer_scores if 0.0 < c <= 0.05)
    good = sum(1 for c in cer_scores if 0.05 < c <= 0.15)
    bad = sum(1 for c in cer_scores if c > 0.30)

    print("\n" + "="*60)
    print(f"📊 アライメント品質プロファイル V3 (Japanese Fixed)")
    print("="*60)
    print(f"📁 分析対象数       : {valid_count:,} 件")
    if no_assistant_text > 0:
        print(f"ℹ️  Asst発話なし     : {no_assistant_text:,} 件")
    
    print("-" * 60)
    print(f"📉 全体平均 CER     : {avg_cer:.2%} (低いほど良い)")
    print(f"🎯 中央値 CER       : {median_cer:.2%}")
    print(f"σ  標準偏差         : {std_cer:.2f}")
    
    print("-" * 60)
    print("📋 品質分布:")
    print(f"  ✨ 完全一致 (0%)    : {perfect:,} 件 ({perfect/valid_count:.1%})")
    print(f"  🟢 高品質 (0-5%)    : {excellent:,} 件 ({excellent/valid_count:.1%})")
    print(f"  🟡 良   好 (5-15%)  : {good:,} 件 ({good/valid_count:.1%})")
    print(f"  🔴 崩   壊 (>30%)   : {bad:,} 件 ({bad/valid_count:.1%})")

    if error_samples:
        print("-" * 60)
        print("🚨 ワーストケース例 (CER > 30%):")
        sorted_errors = sorted(error_samples, key=lambda x: x['cer'], reverse=True)[:3]
        for i, err in enumerate(sorted_errors):
            print(f"\n  [{i+1}] CER: {err['cer']:.2%} | Path: {os.path.basename(err['path'])}")
            print(f"      Ref: {err['ref_sample']}")
            print(f"      Hyp: {err['hyp_sample']}")

    print("="*60)

    # --- グラフ ---
    print(f"📈 ヒストグラムを作成中: {args.output_img}")
    plt.figure(figsize=(10, 6))
    plt.hist(cer_scores, bins=50, color='lightgreen', edgecolor='black', range=(0, 1.0))
    plt.title('Distribution of Assistant CER (Fixed)')
    plt.xlabel('CER (0.0 = Perfect, 1.0 = Bad)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    stats_text = f"Mean: {avg_cer:.3f}\nMedian: {median_cer:.3f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(args.output_img)
    print("Done.")

if __name__ == "__main__":
    main()