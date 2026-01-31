"""
Script: analyze_dialogue_json_turns_with_plot.py
Purpose:
    JSONLファイル内の 'path' (音声パス) をもとに、
    同一ディレクトリにある 'dialogue.json' を読み込み、
    正確な対話ターン数を分析し、その分布をグラフ化する。

Usage:
    python3 analyze_dialogue_json_turns_with_plot.py --input /workspace/data/speech/train_data_refined_a.jsonl
"""

import json
import argparse
import statistics
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Analyze turn counts from dialogue.json files.")
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl", help="Path to JSONL file.")
    
    # パス置換用オプション (環境によるパスの差異を吸収するため)
    parser.add_argument("--src-prefix", default="/gpu-server/user/yoshiki/j-moshivis", help="Source path prefix in JSONL to replace.")
    parser.add_argument("--dst-prefix", default="/workspace", help="Destination path prefix to replace with.")
    parser.add_argument("--output-img", default="turn_distribution.png", help="Output filename for the distribution plot.")
    
    args = parser.parse_args()

    jsonl_path = args.input
    turn_counts = []
    missing_files = 0
    valid_files = 0

    print(f"📂 JSONLファイル読み込み中: {jsonl_path} ...")
    
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: JSONL file not found at {jsonl_path}")
        return

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    
                    if "path" not in data:
                        continue

                    # 1. 音声パスを取得
                    audio_path = data["path"]
                    
                    # 2. パスを dialogue.json に変換
                    target_json_path = audio_path.replace("stereo_dialogue.wav", "dialogue.json")

                    # 3. パスのプレフィックス置換（環境適応）
                    final_path = target_json_path
                    if not os.path.exists(final_path):
                        if args.src_prefix in target_json_path:
                            replaced_path = target_json_path.replace(args.src_prefix, args.dst_prefix, 1)
                            if os.path.exists(replaced_path):
                                final_path = replaced_path
                    
                    # 4. dialogue.json を読み込んでターン数をカウント
                    if os.path.exists(final_path):
                        try:
                            with open(final_path, "r", encoding="utf-8") as df:
                                dialogue_data = json.load(df)
                                
                                if "dialogue" in dialogue_data and isinstance(dialogue_data["dialogue"], list):
                                    turns = len(dialogue_data["dialogue"])
                                    turn_counts.append(turns)
                                    valid_files += 1
                                else:
                                    pass
                        except json.JSONDecodeError:
                            print(f"⚠️ JSON Decode Error at: {final_path}")
                    else:
                        missing_files += 1
                        if missing_files <= 3:
                            print(f"⚠️ File not found (Sample): {final_path}")
                            if missing_files == 3:
                                print(f"   (以降のファイルが見つからないエラーは省略します...)")

                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

    # --- 集計と出力 ---
    total_processed = valid_files + missing_files
    
    print("\n" + "="*50)
    print(f"📊 データセット分析結果 (dialogue.jsonベース)")
    print("="*50)
    print(f"📁 処理対象ファイル数 : {total_processed:,} 件")
    print(f"✅ 有効データ数       : {valid_files:,} 件")
    if missing_files > 0:
        print(f"❌ 見つからないファイル : {missing_files:,} 件")
        print(f"   (パス置換設定: '{args.src_prefix}' -> '{args.dst_prefix}')")

    print("-" * 50)
    print("💬 【ターン数 (Turns)】")
    
    if turn_counts:
        avg_turns = sum(turn_counts) / len(turn_counts)
        max_turns = max(turn_counts)
        min_turns = min(turn_counts)
        median_turns = statistics.median(turn_counts)
        
        even_turns = sum(1 for t in turn_counts if t % 2 == 0)
        odd_turns = sum(1 for t in turn_counts if t % 2 != 0)
        
        print(f"   平均ターン数     : {avg_turns:.2f} 回")
        print(f"   中央値           : {median_turns} 回")
        print(f"   最大 / 最小      : {max_turns} / {min_turns} 回")
        print(f"   🟢 偶数ターン     : {even_turns:,} 件 ({(even_turns/len(turn_counts))*100:.1f}%)")
        print(f"   🔴 奇数ターン     : {odd_turns:,} 件 ({(odd_turns/len(turn_counts))*100:.1f}%)")
        
        if odd_turns > 0:
             print("-" * 50)
             print("💡 補足: 奇数ターンのデータが含まれています。")
             print("   (通常、ユーザー始動でアシスタント終了なら偶数になるはずです)")

        # --- グラフ描画 ---
        print("-" * 50)
        print(f"📈 ターン数分布のグラフを作成中...")
        
        counts = Counter(turn_counts)
        x_values = sorted(counts.keys())
        y_values = [counts[x] for x in x_values]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(x_values, y_values, color='skyblue', edgecolor='black', alpha=0.7)
        
        # バーの上に数値を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom')

        plt.xlabel('Turn Counts')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Dialogue Turn Counts')
        plt.xticks(x_values)  # 全てのターン数をX軸に表示
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(args.output_img)
        print(f"💾 グラフを保存しました: {args.output_img}")

    else:
        print("   (ターン情報の取得に失敗しました)")

    print("="*50)

if __name__ == "__main__":
    main()