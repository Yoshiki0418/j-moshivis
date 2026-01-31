"""
Script: analyze_audio_duration_v2.py
Purpose:
    JSONLファイル内の音声ファイルの再生時間(duration)を分析する。
    基本統計量（平均、最大、最小、中央値）に加え、
    「指定した秒数（閾値）以上のデータ数と割合」を算出する。

Usage:
    python3 analyze_audio_duration_v2.py --input /path/to/file.jsonl --threshold 100
"""

import json
import argparse
import statistics

def format_time(seconds: float) -> str:
    """秒数を 時間・分・秒 の文字列に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}時間{minutes}分{secs}秒"

def format_sec(seconds: float) -> str:
    """秒数を小数点以下2桁までの文字列に変換"""
    return f"{seconds:.2f}秒"

def main():
    parser = argparse.ArgumentParser(description="Analyze durations from JSONL file.")
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl", help="Path to JSONL file.")
    parser.add_argument("--threshold", type=float, default=110.0, help="Threshold in seconds to count files (default: 100.0).")
    args = parser.parse_args()

    durations = []

    print(f"📂 ファイル読み込み中: {args.input} ...")

    # ファイル読み込み
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "duration" in data:
                        durations.append(float(data["duration"]))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Error: File not found at {args.input}")
        return

    count = len(durations)

    if count == 0:
        print("⚠️ 音声データ(duration)が見つかりませんでした。")
        return

    # 統計量の計算
    total_duration = sum(durations)
    avg_duration = total_duration / count
    max_duration = max(durations)
    min_duration = min(durations)
    median_duration = statistics.median(durations)

    # 指定秒数以上のデータをカウント
    over_threshold_count = sum(1 for d in durations if d >= args.threshold)
    over_threshold_ratio = (over_threshold_count / count) * 100

    # 結果の出力
    print("\n" + "="*50)
    print(f"📊 データセット分析結果")
    print("="*50)
    print(f"📁 総ファイル数    : {count:,} 件")
    print(f"⏱️  総再生時間      : {format_time(total_duration)} ({total_duration:,.2f}秒)")
    print("-" * 50)
    print(f"📏 平均再生時間    : {format_sec(avg_duration)}")
    print(f"🎯 中央値          : {format_sec(median_duration)}")
    print(f"🔼 最大再生時間    : {format_time(max_duration)} ({format_sec(max_duration)})")
    print(f"🔽 最小再生時間    : {format_sec(min_duration)}")
    print("-" * 50)
    print(f"🔍 {args.threshold}秒以上のデータ : {over_threshold_count:,} 件")
    print(f"📈 全体に占める割合: {over_threshold_ratio:.2f}%")
    print("="*50)

    # アドバイス表示
    if over_threshold_ratio < 10:
        print(f"💡 {args.threshold}秒以上のデータが少ないです。")
        print("   短い会話が多い場合でも、設定を長くして複数の会話を連結して学習させるか、")
        print("   パディングを含めて文脈を確保する設定(100秒以上)が推奨されます。")
    elif over_threshold_ratio > 50:
        print(f"💡 半分以上が{args.threshold}秒を超えています。")
        print(f"   設定を {args.threshold}秒 にすることで、多くの会話文脈を保持でき、")
        print("   推論精度の大幅な向上が期待できます。")

if __name__ == "__main__":
    main()