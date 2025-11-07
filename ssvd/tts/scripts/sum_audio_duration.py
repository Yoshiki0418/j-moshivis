"""
Script: sum_audio_duration.py
Purpose:
    JSONLファイル内のすべての音声ファイルの再生時間(duration)を合計し、
    総再生時間を「〇〇時間〇〇分〇〇秒」で出力する。

Usage:
    python3 sum_audio_duration.py --input /path/to/file.jsonl
"""

import json
import argparse


def format_time(seconds: float) -> str:
    """秒数を 時間・分・秒 の文字列に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}時間{minutes}分{secs}秒"


def main():
    parser = argparse.ArgumentParser(description="Sum durations from JSONL file.")
    parser.add_argument("--input", default="/workspace/data/speech/data.jsonl", help="Path to JSONL file.")
    args = parser.parse_args()

    total_duration = 0.0
    count = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "duration" in data:
                total_duration += data["duration"]
                count += 1

    print(f"📁 ファイル数: {count}")
    print(f"⏱️  総再生時間: {format_time(total_duration)}")


if __name__ == "__main__":
    main()
