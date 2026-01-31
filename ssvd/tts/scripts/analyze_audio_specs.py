"""
Script: analyze_audio_specs.py
Purpose:
    JSONLファイル内の 'path' (音声パス) をもとに、
    音声ファイル(WAV)の物理的な仕様を解析する。
    - サンプルレート (Sample Rate)
    - ビット深度 (Bit Depth)
    - チャンネル数 (Channels)

Usage:
    python3 analyze_audio_specs.py --input /workspace/data/speech/train_data_refined_a.jsonl
"""

import json
import argparse
import os
import wave
import contextlib
from collections import Counter

def get_wav_info(file_path):
    """WAVファイルのヘッダー情報を取得する"""
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            rate = f.getframerate()
            channels = f.getnchannels()
            sampwidth = f.getsampwidth()
            bit_depth = sampwidth * 8
            return {
                "sample_rate": rate,
                "bit_depth": bit_depth,
                "channels": channels
            }
    except wave.Error as e:
        return {"error": f"Wave Error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {e}"}

def format_freq(hz):
    """周波数を読みやすくフォーマット"""
    return f"{hz/1000:.1f}kHz" if hz >= 1000 else f"{hz}Hz"

def main():
    parser = argparse.ArgumentParser(description="Analyze audio specifications (Sample Rate, Bit Depth, Channels).")
    parser.add_argument("--input", default="/workspace/data/speech/train_data_refined_a.jsonl", help="Path to JSONL file.")
    
    # パス置換用オプション (環境によるパスの差異を吸収するため)
    parser.add_argument("--src-prefix", default="/gpu-server/user/yoshiki/j-moshivis", help="Source path prefix in JSONL to replace.")
    parser.add_argument("--dst-prefix", default="/workspace", help="Destination path prefix to replace with.")
    
    args = parser.parse_args()

    jsonl_path = args.input
    
    # 統計用カウンター: (rate, depth, channels) のタプルをキーにする
    specs_counter = Counter()
    missing_files = 0
    valid_files = 0
    error_files = 0

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
                    
                    # 2. パスのプレフィックス置換（環境適応）
                    final_path = audio_path
                    if not os.path.exists(final_path):
                        if args.src_prefix in audio_path:
                            replaced_path = audio_path.replace(args.src_prefix, args.dst_prefix, 1)
                            if os.path.exists(replaced_path):
                                final_path = replaced_path
                    
                    # 3. 音声ファイルを解析
                    if os.path.exists(final_path):
                        info = get_wav_info(final_path)
                        
                        if "error" in info:
                            # WAVとして読み込めなかった場合など
                            error_files += 1
                            if error_files <= 3:
                                print(f"⚠️ Read Error at {os.path.basename(final_path)}: {info['error']}")
                        else:
                            # 成功: 仕様を記録
                            spec_key = (info["sample_rate"], info["bit_depth"], info["channels"])
                            specs_counter[spec_key] += 1
                            valid_files += 1
                    else:
                        missing_files += 1
                        if missing_files <= 3:
                            print(f"⚠️ File not found (Sample): {final_path}")

                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

    # --- 結果出力 ---
    total_processed = valid_files + missing_files + error_files
    
    print("\n" + "="*50)
    print(f"🎵 音声データプロファイル分析結果")
    print("="*50)
    print(f"📁 処理対象ファイル数 : {total_processed:,} 件")
    print(f"✅ 有効データ数       : {valid_files:,} 件")
    
    if missing_files > 0:
        print(f"❌ 見つからないファイル : {missing_files:,} 件")
        print(f"   (パス設定を確認してください: {args.src_prefix} -> {args.dst_prefix})")
    if error_files > 0:
        print(f"⚠️  読み込みエラー       : {error_files:,} 件 (非WAV形式、破損など)")

    print("-" * 50)
    print("📊 【検出された仕様 (Sample Rate / Bit Depth / Channels)】")
    
    if specs_counter:
        # 多い順にソートして表示
        for spec, count in specs_counter.most_common():
            rate, depth, ch = spec
            ch_str = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}ch"
            ratio = (count / valid_files) * 100
            
            print(f"   🔹 {format_freq(rate)} / {depth}bit / {ch_str} ({ch}ch)")
            print(f"      Count: {count:,} 件 ({ratio:.1f}%)")
            
            # 警告ロジック
            warnings = []
            if ch != 2:
                warnings.append("ステレオ(2ch)ではありません")
            if rate < 24000:
                warnings.append("サンプルレートが低めです(Moshiは通常24kHz)")
            
            if warnings:
                print(f"      ⚠️  注意: {' / '.join(warnings)}")
            print("")
    else:
        print("   (有効な音声データが見つかりませんでした)")

    print("="*50)

if __name__ == "__main__":
    main()