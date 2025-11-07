"""
create_jsonl.py

📘 概要:
  このスクリプトは、指定したディレクトリ配下のすべての `.wav` 音声ファイルを再帰的に探索し、
  各ファイルの「絶対パス」と「音声の長さ（秒）」を取得して JSONL 形式でまとめるツールです。

  生成される `data.jsonl` は、Whisper などの音声アノテーション処理（例: annotate.py）の
  入力リストとして利用できます。

🧩 主な機能:
  - ディレクトリ内の音声ファイル（.wav）を再帰的に探索
  - sphn ライブラリを用いて音声長を一括取得
  - {"path": <音声パス>, "duration": <再生時間[秒]>} の形式で JSONL 出力

📂 出力例:
  {"path": "/workspace/data/speech/data_stereo/sample_001.wav", "duration": 5.23}
  {"path": "/workspace/data/speech/data_stereo/sample_002.wav", "duration": 7.48}

💡 使用例:
  python3 create_jsonl.py \
      --wav-dir /workspace/data/speech/data_stereo \
      --out-dir /workspace/data/speech

🧠 注意:
  - `sphn` パッケージが必要です（`pip install sphn`）
  - 長さが取得できないファイル（破損ファイルなど）はスキップされます
  - 出力先ディレクトリが存在しない場合は自動で作成されます
"""

import sphn
import json
from pathlib import Path
import argparse


def create_jsonl(wavdir_path: str, output_dir: str) -> None:
    """Create a JSONL file with audio paths and durations."""
    paths = [str(f) for f in Path(wavdir_path).glob("**/*.wav")]
    durations = sphn.durations(paths)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "data.jsonl", "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")
            print(f"✅ {p} (duration: {d:.2f} sec)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSONL file with audio paths and durations.")
    parser.add_argument("--wav-dir", type=str, default="/workspace/data/speech/data_stereo", help="Directory containing WAV files.")
    parser.add_argument("--out-dir", type=str, default="/workspace/data/speech", help="Output JSONL file.")
    args = parser.parse_args()
    create_jsonl(args.wav_dir, args.out_dir)    
