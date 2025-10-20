"""
safetensors → pth 変換スクリプト（推論専用）

このスクリプトは Style-Bert-VITS2 などのモデルを
「推論で利用するため」に safetensors ファイルを pth に変換します。

⚠️ 注意:
- 変換後の pth は **推論専用** です。
- 再学習 (fine-tuning / resume training) には利用できません。
- checkpoint の学習情報（iteration, optimizer 等）はダミー値を埋めています。
"""

from pathlib import Path
import argparse
import torch
from safetensors.torch import load_file


def convert_safetensors_to_pth(directory: str, overwrite: bool = False):
    """
    指定したディレクトリ直下の .safetensors を .pth に変換する (推論専用)

    Args:
        directory (str | Path): 変換対象ディレクトリ
        overwrite (bool): 既に pth が存在する場合に上書きするかどうか
    """
    directory = Path(directory)
    safetensors_files = list(directory.glob("*.safetensors"))

    if not safetensors_files:
        print(f"⚠️ {directory} に .safetensors ファイルが見つかりませんでした。")
        return

    for safetensor_path in safetensors_files:
        pth_path = safetensor_path.with_suffix(".pth")
        if pth_path.exists() and not overwrite:
            print(f"⏩ Skip (already exists): {pth_path}")
            continue

        tensors = load_file(safetensor_path)

        # 🔑 Style-Bert-VITS2 が期待する checkpoint 形式にラップ
        checkpoint = {
            "model": tensors,        # 実際の重み
            "optimizer": {},         # ダミー
            "learning_rate": 0.0,    # ダミー
            "iteration": 0,          # ダミー
            "epoch": 0,              # ダミー
        }

        torch.save(checkpoint, pth_path)
        print(f"✅ Converted: {pth_path}")

    print("🎉 すべての変換処理が完了しました！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="safetensors を推論専用の pth に変換するスクリプト"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="変換対象ディレクトリ（例: /workspace/ssvd/tts_model/models/kouon28）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の pth がある場合に上書きする"
    )
    args = parser.parse_args()

    convert_safetensors_to_pth(args.target_dir, overwrite=args.overwrite)
