import torch
import soundfile as sf
import argparse
import os
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# プロジェクト内のaudio_codec.pyから関数をインポートします
from jmvis.utils.audio_codec import encode as audio_encode, decode as audio_decode

# ===== ★ 修正箇所 1 ★ =====
# 不安定な確率（inf, nan）を安全な値に置き換えるためのクラス
class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # infを大きな有限値に、nanを0に置き換える
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Moshiモデルを使用して音声ファイルに応答を生成します。")
    parser.add_argument("--input_wav", required=True, help="入力となるWAVファイルのパス")
    parser.add_argument("--output_wav", default="response.wav", help="生成された応答を保存するWAVファイルのパス")
    # ===== ★ 修正箇所 2 ★ =====
    # デフォルトのモデルを、ファインチューニング済みの日本語モデルに変更
    parser.add_argument("--model_name", default="nu-dialogue/j-moshi", help="使用するHugging Faceモデル名")
    # ==========================
    parser.add_argument("--max_new_tokens", type=int, default=750, help="生成する新しい音声トークンの最大数（約15秒）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # 1. モデルのロード
    print(f"モデル '{args.model_name}' をロード中...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用するデータ型: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto"
    ).eval()
    print("モデルのロード完了。")

    # 2. 入力音声ファイルの読み込みとエンコード
    print(f"入力ファイル '{args.input_wav}' を読み込み中...")
    try:
        wav, sr = sf.read(args.input_wav)
    except Exception as e:
        print(f"エラー: 音声ファイルの読み込みに失敗しました - {e}")
        return
        
    audio_tensor = torch.from_numpy(wav).float()
    if audio_tensor.ndim > 1:
        audio_tensor = torch.mean(audio_tensor, dim=1)
        
    print("音声をトークンにエンコード中...")
    input_tokens = audio_encode(audio_tensor)
    
    if not input_tokens:
        print("エラー: 音声からトークンをエンコードできませんでした。音声が短すぎる可能性があります。")
        return

    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    max_input_length = 4096
    if input_ids.shape[1] > max_input_length:
        print(f"警告: 入力トークン長 ({input_ids.shape[1]}) が長すぎるため、末尾の {max_input_length} トークンに切り詰めます。")
        input_ids = input_ids[:, -max_input_length:]
    
    print(f"エンコード完了。入力トークン数: {input_ids.shape[1]}")

    # 3. 応答の生成
    print("応答音声を生成中...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            # ===== ★ 修正箇所 3 ★ =====
            # ファインチューニング済みモデルでは、より自然なサンプリングが期待できる
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            logits_processor=LogitsProcessorList([SafeLogitsProcessor()]), # 安定化のための処理を追加
            # ==========================
            use_cache=True,
        )
    
    generated_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    print(f"生成完了。出力トークン数: {len(generated_tokens)}")

    # 4. トークンを音声にデコード
    print("トークンを音声波形にデコード中...")
    output_wav_tensor = audio_decode(generated_tokens)
    
    # 5. 音声ファイルとして保存
    output_sr = 16000
    if output_wav_tensor.numel() > 0:
        sf.write(args.output_wav, output_wav_tensor.squeeze(0).cpu().numpy(), output_sr)
        print(f"✅ 応答音声を '{args.output_wav}' に保存しました。")
    else:
        print("警告: デコードされた音声が空のため、ファイルは保存されませんでした。")


if __name__ == "__main__":
    main()