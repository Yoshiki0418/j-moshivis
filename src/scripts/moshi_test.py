import torch
import soundfile as sf
import argparse
import huggingface_hub
import rustymimi
import numpy as np
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# ===== ★★★ 最終修正箇所 1 ★★★ =====
# モデルの出力トークンを特定の範囲に制限するためのクラス
class ClampLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 語彙サイズを超えるトークンの確率を-infに設定し、選択されないようにする
        scores[:, self.vocab_size:] = -float("inf")
        return scores
# =================================

class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores

def main():
    parser = argparse.ArgumentParser(description="Moshiモデルを使用して音声ファイルに応応答を生成します。")
    parser.add_argument("--input_wav", required=True, help="入力となるWAVファイルのパス")
    parser.add_argument("--output_wav", default="response.wav", help="生成された応答を保存するWAVファイルのパス")
    # parser.add_argument("--model_name", default="nu-dialogue/j-moshi", help="使用するHugging Faceモデル名")
    parser.add_argument("--model_name", default="kyutai/moshiko-pytorch-bf16", help="使用するHugging Faceモデル名")
    parser.add_argument("--max_new_tokens", type=int, default=192, help="生成する新しい音声トークンの最大数（8の倍数を推奨）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    print(f"モデル '{args.model_name}' をロード中...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用するデータ型: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto"
    ).eval()
    print("モデルのロード完了。")

    print("Mimiオーディオコーデックの重みをダウンロード中...")
    mimi_weight_path = huggingface_hub.hf_hub_download(
        repo_id="kyutai/moshiko-mlx-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    print(f"Mimiの重みをロード中: {mimi_weight_path}")
    audio_tokenizer = rustymimi.Tokenizer(mimi_weight_path)

    print(f"入力ファイル '{args.input_wav}' を読み込み中...")
    wav, sr = sf.read(args.input_wav, dtype='float32')

    if sr != 2048:
        print(f"警告: 入力音声が{sr}Hzです。24000Hzにリサンプリングします。")
        try:
            import torchaudio.transforms as T
            wav_tensor = torch.from_numpy(wav).float()
            if wav_tensor.ndim > 1:
                 wav_tensor = wav_tensor.T
            else:
                 wav_tensor = wav_tensor.unsqueeze(0)
            resampler = T.Resample(sr, 24000)
            wav = resampler(wav_tensor).numpy()
            if wav.ndim > 1:
                 wav = wav.T
        except ImportError:
            print("torchaudioが見つかりません。簡易的なリサンプリングを試みます。")
            num_samples = int(len(wav) * 24000 / sr)
            wav = np.interp(np.linspace(0, len(wav), num_samples), np.arange(len(wav)), wav)
    
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    print("音声をトークンにエンコード中...")
    wav_tensor_3d = wav.reshape(1, 1, -1)
    input_tokens = np.array(audio_tokenizer.encode(wav_tensor_3d), dtype=np.uint32)
    input_tokens = np.transpose(input_tokens, (0, 2, 1)) 
    input_tokens = input_tokens[:, :, :8].reshape(1, -1)
    
    input_ids = torch.from_numpy(input_tokens).to(device, dtype=torch.long)
    
    max_input_length = 4096
    if input_ids.shape[1] > max_input_length:
        print(f"警告: 入力トークン長 ({input_ids.shape[1]}) が長すぎるため、末尾の {max_input_length} トークンに切り詰めます。")
        input_ids = input_ids[:, -max_input_length:]
    
    print(f"エンコード完了。入力トークン数: {input_ids.shape[1]}")

    print("応答音声を生成中...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            # ===== ★★★ 最終修正箇所 2 ★★★ =====
            # 2つのLogitsProcessorをリストで渡す
            # mimiの語彙サイズは2048
            logits_processor=LogitsProcessorList([SafeLogitsProcessor(), ClampLogitsProcessor(2048)]),
            # =================================
            use_cache=True,
        )
    
    generated_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    print(f"生成完了。出力トークン数: {len(generated_tokens)}")

    print("トークンを音声波形にデコード中...")

    num_tokens = len(generated_tokens)
    tokens_to_decode_len = num_tokens - (num_tokens % 8)
    tokens_to_decode = np.array(generated_tokens[:tokens_to_decode_len]).reshape(1, -1, 8)
    
    tokens_to_decode = np.transpose(tokens_to_decode, (0, 2, 1)).astype(np.uint32).copy()
    
    output_wav_numpy = np.array(audio_tokenizer.decode(tokens_to_decode), dtype=np.float32)

    output_sr = 24000
    sf.write(args.output_wav, output_wav_numpy.flatten(), output_sr)
    print(f"✅ 応答音声を '{args.output_wav}' に保存しました。")


if __name__ == "__main__":
    main()