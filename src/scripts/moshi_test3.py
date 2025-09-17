import torch
import soundfile as sf
import argparse
import huggingface_hub
import rustymimi
import numpy as np
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import sentencepiece

# RuntimeErrorを防ぐため、SafeLogitsProcessorを再度導入します
class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # infを大きな有限値に、nanを0に置き換える
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores

def main():
    parser = argparse.ArgumentParser(description="Moshiモデルを使用して音声ファイルに応答を生成します。")
    parser.add_argument("--input_wav", required=True, help="入力となるWAVファイルのパス")
    parser.add_argument("--output_wav", default="response.wav", help="生成された応答を保存するWAVファイルのパス")
    parser.add_argument("--model_name", default="kyutai/moshiko-pytorch-bf16", help="使用するHugging Faceモデル名")
    parser.add_argument("--max_new_tokens", type=int, default=752, help="生成する新しいトークンの最大数（音声＋テキスト）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # 1. モデルと各種Tokenizerのロード
    print(f"モデル '{args.model_name}' をロード中...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用するデータ型: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto"
    ).eval()
    
    print("テキストトークナイザーをダウンロード中...")
    text_tokenizer_path = huggingface_hub.hf_hub_download(
        repo_id="kyutai/moshiko-pytorch-bf16",
        filename="tokenizer_spm_32k_3.model"
    )
    text_tokenizer = sentencepiece.SentencePieceProcessor(model_file=text_tokenizer_path)
    print("モデルのロード完了。")

    print("Mimiオーディオコーデックの重みをダウンロード中...")
    mimi_weight_path = huggingface_hub.hf_hub_download(
        repo_id="kyutai/moshiko-mlx-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    print(f"Mimiの重みをロード中: {mimi_weight_path}")
    audio_tokenizer = rustymimi.Tokenizer(mimi_weight_path)

    # 2. 入力音声の準備
    print(f"入力ファイル '{args.input_wav}' を読み込み中...")
    wav, sr = sf.read(args.input_wav, dtype='float32')

    if sr != 24000:
        print(f"警告: 入力音声が{sr}Hzです。24000Hzにリサンプリングします。")
        try:
            import torchaudio.transforms as T
            wav_tensor = torch.from_numpy(wav).float()
            if wav_tensor.ndim > 1: wav_tensor = wav_tensor.T
            else: wav_tensor = wav_tensor.unsqueeze(0)
            resampler = T.Resample(sr, 24000)
            wav = resampler(wav_tensor).numpy()
            if wav.ndim > 1: wav = wav.T
        except ImportError:
            num_samples = int(len(wav) * 24000 / sr)
            wav = np.interp(np.linspace(0, len(wav), num_samples), np.arange(len(wav)), wav)
    
    if wav.ndim > 1: wav = np.mean(wav, axis=1)

    print("音声をトークンにエンコード中...")
    wav_tensor_3d = wav.reshape(1, 1, -1)
    input_tokens_np = np.array(audio_tokenizer.encode(wav_tensor_3d), dtype=np.uint32)
    input_tokens_np = np.transpose(input_tokens_np, (0, 2, 1))
    input_tokens_np = input_tokens_np[:, :, :8].reshape(1, -1)
    input_ids = torch.from_numpy(input_tokens_np).long()

    max_input_length = 4096
    if input_ids.shape[1] > max_input_length:
        print(f"警告: 入力トークン長が長すぎるため切り詰めます。")
        input_ids = input_ids[:, -max_input_length:]
    
    # ===== ★★★ 最終改善箇所 ★★★ =====
    # 3. .generate()を正しく動作させるためのattention_maskを作成
    attention_mask = torch.ones_like(input_ids)
    # =================================

    print(f"エンコード完了。総入力トークン数: {input_ids.shape[1]}")

    # 4. 応答生成
    print("\n--- 応答生成開始 ---")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device), # attention_maskを渡す
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.3, # 雑音を抑え、テキストと音声のバランスをとるため低めに設定
            top_p=0.95,
            pad_token_id=text_tokenizer.pad_id(), # pad_token_idを明示的に設定
            eos_token_id=text_tokenizer.eos_id(),
            logits_processor=LogitsProcessorList([SafeLogitsProcessor()]), # 安定化のため再導入
            use_cache=True,
        )
    
    # 5. 生成されたトークンから音声とテキストを分離
    generated_ids = output_ids[0, input_ids.shape[1]:].cpu().tolist()
    
    generated_audio_tokens = [token for token in generated_ids if token < 2048]
    generated_text_ids = [token for token in generated_ids if token >= 2048]

    print("\n--- 応答生成完了 ---")
    print(f"生成された総トークン数: {len(generated_ids)}")
    print(f" ▶ 音声トークン数: {len(generated_audio_tokens)}")
    
    decoded_text = text_tokenizer.decode(generated_text_ids)
    print(f" ▶ デコードされたテキスト: {decoded_text}")

    # 6. 音声のデコードと保存
    if not generated_audio_tokens:
        print("警告: 音声トークンが生成されませんでした。")
        return

    num_audio_tokens = len(generated_audio_tokens)
    len_to_decode = num_audio_tokens - (num_audio_tokens % 8)
    
    if len_to_decode == 0:
        print("警告: デコード可能な音声トークンが8個未満です。")
        return
        
    tokens_to_decode = np.array(generated_audio_tokens[:len_to_decode]).reshape(1, -1, 8)
    tokens_to_decode = np.transpose(tokens_to_decode, (0, 2, 1)).astype(np.uint32).copy()
    
    output_wav_numpy = np.array(audio_tokenizer.decode(tokens_to_decode), dtype=np.float32)

    sf.write(args.output_wav, output_wav_numpy.flatten(), 24000)
    print(f"✅ 応答音声を '{args.output_wav}' に保存しました。")


if __name__ == "__main__":
    main()