import torch
import soundfile as sf
import argparse
import huggingface_hub
import rustymimi
import numpy as np
import sentencepiece
from tqdm import tqdm

from moshi.models.lm import LMModel, LMGen
from safetensors.torch import load_file


# ======== Moshi LMModel ローダー ========
def load_lm_model(model_name: str, device="cpu", dtype=torch.float16):
    print("Hugging Face Hub から checkpoint をダウンロード中...")
    ckpt_path = huggingface_hub.hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors"
    )

    num_codebooks = 8 + 1  # n_q + 1
    delays = [0] * num_codebooks

    # Moshi の LMModel を初期化（ハイパーパラメータはモデル依存。必要に応じて変更）
    lm_model = LMModel(
        delays=delays,
        n_q=8,
        dep_q=8,
        card=1024,
        text_card=32000,
        dim=128,
        num_heads=8,
        hidden_scale=4,
        depformer_dim=256,
        num_layers=12,
    ).to(device, dtype)

    # safetensors 読み込み
    state_dict = load_file(ckpt_path, device=device)
    lm_model.load_state_dict(state_dict)

    print("✅ LMModel のロード完了")
    return lm_model.eval()


# ======== Audio モード ========
def run_audio_mode(lm_gen, audio_tokenizer, text_tokenizer, wav, args):
    sr = 24000
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)  # モノラル化
    wav_tensor_3d = wav.reshape(1, 1, -1)

    # 音声をトークン化
    input_tokens_np = np.array(audio_tokenizer.encode(wav_tensor_3d), dtype=np.uint32)
    input_tokens_np = np.transpose(input_tokens_np, (0, 2, 1))  # (B, K, T)
    input_tokens = torch.from_numpy(input_tokens_np).long().to(lm_gen.lm_model.device)

    generated_audio_tokens = []
    generated_text_ids = []

    print("\n--- LMGen による音声応答生成 ---")
    with lm_gen.streaming(batch_size=1):
        for _ in tqdm(range(args.max_new_tokens), desc="Generating"):
            out = lm_gen.step(input_tokens[:, :, -1:])  # 直近1ステップを入力
            if out is None:
                continue

            # テキストトークン
            text_token = out[:, 0, -1].item()
            if text_token == text_tokenizer.eos_id():
                print("[EOS 検出 → 終了]")
                break
            elif text_token != text_tokenizer.pad_id():
                generated_text_ids.append(text_token)

            # 音声トークン
            audio_tokens_step = out[:, 1:, -1].cpu().numpy()  # (B, dep_q)
            generated_audio_tokens.extend(audio_tokens_step.flatten())

            # 次の入力に渡す
            input_tokens = out

    # テキスト復号
    decoded_text = text_tokenizer.decode(generated_text_ids)
    print(f"\n生成テキスト: {decoded_text}")

    # 音声復号
    if generated_audio_tokens:
        num_audio_tokens = len(generated_audio_tokens)
        len_to_decode = num_audio_tokens - (num_audio_tokens % 8)
        if len_to_decode > 0:
            tokens_to_decode = np.array(
                generated_audio_tokens[:len_to_decode]
            ).reshape(1, -1, 8)
            tokens_to_decode = np.transpose(tokens_to_decode, (0, 2, 1)).astype(np.uint32)
            output_wav_numpy = np.array(audio_tokenizer.decode(tokens_to_decode), dtype=np.float32)
            sf.write(args.output_wav, output_wav_numpy.flatten(), sr)
            print(f"✅ 音声を保存: {args.output_wav}")
        else:
            print("⚠️ 8 の倍数未満のため音声デコード不可")
    else:
        print("⚠️ 音声トークンが生成されませんでした")


# ======== Text モード ========
def run_text_mode(lm_gen, text_tokenizer, args):
    input_tensor = torch.tensor(
        [[[text_tokenizer.bos_id()]]], device=lm_gen.lm_model.device
    )  # (B=1, K=1, T=1)

    generated_ids = []

    print("\n--- LMGen によるテキスト生成 ---")
    with lm_gen.streaming(batch_size=1):
        for _ in range(args.max_new_tokens):
            out = lm_gen.step(input_tensor)
            if out is None:
                continue
            token_id = out[:, 0, -1].item()
            if token_id == text_tokenizer.eos_id():
                print("[EOS 検出 → 終了]")
                break
            generated_ids.append(token_id)
            input_tensor = torch.tensor([[[token_id]]], device=lm_gen.lm_model.device)

    decoded_text = text_tokenizer.decode(generated_ids)
    print(f"\n生成テキスト:\n{decoded_text}")


# ======== Main ========
def main():
    parser = argparse.ArgumentParser(description="Moshi LMGen テストスクリプト")
    parser.add_argument("mode", choices=["text", "audio"], help="モード選択 ('text' または 'audio')")
    parser.add_argument("--input_wav", help="[audio モード] 入力 WAV ファイル")
    parser.add_argument("--output_wav", default="response.wav", help="[audio モード] 出力 WAV ファイル")
    parser.add_argument("--model_name", default="kyutai/moshiko-pytorch-bf16", help="Hugging Face モデル名")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成最大トークン数")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用デバイス: {device}, dtype={dtype}")

    # Moshi LMModel をロード
    lm_model = load_lm_model(args.model_name, device=device, dtype=dtype)
    lm_gen = LMGen(lm_model, temp_text=0.7, top_k_text=25)

    # SentencePiece トークナイザー
    print("テキストトークナイザーをロード中...")
    text_tokenizer_path = huggingface_hub.hf_hub_download(
        repo_id="kyutai/moshiko-bf16",
        filename="tokenizer_spm_32k_3.model"
    )
    text_tokenizer = sentencepiece.SentencePieceProcessor(model_file=text_tokenizer_path)

    if args.mode == "audio":
        if not args.input_wav:
            parser.error("--input_wav は audio モードで必須です")
        wav, sr = sf.read(args.input_wav, dtype="float32")
        if sr != 24000:
            print(f"⚠️ 入力サンプリングレート {sr}Hz → 24kHz に変換してください（現状未実装）")

        print("Mimi コーデックをロード中...")
        mimi_weight_path = huggingface_hub.hf_hub_download(
            repo_id="kyutai/moshiko-mlx-bf16",
            filename="tokenizer-e351c8d8-checkpoint125.safetensors"
        )
        audio_tokenizer = rustymimi.Tokenizer(mimi_weight_path)

        run_audio_mode(lm_gen, audio_tokenizer, text_tokenizer, wav, args)

    elif args.mode == "text":
        run_text_mode(lm_gen, text_tokenizer, args)


if __name__ == "__main__":
    main()
