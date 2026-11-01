import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
import sentencepiece as spm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis

TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


def gate_to_float(g):
    if isinstance(g, torch.Tensor):
        return g.detach().float().mean().item()
    return float(g)


@torch.inference_mode()
def encode_user_audio_to_codes(mimi, wav_path: str, user_channel: int = 0, tail_sec: float = 5.0):
    wav, sr = torchaudio.load(wav_path)  # [C, T]

    # ★重要: stereoなら user_channel のみを使う（meanしない）
    if wav.size(0) > 1:
        wav = wav[user_channel:user_channel+1]  # [1, T]
    else:
        wav = wav[:1]

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # ★重要: delays の尻尾を出すために無音を足す（encode前）
    if tail_sec > 0:
        tail = torch.zeros(1, int(tail_sec * TARGET_SR))
        wav = torch.cat([wav, tail], dim=1)

    wav = wav.unsqueeze(0).to(device)  # [B=1, 1, T]
    codes = mimi.encode(wav)           # [B, 8, T_frames]
    return codes


@torch.inference_mode()
def encode_image_to_ca_src(image_embedder, img_path: str):
    image_processor = ImageProcessor()
    img = image_processor(img_path).unsqueeze(0).to(device=device, dtype=dtype)
    out = image_embedder(img)
    return out["cross_attention_src"]  # [B, S, D]


def main():
    # ===== Config =====
    kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

    # ===== Tokenizer =====
    sp = spm.SentencePieceProcessor()
    sp.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

    # ===== Mimi =====
    mimi_weight = hf_hub_download(
        repo_id="kyutai/moshika-vis-pytorch-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    mimi = get_mimi(mimi_weight, device).eval()

    # ===== MoshiVisGen + ImageProjection =====
    ckpt_path = "/workspace/j-moshivis/checkpoints/step_6000.safetensors"
    moshi_vis, image_embedder = get_moshi_vis(
        kyuteye_config,
        moshi_weight=ckpt_path,
        device=device,
        dtype=dtype,
        gen_kwargs=dict(
            use_sampling=True,
            temp=0.7,
            top_k=50,
            temp_text=0.7,
            top_k_text=50,
            check=False,
        ),
    )
    moshi_vis.eval()
    image_embedder.eval()

    # ===== 仕様チェック（Pattern A になってるか確認）=====
    lm = moshi_vis.lm_model
    needed = lm.num_codebooks - lm.num_audio_codebooks_out - 1
    print("n_q (audio in):", lm.n_q, "dep_q (audio out):", lm.dep_q, "num_codebooks:", lm.num_codebooks)
    print("needed_tokens (expected user codebooks):", needed)
    # Pattern A なら needed == 8 のはず
    # needed == 0 なら、そもそも duplex 前提になってない設定です。

    # ===== Inputs =====
    img_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1cb540373cec900c0a05f8512933360dcda6399315b72de3ae552e4d5a667cd6/image.jpg"
    wav_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1cb540373cec900c0a05f8512933360dcda6399315b72de3ae552e4d5a667cd6/stereo_dialogue.wav"

    ca_src = encode_image_to_ca_src(image_embedder, img_path)
    user_codes = encode_user_audio_to_codes(mimi, wav_path, user_channel=0, tail_sec=5.0)  # ★userチャンネル指定

    assert user_codes.shape[1] == needed, f"Input codes has {user_codes.shape[1]} codebooks but needed={needed}"

    # ===== Streaming Generation =====
    gen_audio_chunks = []
    gen_text_ids = []
    gate_values = []

    with torch.inference_mode(), moshi_vis.streaming():
        T = user_codes.shape[-1]
        for t in range(T):
            user_chunk = user_codes[:, :, t:t+1]  # [B, needed(=8), 1]
            out, gate = moshi_vis.step(user_chunk, ca_src=ca_src)

            gate_values.append(gate_to_float(gate))

            if out is None:
                continue  # warm-up (<= max_delay)

            # out: [B, 1 + dep_q(=8), 1]
            text_id = int(out[0, 0, 0].item())
            gen_text_ids.append(text_id)

            audio_tok = out[:, 1:, :]  # assistant audio 8 codebooks
            gen_audio_chunks.append(audio_tok.cpu())

    # ===== Decode =====
    if gen_audio_chunks:
        gen_audio_tokens = torch.cat(gen_audio_chunks, dim=-1).to(device)  # [B, 8, T']
        gen_wav = mimi.decode(gen_audio_tokens)  # [B, 1, samples] など実装依存
        gen_wav = gen_wav.squeeze().detach().cpu().numpy()
        sf.write("generated_assistant.wav", gen_wav, TARGET_SR)
        print("✅ Saved: generated_assistant.wav")
    else:
        print("❌ No audio tokens generated (out was always None). tail_sec が足りない/入力長が短すぎる可能性。")

    gen_text = sp.decode(gen_text_ids) if gen_text_ids else ""
    print("Generated text:", gen_text)
    print("Gate mean:", sum(gate_values)/max(1, len(gate_values)))


if __name__ == "__main__":
    main()
