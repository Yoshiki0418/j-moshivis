import torch
import torchaudio
import sentencepiece as spm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor, ImageProjection
from moshi.models.loaders import get_mimi
from jmoshivis.models.moshivis import MoshiVisGen

# =================================================================
# ⚙️ 設定 (重要: ここを環境に合わせて調整してください)
# =================================================================
PATHS = {
    "config_jmoshi": "/workspace/j-moshivis/configs/j-moshi-vis.yaml",
    "tokenizer": "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model",
    "checkpoint": "/workspace/j-moshivis/checkpoints/step_100.safetensors",
    "mimi_repo": "kyutai/moshika-vis-pytorch-bf16",
    "mimi_file": "tokenizer-e351c8d8-checkpoint125.safetensors",
    "image": "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1cb540373cec900c0a05f8512933360dcda6399315b72de3ae552e4d5a667cd6/image.jpg",
    "audio": "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/1cb540373cec900c0a05f8512933360dcda6399315b72de3ae552e4d5a667cd6/stereo_dialogue.wav",
}

# ★重要: 学習に使われたセグメントの開始時間（秒）
# JSON (assistant_dialogue.refined.json) を見て、最初のセグメントの開始時刻を入れてください。
# 分からなければ 0.0 から少しずつずらして試す必要があります。
START_SEC = 0.0  
DURATION_SEC = 10.0 # 学習時の duration

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# =================================================================
# 🛠️ ロード関数 (堅牢版)
# =================================================================
def load_moshi_vis_robust(config_path, checkpoint_path, device, dtype):
    config = KyuteyeConfig.from_yml(config_path)
    print(f"📂 Loading weights from: {checkpoint_path}")
    loaded_weights = load_file(checkpoint_path, device="cpu")
    
    image_proj_state = {}
    model_state = {}

    for key, v in loaded_weights.items():
        clean_key = key
        is_image_proj = False
        if clean_key.startswith("image_prefix."):
            clean_key = clean_key[len("image_prefix."):]
            is_image_proj = True
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]

        if is_image_proj:
            image_proj_state[clean_key] = v
        else:
            model_state[clean_key] = v

    moshi_vis = MoshiVisGen.from_config(config, model_state, device, dtype)
    image_embedder = ImageProjection.from_config(config, moshi_vis.model_dim, image_proj_state, device)
    return moshi_vis.to(dtype), image_embedder.to(dtype)

# =================================================================
# 🚀 メイン処理
# =================================================================
def main():
    print("🚀 Verifying Overfitting (Generation Mode)...")

    # 1. モデルロード
    moshi_vis, image_embedder = load_moshi_vis_robust(PATHS["config_jmoshi"], PATHS["checkpoint"], DEVICE, DTYPE)
    moshi_vis.eval()
    image_embedder.eval()
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(PATHS["tokenizer"])
    
    mimi = get_mimi(hf_hub_download(PATHS["mimi_repo"], PATHS["mimi_file"]), DEVICE)

    # 2. 画像処理
    print("🖼️ Processing Image...")
    image_processor = ImageProcessor()
    image_tensor = image_processor(PATHS["image"]).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        img_out = image_embedder(image_tensor)
        ca_src = img_out["cross_attention_src"]

    # 3. 音声処理 (切り出し & ステレオ)
    print(f"🎵 Processing Audio (Segment: {START_SEC}s ~ {START_SEC+DURATION_SEC}s)...")
    waveform, sr = torchaudio.load(PATHS["audio"])
    
    # チャンネル調整
    if waveform.shape[0] == 1: 
        waveform = waveform.repeat(2, 1) # モノラルなら複製
    if sr != 24000: 
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    
    # ★重要: 学習時と同じ区間を切り出す
    start_frame = int(START_SEC * 24000)
    end_frame = start_frame + int(DURATION_SEC * 24000)
    
    # 音声長が足りない場合のパディング処理
    if waveform.shape[1] < end_frame:
        pad_len = end_frame - waveform.shape[1]
        waveform = torch.cat([waveform, torch.zeros(2, pad_len)], dim=1)
        
    waveform_segment = waveform[:, start_frame:end_frame]
    
    # 推論用に少し余白を持たせてパディング
    waveform_input = torch.cat([waveform_segment, torch.zeros(2, 24000 * 2)], dim=1).to(DEVICE)

    # Mimi Encode: [2, 1, T] -> [1, 16, T]
    with torch.no_grad():
        codes = mimi.encode(waveform_input.unsqueeze(1))
        codes = codes.view(1, 16, -1)
    
    print(f"✅ Input Audio Shape: {codes.shape} (Includes User & Moshi channels)")

    # 4. 生成ループ (step)
    print("🤖 Starting Autoregressive Generation (Greedy)...")
    
    # 完全決定論的生成 (Greedy)
    moshi_vis.update_gen_kwargs(temp=0.0, temp_text=0.0, top_k=0, top_k_text=0)
    
    generated_ids = []
    
    # 最初のトークン(BOS)はモデル内部で処理されるか、自然発生を待ちます
    with torch.no_grad(), moshi_vis.streaming():
        for t in range(codes.shape[-1]):
            # MoshiVisGen.step() は「User Audio (8ch)」のみを入力として受け取ります
            # 全16chのうち、前半8ch (User) を切り出して渡します
            full_chunk = codes[:, :, t:t+1]
            user_chunk = full_chunk[:, :8, :] 
            
            # 推論実行
            tokens, gate = moshi_vis.step(user_chunk, ca_src=ca_src)
            
            if tokens is not None:
                # tokens: [Batch, 1(Text) + 8(Audio), 1]
                # テキストトークン (Channel 0) を保存
                text_token = tokens[0, 0, 0].item()
                generated_ids.append(text_token)
            
            if t % 50 == 0:
                print(".", end="", flush=True)

    print("\n✅ Generation Finished.")

    # 5. 結果表示
    # 生IDの確認 (3ばかりでないことを祈る)
    print(f"\nRaw IDs (First 50): {generated_ids[:50]}")

    decoded_text = tokenizer.decode(generated_ids)

    print("\n" + "="*40)
    print("📝 GENERATED TEXT")
    print("="*40)
    print(decoded_text)
    print("="*40)
    
    if len(decoded_text.strip()) > 0:
        print("🎉 SUCCESS: Text generated!")
        print("   If the text matches the audio content, overfitting is verified.")
    else:
        print("⚠️ FAILURE: Text is empty.")
        print("   Possible reasons:")
        print("   1. START_SEC mismatch (Model context is wrong).")
        print("   2. Audio normalization mismatch.")

if __name__ == "__main__":
    main()