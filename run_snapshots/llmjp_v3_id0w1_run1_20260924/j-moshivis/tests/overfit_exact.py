import torch
import hydra
from omegaconf import DictConfig
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from safetensors.torch import load_file
import sys
import os
import torchaudio

# プロジェクトルートにパスを通す
sys.path.append(os.getcwd())

from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis_train
from jmoshivis.models.moshivis import MoshiVisGen, MoshiVis
from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.datasets.interleaver import InterleavedTokenizer, Interleaver
from jmoshivis.datasets.data_loader import build_data_loader
from jmoshivis.config.kyuteye_config import KyuteyeConfig

# =================================================================
# 🛠️ ロード関数
# =================================================================
def load_moshi_vis_robust(config, checkpoint_path, device, dtype):
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

def safe_decode(mimi, codes):
    """
    Mimiでのデコード前に、コードブックの範囲外(Padding等)の値を
    安全な値(0)に置換するヘルパー関数。
    codes: [B, K, T]
    """
    # Mimiのコードブックサイズ (通常2048)
    # mimi.quantizer.bins などから取れるが、固定で2048と仮定しても安全
    vocab_size = 2048
    
    # 範囲外の値を検出
    invalid_mask = (codes < 0) | (codes >= vocab_size)
    
    if invalid_mask.any():
        print(f"   ⚠️ Warning: Found {invalid_mask.sum().item()} invalid codes (pad/-1). Replaced with 0 for decoding.")
        # クローンして書き換え（元のテンソルは変更しない）
        codes = codes.clone()
        codes[invalid_mask] = 0 # 0番目のコード（無音に近いもの）に置換
        
    return mimi.decode(codes)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig):
    print("🚀 Starting Exact Verification (Safe Decode Mode)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 1. Config & Tokenizer
    kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

    # 2. Mimi Load (デコード用)
    mimi_weight = hf_hub_download(repo_id=args.repo_id, filename=args.mimi_name)
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()

    # 3. MoshiVis Load (学習済みチェックポイント)
    checkpoint_path = "/workspace/j-moshivis/checkpoints/step_6000.safetensors" 
    moshi_vis, image_embedder = load_moshi_vis_robust(kyuteye_config, checkpoint_path, device, dtype)
    moshi_vis.eval()
    image_embedder.eval()

    # 4. Data Loader
    print("🏗️ Building Data Loader...")
    interleaver = Interleaver(
        tokenizer,
        mimi.frame_rate,
        moshi_vis.lm_model.text_padding_token_id,
        moshi_vis.lm_model.end_of_text_padding_id,
        moshi_vis.lm_model.zero_token_id,
        keep_main_only=True,
        device=device,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )
    target_len = int(mimi.frame_rate * args.duration_sec)
    
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=1, 
        seed=args.train.seed,
        rank=0,
        world_size=1,
        is_eval=False,
        image_root=args.data.image_root,
        image_embedder=image_embedder,
        device=device,
        mode="speech",
        text_tokenizer=tokenizer,
        target_len=target_len
    )

    # 5. Get Batch & Forward
    print("📦 Fetching batch & Running Forward...")
    batch = next(iter(data_loader))
    codes = batch.codes.to(device) # Shape: [Batch, 17, Time]
    
    # 画像Embed準備
    image_input = None
    if isinstance(batch.condition_attributes, list):
        tensors = []
        for ca in batch.condition_attributes:
            if hasattr(ca, "tensor") and "image" in ca.tensor:
                tensors.append(ca.tensor["image"].tensor.to(device).to(dtype))
        if tensors:
            image_input = torch.cat(tensors, dim=0)

    with torch.no_grad():
        cross_attention_src = None
        if image_input is not None:
            embedder_out = image_embedder(image_input)
            cross_attention_src = embedder_out["cross_attention_src"]

        # Forward
        outputs = moshi_vis.lm_model.forward_speech(
            input_ids=codes,
            cross_attention_src=cross_attention_src
        )
        text_logits = outputs["text_logits"]
        audio_logits = outputs["audio_logits"]

    # =================================================================
    # 🎯 TARGET EXTRACTION
    # =================================================================
    model = moshi_vis.lm_model 
    audio_offset = model.audio_offset # 通常 1
    
    # dep_q取得
    if hasattr(model, "dep_q"):
        dep_q = model.dep_q
    else:
        dep_q = audio_logits.shape[1]
    
    print(f"ℹ️ Model Config: audio_offset={audio_offset}, dep_q={dep_q}")

    # 正解データの切り出し
    text_target = codes[:, :audio_offset] 
    audio_target = codes[:, audio_offset : audio_offset + dep_q]

    print(f"📊 Shapes Check:")
    print(f"   Input Codes : {codes.shape}")
    print(f"   Text Target : {text_target.shape}")
    print(f"   Audio Target: {audio_target.shape} (Ground Truth)")
    print(f"   Audio Logits: {audio_logits.shape} (Prediction)")

    # =================================================================
    # 🔍 A. テキスト検証
    # =================================================================
    print("\n📝 [Text Validation]")
    pred_text_ids = torch.argmax(text_logits, dim=-1)[0, 0]
    gt_text_ids = text_target[0, 0]
    
    valid_gt_text = [t.item() for t in gt_text_ids if t >= 0]
    valid_pred_text = [p.item() for p in pred_text_ids if p >= 0]
    
    print(f"   Target: {tokenizer.decode(valid_gt_text)[:100]}...")
    print(f"   Pred  : {tokenizer.decode(valid_pred_text)[:100]}...")

    # =================================================================
    # 🔊 B. 音声検証 (安全なデコード)
    # =================================================================
    print("\n🔊 [Audio Validation]")
    
    # 1. 予測 (Prediction)
    pred_audio_codes = torch.argmax(audio_logits, dim=-1) # [B, dep_q, T]
    
    # 2. 一致率計算
    match_count = (pred_audio_codes == audio_target).sum().item()
    total_tokens = audio_target.numel()
    print(f"   ✅ Audio Code Match Rate: {match_count / total_tokens:.2%}")

    # 3. Mimiでデコード (WAV化) - ★ここを修正しました
    print("   Decoding waveforms (with padding handling)...")
    
    # --- Ground Truth (正解) のデコード ---
    with torch.no_grad():
        # safe_decodeを使ってパディング(-1など)を0に置換してデコード
        wav_gt = safe_decode(mimi, audio_target) 
    
    path_gt = "val_ground_truth.wav"
    torchaudio.save(path_gt, wav_gt[0].cpu().float(), mimi.sample_rate)
    print(f"   💾 Saved Ground Truth: {path_gt}")

    # --- Prediction (予測) のデコード ---
    with torch.no_grad():
        wav_pred = safe_decode(mimi, pred_audio_codes)
        
    path_pred = "val_prediction_tf.wav"
    torchaudio.save(path_pred, wav_pred[0].cpu().float(), mimi.sample_rate)
    print(f"   💾 Saved Prediction  : {path_pred}")

    print("\n💡 確認方法:")
    print("   1. 'val_ground_truth.wav' -> アシスタントの声が正しく再生されるか？")
    print("   2. 'val_prediction_tf.wav' -> 正解に近いか？")

if __name__ == "__main__":
    main()