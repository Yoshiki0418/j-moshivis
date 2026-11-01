import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from tqdm import tqdm

# jmvis パッケージからのインポート (環境に合わせて調整してください)
from moshi.models.loaders import get_mimi
from jmoshivis.models.loaders import get_moshi_vis_train
from jmoshivis.datasets.interleaver import InterleavedTokenizer, Interleaver
from jmoshivis.datasets.data_loader import build_data_loader
from jmoshivis.distributed import get_rank, get_world_size
from jmoshivis.config.kyuteye_config import KyuteyeConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig):
    # 分析対象の Duration リスト (秒)
    durations = [10, 100]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(mixed_precision="bf16")
    
    # コンフィグのパスは環境に合わせて修正してください
    config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
    if os.path.exists(config_path):
        kyuteye_config = KyuteyeConfig.from_yml(config_path)
    else:
        print(f"Warning: Config not found at {config_path}. Using default or args.")
        # 必要に応じてフォールバック処理
        return

    # --- Tokenizer / Processor ---
    tokenizer_path = "/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model"
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    if accelerator.is_main_process:
        print("Loading Mimi and MoshiVis...")
    
    # Mimi のロード
    mimi_weight = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.mimi_name,
    )
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # MoshiVis のロード
    print("Start get_moshi_vis_train")
    moshi_vis, image_embedder = get_moshi_vis_train(
        kyuteye_config=kyuteye_config,
        moshivis_weight="/workspace/j-moshivis/model_merged_bf16.safetensors",
        device=device,
        dtype=torch.bfloat16,
        strict=False,
        freeze_backbone=True,
    )

    # PADトークンIDの取得
    pad_token_id = moshi_vis.text_padding_token_id
    print(f"Text Padding Token ID: {pad_token_id}")

    print("\n" + "="*60)
    print("📊 データセット テキストトークン分析開始")
    print("="*60)

    results = []

    # 各 Duration 設定でループ
    for duration_sec in durations:
        if accelerator.is_main_process:
            print(f"\nAnalyzing for Duration: {duration_sec} seconds...")

        # Interleaver の再構築 (duration に依存しない部分は外でも良いが念のため)
        interleaver = Interleaver(
            tokenizer,
            mimi.frame_rate,
            moshi_vis.text_padding_token_id,
            moshi_vis.end_of_text_padding_id,
            moshi_vis.zero_token_id,
            keep_main_only=True,
            device=device,
        )
        
        # InterleavedTokenizer の構築 (duration を更新)
        interleaved_tokenizer = InterleavedTokenizer(
            mimi, interleaver, duration_sec=duration_sec
        )

        target_len = int(mimi.frame_rate * duration_sec)
        
        # DataLoader の構築
        # バッチサイズはVRAMに合わせて調整してください (分析用なので大きめでも可)
        analyze_batch_size = args.train.batch_size
        
        data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=analyze_batch_size,
            seed=args.train.seed,
            rank=get_rank(),
            world_size=get_world_size(),
            is_eval=True,
            image_root=args.data.image_root,
            image_embedder=image_embedder,
            device=device,
            mode="speech",
            text_tokenizer=tokenizer,
            target_len=target_len
        )

        total_valid_tokens = 0  # 分母: -1以外のトークン総数
        total_pad_tokens = 0    # 分子: 有効範囲内のPADトークン数
        total_samples = 0

        # データセット走査
        for batch in tqdm(data_loader, desc=f"Dur {duration_sec}s", disable=not accelerator.is_main_process):
            # batch.codes shape: [B, D, T]
            # Moshiの仕様では Codebook 0 がテキストトークン (Vocabulary size ~32k)
            # Codebook 1-16 が音声トークン (Vocabulary size 2048)
            
            codes = batch.codes.to(device)
            text_codes = codes[:, 0, :]  # [B, T]
            
            # --- 修正箇所: -1 (無効なパディング) を除外して計算 ---
            ignore_token_id = -1

            # -1 以外の部分を有効とするマスクを作成
            valid_mask = (text_codes != ignore_token_id)

            # 分母: 有効なトークン数 (-1 以外)
            num_valid = valid_mask.sum().item()

            # 分子: 有効な範囲内で、かつ PADトークン (ID: 3) であるもの
            # 論理積 (&) を取ることで、万が一 -1 の埋め草部分に 3 が入っていてもカウントしないようにする
            num_pads = ((text_codes == pad_token_id) & valid_mask).sum().item()
            
            total_valid_tokens += num_valid
            total_pad_tokens += num_pads
            total_samples += codes.shape[0]

        # 結果集計
        if total_valid_tokens > 0:
            pad_ratio = (total_pad_tokens / total_valid_tokens) * 100
            result_str = (
                f"Duration: {duration_sec:3}s | "
                f"Valid Tokens: {total_valid_tokens:12,} | "
                f"PAD Tokens: {total_pad_tokens:12,} | "
                f"PAD Ratio: {pad_ratio:.2f}% (excluding -1 padding)"
            )
            results.append(result_str)
            if accelerator.is_main_process:
                print(f"👉 {result_str}")
        else:
            if accelerator.is_main_process:
                print(f"⚠️ Duration {duration_sec}s: No valid tokens found.")

    # 最終レポート
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("📑 最終分析レポート")
        print("="*60)
        for res in results:
            print(res)
        print("="*60)

if __name__ == "__main__":
    main()