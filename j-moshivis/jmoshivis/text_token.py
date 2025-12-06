import os
import torch
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from moshi.models.loaders import get_mimi
from .models.loaders import get_moshi_vis_train
from .datasets.interleaver import InterleavedTokenizer, Interleaver
from .datasets.data_loader import build_data_loader
from .distributed import get_rank, get_world_size
from .config.kyuteye_config import KyuteyeConfig
from .trainer import JmoshiVisTrainer
from torch.optim import AdamW
from jmoshivis.tools import WandBMetricsWriter


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(mixed_precision="bf16")
    kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

    # --- Tokenizer / Processor ---
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

    print("Loading Mimi and MoshiVis...")
    mimi_weight = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.mimi_name,
    )
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    print("Start get_moshi_vis_train")
    moshi_vis, image_embedder = get_moshi_vis_train(
        kyuteye_config=kyuteye_config,
        moshivis_weight="/workspace/j-moshivis/model_merged_bf16.safetensors",
        device=device,
        dtype=torch.bfloat16,
        strict=False,
        freeze_backbone=True,
    )

    interleaver = Interleaver(
        tokenizer,
        mimi.frame_rate,
        moshi_vis.text_padding_token_id,
        moshi_vis.end_of_text_padding_id,
        moshi_vis.zero_token_id,
        keep_main_only=True,
        device=device,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    """
    ───────────────────────────────────────────────
    🧠 DataLoader 構成概要（J-MoshiVis）
    ───────────────────────────────────────────────
    ・本セクションでは、Interleaver（音声・テキスト統合トークナイザ）を介して、
    MoshiVis が期待するフォーマットの学習データを読み込みます。

    ・出力は `Batch` オブジェクトであり、
    以下の2つの主構成を持ちます：

        Batch(
            codes: torch.Tensor,                # 量子化コード列 [B, D, T]
            condition_attributes: Optional[...] # 補助条件（例: 画像特徴やアライメント情報）
        )

    ───────────────────────────────────────────────
    📦 Shape specification
    ───────────────────────────────────────────────
    - codes: Tensor of shape [B, D, T]
        B : Batch size
            └─ 各ステップで並列処理するサンプル数
        D : Depth axis
            └─ Residual Quantizer（RQ）層の数
            例: Mimiでは17層（=17コードブック）を使用
        T : Time axis
            └─ 音声を一定フレーム幅で分割した系列長
            例: 約125ステップ ≒ 10秒前後の音声長

    例:
        >>> batch.codes.shape
        torch.Size([2, 17, 125])
        → 2サンプル / 各サンプル17層 / 125フレームのトークン列
    """
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.train.batch_size,
        seed=args.train.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
        image_root=args.data.image_root,
        image_embedder=image_embedder,
        device=device,
    )

    # --- 1. データローダ生成後に1バッチだけ取り出す ---
    data_iter = iter(data_loader)
    first_batch = next(data_iter)

    sample = next(iter(data_loader))
    print(f"sample: {sample}")

    print("=== Batch object ===")
    print(type(first_batch))
    print(first_batch)

    # --- 2. 中身を要素ごとに確認 ---
    if hasattr(first_batch, "codes"):
        print("\n[Shape] codes:", first_batch.codes.shape)
        if first_batch.condition_attributes:
            print("[Type] condition_attributes:", type(first_batch.condition_attributes))
            print("[Count] len(condition_attributes):", len(first_batch.condition_attributes))
            print("[Sample 0] condition_attributes[0]:", first_batch.condition_attributes[0])
    else:
        # もしBatchクラスでなくlist形式のままならこちら
        print("\nFirst element sample keys:", first_batch[0].keys())
        print("First element sample detail:\n", first_batch[0])

    # # ============================
    # # 🧪 テキストトークン分布の調査
    # # ============================
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from collections import Counter

    # print("\n\n==== Inspecting TEXT TOKEN distribution ====\n")

    # # --- 1. まず codes の形状から text トークンの場所を推測 ---
    # print("codes shape:", first_batch.codes.shape)
    # B, D, T = first_batch.codes.shape
    # print(f"[Info] Batch size={B}, Codebooks={D}, Time steps={T}")

    # # --- 2. ここで text code を取り出す（通常は codebook=0） ---
    # # 必要なら後で確認して修正します
    # text_tokens_batch = first_batch.codes[:, 0, :]  # shape [B, T]
    # print("\n[Sample text tokens for batch 0]:")
    # print(text_tokens_batch[0][:50])  # 先頭50個を表示

    # # flatten
    # text_tokens = text_tokens_batch.reshape(-1).cpu().numpy()

    # print("\nUnique tokens:", len(np.unique(text_tokens)))
    # print("Token min:", np.min(text_tokens))
    # print("Token max:", np.max(text_tokens))

    # # --- 3. 出現頻度をカウント ---
    # counter = Counter(text_tokens)
    # print("\nTop 50 most common tokens:")
    # for tok, cnt in counter.most_common(50):
    #     print(f"{tok}: {cnt}")

    # # --- 4. ヒストグラムで頻度分布を描画 ---
    # plt.figure(figsize=(12,6))
    # plt.hist(text_tokens, bins=200)
    # plt.title("Text Token Frequency Distribution")
    # plt.xlabel("Token ID")
    # plt.ylabel("Count")
    # plt.grid(True, alpha=0.4)

    # # ローカル or コンテナ内表示
    # plt.tight_layout()
    # plt.savefig("text_token_histogram.png")
    # print("\n📊 Saved histogram to 'text_token_histogram.png'")
    # plt.close()

    # # --- 5. 上位20件を棒グラフで可視化 ---
    # top20 = counter.most_common(20)
    # tokens_top20 = [x[0] for x in top20]
    # counts_top20 = [x[1] for x in top20]

    # plt.figure(figsize=(10,6))
    # plt.bar(tokens_top20, counts_top20)
    # plt.title("Top 20 Most Frequent Text Tokens")
    # plt.xlabel("Token ID")
    # plt.ylabel("Frequency")
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig("text_token_top20.png")
    # print("📊 Saved top20 bar plot to 'text_token_top20.png'")
    # plt.close()

    # ============================================
    # 🧪 全データセットの TEXT TOKEN 分布解析（最大5000バッチ）
    # ============================================
    print("\n\n========== GLOBAL TEXT TOKEN ANALYSIS ==========\n")

    import numpy as np
    from collections import Counter
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    global_counter = Counter()
    num_samples = 0
    sample_unique_counts = []

    MAX_ANALYZE = 5000  # ★ ここで上限を設定（5000バッチ）
    print(f"Iterating over dataset for full token analysis (max {MAX_ANALYZE} batches)...")

    for i, batch in enumerate(tqdm(data_loader)):
        if i >= MAX_ANALYZE:     # ★ 解析を5000バッチで打ち切る
            break

        # codes: [B, D, T]
        codes = batch.codes  # Tensor on CUDA
        text_tokens = codes[:, 0, :].detach().cpu().numpy()  # [B, T]

        # flatten
        flat = text_tokens.reshape(-1)

        # update global counter
        global_counter.update(flat.tolist())

        # track unique count per sample
        for row in text_tokens:
            sample_unique_counts.append(len(np.unique(row)))
            num_samples += 1

    print("\n=== RESULT SUMMARY ===")
    print(f"Total samples processed: {num_samples}")
    print(f"Total unique tokens: {len(global_counter)}")
    print(f"Token min: {min(global_counter.keys())}")
    print(f"Token max: {max(global_counter.keys())}")

    print("\nTop 50 most common tokens:")
    for tok, cnt in global_counter.most_common(50):
        print(f"{tok}: {cnt}")

    # ★ 除外対象のトークン
    SPECIAL_TOKENS = {-1, 0, 3, 9}

    # ========================================================
    # 元のヒストグラム（全トークン）
    # ========================================================
    all_tokens = list(global_counter.elements())

    plt.figure(figsize=(12, 6))
    plt.hist(all_tokens, bins=300)
    plt.title("Global Text Token Distribution (ALL TOKENS)")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("global_text_token_hist.png")
    plt.close()
    print("📊 Saved global histogram to 'global_text_token_hist.png'")

    # ========================================================
    # ★ 特殊トークンを除いたヒストグラム（filtered）
    # ========================================================
    filtered_tokens = [t for t in all_tokens if t not in SPECIAL_TOKENS]

    plt.figure(figsize=(12, 6))
    plt.hist(filtered_tokens, bins=300)
    plt.title("Global Text Token Distribution (FILTERED: remove -1,0,3,9)")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("global_text_token_hist_filtered.png")
    plt.close()
    print("📊 Saved filtered histogram to 'global_text_token_hist_filtered.png'")

    # ========================================================
    # トップ20（全体）
    # ========================================================
    top20 = global_counter.most_common(20)
    tokens_top20 = [x[0] for x in top20]
    counts_top20 = [x[1] for x in top20]

    plt.figure(figsize=(10, 6))
    plt.bar(tokens_top20, counts_top20)
    plt.title("Top 20 Most Frequent Text Tokens (ALL TOKENS)")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("global_text_token_top20.png")
    plt.close()
    print("📊 Saved global top20 bar plot to 'global_text_token_top20.png'")

    # ========================================================
    # ★ トップ20（特殊トークン除いた version）
    # ========================================================
    # Counter から除外して再作成
    filtered_counter = {
        tok: cnt for tok, cnt in global_counter.items() if tok not in SPECIAL_TOKENS
    }
    filtered_counter = Counter(filtered_counter)

    top20_filt = filtered_counter.most_common(20)
    tokens_top20_filt = [x[0] for x in top20_filt]
    counts_top20_filt = [x[1] for x in top20_filt]

    plt.figure(figsize=(10, 6))
    plt.bar(tokens_top20_filt, counts_top20_filt)
    plt.title("Top 20 Most Frequent Text Tokens (FILTERED)")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("global_text_token_top20_filtered.png")
    plt.close()
    print("📊 Saved filtered top20 bar plot to 'global_text_token_top20_filtered.png'")


    # Additional statistics
    print("\n=== Sample-level token statistics ===")
    print(f"Avg unique tokens per sample: {np.mean(sample_unique_counts):.2f}")
    print(f"Median unique tokens per sample: {np.median(sample_unique_counts):.2f}")
    print(f"Min unique tokens in sample: {np.min(sample_unique_counts)}")
    print(f"Max unique tokens in sample: {np.max(sample_unique_counts)}")



if __name__ == "__main__":
    main()
