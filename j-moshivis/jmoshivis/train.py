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

    if accelerator.is_main_process:
        print("Loading Mimi and MoshiVis...")
    mimi_weight = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.mimi_name,
    )
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # Config からDepformer/Gate/CrossAttention設定を取得
    depformer_config = dict(args.get("depformer", {})) if "depformer" in args else None
    gate_config = dict(args.get("gate", {})) if "gate" in args else None
    cross_attention_config = (
        dict(args.get("cross_attention", {})) if "cross_attention" in args else None
    )

    print("Start get_moshi_vis_train")
    print(f"  Depformer config: {depformer_config}")
    print(f"  Gate config: {gate_config}")
    print(f"  CrossAttention config: {cross_attention_config}")

    moshi_vis, image_embedder = get_moshi_vis_train(
        kyuteye_config=kyuteye_config,
        moshivis_weight="/workspace/j-moshivis/model_merged_bf16.safetensors",
        device=device,
        dtype=torch.bfloat16,
        strict=False,
        freeze_backbone=True,
        depformer_config=depformer_config,
        gate_config=gate_config,
        cross_attention_config=cross_attention_config,
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
    target_len = int(mimi.frame_rate * args.duration_sec)
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
        mode="mixed",
        text_tokenizer=tokenizer,
        target_len=target_len
    )

    # # --- 1. データローダ生成後に1バッチだけ取り出す ---
    # data_iter = iter(data_loader)
    # first_batch = next(data_iter)

    # print("=== Batch object ===")
    # print(type(first_batch))
    # print(first_batch)

    # # --- 2. 中身を要素ごとに確認 ---
    # if hasattr(first_batch, "codes"):
    #     print("\n[Shape] codes:", first_batch.codes.shape)
    #     if first_batch.condition_attributes:
    #         print("[Type] condition_attributes:", type(first_batch.condition_attributes))
    #         print("[Count] len(condition_attributes):", len(first_batch.condition_attributes))
    #         print("[Sample 0] condition_attributes[0]:", first_batch.condition_attributes[0])
    # else:
    #     # もしBatchクラスでなくlist形式のままならこちら
    #     print("\nFirst element sample keys:", first_batch[0].keys())
    #     print("First element sample detail:\n", first_batch[0])

    cross_attn_params = []
    gate_params = []
    depformer_params = []
    # other_params は今回空になるのが理想ですが、念のため残します
    other_params = []

    for name, p in moshi_vis.named_parameters():
        if not p.requires_grad:
            continue
        # Gate パラメータ (実際の名前は cross_attention.gate.alpha.*)
        if "gate" in name and "alpha" in name:
            gate_params.append(p)
        # Depformer パラメータ
        elif "depformer" in name or "audio_linears" in name:
            depformer_params.append(p)
        # Cross-Attention パラメータ

        # 【修正2】 norm_cross も CrossAttention グループに含める
        elif "cross_attention" in name or "xa" in name or "norm_cross" in name:
            cross_attn_params.append(p)

        else:
            # ここに出るものがなければOK
            print("[WARN] Unexpected trainable param:", name)
            other_params.append(p)

    embedder_params = [p for p in image_embedder.parameters() if p.requires_grad]

    print(f"Trainable params: CrossAttn={len(cross_attn_params)}, Gate={len(gate_params)}, Depformer={len(depformer_params)}, Embedder={len(embedder_params)}")

    if accelerator.is_main_process:
        def print_trainable_parameters(model, model_name="Model"):
            print(f"\n=== Trainable Parameters in {model_name} ===")
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    total_params += num_params
                    print(f"{name}: {num_params:,} params | Shape: {list(param.shape)}")
            print(f"--- Total Trainable Params in {model_name}: {total_params:,} ---\n")
            return total_params

        # MoshiVis本体の学習対象パラメータを表示
        moshi_params = print_trainable_parameters(moshi_vis, "MoshiVis (Adapters)")

        # ImageEmbedderの学習対象パラメータを表示
        embedder_params_count = print_trainable_parameters(image_embedder, "ImageEmbedder (Projection)")

        print(f"🔥 Grand Total Trainable Parameters: {moshi_params + embedder_params_count:,}")

        # もし other_params に何か残っていたら、それも学習対象に加えるべきですが、
        # 上記の修正で norm_cross は CrossAttn に入るため、基本的には空になるはずです。

    # オプティマイザのパラメータグループを構築
    param_groups = [
        {"params": cross_attn_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": embedder_params,   "lr": 1e-4, "weight_decay": 0.0},
    ]

    # Gate パラメータ (凍結されていなければ追加)
    if gate_params:
        gate_lr = gate_config.get("learning_rate", 1e-5) if gate_config else 1e-5
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})
        print(f"🔧 Gate params added to optimizer (lr={gate_lr})")

    # Depformer パラメータ (設定されていれば追加)
    if depformer_params:
        depformer_lr = depformer_config.get("learning_rate", 5e-6) if depformer_config else 5e-6
        param_groups.append({"params": depformer_params, "lr": depformer_lr, "weight_decay": 0.0})
        print(f"🔧 Depformer params added to optimizer (lr={depformer_lr})")

    optimizer = torch.optim.AdamW(param_groups, fused=True)

    moshi_vis.forward = moshi_vis.forward_speech

    if accelerator.is_main_process:
        writer = WandBMetricsWriter(project_name="J-MoshiVis-Training",
                                    model_name="j-moshivis")

    # DDP準備
    moshi_vis, image_embedder, optimizer, data_loader = accelerator.prepare(
        moshi_vis, image_embedder, optimizer, data_loader
    )

    writer = None

    # メインプロセスのみ WandB を起動し、writer に代入
    if accelerator.is_main_process:
        writer = WandBMetricsWriter(
            project_name="J-MoshiVis-Training",
            model_name="j-moshivis"
        )

    # --- Trainer Setup ---
    trainer = JmoshiVisTrainer(moshi_vis, optimizer, device, args.trainer, accelerator, image_embedder=image_embedder, writer=writer, tokenizer=tokenizer)

    # --- Training ---
    epochs = 3
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(data_loader, epoch)

    # --- Save ---
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(moshi_vis.state_dict(), f"{save_dir}/jmoshivis_dummy.pt")
    print(f"💾 Saved model to {save_dir}/jmoshivis_dummy.pt")


if __name__ == "__main__":
    main()
