import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.loaders import get_moshi_vis
from moshi.models.loaders import get_mimi
from jmoshivis.data.dataset import JMVisionSpeechDataset   # <- あなたのデータクラスに置き換え
from jmoshivis.data.collate import collate_fn             # <- collate関数

from huggingface_hub import hf_hub_download


def main():
    # ===== 1. 設定 =====
    cfg = KyuteyeConfig.from_yml("configs/moshi-vis.yaml")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # ===== 2. 重みロード =====
    jmoshi_weight = "/workspace/j-moshi/jmoshi.safetensors"   # J-Moshi学習済み重み
    moshi_vis_weight = None  # Cross-Attnを新規初期化したいのでNone

    mimi_weight = hf_hub_download(
        repo_id="kyutai/moshika-vis-pytorch-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors"
    )

    moshi_vis, image_embedder = get_moshi_vis(
        cfg,
        moshi_weight=moshi_vis_weight,
        device=device,
        dtype=dtype,
    )
    mimi = get_mimi(mimi_weight, device)

    # ===== 3. J-Moshi の state_dict を流し込み =====
    base_sd = torch.load(jmoshi_weight, map_location="cpu")
    missing, unexpected = moshi_vis.moshi.load_state_dict(base_sd, strict=False)
    print("Missing keys:", missing[:10], "...")   # 視覚周りが出るのはOK
    print("Unexpected keys:", unexpected)

    # ===== 4. パラメータ凍結 =====
    for p in moshi_vis.moshi.parameters():
        p.requires_grad = False
    for p in image_embedder.parameters():
        p.requires_grad = False
    for p in moshi_vis.vision_adapters.parameters():
        p.requires_grad = True

    # ===== 5. Dataset & Dataloader =====
    train_dataset = JMVisionSpeechDataset(
        data_root="/workspace/data/train.jsonl",   # 画像 + 音声トークンのペア
        mimi=mimi,
        image_transform=image_embedder.transform, # 224x224 + normalize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    # ===== 6. Optimizer =====
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, moshi_vis.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # ===== 7. 損失関数 =====
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 音声token用
    gate_loss = nn.BCEWithLogitsLoss()                  # 発話開始制御

    # ===== 8. 学習ループ =====
    moshi_vis.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

    for epoch in range(cfg.train.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch: { "codes": [B, 8, T], "images": [B, 3, 224, 224], "labels": [B, 8, T], "gate": [B, T] }
            codes = batch["codes"].to(device)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            gate_targets = batch["gate"].to(device)

            with torch.cuda.amp.autocast(dtype=dtype):
                img_out = image_embedder(images)
                ca_src = img_out["cross_attention_src"]
