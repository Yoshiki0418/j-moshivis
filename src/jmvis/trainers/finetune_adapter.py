"""Adapter fine‑tuning script for J‑MoshiVis
==========================================
This script trains *only* the parameters that have ``requires_grad=True`` in
:class:`jmvis.models.jm_vis_model.JMoshiVisModel` – i.e. the **Cross‑Attention
Gated Adapter** (and optional vision projection layer).  The language backbone
and vision encoder remain frozen, preserving real‑time latency.

Usage (stand‑alone example)
---------------------------
```bash
python -m jmvis.trainers.finetune_adapter \
    --metadata data/ja_captions.jsonl \
    --root /datasets/ja_captions \
    --output ./checkpoints \
    --epochs 3 --batch_size 32 --lr 2e-4
```

For larger scale training consider wrapping with **🤗 Accelerate** config and
running `accelerate launch` to spawn multi‑GPU jobs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from accelerate import Accelerator
except ModuleNotFoundError as e:  # pragma: no cover – accelerate optional
    raise RuntimeError("accelerate>=0.25 is required: pip install accelerate") from e

from jmvis.models.jm_vis_model import JMoshiVisModel
from jmvis.data.datasets import JMVisDataset
from jmvis.data.collate import JMVisCollator
from jmvis.utils.audio_codec import encode as mimi_encode

# ---------------------------------------------------------------------------
# Args / Config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Fine‑tune Adapter for J‑MoshiVis")
    p.add_argument("--metadata", type=str, required=True, help="JSONL metadata file")
    p.add_argument("--root", type=str, required=True, help="Dataset root directory")
    p.add_argument("--output", type=str, default="checkpoints", help="Save dir")
    p.add_argument("--model_name", type=str, default="kyutai/moshiko-pytorch-bf16", help="Base Moshi model")
    p.add_argument("--jm_ckpt", type=str, default=None, help="Path to J‑Moshi checkpoint")
    p.add_argument("--vision_name", type=str, default="google/siglip-so400m-patch14-384")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=-1, help="Override epoch * steps")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, lr: float):  # noqa: D401
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


# Shift‑left labels for causal LM

def lm_shift(input_ids: torch.Tensor, pad_id: int):  # noqa: D401
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    return labels


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()

    # ── Tokenizer / Model ───────────────────────────────────────────────
    model = JMoshiVisModel(
        moshi_name=args.model_name,
        jm_checkpoint=args.jm_ckpt,
        vision_name=args.vision_name,
        freeze_backbone=True,
        freeze_vision=True,
    )
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Dataset / Dataloader ────────────────────────────────────────────
    dataset = JMVisDataset(
        root=args.root,
        metadata=args.metadata,
        tokenizer=tokenizer,
        audio_tokenizer=mimi_encode,
    )
    collator = JMVisCollator(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )

    optimizer = build_optimizer(model, args.lr)

    # Prepare for multiple GPUs
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * math.ceil(len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step = 0
    model.train()
    with tqdm(total=total_steps, unit="step") as pbar:
        for epoch in range(args.epochs):
            for batch in loader:
                # Forward ---------------------------------------------------
                logits = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                )
                labels = lm_shift(batch["input_ids"], pad_id)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=pad_id,
                )

                # Backward --------------------------------------------------
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Logging ---------------------------------------------------
                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
                pbar.update(1)
                global_step += 1

                # Save adapter only ----------------------------------------
                if (
                    args.save_every > 0
                    and global_step % args.save_every == 0
                    and accelerator.is_main_process
                ):
                    save_path = out_dir / f"adapter_step{global_step}.pt"
                    state_dict: Dict[str, torch.Tensor] = {
                        k: v.cpu()
                        for k, v in model.adapter.state_dict().items()
                    }
                    if hasattr(model, "vision_proj") and any(p.requires_grad for p in model.vision_proj.parameters()):
                        state_dict.update({f"vision_proj.{k}": v.cpu() for k, v in model.vision_proj.state_dict().items()})
                    torch.save(state_dict, save_path)
                    print(f"✅ Saved checkpoint → {save_path}")

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_ckpt = out_dir / "adapter_final.pt"
        torch.save(
            {k: v.cpu() for k, v in model.adapter.state_dict().items()},
            final_ckpt,
        )
        print(f"🎉 Training completed – final adapter saved to {final_ckpt}")


if __name__ == "__main__":
    main()
