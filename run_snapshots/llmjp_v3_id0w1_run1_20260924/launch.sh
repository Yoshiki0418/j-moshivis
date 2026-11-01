#!/bin/bash
set -euo pipefail

snapshot_root="$(cd "$(dirname "$0")" && pwd)"
workspace_root="/gpu-server/yamamoto/j-moshivis"

cd "$snapshot_root/j-moshivis"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$snapshot_root/j-moshivis${PYTHONPATH:+:$PYTHONPATH}"

exec "$workspace_root/.venv-train/bin/accelerate" launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --main_process_port 29551 \
  -m jmoshivis.train \
  paths.kyuteye_config="$snapshot_root/j-moshivis/configs/j-moshi-vis.yaml" \
  trainer.text_padding_weight=0.5 \
  trainer.end_of_text_padding_weight=1.0 \
  train.batch_size=1 \
  train.target_batch_size=16 \
  train.max_steps=20000 \
  train.save_every=1000 \
  train.keep_last=20 \
  train.checkpoint_dir="$workspace_root/checkpoints/llmjp_v3_id0w1_run1" \
  train.resume_from=null \
  train.seed=42
