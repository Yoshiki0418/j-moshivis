#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_ROOT=/gpu-server/yamamoto/j-moshivis/run_snapshots/llmjp_v3_run1_20260923
REPO_ROOT=/gpu-server/yamamoto/j-moshivis
RUN_LOG_ROOT="$REPO_ROOT/logs/experiments/llmjp_v3_run1"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_PORT="${MASTER_PORT:-29541}"
export WANDB_DIR="$RUN_LOG_ROOT/wandb"
export PYTHONDONTWRITEBYTECODE=1

RESUME_FROM="${JMOSHIVIS_RESUME_FROM:-null}"
HYDRA_RUN_DIR="${JMOSHIVIS_HYDRA_RUN_DIR:-$RUN_LOG_ROOT/hydra}"

mkdir -p "$WANDB_DIR" "$HYDRA_RUN_DIR"

exec bash "$SNAPSHOT_ROOT/run_train.sh" \
  paths.kyuteye_config="$SNAPSHOT_ROOT/j-moshivis/configs/j-moshi-vis.yaml" \
  paths.tokenizer="$REPO_ROOT/models/j-moshi/tokenizer_spm_32k_3.model" \
  paths.moshivis_weight="$REPO_ROOT/models/model_merged_llmjp_bf16.safetensors" \
  data.train_data="$REPO_ROOT/data/speech_v3/train.jsonl" \
  data.eval_data="$REPO_ROOT/data/speech_v3/eval.jsonl" \
  data.asset_manifest="$REPO_ROOT/data/speech_v3/asset_manifest.v1.jsonl" \
  data.asset_manifest_verification=fast \
  data.shuffle=true \
  data.speech_ratio=0.25 \
  data.speech_chunk_policy=one_random_crop_per_row \
  data.speech_decode_workers=4 \
  train.batch_size=1 \
  train.target_batch_size=16 \
  train.learning_rate=5.0e-5 \
  train.weight_decay=0.0 \
  train.warmup_steps=500 \
  train.max_steps=20000 \
  train.min_lr_ratio=0.1 \
  train.save_every=1000 \
  train.keep_last=20 \
  train.checkpoint_dir="$REPO_ROOT/checkpoints/llmjp_v3_run1" \
  train.resume_from="$RESUME_FROM" \
  train.log_interval=10 \
  train.seed=42 \
  train.num_workers=0 \
  trainer.text_padding_weight=0.5 \
  trainer.first_codebook_weight_multiplier=3.0 \
  depformer.train=false \
  gate.freeze=false \
  gate.learning_rate=1.0e-4 \
  cross_attention.visual_adapter_init=inherit_visual_adapter \
  hydra.run.dir="$HYDRA_RUN_DIR"
