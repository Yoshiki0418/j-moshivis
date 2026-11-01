#!/bin/bash
# 本番学習の起動スクリプト (2×A6000, accelerate DDP + 勾配累積)
# 使い方: bash run_train.sh [hydra overrides...]   例: bash run_train.sh train.max_steps=3
set -e
cd "$(dirname "$0")/j-moshivis"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export WANDB_MODE=offline
export TORCHDYNAMO_DISABLE=1         # triton JIT が python3.13-dev 不足で失敗するため eager 実行
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):$PYTHONPATH"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
exec ../.venv-train/bin/accelerate launch --num_processes "$NPROC" --mixed_precision bf16 \
  --main_process_port "${MASTER_PORT:-29511}" -m jmoshivis.train "$@"
