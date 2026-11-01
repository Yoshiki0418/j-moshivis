#!/bin/bash
# J-MoshiVis リアルタイムWebデモ起動 (学習済みチェックポイントで対話)
# 使い方: bash run_web_demo.sh [--checkpoint <path>] [追加引数]
# アクセス: 手元PCから  ssh -L 8998:localhost:8998 <このサーバー>  して http://localhost:8998
#           (マイク使用のため localhost アクセスが必要。http の非localhostはブラウザがマイクを拒否する)
set -e
cd "$(dirname "$0")/j-moshivis"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}  # GPU1既定 (GPU0は他ユーザーのジョブと競合しやすい)
export TORCHDYNAMO_DISABLE=1
export NO_TORCH_COMPILE=1
export OMP_NUM_THREADS=8
export PYTHONPATH="$(pwd):$PYTHONPATH"
exec ../.venv-train/bin/python run_realtime_demo.py \
  --config  /gpu-server/yamamoto/j-moshivis/j-moshivis/configs/j-moshi-vis.yaml \
  --tokenizer /gpu-server/yamamoto/j-moshivis/models/j-moshi/tokenizer_spm_32k_3.model \
  --static  /gpu-server/yamamoto/j-moshivis/j-moshivis/web \
  --checkpoint /gpu-server/yamamoto/j-moshivis/checkpoints/depth_audio_timing_pilot_20261012/final_step_200.safetensors \
  --host 127.0.0.1 --port 8998 "$@"
