#!/bin/bash
# DOCCI+COCO 対話生成のシャード実行 (GPU 2枚並列用)
# 使い方: bash ssvd/dialogue/run_qwen_gen_shard.sh <GPU番号> <meta.jsonl> <出力.jsonl> \
#           [task mode] [shard index] [num shards] [registry dir]
# 例:     nohup bash ssvd/dialogue/run_qwen_gen_shard.sh 0 data/images/docci_coco_shard0.jsonl \
#           data/dialogues/qwen3vl_30b_docci_coco_s0.jsonl > data/dialogues/qwen_s0.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1
GPU="$1"; META="$2"; OUT="$3"; TASK_MODE="${4:-${TASK_MODE:-normal}}"
SHARD_INDEX="${5:-0}"; NUM_SHARDS="${6:-1}"; REGISTRY_DIR="${7:-$(dirname "$OUT")}" 
NORMAL_STRATEGY="${NORMAL_STRATEGY:-standard}"
REGISTRY_SCOPE="${REGISTRY_SCOPE:-task}"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/gpu-server/yamamoto/miniforge3/envs/vlm
export PATH=/gpu-server/yamamoto/miniforge3/envs/vlm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

pass=0
while true; do
  pass=$((pass+1))
  /gpu-server/yamamoto/miniforge3/envs/vlm/bin/python ssvd/dialogue/generate_local_vlm.py \
    --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ --backend vllm \
    --task-mode "$TASK_MODE" \
    --normal-strategy "$NORMAL_STRATEGY" --registry-scope "$REGISTRY_SCOPE" \
    --shard "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
    --meta "$META" \
    --out "$OUT" \
    --registry_dir "$REGISTRY_DIR" \
    --batch_size 64 --gpu_memory_utilization 0.85
  code=$?
  if [ $code -eq 3 ]; then echo "[loop] shard done — finished at $(date)"; break; fi
  if [ $pass -ge 8 ]; then echo "[loop] 8 passes reached — stopping (persistent failures remain)"; break; fi
  echo "[loop] pass $pass ended (exit $code) — retrying failed items in 60s"; sleep 60
done
