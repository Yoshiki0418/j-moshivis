#!/bin/bash
# Qwen3-VL-30B 対話生成の自動リトライランナー (パース失敗分を再試行、全件完了で停止)
cd /gpu-server/yamamoto/j-moshivis || exit 1
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/gpu-server/yamamoto/miniforge3/envs/vlm
export PATH=/gpu-server/yamamoto/miniforge3/envs/vlm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

pass=0
while true; do
  pass=$((pass+1))
  /gpu-server/yamamoto/miniforge3/envs/vlm/bin/python ssvd/dialogue/generate_local_vlm.py \
    --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ --backend vllm \
    --meta data/images/filtered_phase2.jsonl data/images/filtered_phase3.jsonl \
    --out data/dialogues/qwen3vl_30b_main.jsonl \
    --batch_size 16 --gpu_memory_utilization 0.7
  code=$?
  if [ $code -eq 3 ]; then echo "[loop] all images done — finished at $(date)"; break; fi
  if [ $pass -ge 5 ]; then echo "[loop] 5 passes reached — stopping (persistent failures remain)"; break; fi
  echo "[loop] pass $pass ended (exit $code) — retrying failed items in 60s"; sleep 60
done
