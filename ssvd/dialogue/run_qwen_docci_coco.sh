#!/bin/bash
# DOCCI (14,847) + COCO (123,287) の全画像に Qwen3-VL-30B で対話テキストを生成する自動リトライランナー
# - DOCCI を先に完了させるため meta の先頭に置く
# - レジストリは data/dialogues/*.jsonl 全体 (既存生成分と重複しない)
# 使い方: nohup bash ssvd/dialogue/run_qwen_docci_coco.sh > data/dialogues/qwen_docci_coco.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=/gpu-server/yamamoto/miniforge3/envs/vlm
export PATH=/gpu-server/yamamoto/miniforge3/envs/vlm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

pass=0
while true; do
  pass=$((pass+1))
  /gpu-server/yamamoto/miniforge3/envs/vlm/bin/python ssvd/dialogue/generate_local_vlm.py \
    --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ --backend vllm \
    --meta data/images/docci/metadata.jsonl \
           data/images/coco/metadata_val2017.jsonl \
           data/images/coco/metadata_train2017.jsonl \
    --out data/dialogues/qwen3vl_30b_docci_coco.jsonl \
    --batch_size 16 --gpu_memory_utilization 0.7
  code=$?
  if [ $code -eq 3 ]; then echo "[loop] all images done — finished at $(date)"; break; fi
  if [ $pass -ge 8 ]; then echo "[loop] 8 passes reached — stopping (persistent failures remain)"; break; fi
  echo "[loop] pass $pass ended (exit $code) — retrying failed items in 60s"; sleep 60
done
