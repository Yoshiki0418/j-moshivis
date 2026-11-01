#!/bin/bash
# 音声データセット構築をGPU並列で起動 (各GPUに複数シャード)。再開可能 (既存UIDはスキップ)
# 使い方: bash ssvd/tts/run_build_speech.sh [dialogues.jsonl] [outdir] [shards_per_gpu] [gpus]
cd /gpu-server/yamamoto/j-moshivis || exit 1
DIALOGUES=${1:-data/dialogues_merged/dialogues_final.jsonl}
OUTDIR=${2:-data/speech/data_stereo}
PER_GPU=${3:-3}
GPUS=${4:-"0 1"}
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
mkdir -p data/speech/logs
NGPU=$(echo $GPUS | wc -w)
TOTAL=$((PER_GPU * NGPU))
i=0
for g in $GPUS; do
  for _ in $(seq 1 $PER_GPU); do
    CUDA_VISIBLE_DEVICES=$g nohup $PY ssvd/tts/build_speech_dataset.py \
      --dialogues "$DIALOGUES" --outdir "$OUTDIR" --shard $i --num_shards $TOTAL \
      > data/speech/logs/build_shard$i.log 2>&1 &
    echo "shard $i on GPU $g (PID $!)"
    i=$((i+1))
  done
done
