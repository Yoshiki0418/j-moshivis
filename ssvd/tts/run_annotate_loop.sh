#!/bin/bash
# アライメント付与を構築と並行して回すランナー。新しく出来た UID を拾いながら繰り返し、
# 構築プロセスが終了し未処理が無くなったら停止する。各パス後に旧形式の refined.json を最新形式へ更新する。
# 使い方: bash ssvd/tts/run_annotate_loop.sh <shard> <num_shards> <gpu> [datadir] [model]
cd /gpu-server/yamamoto/j-moshivis || exit 1
SHARD=$1; NUM=$2; GPU=$3
DATADIR=${4:-data/speech/data_stereo}
MODEL=${5:-turbo}
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
while true; do
  CUDA_VISIBLE_DEVICES=$GPU $PY ssvd/tts/annotate_alignments.py --datadir "$DATADIR" --shard $SHARD --num_shards $NUM --model $MODEL 2>&1 | grep -vE "it/s\]|Warning|warn"
  # 旧形式 (v1 等) が残っていれば最新形式へ再精緻化 (whisper_words.json から数秒で再生成)
  [ "$SHARD" -eq 0 ] && $PY ssvd/tts/refine_alignments.py --datadir "$DATADIR" 2>&1 | grep -v Warning
  building=$(pgrep -fc "build_speech_dataset[.]py")
  pending=$(find "$DATADIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
  done_n=$(find "$DATADIR" -mindepth 2 -maxdepth 2 -name whisper_words.json | wc -l)
  echo "[annotate-loop $SHARD] pass done: $done_n/$pending annotated, build procs=$building ($(date))"
  if [ "$building" -eq 0 ] && [ "$done_n" -ge "$pending" ]; then echo "[annotate-loop $SHARD] finished"; break; fi
  sleep 300
done
