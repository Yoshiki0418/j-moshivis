#!/bin/bash
# p_concat 用の非画像対話を、単一の物理GPU上で4シャード並列合成する。
set -euo pipefail

cd /gpu-server/yamamoto/j-moshivis || exit 1

DIALOGUES=${1:-data/dialogues_nonvisual/claude_sonnet5_nonvisual_tts_v1.jsonl}
OUTDIR=${2:-data/speech_nonvisual_v1/data_stereo}
GPU=${3:-1}
NUM_SHARDS=${4:-4}
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
LOGDIR=data/speech_nonvisual_v1/logs

if [ ! -f "$DIALOGUES" ]; then
  echo "missing input: $DIALOGUES" >&2
  exit 2
fi

mkdir -p "$OUTDIR" "$LOGDIR"
pids=()

terminate_children() {
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u ssvd/tts/build_speech_dataset.py \
    --dialogues "$DIALOGUES" \
    --outdir "$OUTDIR" \
    --allow-no-image \
    --shard "$shard" \
    --num_shards "$NUM_SHARDS" \
    > "$LOGDIR/build_shard${shard}.log" 2>&1 &
  pids+=("$!")
  echo "shard $shard/$NUM_SHARDS on physical GPU $GPU (PID $!)"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
echo "all shards exited with aggregate status $status at $(date)"
exit "$status"
