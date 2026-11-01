#!/bin/bash
# Follow the adaptive-normal dialogues generated for counting_v4 and build
# speech + alignment on one physical GPU.  TTS and Whisper run in separate
# phases so they never contend for GPU memory.  Every stage is restartable.

set -uo pipefail
cd /gpu-server/yamamoto/j-moshivis || exit 1

GPU=${1:-1}
OUTDIR=${2:-data/speech_counting_v1/data_stereo}
INPUT0=data/dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl
INPUT1=data/dialogues_counting/full/qwen3vl_30b_normal_s1.jsonl
LOGDIR=data/speech_counting_v1/logs
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
STOP_FILE=data/speech_counting_v1/STOP_NORMAL_PIPELINE
MAX_STAGNANT_PASSES=3

mkdir -p "$OUTDIR" "$LOGDIR"

count_inputs() {
  local total=0 count input
  for input in "$INPUT0" "$INPUT1"; do
    if [ -f "$input" ]; then
      count=$(wc -l < "$input")
      total=$((total + count))
    fi
  done
  echo "$total"
}

count_normal_artifact() {
  local artifact=$1
  find "$OUTDIR" -mindepth 2 -maxdepth 2 -type f -name "$artifact" -print |
    awk 'index($0, "__task_") == 0 { n++ } END { print n + 0 }'
}

generator_running() {
  pgrep -f 'generate_local_vlm[.]py.*data/dialogues_counting/full/qwen3vl_30b_normal_s[01][.]jsonl' >/dev/null
}

cycle=0
stagnant=0
last_signature=""

while true; do
  if [ -f "$STOP_FILE" ]; then
    echo "[$(date)] stop file found: $STOP_FILE"
    exit 0
  fi

  cycle=$((cycle + 1))
  snapshot_inputs=$(count_inputs)
  echo "[$(date)] cycle=$cycle input_snapshot=$snapshot_inputs: TTS start on physical GPU $GPU"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u ssvd/tts/build_speech_dataset.py \
    --dialogues "$INPUT0" --outdir "$OUTDIR" --shard 0 --num_shards 1 \
    >> "$LOGDIR/normal_build_s0.log" 2>&1 &
  tts0=$!
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u ssvd/tts/build_speech_dataset.py \
    --dialogues "$INPUT1" --outdir "$OUTDIR" --shard 0 --num_shards 1 \
    >> "$LOGDIR/normal_build_s1.log" 2>&1 &
  tts1=$!

  wait "$tts0"; tts0_status=$?
  wait "$tts1"; tts1_status=$?
  echo "[$(date)] cycle=$cycle TTS done statuses=$tts0_status,$tts1_status: alignment start"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u ssvd/tts/annotate_alignments.py \
    --datadir "$OUTDIR" --shard 0 --num_shards 2 --model turbo \
    >> "$LOGDIR/normal_align_shard0.log" 2>&1 &
  align0=$!
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u ssvd/tts/annotate_alignments.py \
    --datadir "$OUTDIR" --shard 1 --num_shards 2 --model turbo \
    >> "$LOGDIR/normal_align_shard1.log" 2>&1 &
  align1=$!

  wait "$align0"; align0_status=$?
  wait "$align1"; align1_status=$?
  "$PY" -u ssvd/tts/refine_alignments.py --datadir "$OUTDIR" \
    >> "$LOGDIR/normal_refine.log" 2>&1
  refine_status=$?

  current_inputs=$(count_inputs)
  speech_count=$(count_normal_artifact stereo_dialogue.wav)
  aligned_count=$(count_normal_artifact assistant_dialogue.refined.json)
  signature="$current_inputs:$speech_count:$aligned_count"
  running=0
  generator_running && running=1
  echo "[$(date)] cycle=$cycle statuses align=$align0_status,$align1_status refine=$refine_status; input=$current_inputs speech=$speech_count aligned=$aligned_count generator_running=$running"

  if [ "$running" -eq 0 ] && [ "$current_inputs" -eq "$snapshot_inputs" ] && \
     [ "$speech_count" -ge "$current_inputs" ] && \
     [ "$aligned_count" -ge "$speech_count" ]; then
    echo "[$(date)] counting normal speech/alignment pipeline finished"
    exit 0
  fi

  if [ "$running" -eq 0 ] && [ "$signature" = "$last_signature" ]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  if [ "$stagnant" -ge "$MAX_STAGNANT_PASSES" ]; then
    echo "[$(date)] stopped after $stagnant no-progress passes; inspect per-stage logs"
    exit 1
  fi
  last_signature="$signature"
  sleep 60
done
