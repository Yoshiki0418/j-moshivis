#!/bin/bash
# 非画像 p_concat 音声を2 GPU・6シャードでアライメントし、全件refine/verifyする。
set -euo pipefail

cd /gpu-server/yamamoto/j-moshivis || exit 1

DATADIR=${1:-data/speech_nonvisual_v1/data_stereo}
LOGDIR=${2:-data/speech_nonvisual_v1/logs}
EXPECTED=${EXPECTED:-11886}
NUM_SHARDS=6
MAX_PASSES=3
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python

mkdir -p "$LOGDIR"
pids=()

terminate_children() {
  local exit_code=${1:-130}
  trap - HUP INT TERM
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap 'terminate_children 129' HUP
trap 'terminate_children 130' INT
trap 'terminate_children 143' TERM

# 同じデータセットへ複数ランナーが同時に書き込む事故を防ぐ。
exec 9>"$LOGDIR/alignment.lock"
if ! flock -n 9; then
  echo "another nonvisual alignment runner is already active" >&2
  exit 4
fi

count_artifact() {
  local name=$1
  find "$DATADIR" -mindepth 2 -maxdepth 2 -name "$name" | wc -l
}

launch_pass() {
  local pass=$1
  local status=0
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu=0
    if [ "$shard" -ge 3 ]; then
      gpu=1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u ssvd/tts/annotate_alignments.py \
      --datadir "$DATADIR" \
      --shard "$shard" \
      --num_shards "$NUM_SHARDS" \
      --model turbo \
      > "$LOGDIR/align_pass${pass}_shard${shard}.log" 2>&1 &
    pids+=("$!")
    echo "pass $pass shard $shard/$NUM_SHARDS on physical GPU $gpu (PID $!)"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  pids=()
  return "$status"
}

total_dirs=$(find "$DATADIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$total_dirs" -ne "$EXPECTED" ]; then
  echo "expected $EXPECTED dialogue directories, found $total_dirs" >&2
  exit 2
fi

for pass in $(seq 1 "$MAX_PASSES"); do
  before=$(count_artifact whisper_words.json)
  echo "starting pass $pass: whisper=$before/$EXPECTED at $(date)"
  if ! launch_pass "$pass"; then
    echo "one or more workers exited non-zero in pass $pass" >&2
  fi

  whisper=$(count_artifact whisper_words.json)
  asr=$(count_artifact asr_check.json)
  refined=$(count_artifact assistant_dialogue.refined.json)
  echo "pass $pass complete: whisper=$whisper asr=$asr refined=$refined expected=$EXPECTED at $(date)"
  if [ "$whisper" -eq "$EXPECTED" ] && [ "$asr" -eq "$EXPECTED" ] && [ "$refined" -eq "$EXPECTED" ]; then
    break
  fi
  if [ "$whisper" -le "$before" ] && [ "$pass" -lt "$MAX_PASSES" ]; then
    echo "no whisper progress in pass $pass; retrying partial outputs once more" >&2
  fi
done

whisper=$(count_artifact whisper_words.json)
asr=$(count_artifact asr_check.json)
refined=$(count_artifact assistant_dialogue.refined.json)
if [ "$whisper" -ne "$EXPECTED" ] || [ "$asr" -ne "$EXPECTED" ] || [ "$refined" -ne "$EXPECTED" ]; then
  echo "alignment incomplete after $MAX_PASSES passes: whisper=$whisper asr=$asr refined=$refined" >&2
  exit 3
fi

echo "running final refine sweep at $(date)"
"$PY" -u ssvd/tts/refine_alignments.py --datadir "$DATADIR" \
  > "$LOGDIR/refine_final.log" 2>&1

echo "running alignment verification at $(date)"
PYTHONPATH=j-moshivis .venv-train/bin/python -u ssvd/tts/verify_alignments.py \
  --datadir "$DATADIR" --expected "$EXPECTED" --strict \
  > "$LOGDIR/verify_final.log" 2>&1

echo "nonvisual alignment and verification complete at $(date)"
