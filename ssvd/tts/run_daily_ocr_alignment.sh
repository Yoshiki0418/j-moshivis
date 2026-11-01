#!/usr/bin/env bash
# Follow completed stereo speech on GPU0, independent of GPU1 TTS.
set -uo pipefail
cd /gpu-server/yamamoto/j-moshivis || exit 1
GPU=${1:-0}
WORKERS=${2:-6}
[[ "$GPU" =~ ^[0-9]+$ && "$WORKERS" =~ ^[1-6]$ ]] || exit 2
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
LOGDIR=data/speech_japanese_daily_v1/logs
mkdir -p "$LOGDIR"
exec 9>"$LOGDIR/daily_ocr_alignment.lock"
flock -n 9 || { echo "Alignment launcher already active"; exit 1; }
echo "$$" > "$LOGDIR/alignment_launcher.pid"
pids=()
stop_workers() {
    trap - INT TERM
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit 0
}
trap stop_workers INT TERM
for ((shard=0; shard<WORKERS; shard++)); do
    (
        cycle=0
        while true; do
            cycle=$((cycle + 1))
            echo "[$(date --iso-8601=seconds)] cycle=$cycle shard=$shard/$WORKERS GPU=$GPU"
            CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                "$PY" -u ssvd/tts/annotate_alignments.py \
                --datadir data/speech_japanese_daily_v1/data_stereo \
                --extra-datadir data/speech_ocr_v1/data_stereo \
                --stable-sharding --shard "$shard" --num_shards "$WORKERS" \
                --model turbo --language ja &
            child=$!
            trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 0' INT TERM
            wait "$child"; status=$?
            echo "[$(date --iso-8601=seconds)] cycle=$cycle exit=$status; recheck completed speech in 60s"
            sleep 60 & child=$!
            wait "$child"
        done
    ) >> "$LOGDIR/alignment_shard${shard}.log" 2>&1 &
    pids+=("$!")
    echo "[$(date --iso-8601=seconds)] shard=$shard pid=$! GPU=$GPU"
done
wait "${pids[@]}"
