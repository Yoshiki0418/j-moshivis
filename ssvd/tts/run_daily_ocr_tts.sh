#!/usr/bin/env bash
# GPU1 only: restartable speech synthesis for completed Daily/JSSODa dialogues.
# The builder applies exact-content human-review gates and revision overlays.
set -uo pipefail
cd /gpu-server/yamamoto/j-moshivis || exit 1

GPU=${1:-1}
WORKERS=${2:-4}
[[ "$GPU" =~ ^[0-9]+$ && "$WORKERS" =~ ^[1-4]$ ]] || {
    echo "usage: run_daily_ocr_tts.sh [physical GPU] [concurrent workers: 1..4]" >&2
    exit 2
}
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
LOGDIR=data/speech_japanese_daily_v1/logs
mkdir -p "$LOGDIR" data/speech_japanese_daily_v1/data_stereo data/speech_ocr_v1/data_stereo
exec 9>"$LOGDIR/tts_launcher.lock"
flock -n 9 || { echo "TTS launcher already running"; exit 1; }
echo "$$" > "$LOGDIR/tts_launcher.pid"
declare -A active=()

stop_workers() {
    trap - INT TERM
    if ((${#active[@]})); then
        kill -TERM "${!active[@]}" 2>/dev/null || true
        wait "${!active[@]}" 2>/dev/null || true
    fi
    exit 0
}
trap stop_workers INT TERM

wait_one() {
    local finished="" status=0
    wait -n -p finished "${!active[@]}" || status=$?
    if [[ -n "$finished" ]]; then
        echo "[$(date --iso-8601=seconds)] finished ${active[$finished]} pid=$finished status=$status"
        unset 'active[$finished]'
    fi
    if ((status != 0)); then failures=$((failures + 1)); fi
}

cycle=0
while true; do
    cycle=$((cycle + 1)); failures=0
    echo "[$(date --iso-8601=seconds)] cycle=$cycle GPU=$GPU concurrency=$WORKERS"
    # Interleave sources so both Daily and OCR synthesis begin immediately.
    # Freed slots take the next shard without waiting for the whole source.
    for shard in 0 1 2 3; do
        for source in daily ocr0 ocr1; do
            while ((${#active[@]} >= WORKERS)); do wait_one; done
            case "$source" in
                daily)
                    input=data/dialogues_japanese_daily_v1/full/qwen3vl_30b_normal_adaptive.jsonl
                    out=data/speech_japanese_daily_v1/data_stereo ;;
                ocr0|ocr1)
                    input="data/dialogues_ocr/jssoda_v1/shard${source#ocr}.jsonl"
                    out=data/speech_ocr_v1/data_stereo ;;
            esac
            [[ -f "$input" ]] || continue
            CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                "$PY" -u ssvd/tts/build_speech_dataset.py \
                --dialogues "$input" --outdir "$out" --device cuda \
                --shard "$shard" --num_shards 4 \
                >> "$LOGDIR/tts_${source}_shard${shard}.log" 2>&1 &
            pid=$!; active[$pid]="${source}_shard${shard}"
            echo "[$(date --iso-8601=seconds)] started ${active[$pid]} pid=$pid"
        done
    done
    while ((${#active[@]})); do wait_one; done
    echo "[$(date --iso-8601=seconds)] cycle=$cycle finished failures=$failures; recheck approvals in 300s"
    # Existing complete WAV+metadata are skipped. Newly approved/edited text
    # is picked up on the next pass, without modifying old revision audio.
    sleep 300 &
    pid=$!; active[$pid]=approval_wait
    wait_one
done
