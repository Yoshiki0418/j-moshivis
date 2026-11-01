#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/gpu-server/yamamoto/j-moshivis"
PYTHON_BIN="/gpu-server/yamamoto/miniforge3/envs/vlm/bin/python"
CUDA_ROOT="/gpu-server/yamamoto/miniforge3/envs/vlm"
RUN_DIR="$PROJECT_ROOT/data/dialogues_ocr/jssoda_v1"
LOG_DIR="$RUN_DIR/logs"
MODEL="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
echo "$$" > "$RUN_DIR/launcher.pid"

run_worker() {
    local gpu="$1" shard="$2" target="$3"
    local output="$RUN_DIR/shard${shard}.jsonl" log="$LOG_DIR/shard${shard}.log"
    local pass=0
    while true; do
        pass=$((pass + 1))
        current=$(wc -l < "$output" 2>/dev/null || echo 0)
        if [[ "$current" -ge "$target" ]]; then
            echo "[$(date --iso-8601=seconds)] shard=$shard complete $current/$target" >> "$log"
            break
        fi
        echo "[$(date --iso-8601=seconds)] retry pass=$pass GPU=$gpu shard=$shard current=$current/$target" >> "$log"
        env CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME="$CUDA_ROOT" \
            PATH="$CUDA_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
            "$PYTHON_BIN" -u -m ssvd.dialogue.generate_jssoda_ocr_dataset_runner_v4 \
                --out "$output" --shard-id "$shard" --num-shards 2 \
                --batch-size 8 --max-attempts 3 --gpu-memory-utilization 0.82 \
                --model "$MODEL" >> "$log" 2>&1
        status=$?
        current=$(wc -l < "$output" 2>/dev/null || echo 0)
        echo "[$(date --iso-8601=seconds)] pass=$pass exited status=$status current=$current/$target" >> "$log"
        sleep 30
    done
}

run_worker 0 0 8996 & worker0=$!; echo "$worker0" > "$RUN_DIR/shard0.pid"
run_worker 1 1 8995 & worker1=$!; echo "$worker1" > "$RUN_DIR/shard1.pid"
trap 'kill -TERM "$worker0" "$worker1" 2>/dev/null || true; wait "$worker0" "$worker1" 2>/dev/null || true' INT TERM
wait "$worker0" "$worker1"

