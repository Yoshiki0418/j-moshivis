#!/usr/bin/env bash
# Resume Japanese Daily v1 dialogue generation after SSH disconnection or a
# transient worker failure. Run detached with nohup/setsid; only GPU 1 is used.
set -uo pipefail

PROJECT_ROOT="/gpu-server/yamamoto/j-moshivis"
PYTHON_BIN="/gpu-server/yamamoto/miniforge3/envs/vlm/bin/python"
CUDA_ROOT="/gpu-server/yamamoto/miniforge3/envs/vlm"
MANIFEST="$PROJECT_ROOT/data/images/japanese_daily_v1/accepted_images.jsonl"
RUN_DIR="$PROJECT_ROOT/data/dialogues_japanese_daily_v1/full"
OUTPUT="$RUN_DIR/qwen3vl_30b_normal_adaptive.jsonl"
LOCK_FILE="$RUN_DIR/generation.lock"

mkdir -p "$RUN_DIR"
cd "$PROJECT_ROOT" || exit 1
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] Japanese Daily v1 generator is already running"
    exit 1
fi
echo "$$" > "$RUN_DIR/launcher.pid"

no_progress_passes=0
pass=0
while true; do
    pass=$((pass + 1))
    if [[ -f "$OUTPUT" ]]; then
        before=$(wc -l < "$OUTPUT")
    else
        before=0
    fi
    echo "[$(date --iso-8601=seconds)] pass=$pass GPU=1 generated=$before manifest=$MANIFEST"
    env CUDA_VISIBLE_DEVICES=1 CUDA_HOME="$CUDA_ROOT" \
        PATH="$CUDA_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT" \
        PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        "$PYTHON_BIN" -u ssvd/dialogue/generate_local_vlm.py \
            --task-mode normal --normal-strategy vlm_adaptive \
            --registry-scope uid --registry_dir "$RUN_DIR" \
            --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ --backend vllm \
            --meta "$MANIFEST" --out "$OUTPUT" \
            --batch_size 8 --gpu_memory_utilization 0.80
    status=$?
    after=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    echo "[$(date --iso-8601=seconds)] pass=$pass exited=$status generated=$after"
    if [[ "$status" -eq 3 ]]; then
        echo "[$(date --iso-8601=seconds)] all eligible images have dialogues"
        break
    fi
    if [[ "$after" -gt "$before" ]]; then
        no_progress_passes=0
    else
        no_progress_passes=$((no_progress_passes + 1))
    fi
    if [[ "$no_progress_passes" -ge 5 ]]; then
        echo "[$(date --iso-8601=seconds)] stopped after 5 passes without progress; inspect the log"
        exit 1
    fi
    sleep 60
done
