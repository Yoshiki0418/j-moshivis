#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/gpu-server/yamamoto/j-moshivis"
RUN_NAME="llmjp_v3_id0w1_run1_step20000"
OUT_DIR="$PROJECT_ROOT/eval_results/$RUN_NAME/production_audio_greedy_20260930"
MANIFEST="$PROJECT_ROOT/data/evaluation/vision_v1/manifest_audio.jsonl"
CHECKPOINT="$PROJECT_ROOT/checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors"
PREDICTIONS="$OUT_DIR/vision_v1_audio_greedy.jsonl"

mkdir -p "$OUT_DIR"
echo "$$" > "$OUT_DIR/pipeline.pid"
date --iso-8601=seconds > "$OUT_DIR/started_at.txt"

cd "$PROJECT_ROOT"

PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.validate_manifest \
  --manifest "$MANIFEST" \
  --require-audio --require-overlap-clean --strict-checksums \
  > "$OUT_DIR/manifest_validation.json"

PYTHONPATH=j-moshivis .venv-train/bin/python -u \
  -m jmoshivis.evaluation.run_inference \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output "$PREDICTIONS" \
  --device cuda:1

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.score \
  --manifest "$MANIFEST" \
  --predictions "$PREDICTIONS" \
  --scores "$OUT_DIR/vision_v1_scores.jsonl" \
  --summary "$OUT_DIR/vision_v1_summary.json" \
  --official-rouge-ja \
  --lid176-model models/evaluation/lid.176.ftz

PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.judge prepare \
  --manifest "$MANIFEST" \
  --predictions "$PREDICTIONS" \
  --output "$OUT_DIR/jammeval_judge_requests.jsonl" \
  --model gpt-5.1-2025-11-13 \
  --n-repeats 3

date --iso-8601=seconds > "$OUT_DIR/completed_at.txt"
