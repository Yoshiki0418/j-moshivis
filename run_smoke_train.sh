#!/bin/bash
# Isolated one-update training smoke test (single GPU by default).
# Synthetic data, manifest, and checkpoints are created under a fresh /tmp dir;
# production v3 indexes and checkpoint directories are never selected.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
python_bin="$repo_root/.venv-train/bin/python"

# A production-sized smoke checkpoint is about 17.44 GB and its model-only
# export is about 16.61 GB.  The first export is removed before the resume
# check below, so 40 GiB safely covers one complete resumable state, one atomic
# export temporary, the synthetic inputs, and filesystem headroom.  CI or a
# differently sized model may override the byte threshold explicitly.
gib_bytes=$((1024 * 1024 * 1024))
minimum_tmp_free_bytes="${JMOSHIVIS_SMOKE_MIN_TMP_FREE_BYTES:-$((40 * gib_bytes))}"
case "$minimum_tmp_free_bytes" in
  ''|*[!0-9]*)
    echo "JMOSHIVIS_SMOKE_MIN_TMP_FREE_BYTES must be a non-negative integer, got: $minimum_tmp_free_bytes" >&2
    exit 1
    ;;
esac
if ! tmp_available_bytes="$(df -PB1 /tmp | awk 'NR == 2 {print $4}')"; then
  echo "Unable to determine free space in /tmp; refusing to start the smoke run." >&2
  exit 1
fi
case "$tmp_available_bytes" in
  ''|*[!0-9]*)
    echo "Unable to parse free space in /tmp: $tmp_available_bytes" >&2
    exit 1
    ;;
esac
if (( tmp_available_bytes < minimum_tmp_free_bytes )); then
  echo "Insufficient free space in /tmp for the J-MoshiVis smoke run." >&2
  echo "Available: $tmp_available_bytes bytes; required: $minimum_tmp_free_bytes bytes (40 GiB by default)." >&2
  echo "Free /tmp space or set JMOSHIVIS_SMOKE_MIN_TMP_FREE_BYTES for a deliberately smaller model." >&2
  exit 1
fi

smoke_root="$(mktemp -d /tmp/jmoshivis-smoke.XXXXXX)"

if [[ ! -x "$python_bin" ]]; then
  echo "training Python not found: $python_bin" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_MODE=offline
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_root/j-moshivis${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a smoke_devices <<< "$CUDA_VISIBLE_DEVICES"
num_processes="${#smoke_devices[@]}"

echo "Smoke artifacts will remain at: $smoke_root"
"$python_bin" "$repo_root/sample_test/make_smoke_data.py" \
  --output-dir "$smoke_root/data" \
  --train-samples 4 --eval-samples 2 --duration-seconds 2.0

cd "$repo_root/j-moshivis"
"$python_bin" -m jmoshivis.datasets.asset_manifest generate \
  --train-jsonl "$smoke_root/data/train_smoke.jsonl" \
  --eval-jsonl "$smoke_root/data/eval_smoke.jsonl" \
  --output "$smoke_root/data/asset_manifest.v1.jsonl" \
  --workers 2
"$python_bin" -m jmoshivis.datasets.asset_manifest verify \
  --train-jsonl "$smoke_root/data/train_smoke.jsonl" \
  --eval-jsonl "$smoke_root/data/eval_smoke.jsonl" \
  --manifest "$smoke_root/data/asset_manifest.v1.jsonl" \
  --mode full

# User-supplied model/path overrides are accepted first.  Safety-critical data,
# output, duration, and step limits follow and therefore remain isolated.
common_args=(
  "$@"
  duration_sec=2.0
  data.train_data="$smoke_root/data/train_smoke.jsonl"
  data.eval_data="$smoke_root/data/eval_smoke.jsonl"
  data.asset_manifest="$smoke_root/data/asset_manifest.v1.jsonl"
  data.asset_manifest_verification=full
  data.image_root="$smoke_root/data"
  data.speech_ratio=0.25
  data.speech_decode_workers=1
  train.batch_size=1
  train.target_batch_size="$((4 * num_processes))"
  train.max_steps=1
  train.save_every=1
  train.keep_last=1
  train.checkpoint_dir="$smoke_root/checkpoints"
  train.log_interval=1
)

run_training() {
  local resume_value="$1"
  # Use the same Accelerate path for one and multiple GPUs.  Directly invoking
  # Python on a single GPU can yield an unindexed ``cuda`` device, while the
  # training runtime deliberately requires an explicit rank-local device.
  "$repo_root/.venv-train/bin/accelerate" launch \
    --num_processes "$num_processes" --mixed_precision bf16 \
    --main_process_port "${MASTER_PORT:-29512}" \
    -m jmoshivis.train "${common_args[@]}" "train.resume_from=$resume_value"
}

run_training null
# The first run already proved that model-only export succeeds.  Resume needs
# only step_1, and retaining this synthetic export would force the second
# atomic export to coexist with two 16.6 GB copies.  Remove only the artifact
# owned by this fresh smoke directory; the resumed run recreates it below.
first_export="$smoke_root/checkpoints/final_step_1.safetensors"
if [[ ! -f "$first_export" ]]; then
  echo "First smoke run did not produce its expected inference export: $first_export" >&2
  exit 1
fi
rm -- "$first_export"
echo "Reloading the completed step_1 checkpoint to verify strict resume..."
run_training latest
echo "Smoke save/resume completed. Artifacts: $smoke_root"
