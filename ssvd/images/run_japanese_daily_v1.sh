#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/data/images/japanese_daily_v1"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_PATH="${LOG_DIR}/collection.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date --iso-8601=seconds)] Japanese Daily v1 metadata harvest started"
python3 -u "${REPO_ROOT}/ssvd/images/collect_japanese_daily_v1.py" \
  harvest \
  --output-dir "${OUTPUT_DIR}"

echo "[$(date --iso-8601=seconds)] download and deduplication started"
python3 -u "${REPO_ROOT}/ssvd/images/collect_japanese_daily_v1.py" \
  download \
  --output-dir "${OUTPUT_DIR}"

echo "[$(date --iso-8601=seconds)] Japanese Daily v1 collection finished"
