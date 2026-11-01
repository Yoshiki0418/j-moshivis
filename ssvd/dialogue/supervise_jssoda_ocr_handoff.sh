#!/usr/bin/env bash
set -uo pipefail

CURRENT_LAUNCHER_PID="$1"
PROJECT_ROOT="/gpu-server/yamamoto/j-moshivis"
RUN_DIR="$PROJECT_ROOT/data/dialogues_ocr/jssoda_v1"

echo "$$" > "$RUN_DIR/supervisor.pid"
echo "[$(date --iso-8601=seconds)] waiting for initial launcher $CURRENT_LAUNCHER_PID"
while kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date --iso-8601=seconds)] initial pass ended; starting retry-until-complete launcher"
exec bash "$PROJECT_ROOT/ssvd/dialogue/run_jssoda_ocr_until_complete.sh"
