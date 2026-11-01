#!/bin/bash
# 非画像(日常/一般知識)対話を Claude Sonnet 5 で生成する自動再開ランナー
# 使い方: nohup bash ssvd/dialogue/run_nonvisual_loop.sh > data/dialogues_nonvisual/nonvisual.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1

NUM=${NUM:-12000}
MAX_RETRY_PASSES=${MAX_RETRY_PASSES:-5}
retry=0

while true; do
  .venv-data/bin/python -u ssvd/dialogue/generate_nonvisual_cli.py \
    --out data/dialogues_nonvisual/claude_sonnet5_nonvisual.jsonl \
    --num "$NUM" --workers 4
  code=$?
  case $code in
    3) echo "[loop] all dialogues done — finished at $(date)"; break ;;
    2) echo "[loop] usage limit suspected — sleeping 15 min ($(date))"; sleep 900 ;;
    *) retry=$((retry+1))
       if [ $retry -ge $MAX_RETRY_PASSES ]; then
         echo "[loop] $MAX_RETRY_PASSES retry passes reached — stopping $(date)"; break
       fi
       echo "[loop] pass ended (exit $code) — retry $retry/$MAX_RETRY_PASSES in 60s ($(date))"; sleep 60 ;;
  esac
done
