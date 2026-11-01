#!/bin/bash
# Qwen生成分 20,289 対話を Claude Sonnet 5 で置き換え生成する自動再開ランナー
# - registry_glob を claude_*.jsonl に絞ることで、Qwen生成済みUIDも「未生成」として再生成する
# - 出力は data/dialogues/claude_sonnet5_replace.jsonl (マージ時に Sonnet 優先で置き換わる)
# 使い方: nohup bash ssvd/dialogue/run_sonnet_replace_loop.sh > data/dialogues/sonnet_replace.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1

MAX_RETRY_PASSES=${MAX_RETRY_PASSES:-5}
retry=0

while true; do
  .venv-data/bin/python -u ssvd/dialogue/generate_claude_cli.py \
    --meta data/images/replace_qwen_with_sonnet.jsonl \
    --out data/dialogues/claude_sonnet5_replace.jsonl \
    --registry_glob 'claude_*.jsonl' \
    --workers 4
  code=$?
  case $code in
    3) echo "[loop] all images done — finished at $(date)"; break ;;
    2) echo "[loop] usage limit suspected — sleeping 15 min ($(date))"; sleep 900 ;;
    *) retry=$((retry+1))
       if [ $retry -ge $MAX_RETRY_PASSES ]; then
         echo "[loop] $MAX_RETRY_PASSES retry passes reached — stopping (persistent failures remain) $(date)"; break
       fi
       echo "[loop] pass ended (exit $code) — retry $retry/$MAX_RETRY_PASSES in 60s ($(date))"; sleep 60 ;;
  esac
done
