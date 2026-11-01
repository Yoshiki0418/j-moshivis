#!/bin/bash
# Claude CLI 対話生成の自動再開ランナー
# - exit 3: 全画像生成済み → 完了して終了
# - exit 2: 連続失敗 (使用制限疑い) → 15分待機して再実行
# - その他: 1パス完了 (一部失敗あり) → 60秒後に失敗分だけ再実行
# 使い方: nohup bash ssvd/dialogue/run_gen_loop.sh <meta...> -- <out.jsonl> > log 2>&1 &
# task切替: TASK_MODE=counting nohup bash ... (既定: normal)
cd /gpu-server/yamamoto/j-moshivis || exit 1

METAS=()
while [ "$1" != "--" ]; do METAS+=("$1"); shift; done
shift
OUT="$1"
MAX_RETRY_PASSES=${MAX_RETRY_PASSES:-5}   # 失敗分の再試行パス上限 (課金の無限ループ防止)
TASK_MODE=${TASK_MODE:-normal}
NORMAL_STRATEGY=${NORMAL_STRATEGY:-standard}
REGISTRY_SCOPE=${REGISTRY_SCOPE:-task}
retry=0

while true; do
  .venv-data/bin/python -u ssvd/dialogue/generate_claude_cli.py \
    --task-mode "$TASK_MODE" --normal-strategy "$NORMAL_STRATEGY" \
    --registry-scope "$REGISTRY_SCOPE" \
    --meta "${METAS[@]}" --out "$OUT" --workers 4
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
