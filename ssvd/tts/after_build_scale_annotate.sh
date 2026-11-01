#!/bin/bash
# 構築プロセス終了を待ち、アライメントループを2→4シャード(各GPU 2)へ増強する
cd /gpu-server/yamamoto/j-moshivis || exit 1
until [ "$(pgrep -fc 'build_speech_dataset[.]py')" -eq 0 ]; do sleep 120; done
echo "[scale] build finished at $(date) — restarting annotate loops with 4 shards"
for pid in $(pgrep -f 'run_annotate_[l]oop'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'annotate_alignments[.]py'); do kill "$pid" 2>/dev/null; done
sleep 5
i=0
for g in 0 0 1 1; do
  nohup bash ssvd/tts/run_annotate_loop.sh $i 4 $g > data/speech/logs/annotate4_shard$i.log 2>&1 &
  echo "[scale] annotate shard $i on GPU $g (PID $!)"
  i=$((i+1))
done
