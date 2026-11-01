#!/bin/bash
# v2 データセットの後段自動化:
#  1) 音声構築(build)完了を検知 → アライメントを 6→12 シャードに増強
#  2) アライメント完了を検知 → ループ停止 → 再精緻化 → 全件検証 → 学習用JSONL作成
# 使い方: nohup bash ssvd/tts/v2_scale_and_finalize.sh > data/speech_v2/logs/finalize.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1
D=data/speech_v2/data_stereo
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
cnt() { find $D -mindepth 2 -maxdepth 2 -name whisper_words.json | wc -l; }
dirs() { find $D -mindepth 1 -maxdepth 1 -type d | wc -l; }

# --- 1) build 完了待ち → アノテーション増強 (6 → 12) ---
while [ "$(pgrep -fc 'build_speech_dataset[.]py')" -gt 0 ]; do sleep 300; done
echo "[v2] build complete: $(dirs) dirs at $(date)"
for pid in $(pgrep -f 'run_annotate_[l]oop'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'annotate_alignments[.]py'); do kill "$pid" 2>/dev/null; done
sleep 10
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  g=$((s/6))
  nohup bash ssvd/tts/run_annotate_loop.sh $s 12 $g $D > data/speech_v2/logs/annotate12_shard$s.log 2>&1 &
done
echo "[v2] annotation scaled to 12 shards at $(date)"

# --- 2) アライメント完了待ち → 仕上げ ---
total=$(dirs)
until [ "$(cnt)" -ge "$total" ]; do sleep 300; done
echo "[v2] annotation complete: $(cnt)/$total at $(date)"
sleep 60
for pid in $(pgrep -f 'run_annotate_[l]oop'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'annotate_alignments[.]py'); do kill "$pid" 2>/dev/null; done
sleep 5
echo "[v2] refine sweep"; $PY ssvd/tts/refine_alignments.py --datadir $D 2>&1 | grep -v Warning
echo "[v2] verify"; PYTHONPATH=j-moshivis .venv-train/bin/python ssvd/tts/verify_alignments.py --datadir $D 2>&1 | grep -v Warning
echo "[v2] make train jsonl"; .venv-train/bin/python ssvd/tts/make_train_jsonl.py --datadir $D --out_train data/speech_v2/train.jsonl --out_eval data/speech_v2/eval.jsonl --max_cer 0.3 --require_asr
echo "[v2] done at $(date)"
