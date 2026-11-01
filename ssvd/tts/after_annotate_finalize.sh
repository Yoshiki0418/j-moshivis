#!/bin/bash
# アライメント完了を待って: ループ停止 → 旧形式の再精緻化 → 全件検証 → 学習用JSONL作成
cd /gpu-server/yamamoto/j-moshivis || exit 1
D=data/speech/data_stereo
cnt() { find $D -mindepth 2 -maxdepth 2 -name whisper_words.json | wc -l; }
total=$(find $D -mindepth 1 -maxdepth 1 -type d | wc -l)
until [ "$(cnt)" -ge "$total" ]; do sleep 300; done
echo "[finalize] annotation complete: $(cnt)/$total at $(date)"
sleep 60
for pid in $(pgrep -f 'run_annotate_[l]oop'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'annotate_alignments[.]py'); do kill "$pid" 2>/dev/null; done
sleep 5
echo "[finalize] refine sweep"; /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/refine_alignments.py --datadir $D 2>&1 | grep -v Warning
echo "[finalize] verify"; PYTHONPATH=j-moshivis .venv-train/bin/python ssvd/tts/verify_alignments.py --datadir $D 2>&1 | grep -v Warning
echo "[finalize] make train jsonl"; .venv-train/bin/python ssvd/tts/make_train_jsonl.py --datadir $D --out_train data/speech/train.jsonl --out_eval data/speech/eval.jsonl --max_cer 0.3 --require_asr
echo "[finalize] done at $(date)"
