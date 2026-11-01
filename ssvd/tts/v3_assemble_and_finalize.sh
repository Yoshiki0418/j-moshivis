#!/bin/bash
# Sonnet分のアライメント完了を待って v3 データセットを組み立てる:
#  1) アライメント完了検知 → ループ停止 → 再精緻化 → 検証
#  2) v3 = (v2 のうち Sonnet 差分以外) ハードリンク + (Sonnet 差分) ハードリンク
#  3) 全件検証 → 学習用 JSONL 作成
# 使い方: nohup bash ssvd/tts/v3_assemble_and_finalize.sh > data/speech_v3/logs/finalize.log 2>&1 &
cd /gpu-server/yamamoto/j-moshivis || exit 1
S=data/speech_v3_sonnet/data_stereo
V2=data/speech_v2/data_stereo
V3=data/speech_v3/data_stereo
PY=/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python
mkdir -p "$V3" data/speech_v3/logs

cnt() { find $S -mindepth 2 -maxdepth 2 -name whisper_words.json | wc -l; }
total=$(find $S -mindepth 1 -maxdepth 1 -type d | wc -l)
until [ "$(cnt)" -ge "$total" ]; do sleep 300; done
echo "[v3] sonnet annotation complete: $(cnt)/$total at $(date)"
sleep 60
for pid in $(pgrep -f 'run_annotate_[l]oop'); do kill "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'annotate_alignments[.]py'); do kill "$pid" 2>/dev/null; done
sleep 5

echo "[v3] refine sweep (sonnet)"; $PY ssvd/tts/refine_alignments.py --datadir $S 2>&1 | grep -v Warning

# --- v3 組み立て (ハードリンクのためディスク追加消費はほぼゼロ) ---
echo "[v3] assembling at $(date)"
.venv-train/bin/python - <<'PYEOF'
import json, os, subprocess
from pathlib import Path
delta = {json.loads(l)["uid"] for l in open("data/dialogues_merged/sonnet_delta.jsonl")}
V2, S, V3 = Path("data/speech_v2/data_stereo"), Path("data/speech_v3_sonnet/data_stereo"), Path("data/speech_v3/data_stereo")
V3.mkdir(parents=True, exist_ok=True)
n_s = n_v2 = miss = 0
for l in open("data/dialogues_merged/dialogues_final_v3.jsonl"):
    uid = json.loads(l)["uid"]
    dst = V3 / uid
    if dst.exists():
        continue
    src = (S if uid in delta else V2) / uid
    if not src.is_dir():
        miss += 1
        continue
    subprocess.run(["cp", "-al", str(src), str(dst)], check=False)
    if uid in delta: n_s += 1
    else: n_v2 += 1
print(f"[v3] linked: sonnet={n_s} v2={n_v2} missing={miss}")
PYEOF

echo "[v3] verify"; PYTHONPATH=j-moshivis .venv-train/bin/python ssvd/tts/verify_alignments.py --datadir $V3 2>&1 | grep -v Warning
echo "[v3] make train jsonl"; .venv-train/bin/python ssvd/tts/make_train_jsonl.py --datadir $V3 --out_train data/speech_v3/train.jsonl --out_eval data/speech_v3/eval.jsonl --max_cer 0.3 --require_asr
echo "[v3] done at $(date)"
