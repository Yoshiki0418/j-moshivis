#!/bin/bash
# Irodori-TTS VoiceDesign サンプル合成 (参照音声なし、キャプションで女性話者を指定)
cd /gpu-server/yamamoto/Irodori-TTS || exit 1
export PATH=/gpu-server/yamamoto/miniforge3/envs/tts/bin:$PATH
OUT=/gpu-server/yamamoto/j-moshivis/data/tts_samples
CAPTION="落ち着いた、近い距離感の二十代後半の女性話者。親しみやすく自然な会話口調"
python3 - <<'PY' > /tmp/irodori_sents.tsv
import json
for s in json.load(open("/gpu-server/yamamoto/j-moshivis/data/tts_samples/sentences.json", encoding="utf-8")):
    print(f"{s['id']}\t{s['text']}")
PY
for ckpt in "Aratako/Irodori-TTS-600M-v3-VoiceDesign:irodori_600M_v3" "Aratako/Irodori-TTS-v4.1-Small:irodori_v4.1_small"; do
  hf=${ckpt%%:*}; tag=${ckpt##*:}
  while IFS=$'\t' read -r id text; do
    uv run --no-sync python infer.py --hf-checkpoint "$hf" --text "$text" --caption "$CAPTION" --no-ref --output-wav "$OUT/${tag}__${id}.wav" 2>&1 | tail -1
  done < /tmp/irodori_sents.tsv
  echo "✅ $tag"
done
