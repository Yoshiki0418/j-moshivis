# STAIR日本語キャプション評価セット

STAIR Captions v1.2 のvalidation注釈を使い、画像1枚に対して5文の日本語正解キャプションを持つ評価セットを構築した。質問は全件「この画像の内容を一文で説明してください。」とし、音声入力でもテキスト入力でも同じ文を使う。モデルの同期テキストを採点し、出力音声のASRは使わない。

## 現在の成果物

| 項目 | 内容 |
|---|---|
| 元データ | STAIR Captions v1.2 validation、40,504画像・202,520文 |
| v3学習COCO IDに重複 | 40,437画像 |
| ファイル・画素・近似画像まで重複なし | **67画像・67問** |
| 参照キャプション | 各画像5文、計335文 |
| 固定質問音声 | 67 WAV、24 kHz mono、男性39・女性28 |
| 主指標 | 日本語MeCab/UniDicで分かち書きしたCIDEr (`cider_ja`) |

データ: [公式STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions)、commit `6ac656e65d80cbe2c914d93626a1d5454e35a5d6`。元注釈archive SHA-256は `8d47b7971d2883bffcba92e2ab44918e70e22c434d80dfffc01670c1654a7735`。注釈はCC BY 4.0、画像のライセンスはCOCO側の各画像のmetadataに依存する。論文ではSTAIRの著者を引用する。

ファイル:

- `data/evaluation/stair_caption_v1/source/stair_captions_v1.2.tar.gz`: 公式注釈archive
- `data/evaluation/stair_caption_v1/manifest_val_source_id_candidates.jsonl`: COCO IDによる初期除外後
- `data/evaluation/stair_caption_v1/manifest_val_overlap_clean.jsonl`: 画像内容の重複監査後
- `data/evaluation/stair_caption_v1/manifest_audio.jsonl`: 固定質問音声付きの評価正本
- `data/evaluation/stair_caption_v1/overlap_val_report.summary.json`: 重複監査とtrain/indexのhash
- `data/evaluation/stair_caption_v1/validation_audio.json`: 画像・音声hash検証
- `j-moshivis/evaluation/stair_caption_ja_val_clean_v1.lock.json`: manifest、音声、greedy、採点器を凍結したsuite lock

キャプションには5件の参照文を保持する。`cider_ja` は予測と正解の双方にNFKC正規化、fugashi/UniDic分割、句読点の除去を施し、`pycocoevalcap==1.2`のCIDErでコーパス全体を一度に採点する。日本語向けに適応した指標なので、英語COCOを使うMoshiVis論文のCIDEr数値と直接比較しない。67問という規模も小さいため、まず動作・傾向確認に使い、論文の主結果にする際は別画像源の追加を検討する。

## 再構築と評価

コマンドはworkspace rootから実行する。既存manifestとlockは上書きしないため、再構築時は別バージョンの出力先を指定する。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python -m jmoshivis.evaluation.datasets.stair_captions \
  --archive data/evaluation/stair_caption_v1/source/stair_captions_v1.2.tar.gz \
  --coco-root data/images/coco \
  --train-manifest data/speech_v3/train.jsonl --train-manifest data/speech_v3/eval.jsonl \
  --output data/evaluation/stair_caption_v2/manifest_val_source_id_candidates.jsonl

PYTHONPATH=j-moshivis .venv-train/bin/python -m jmoshivis.evaluation.overlap.audit \
  --manifest data/evaluation/stair_caption_v2/manifest_val_source_id_candidates.jsonl \
  --train-manifest data/speech_v3/train.jsonl --train-manifest data/speech_v3/eval.jsonl \
  --image-index data/evaluation/v3_image_index.jsonl \
  --clean-manifest data/evaluation/stair_caption_v2/manifest_val_overlap_clean.jsonl \
  --report data/evaluation/stair_caption_v2/overlap_val_report.jsonl --exclude-near
```

既存の質問音声を使う音声入力評価:

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/stair_caption_v1/manifest_audio.jsonl \
  --suite-lock j-moshivis/evaluation/stair_caption_ja_val_clean_v1.lock.json \
  --checkpoint checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors \
  --output eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/audio_greedy67.jsonl \
  --input-modality audio --response-seconds 10.0 --response-overlap-seconds 0.32 \
  --audio-dialogue-protocol bare_question_v1 --device cuda:1

PYTHONPATH=j-moshivis .venv-eval/bin/python -m jmoshivis.evaluation.score_moshivis \
  --manifest data/evaluation/stair_caption_v1/manifest_audio.jsonl \
  --suite-lock j-moshivis/evaluation/stair_caption_ja_val_clean_v1.lock.json \
  --predictions eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/audio_greedy67.jsonl \
  --scores eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/audio_scores.jsonl \
  --summary eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/audio_summary.json
```

同じmanifestで `--input-modality text --max-new-tokens 96` に変更するとspeechlessテキスト入力も比較できる。予測結果とsummaryの保存先は音声入力とは分ける。限定件数の`--diagnostic`実行は完走した正式結果としては扱わない。
