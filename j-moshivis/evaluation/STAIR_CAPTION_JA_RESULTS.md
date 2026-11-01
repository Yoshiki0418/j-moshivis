# STAIR日本語キャプション評価: 20k-step動作確認

評価対象は `checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors`。両条件とも同じ67画像・質問を使用し、greedyで生成した同期テキストを採点した。元データと指標の作り方は [STAIR_CAPTION_JA.md](STAIR_CAPTION_JA.md) を参照。

| 入力 | 件数 | 日本語CIDEr ×100 | 空回答 | 推論エラー | 生成文の中央値 |
|---|---:|---:|---:|---:|---:|
| 質問音声 | 67 | 26.4122 | 1 | 0 | 32文字 |
| 質問テキスト | 67 | 0.000537 | 0 | 0 | 147文字 |

5件ずつの参照キャプションは中央値20文字。テキスト入力では65/67件の出力が100文字以上で、多くが画像説明の後に自発的な模擬対話を続け、96トークンで打ち切られた。CIDErには長さに対するペナルティがあるため、このテキスト入力の値だけから画像理解能力を判断しない。音声入力にも、文字列の反復や空白などでキャプションにならない例がある。追加の評価では一文で終了させる条件を事前に固定し、別suiteとして比較する。

67件の同一条件内では採点器・音声・画像のハッシュがsuite lockと一致し、予測の欠損、失敗、余剰は全て0。出力:

- `eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/audio_summary.json`
- `eval_results/llmjp_v3_id0w1_run1_step20000/stair_caption_ja_val_clean_v1/text_summary.json`
- 同ディレクトリの `audio_scores.jsonl` / `text_scores.jsonl` に各画像の予測と参照文、スコアを記録

この67画像はSTAIR validationのうちv3学習画像と重複しない部分集合であり、STAIR全体の公式スコアではない。日本語分かち書きCIDErは英語COCOのCIDEr値とも直接比較しない。
