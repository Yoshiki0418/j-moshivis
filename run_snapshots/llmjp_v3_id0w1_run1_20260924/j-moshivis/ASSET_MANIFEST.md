# v3 学習資産マニフェスト

`train.jsonl` / `eval.jsonl` だけでは、実際に教師として読む WAV・画像・
alignment の内容を固定できません。本番学習前に一度、全資産を SHA-256 で
固定します。マニフェストが無い、JSONL が変わった、資産が欠落・変更された
場合は学習起動が失敗します。

> **I/O 注意:** `generate` は全 WAV・画像・alignment の全バイトを1回読みます。
> 現行 v3（train約2,465時間 + eval約50時間、24 kHz/16-bit/stereo WAV）では WAV だけでも理論上
> 約0.87 TBです。`verify --mode full` は同量をもう1回読むため、共有ストレージの
> 混雑時間を避けて実行してください。通常起動の `fast` は資産本体を再読せず、
> statとWAVヘッダを全件取得します（JSONLとmanifest自体はSHA-256計算のため読みます）。

リポジトリ直下からの生成コマンド:

```bash
cd j-moshivis
../.venv-train/bin/python -m jmoshivis.datasets.asset_manifest generate \
  --train-jsonl ../data/speech_v3/train.jsonl \
  --eval-jsonl ../data/speech_v3/eval.jsonl \
  --output ../data/speech_v3/asset_manifest.v1.jsonl \
  --workers 8
```

生成後に一度、全ファイルの SHA-256 を再計算して確認します。

```bash
../.venv-train/bin/python -m jmoshivis.datasets.asset_manifest verify \
  --train-jsonl ../data/speech_v3/train.jsonl \
  --eval-jsonl ../data/speech_v3/eval.jsonl \
  --manifest ../data/speech_v3/asset_manifest.v1.jsonl \
  --mode full
```

通常の学習起動時は `fast`（path・size・mtime_ns・WAVヘッダ）で全件確認します。厳密な
再監査時は `data.asset_manifest_verification=full` を指定します。`fast` でも
train/eval JSONL とマニフェスト自体の SHA-256 は毎回計算され、検証結果と
マニフェスト SHA は `run.json` の run fingerprint に含まれます。

1サンプルにつき、以下の3資産が正本です。

- `row.path` の WAV
- WAV と同じディレクトリの `image.jpg`
- WAV と同じディレクトリの `assistant_dialogue.refined.json`

JSONL の `image` が co-located path と異なる場合、UID・資産 path が重複する
場合、WAV/JPEG/alignment を解釈できない場合も生成時に停止します。WAVは2chを必須とし、
JSONLの`duration`が有限・正値でWAVヘッダのframes/sample_rateと1 ms以内で一致することも
生成時と起動時の両方で検証します（元sample rateはMimi入力時にresampleするため固定しません）。
alignmentはtoken/text、有限で正順の時刻、duration範囲、USER/ASSISTANT両役割を検証します。
全二重の発話重複ではファイル上の対話順と開始時刻順が異なり得るため、学習loaderはcrop前に
時刻順へ安定sortします。
