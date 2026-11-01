# J-MoshiVis 画像理解評価基盤

論文用の主評価は、**未知画像に対する日本語音声 VQA**です。画像と質問音声を J-MoshiVis に入力し、モデルが音声と同期して生成するテキストトークンを直接採点します。出力音声を ASR に通さないため、ASR 誤りを画像理解の誤りに混ぜません。

`vision_v1` は 2026-09-22 時点で次の内容に固定済みです。正本となる設定・revision・SHA-256 は [`suite_v1.lock.json`](suite_v1.lock.json) にあります。

| 領域 / ベンチマーク | raw 問数 | 重複除外 | 評価問数 | 評価画像 |
|---|---:|---:|---:|---:|
| 一般世界: JA-VG-VQA-500 | 500 | 242 | 258 | 258 |
| 日本: JAMMEval CC-OCR | 145 | 0 | 145 | 145 |
| 日本: JAMMEval CVQA | 200 | 6 | 194 | 91 |
| 日本: JAMMEval Heron-Bench | 88 | 0 | 88 | 21 |
| 日本: JAMMEval JA-VLM-Bench | 49 | 0 | 49 | 42 |
| **合計** | **982** | **248** | **734** | **557** |

固定質問音声は 24 kHz mono、734ファイル、合計 1.93 時間です。一般世界 258問と日本関連 476問は混ぜずに別々に報告し、各ベンチマークの値も残します。

## 評価設計

### 一般世界と日本の両方を見る

- `general_world`: [JA-VG-VQA-500](https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500) の各画像先頭1問。人物、物体、色、数、位置、場所など、日本固有ではない視覚理解を測ります。
- `japan`: [JAMMEval](https://huggingface.co/datasets/llm-jp/JAMMEval) の単一画像4セット。日本語 OCR、日本文化、日本の風景・日常場面を測ります。

JA-VG の Visual Genome 画像には COCO と共通するものがあり、v3 学習データにも COCO が含まれます。このため公開500問をそのまま使わず、v3 の17万画像と照合して未知画像だけを残しました。したがって、258問の値は「JA-VG overlap-clean adaptation」として報告し、公開500問の既報値とは直接比較しません。

### 学習画像との重複を除く

v3 train/eval の全画像に対し、次を順に照合します。

1. COCO source ID
2. 元ファイルの SHA-256
3. 復号した RGB 画素の SHA-256
4. pHash / dHash（Hamming 距離4以下を保守的に除外）

インデックスは170,021画像すべてを正常に処理済みです。JA-VG 242問と CVQA 6問を除き、最終 manifest に `review_near` は残っていません。監査時にはインデックス本体だけでなく、sidecar に保存した v3 manifest のハッシュと fingerprint 仕様も検証します。

### 音声プロンプトを固定する

- SBV2 の男性 `jvnv-M1` と女性 `jvnv-F1` を、`sample_id` のハッシュで固定割当します。
- 現在の `vision_v1` は、prompt、話者モデル、TTS パラメータ、生成 wav の SHA-256 を記録し、**固定 wav 自体を正本**にします。
- 今後 TTS を再生成するコードでは、これらに加えて BERT revision、依存ライブラリ、前処理実装自身の hash も cache key へ記録します。現行734件はその改修前に生成したため、同じ音声を再合成できると主張せず、配布・保全した固定 wav の hash で同一性を保証します。
- 推論時には prompt と wav の `sample_id`・文字列・ハッシュが一致することを再検証します。
- JAMMEval の短答指示や選択肢を含む adapter-defined prompt を、そのまま音声化します。

CVQA だけは音声で A–D が紛らわしいため、選択肢と回答を「1〜4」で読み上げます。これは音声向けの適応プロトコルであり、公式の A–D 条件と同一ではありません。公式ラベル対応は各行の metadata に残しています。

### 生成条件

[MoshiVis 論文](https://arxiv.org/abs/2503.15633)の視覚ベンチマークに合わせ、論文用の主条件は **greedy decoding** です。質問音声末尾の後に10秒の無音を与え、末尾の自然な重なりを許すため0.32秒前から最初の応答を抽出します。

`--sampling` は挙動診断用です。使う場合もサンプルごとの乱数 seed、temperature、top-k を固定・記録するため、中断再開の順序で出力は変わりません。greedy と sampling の結果は混ぜられません。

### 音声入力とテキスト入力を対にする

本家 MoshiVis と同様に、同じ画像・同じ質問を音声入力とspeechlessテキスト入力の両方で実行できます。テキスト入力ではaudio embeddingを`zero_token_id=-1`にし、学習時のspeechless経路と同じ状態で自己回帰生成します。これにより、「画像知識を学べていない」のか、「質問音声からその知識を引き出せない」のかを分離します。

raw audio codeには`zero_token_id=-1`を使い、codebook delayの先頭では学習時の`forward_speech`と同じinitial tokenを入れます。現在の`vision_v1` lockは音声条件だけを凍結しているため、テキスト入力は`--diagnostic`必須です。論文値にする前に、最大生成token数を含むtext protocolを新しいsuite lockとして凍結します。

### 採点

| 対象 | 主指標 | 補助指標 |
|---|---|---|
| JA-VG-VQA-500 clean | SakanaAI 互換 MeCab ROUGE + 日本語判定 | normalized exact、文字 F1、edit similarity |
| JAMMEval CC-OCR / Heron / JA-VLM | pinned `simple-evals-mm` の soft exact-match judge、GPT-5.1で3回採点した mean ± std | normalized exact、文字 F1 |
| JAMMEval CVQA | 1〜4 の strict choice accuracy | 問題カテゴリ別 accuracy |

CVQA は応答全体が `2`、`2番`、`答えは2番です` のように一つの番号を答えた場合だけ正解とし、選択肢本文だけの一致や自由文中の数字は主指標で正解にしません。

JAMMEval judge は grader prompt と model snapshot を固定し、各 request を回答内容の hash に結び付けます。ROUGE は bootstrap seed、実装、MeCab 辞書、正規化、fastText、NumPy の版まで lock file と照合します。モデル失敗または judge 失敗が1件でもあれば論文値として確定しません。`answer-in-response`（参照答えが応答内に含まれるか）は診断値であり、主結果にはしません。

## 凍結済み成果物

- `data/evaluation/vision_v1/manifest_raw.jsonl`: 982問
- `data/evaluation/vision_v1/manifest_clean.jsonl`: 重複除外後734問
- `data/evaluation/vision_v1/manifest_audio.jsonl`: 固定音声付き734問
- `data/evaluation/v3_image_index.jsonl`: v3 画像 fingerprint index
- `j-moshivis/evaluation/suite_v1.lock.json`: 上記の revision、規模、SHA-256、採点規則

`data/` は Git 管理外なので、論文実験前に lock file と実体の SHA-256 が一致することを必ず確認します。

## 実行手順

以下は workspace root `/gpu-server/yamamoto/j-moshivis` から実行します。

### 1. 元ベンチマークを再構築する場合

既に構築済みなので、通常はこの節を再実行する必要はありません。

```bash
PYTHONPATH=j-moshivis .venv-data/bin/python \
  -m jmoshivis.evaluation.datasets.ja_vg_vqa \
  --output-dir data/evaluation/vision_v1/ja_vg \
  --profile official \
  --revision aa13b70cd939faa1783187a752be2c0ca0d8303b \
  --arrow-file /home/yamamoto/.cache/huggingface/datasets/SakanaAI___ja-vg-vqa-500/default/0.0.0/aa13b70cd939faa1783187a752be2c0ca0d8303b/ja-vg-vqa-500-test.arrow

PYTHONPATH=j-moshivis .venv-data/bin/python \
  -m jmoshivis.evaluation.datasets.jammeval \
  --output-dir data/evaluation/vision_v1/jammeval \
  --config all \
  --revision fc602195d5c9142c579e9a0e8434a9d7c54ab1f0

PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.merge_manifests \
  --manifest data/evaluation/vision_v1/ja_vg/manifest_test_official.jsonl \
  --manifest data/evaluation/vision_v1/jammeval/jammeval-cc-ocr-ja-refined/manifest_test.jsonl \
  --manifest data/evaluation/vision_v1/jammeval/jammeval-cvqa-ja-refined/manifest_test.jsonl \
  --manifest data/evaluation/vision_v1/jammeval/jammeval-heron-bench-refined/manifest_test.jsonl \
  --manifest data/evaluation/vision_v1/jammeval/jammeval-ja-vlm-bench-refined/manifest_test.jsonl \
  --output data/evaluation/vision_v1/manifest_raw.jsonl
```

各データセットの画像は再圧縮せず、元の encoded bytes を保存します。JAMMEval は content hash、JA-VG は公式 `image_id` で配置し、どちらも各行の `image_sha256` で元バイト列を固定します。
現在の JA-VG Arrow artifact の SHA-256 は `bde85f6353a5638e7dfd8af55b12b88cb055b9c581626e35de97c07c3eef1684` で、lock file にも保存しています。別環境では同 revision を取得し、同じ hash の Arrow を `--arrow-file` に渡します。

### 2. v3 重複監査

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.overlap.build_index \
  --train-manifest data/speech_v3/train.jsonl \
  --train-manifest data/speech_v3/eval.jsonl \
  --output data/evaluation/v3_image_index.jsonl \
  --workers 8 --resume

PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.overlap.audit \
  --manifest data/evaluation/vision_v1/manifest_raw.jsonl \
  --train-manifest data/speech_v3/train.jsonl \
  --train-manifest data/speech_v3/eval.jsonl \
  --image-index data/evaluation/v3_image_index.jsonl \
  --clean-manifest data/evaluation/vision_v1/manifest_clean.jsonl \
  --report data/evaluation/vision_v1/overlap_report_clean.jsonl \
  --exclude-near
```

### 3. 固定質問音声

```bash
PYTHONPATH=j-moshivis /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python \
  -m jmoshivis.evaluation.prepare_tts \
  --manifest data/evaluation/vision_v1/manifest_clean.jsonl \
  --output-manifest data/evaluation/vision_v2/manifest_audio.jsonl \
  --audio-dir data/evaluation/vision_v2/audio \
  --voices jvnv-M1,jvnv-F1 \
  --prompt-style verbatim
```

中断した同一設定を続けるときだけ `--resume` を付けます。現在の生成コードでは、TTS engine、BERT revision、依存バージョン、話者資産のいずれかが変わると cache key も変わります。既存 manifest の置換はコード側でも拒否します。作り直した場合は上記のように `vision_v2` として検証・凍結し、既存 `vision_v1` へ上書きしません。

### 4. 厳密検証

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.validate_manifest \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --require-audio --require-overlap-clean --strict-checksums
```

現在の正本では、734問、557画像、音声1.9296時間、checksum不一致0件です。

### 5. GPU 推論

新しいv3 runでは、`step_N/`はoptimizer等を含む**学習再開用directory**、
`final_step_N.safetensors`は**推論・評価用export**です。評価CLIには後者を渡します。

最初に、論文に使わない診断出力へ5問だけ流します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --checkpoint checkpoints/<run>/final_step_20000.safetensors \
  --output eval_results/<run>/smoke_greedy.jsonl \
  --device cuda:0 --limit 5 \
  --diagnostic --skip-checkpoint-hash
```

`answer_text`、質問終了 frame、text token、gate を目視し、5問すべてが空応答なら全件実行を始めません。診断が通った後、別の新規出力へ正式実行します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --checkpoint checkpoints/<run>/final_step_20000.safetensors \
  --output eval_results/<run>/vision_v1_greedy.jsonl \
  --device cuda:0
```

中断後は**同じコマンド**に `--resume` を追加します。checkpoint、manifest、画像・音声、tokenizer、Mimi、生成条件、依存バージョン、実装ファイルのどれかが変われば停止します。初回から `--resume` を付けると、安全な sidecar がないためエラーになります。

本家形式の「assistant挨拶→user質問」は、既存の `vision_v1` を変更せず、次の診断プロトコルで比較できます。挨拶WAVはassistant/outputストリームへteacher forcingし、同時間のuser/inputストリームには無音を入れます。WAVとプロトコルはrun metadataとresume fingerprintへ固定されます。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --checkpoint checkpoints/<run>/final_step_20000.safetensors \
  --output eval_results/<run>/prelude_diagnostic.jsonl \
  --audio-dialogue-protocol assistant_prelude_audio_only_v2 \
  --assistant-prelude-wav /path/to/frozen_assistant_greeting_24khz.wav \
  --device cuda:0 --limit 5 --diagnostic
```

`/path/to/...` は未作成資産を示すplaceholderであり、リポジトリ内に固定挨拶WAVが存在するという意味ではありません。現段階では挨拶に同期したtext token資産もないため、プレリュード区間のtextはモデル生成のまま採点から除外します。この条件を論文値に使う前に、24 kHz monoの挨拶WAVについて文言・話者・音声バイト列を確定し、そのSHA-256とプロトコルを新しいv2 suite lockへ固定する必要があります。

同じ質問のspeechlessテキスト入力を診断するときは、別の出力へ次を実行します。テキスト条件ではMimiや質問wavをモデル入力に使いません。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --checkpoint checkpoints/<run>/final_step_20000.safetensors \
  --output eval_results/<run>/vision_v1_text_greedy.jsonl \
  --input-modality text --max-new-tokens 96 \
  --device cuda:0 --diagnostic
```

音声・テキストで同じ採点器を使い、`sample_id`単位でも成否を比較します。音声だけ失敗する場合は音声転移・turn認識、両方失敗する場合は視覚学習または回答生成を優先して調べます。

### 6. ローカル採点

公式互換指標は学習環境と分離した Python 3.11 環境で実行します（fastText 0.9.2 と互換にするため NumPy 1.26.4 を固定）。

```bash
/gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python -m venv .venv-eval
.venv-eval/bin/python -m pip install -r j-moshivis/evaluation/requirements-eval.txt
```

```bash
PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.score \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --predictions eval_results/<run>/vision_v1_greedy.jsonl \
  --scores eval_results/<run>/vision_v1_scores.jsonl \
  --summary eval_results/<run>/vision_v1_summary.json \
  --official-rouge-ja \
  --lid176-model models/evaluation/lid.176.ftz
```

CLI は prediction の `.run.json` と manifest/checkpoint/result hash を検証します。追跡情報のない手作業ファイルを診断採点するときだけ `--allow-untracked-predictions` を使います。

### 7. JAMMEval short-answer judge

まず OpenAI Batch API 用 JSONL をローカル生成します。API 送信自体はこのコマンドでは行いません。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.judge prepare \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --predictions eval_results/<run>/vision_v1_greedy.jsonl \
  --output eval_results/<run>/jammeval_judge_requests.jsonl \
  --model gpt-5.1-2025-11-13 \
  --n-repeats 3
```

Batch の出力 JSONL を保存した後に集計します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.judge score \
  --manifest data/evaluation/vision_v1/manifest_audio.jsonl \
  --predictions eval_results/<run>/vision_v1_greedy.jsonl \
  --batch-requests eval_results/<run>/jammeval_judge_requests.jsonl \
  --judge-results eval_results/<run>/jammeval_judge_results.jsonl \
  --output eval_results/<run>/jammeval_judge_scores.jsonl \
  --summary eval_results/<run>/jammeval_judge_summary.json
```

request body と回答内容は hash で結び付いているため、別 run の judge 結果を同じ sample ID だけで流用できません。

ローカル採点 summary の `ready_for_local_metrics` と judge summary の `ready_for_jammeval_judge` は、異なる成果物の準備完了フラグです。**論文用の一式が完了したと判断するには両方が `true` で、manifest SHA と predictions SHA が一致すること**を確認します。どちらか一方だけを全評価の `paper_ready` とは扱いません。

## MoshiVis / Babillage 互換の3指標

既存の `vision_v1` は日本語主評価として維持し、それとは別に本家 MoshiVis と同じ種類の古典的ベンチマークを採点する CLI を用意しています。

データ構築、固定 revision、公式/overlap-clean の分離、suite freeze、音声・テキスト推論、採点までの完全な手順は [`MOSHIVIS_COMPAT.md`](MOSHIVIS_COMPAT.md) を参照してください。公式データの full download と full 評価はまだ実施していません。

| `metadata.moshivis_metric` | 必要な正解 | 指標 |
|---|---|---|
| `ocr_vqa_accuracy` | 1件以上の正答候補 | 小文字化だけを行う全文一致 Accuracy。空白・句読点は保持 |
| `vqav2_accuracy` | **10人分**の回答 | VQAv2 公式 leave-one-out consensus Accuracy |
| `cider` | 1画像につき複数caption | COCO PTBTokenizer + CIDEr |

既知のbenchmark名 (`ocr-vqa`、`vqav2`、`coco-captions` とBabillage側の名称) からも指標を推定しますが、固定manifestでは `metadata.moshivis_metric` を明記します。COCO は各評価行の `answers` に同じ画像の全正解captionを格納します。単一captionしかない行は、非互換なCIDEr値を誤って出さないよう拒否されます。

CIDEr用の独立依存を導入します。PTBTokenizerの実行にはJava runtimeも必要で、Javaのversion bannerも採点provenanceへ記録されます。

```bash
.venv-eval/bin/python -m pip install \
  -r j-moshivis/evaluation/requirements-moshivis.txt
```

音声入力・greedyで推論した既存形式の結果は、そのまま新CLIへ渡せます。採点対象は `answer_text`（音声と同期したモデルのテキストトークン）であり、出力音声のASR結果ではありません。

```bash
PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.score_moshivis \
  --manifest data/evaluation/moshivis_v1/manifest_audio.jsonl \
  --suite-lock j-moshivis/evaluation/moshivis_v1.lock.json \
  --predictions eval_results/<run>/moshivis_v1_audio_greedy.jsonl \
  --scores eval_results/<run>/moshivis_v1_scores.jsonl \
  --summary eval_results/<run>/moshivis_v1_summary.json
```

summary の `score` は Accuracy/CIDEr の内部値、`paper_score` は本家論文の図と同じ100倍表記です。CLIはmanifestとpredictionのhash、推論 `.run.json`、suite lock、greedy・入力modality・応答窓、checkpoint hash、採点コード・依存バージョンを照合します。すべて一致した場合だけ `paper_ready=true` になります。`--suite-lock` を省略した場合や、手製predictionへ `--allow-untracked-predictions` を指定した場合は診断値です。

このCLIが同じ指標を計算できることと、本家の公開値へ直接比較できることは別です。直接比較には、Babillage/元データのrevision・split、画像単位の正解集合、質問音声と会話プレリュード、生成・応答抽出条件をsuite lockへ固定する必要があります。

## 論文実験での扱い

- 現在の `step_17000` はデータセット構築中の挙動確認用であり、論文結果には含めません。
- 論文値は v3 完成後に同じ手順で学習した全条件だけを比較します。
- 合意済みの比較軸は、`speech_ratio={0,25,50,100%}`、データ量 `{25,50,100%}`、`J-Moshi / LLM-jp-Moshi`、画像 projection/adapter の継承対ランダム初期化です。
- すべての条件へ同一の734画像・同一wav・同一生成条件・同一採点を使います。
- 現チェックポイントの診断では greedy が全PAD、sampling も対象質問には不正解でした。これは評価基盤が失敗を隠さず観測できることの確認であり、性能値としては報告しません。

## テスト

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m unittest discover -s j-moshivis/tests/evaluation -v

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  j-moshivis/tests/evaluation/test_official_rouge.py -v
```
