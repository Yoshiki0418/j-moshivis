# MoshiVis / Babillage 互換評価

この文書は、既存の日本語主評価 `vision_v1` とは独立に、本家
[MoshiVis](https://arxiv.org/abs/2503.15633) の視覚理解評価について、公開情報から確認できる入力データ、greedy生成、同期テキスト採点、指標を互換化する手順です。本家の非公開runtimeまで完全に同一であるとは主張しません。

ここで提供するのは**データセット本体ではなく評価基盤**です。手元の画像・質問・正解を下記 JSONL manifest 契約へ変換すれば、その manifest を suite-lock、推論、採点 CLI へ差し替えて利用できます。Babillage の full download/build は公式公開セットを使う場合だけの任意 adapter 手順であり、基盤を使うための必須条件ではありません。

本家の基本条件は次のとおりです。

- ベンチマーク: COCO Captions 2014、OCR-VQA、VQAv2
- 入力: 同一問題を音声とテキストの両方で評価
- 生成: greedy decoding
- 採点対象: 音声と同時にモデルが生成する同期テキストトークン。生成音声を ASR に通した文字列ではない
- 指標: COCO は CIDEr、OCR-VQA は Accuracy、VQAv2 は VQA Accuracy

`build_babillage_manifest` は、Kyutai が公開した Babillage の質問音声を元ベンチマークの画像・正解と結合します。元の encoded image/audio bytes を再圧縮せず保存し、各バイト列、dataset revision、正解集合を manifest に記録します。

### 任意データを差し込む最小 manifest 契約

全行に、一意な `sample_id`、`benchmark`、`split`、`domain`、`task`、`image_path`、`question`、`answers`、`source.image_sha256` を持たせます。音声入力も評価する行は、さらに `audio_path` と、質問本文および WAV に結び付いた `metadata.question_audio`（`sample_id`、`prompt_text`、prompt の SHA-256、`wav_sha256`）が必要です。パスは manifest からの相対パスを使用できます。

| 評価 | 各行の追加最小契約 |
|---|---|
| OCR-VQA | `task="vqa"`、`answers` に1件以上の正答文字列、`metadata.moshivis_metric="ocr_vqa_accuracy"` |
| VQAv2 | `task="vqa"`、`answers` に annotator 10人分を重複も保持して格納、`metadata.moshivis_metric="vqav2_accuracy"` |
| Caption | `task="caption"`、`answers` に同一画像の参照 caption を2件以上、`metadata.moshivis_metric="cider"` |

`benchmark` 名から指標を推定する互換動作もありますが、独自 manifest では `metadata.moshivis_metric` を明記してください。これによりデータ名と採点方法の取り違えを防げます。

Babillage の公開promptは英語です。本家と同じ入力で比較する公式suiteでは英語音声・英語テキストを変更しません。その結果は J-MoshiVis の英語理解力も含むため、低スコアを直ちに画像理解不足だけへ帰属できません。日本語翻訳 + 日本語TTS版を作る場合は有用な追加評価ですが、別名のadapted suiteとしてfreezeし、本家公表値とは直接比較しません。

> **現在の状態（2026-09-30）:** adapter、採点器、suite-lock v2 と実行 CLI は実装済みですが、公式データの full download と full manifest の生成・推論はまだ実施していません。`data/evaluation/` に Babillage full suite が既にある、または本家の数値を再現済み、という意味ではありません。

## 固定 revision、規模、概算容量

adapter のデフォルトは次の commit に固定されています。論文実験中に安易に `--base-revision` を上書きしないでください。

| データ | 固定 revision | 利用可能 split / Babillage 行数 |
|---|---|---:|
| `kyutai/Babillage` | `3ede8566da87763fffe3e0a1d9b78cd76bf5b85d` | 下記3構成 |
| `HuggingFaceM4/COCO` (`2014`) source | `4d0dfd4a3712a80e7b8498ed76e778f7cea9d21c` | annotation/schemaの元revision |
| `HuggingFaceM4/COCO` auto-converted Parquet（実ロード） | `29bc9e9117edd016db886abecb3c6f5c2e7f7c2b` | `2014/coco-{validation\|test}-*.parquet`、各25,010行 |
| `howard-hou/OCR-VQA` | `88234cc092c5f6d199b5cf3b471e3b490a69c07b` | validation 100,032 / test 100,424 |
| `lmms-lab/VQAv2` | `32665d35052eb4a6d4414851c3c829a72754915a` | validation 214,354 |

COCOの2 revisionは役割が異なります。`4d0d...`は元のdataset
script・annotation定義を固定するsource revisionです。一方、`datasets==5.0.1`は旧
`COCO.py`を実行できないため、builderが実際に読むデータはHugging Faceの
auto-convert成果物 `29bc...` にある固定Parquetです。manifestとsummaryには
`base_dataset_revision`（source）と`base_dataset_storage_revision`（実ロード）を
両方記録し、dry-runにも完全なstorage URIを表示します。

3構成の validation を合わせると 339,396 行です。Babillage 側の Hugging Face metadata による download size は、COCO 約3.76 GB、OCR-VQA 約12.65 GB、VQAv2 約4.78 GB、合計約21.2 GBです。これとは別に元ベンチマークの画像・annotation、Hugging Face cache、content-addressed output が必要です。COCO と VQAv2 の元画像だけでも数十 GBになるため、full 構築時は **少なくとも100 GB程度の空き容量**を確保してください。cache の状態によって必要量は増減します。

339,396 は画像数ではなく**音声prompt行数**です。現在の audio runtime は各行について「質問音声 + 固定tail」の全frameを逐次処理し、EOSによる早期終了をしません。そのため、公開セットを任意構築する場合も、最初は `--limit` の診断を行い、必要ならbenchmarkごとに別のmanifest/lock/runへ分け、同一runの中断には `--resume` を使ってください。現CLIには、同一benchmarkを複数GPUへ正式shardして paper-ready な1結果へmergeする機能は含みません。出力JSONLを手作業で連結して論文値にしないでください。

ライセンスも別々に確認します。Babillage、COCO、VQAv2 の adapter 記録値は CC-BY-4.0 ですが、現在使用している OCR-VQA の Hugging Face 再配布版は dataset card 上でライセンス未宣言です。再配布可否を Babillage のライセンスだけで判断しないでください。

## 1. 公式 CIDEr 環境

評価依存は学習環境から分離します。`pycocoevalcap` の PTBTokenizer は Java を呼ぶため、Python package だけでは CIDEr を実行できません。

```bash
.venv-eval/bin/python -m pip install \
  -r j-moshivis/evaluation/requirements-moshivis.txt

export PATH="$PWD/.venv-eval/java/bin:$PATH"
java -version
.venv-eval/bin/python -c \
  'import importlib.metadata as m; print(m.version("pycocoevalcap"))'
```

この workspace では `.venv-eval/java/bin/java` を使う想定です。別環境では Java runtime を用意し、同じ PATH を suite の freeze と採点の両方で使ってください。freeze 時には `pycocoevalcap` の版、Java version banner、同梱 PTB tokenizer JAR の SHA-256、採点コードの SHA-256 が lock に保存されます。

## 2. Babillage と元正解の結合（任意 adapter）

この節は公開 Babillage をそのまま評価データとして materialize したい場合だけ実行します。独自 manifest を上の契約で用意する場合、full download は不要です。

まず download を行わない dry-run で revision、split、行数を確認します。

```bash
PYTHONPATH=j-moshivis .venv-data/bin/python \
  -m jmoshivis.evaluation.datasets.build_babillage_manifest \
  --output-dir data/evaluation/moshivis_babillage_v1 \
  --config all --split validation --dry-run
```

full validation を構築するコマンドは次です。大容量 download が発生します。

```bash
PYTHONPATH=j-moshivis .venv-data/bin/python \
  -m jmoshivis.evaluation.datasets.build_babillage_manifest \
  --output-dir data/evaluation/moshivis_babillage_v1 \
  --config all --split validation
```

出力は構成ごとに分かれます。

```text
data/evaluation/moshivis_babillage_v1/
├── babillage-coco/manifest_validation.jsonl
├── babillage-ocrvqa/manifest_validation.jsonl
└── babillage-vqav2/manifest_validation.jsonl
```

COCO と OCR-VQA の test も必要なら、validation を上書きせず、別の出力先へ `--split test` で構築します。VQAv2 adapter は正解10件が得られる validation のみを受け付けます。`--limit` は接続・schema確認用であり、その出力を論文値に使いません。

3 manifest を安全に結合します。画像・音声への相対 path は `merge_manifests` が出力先を基準に張り直します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.merge_manifests \
  --manifest data/evaluation/moshivis_babillage_v1/babillage-coco/manifest_validation.jsonl \
  --manifest data/evaluation/moshivis_babillage_v1/babillage-ocrvqa/manifest_validation.jsonl \
  --manifest data/evaluation/moshivis_babillage_v1/babillage-vqav2/manifest_validation.jsonl \
  --output data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl
```

## 3. 学習画像との重複監査

本家との比較用と、未知画像能力を主張するための集合を混同しません。

1. **公式 split:** `manifest_validation_official.jsonl` をそのまま使う。本家と同じ公開 split/protocol に最も近い比較用で、v3 train との重複件数・割合を結果に必ず併記する。
2. **overlap-clean:** 学習画像との一致・近似一致を除いた別 manifest を作る。未知画像への汎化を主張する主評価に使うが、問題集合が変わるため本家の公表値と直接比較しない。

公式 manifest を clean 結果で上書きしてはいけません。v3 index が未作成なら最初に作ります。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.overlap.build_index \
  --train-manifest data/speech_v3/train.jsonl \
  --train-manifest data/speech_v3/eval.jsonl \
  --output data/evaluation/v3_image_index.jsonl \
  --workers 8 --resume
```

続いて audit report と clean manifest を別名で出力します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.overlap.audit \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --train-manifest data/speech_v3/train.jsonl \
  --train-manifest data/speech_v3/eval.jsonl \
  --image-index data/evaluation/v3_image_index.jsonl \
  --clean-manifest data/evaluation/moshivis_babillage_v1/manifest_validation_overlap_clean.jsonl \
  --report data/evaluation/moshivis_babillage_v1/overlap_validation_report.jsonl \
  --exclude-near
```

両方の manifest を別々に検証します。

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.validate_manifest \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --require-audio --strict-checksums

PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.validate_manifest \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_overlap_clean.jsonl \
  --require-audio --require-overlap-clean --strict-checksums
```

## 4. suite を freeze する

full build と重複監査を確認してから、公式 split と overlap-clean に別々の lock を作ります。lock は既存ファイルを上書きしないため、条件を変更したら suite ID とファイル名を進めます。

```bash
export PATH="$PWD/.venv-eval/java/bin:$PATH"

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.freeze_moshivis_suite \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --output j-moshivis/evaluation/moshivis_babillage_validation_v1.lock.json \
  --suite-id moshivis_babillage_validation_v1 \
  --paper-input-modalities audio text \
  --text-max-new-tokens 96 \
  --response-tail-seconds 10.0 \
  --response-overlap-seconds 0.32 \
  --audio-dialogue-protocol bare_question_v1

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.freeze_moshivis_suite \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_overlap_clean.jsonl \
  --output j-moshivis/evaluation/moshivis_babillage_validation_clean_v1.lock.json \
  --suite-id moshivis_babillage_validation_clean_v1 \
  --paper-input-modalities audio text \
  --text-max-new-tokens 96 \
  --response-tail-seconds 10.0 \
  --response-overlap-seconds 0.32 \
  --audio-dialogue-protocol bare_question_v1
```

本家論文は、speech model向けに assistant の短い挨拶を先頭へ置く会話形式も説明しています。しかし現在の `assistant_prelude_audio_only_v2` は挨拶**音声だけ**を注入し、対応する同期 assistant text を teacher-forcing しません。そのため本家と同等の会話履歴とはみなさず、論文用 suite へ freeze できない診断専用条件にしています。論文用 lock は `bare_question_v1` のみです。挨拶の影響を探索するときは、上記の bare-question lock に対して `run_inference --diagnostic --audio-dialogue-protocol assistant_prelude_audio_only_v2 --assistant-prelude-wav <24kHz-mono-WAV>` を使います。この診断値は `paper_ready=true` になりません。

## 5. 音声・テキストの greedy 推論

`--sampling` を付けないことが greedy 条件です。正式結果では `--diagnostic`、`--skip-checkpoint-hash`、`--limit` を使いません。以下は公式 split の例で、overlap-clean は manifest、lock、出力名を clean 版へ置き換えます。

まずbenchmarkごとに `--limit 5 --diagnostic` のような小規模runを通し、画像・音声・同期テキスト抽出が成立することを確認します。`--limit` run の sidecar は `incomplete` のままなので、採点できても `paper_ready=true` にはなりません。公開セット全体を実行する必要がある場合は COCO / OCR-VQA / VQAv2 をそれぞれ別のmanifest・lock・runにできます。ただし、現CLIは同一benchmark内の正式shard mergeを提供しないため、paper-ready結果では1つのrunを `--resume` して完走させます。

音声入力:

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --suite-lock j-moshivis/evaluation/moshivis_babillage_validation_v1.lock.json \
  --checkpoint checkpoints/<run>/final_step_<N>.safetensors \
  --output eval_results/<run>/babillage_validation_audio_greedy.jsonl \
  --input-modality audio \
  --response-seconds 10.0 \
  --response-overlap-seconds 0.32 \
  --audio-dialogue-protocol bare_question_v1 \
  --device cuda:0
```

テキスト入力:

```bash
PYTHONPATH=j-moshivis .venv-train/bin/python \
  -m jmoshivis.evaluation.run_inference \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --suite-lock j-moshivis/evaluation/moshivis_babillage_validation_v1.lock.json \
  --checkpoint checkpoints/<run>/final_step_<N>.safetensors \
  --output eval_results/<run>/babillage_validation_text_greedy.jsonl \
  --input-modality text --max-new-tokens 96 \
  --response-seconds 10.0 \
  --response-overlap-seconds 0.32 \
  --audio-dialogue-protocol bare_question_v1 \
  --device cuda:0
```

中断再開時だけ、まったく同じコマンドへ `--resume` を加えます。実行 sidecar (`*.run.json`) は manifest、suite、checkpoint、tokenizer、Mimi、生成条件、実装を固定し、変更を検出したら再開を拒否します。

## 6. 採点

音声結果とテキスト結果は別々に採点し、同じ指標で並べます。

```bash
export PATH="$PWD/.venv-eval/java/bin:$PATH"

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.score_moshivis \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --suite-lock j-moshivis/evaluation/moshivis_babillage_validation_v1.lock.json \
  --predictions eval_results/<run>/babillage_validation_audio_greedy.jsonl \
  --scores eval_results/<run>/babillage_validation_audio_scores.jsonl \
  --summary eval_results/<run>/babillage_validation_audio_summary.json

PYTHONPATH=j-moshivis .venv-eval/bin/python \
  -m jmoshivis.evaluation.score_moshivis \
  --manifest data/evaluation/moshivis_babillage_v1/manifest_validation_official.jsonl \
  --suite-lock j-moshivis/evaluation/moshivis_babillage_validation_v1.lock.json \
  --predictions eval_results/<run>/babillage_validation_text_greedy.jsonl \
  --scores eval_results/<run>/babillage_validation_text_scores.jsonl \
  --summary eval_results/<run>/babillage_validation_text_summary.json
```

summary の `paper_ready=true` は、manifest・suite・greedy・modality・テキスト最大生成token数・応答窓・**画像/質問音声の frozen asset inventory**・checkpoint hash・採点コード・依存がすべて lock と一致し、欠損や推論失敗がないことを表します。したがって、manifest のパスが同じでも参照画像や質問音声の実体が変われば論文用結果として受理しません。`--suite-lock` なし、または `--allow-untracked-predictions` 付きの採点は診断値です。

### 指標の定義

- **OCR-VQA Accuracy:** 予測と正答候補を小文字化し、全文が一致したときだけ正解です。空白、句読点、説明文を除去しません。正答文字列の部分一致率ではありません。
- **VQAv2 Accuracy:** 10人の annotator 回答へ公式の句読点・数字・冠詞正規化を行い、各 annotator を1人ずつ除外した consensus score を平均します。単純な全文一致でも `min(一致人数/3, 1)` の近似でもありません。
- **COCO CIDEr:** `pycocoevalcap==1.2` の Stanford PTBTokenizer と CIDEr を使う corpus score です。各評価行に同一画像の複数captionを保持します。内部値が約1.2なら、`paper_score` は本家図と同じ約120表記です。

すべて `answer_text`、すなわちモデルの同期テキストストリームを採点します。音声出力の自然さや発音品質はこの3指標には含まれず、別の音声品質評価が必要です。

## 数値比較で必ず付ける注意

本家 Figure 4 の主結果は、一つの汎用会話モデルを3ベンチマークへ zero-shot 評価した値ではありません。OCR-VQA、VQAv2、COCO の**タスクごとに別々に adaptation modules を fine-tune**し、`p_audio` を変えて評価しています。また、この節では synthetic visual dialogue を学習に使っていません。

したがって、汎用対話用 J-MoshiVis をこの基盤へ流した zero-shot 値は、入力形式と指標を揃えた重要な診断・外部評価にはなりますが、本家 Figure 4 との完全な apples-to-apples 比較ではありません。直接比較を主張する場合は、少なくとも次を表に分けます。

- J-MoshiVis conversational zero-shot
- J-MoshiVis task-specific fine-tune（実施した場合）
- MoshiVis 論文の task-specific fine-tune 公表値
- 公式 split と overlap-clean split（後者は本家値と直接比較しない）

さらに、COCO の Babillage は同一画像に複数の音声prompt行があります。この adapter は各promptを全caption参照で採点するため、標準的な「1画像1予測」の COCO 単位とも区別し、manifest の行数と unique image 数を必ず報告します。
