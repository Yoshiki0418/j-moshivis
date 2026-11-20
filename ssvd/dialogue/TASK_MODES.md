# Image dialogue task modes

Both image-dialogue generators use the same task-aware input preparation,
prompt and validation code.

| Mode | Input requirement | Grounding supplied to the VLM | Eligible output |
|---|---|---|---|
| `normal` | `uid` or `candidate_uid`, `path` | none | ordinary image dialogue |
| `counting` | count fact with integer `answer_count >= 2` | original English count condition and canonical answer | first assistant turn contains that exact count naturally |
| `ocr` | exact transcription of 2--24 characters | canonical text | first assistant turn contains the exact text |
| `auto` | `capability` field | selects `counting`, `ocr`, or `normal` per row | same as the selected mode |

## Normal-dialogue strategies

`--normal-strategy standard` is the default and preserves the existing normal
prompt and output contract. `--normal-strategy vlm_adaptive` keeps
`task_mode: normal`, but asks the image-reading VLM to decide for each image
whether one natural follow-up exchange should exercise counting or short
Japanese OCR. There is no hash bucket or target percentage: ordinary dialogue
is the default, and `auxiliary_task.kind` may be `none` for any image.

The `count_balance_v2` prompt first looks for clearly countable groups of
2--10 people or objects. If both counting and OCR fit the image and conversation,
it prefers counting; otherwise it may choose OCR or `none`. This is a visual
decision by the VLM, not a fixed quota or random prompt switch. Canonical output
examples are shown only for the supplied annotation kinds, and unannotated
images explicitly require `vlm_visual`. A worked counting exchange keeps the
target name and zero-based turn indices consistent with validation. New local
adaptive records carry `prompt_version: count_balance_v2`; previously generated
rows are retained unchanged and skipped by UID during resumption.
Explicit quantity questions such as `何人`, `何個`, `何台`, or `いくつ` must
carry counting metadata even if the VLM calls the dialogue `none` or `ocr`.
Unlabelled quantity questions and multiple counting questions are rejected for
retry. Expressions such as `何人か` and `いくつか` are not quantity questions.

The adaptive prompt enforces the following policy:

- the first two exchanges remain ordinary image dialogue;
- at most one counting **or** OCR exchange may appear from turn index 4 onward;
- the user may ask only after the assistant has naturally introduced the
  relevant object or sign, because the user cannot see the image;
- counting prefers a safe canonical candidate when one is present and clearly
  supported by the image; canonical counts of one are allowed, but zero,
  body-part/possession, each/per and ambiguous crowd counts are not candidates;
- without a canonical candidate, the VLM may self-ground one count only when
  2--10 instances of one concrete object type are fully visible, clearly
  separated and unambiguous. It records `grounding_source: vlm_visual`, a
  concrete Japanese `counted_target`, and the integer `answer_count`. The
  validator requires that target in both the preceding assistant setup and the
  user question, no leaked count in either, and the exact count in the answer;
- OCR is allowed only for a clearly legible Japanese string of 2--24
  characters. Exact source transcription is used when available; otherwise
  the record is explicitly marked `grounding_source: vlm_visual`;
- if neither task fits naturally, the VLM emits `kind: none` and writes a fully
  ordinary dialogue.

For a canonical adaptive choice, the generator copies grounding from the
source manifest into the output after validation; it never trusts model-written
grounding as the authoritative source. A self-grounded visual count or OCR
string is retained in `auxiliary_task` with `grounding_source: vlm_visual` and
`verification_status: unverified_vlm_visual`, but does not receive canonical
`grounding`. The generator also records `review_required: true`, the source
image SHA-256, and a content-bound `review_id`. These rows are fail-closed:
they are not eligible for a training JSONL until that exact `review_id` is
approved in the Dataset Viewer.

In the Dataset Viewer, enable **人手確認が必要なデータのみ**, optionally
select **カウント** or **日本語 OCR**, and inspect the highlighted question
and answer. Decisions are appended to
`data/curation/dialogue_review_events.jsonl`. `pending` and `rejected` rows are
excluded by `ssvd/tts/make_train_jsonl.py`; only the latest exact `approved`
decision is included. Regenerating the dialogue or changing the reviewed image
produces a different `review_id`, so an earlier approval cannot carry over.

To generate ordinary dialogues only for images that have no dialogue of any
task mode yet, use `--registry-scope uid`. The default `task` scope remains
backward compatible and skips only an existing `(uid, task_mode)` pair.

For the 6,591-image counting pool, the existing 3,150 counting records and UID
scope leave exactly 3,441 normal-dialogue images:

```bash
/gpu-server/yamamoto/miniforge3/envs/vlm/bin/python \
  ssvd/dialogue/generate_local_vlm.py \
  --task-mode normal \
  --normal-strategy vlm_adaptive \
  --registry-scope uid \
  --registry_dir data/dialogues_counting/full \
  --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --backend vllm \
  --meta data/images/targeted_v4/image_pool_v1/counting_images.jsonl \
  --out data/dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl \
  --shard 0 --num-shards 2
```

Run the second disjoint shard with `--shard 1` and a separate output path.
Review a small pilot before launching the full generation, especially any
`grounding_source: vlm_visual` counting or OCR examples because their answers
are not canonically verified.

The answer is not discovered by rejecting guesses. The code supplies the
verified answer to the VLM before generation, and the VLM writes a natural
Japanese question and answer around it. Post-generation validation is only a
safety net; a validation failure remains retryable. Canonical grounding is
copied from the source manifest into the output by code; model-generated
metadata is never trusted.

For counting, the prompt receives only `question_en` and `answer_count`.
Internal provenance and selection fields are retained in output metadata but
are not exposed to the generator. The first Japanese question must preserve
the counted target and conditions (colour, location, state, and so on), while
the assistant phrases the supplied answer naturally with a suitable counter.

`split_group_key` is also carried through dialogue generation, speech
synthesis and the final train/eval manifest. All task variants of one source
therefore receive one deterministic split assignment. Speech directories keep
legacy normal UIDs and suffix targeted variants with `__task_counting` or
`__task_ocr`, so modes cannot silently overwrite one another.

## Counting with Claude Sonnet

```bash
.venv-data/bin/python ssvd/dialogue/generate_claude_cli.py \
  --task-mode counting \
  --meta data/images/targeted_v4/image_pool_v1/counting_images.jsonl \
  --out data/dialogues_counting/claude_sonnet5_counting.jsonl \
  --workers 2
```

For the resumable wrapper, use:

```bash
TASK_MODE=counting nohup bash ssvd/dialogue/run_gen_loop.sh \
  data/images/targeted_v4/image_pool_v1/counting_images.jsonl -- \
  data/dialogues_counting/claude_sonnet5_counting.jsonl
```

This selects 3,150 of the 6,591 images. Each selected image has at least one
verified count of two or more. One fact is chosen deterministically per image;
facts with answers zero or one are never shown to the VLM in this mode. When
an image has multiple facts, explicit visible person/object counts are chosen
before body-part, intrinsic-possession, or each/per questions. Lower-quality
facts are retained only as flagged fallbacks, so no otherwise eligible image
is silently removed. Current tiers are 1,779 direct visual, 1,211 ordinary,
and 160 fallback rows.

## Counting with a local VLM

```bash
/gpu-server/yamamoto/miniforge3/envs/vlm/bin/python \
  ssvd/dialogue/generate_local_vlm.py \
  --task-mode counting \
  --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --backend vllm \
  --meta data/images/targeted_v4/image_pool_v1/counting_images.jsonl \
  --out data/dialogues_counting/qwen3vl_counting.jsonl
```

Use `--max_samples 5` for the first pilot. Review whether the generated
Japanese question preserves every condition in `grounding.question_en`, such
as colour, position and state, before starting the full run.

## OCR mode

Use the same commands with `--task-mode ocr` and a short scene-text manifest.
Current JSSODa pages are 115 characters or longer and intentionally produce no
eligible OCR rows; the mode is ready for the upcoming shop-sign/telephone/
price dataset.

## 日本の日常画像の要確認対話を新規生成する場合

対象は同じ画像の元の要確認種別が OCR または counting で、人がまだ判断していない Qwen 対話。
2026-11-19 の対象候補 24,228 件のうち、旧対話の `auxiliary_task.question_turn_index` が 4（3 問目）のものは 24,183 件。
旧ラベルの内訳は OCR 16,806 件、カウント 7,422 件。これは元画像を抽出するための内訳で、再生成後の課題数は画像の内容によって決まる。

本家は会話履歴を共有する二つの話者が同じ画像説明を参照し、1 発話ずつ生成する。
論文本文は 8–16 往復、公開コードの既定は 8–16 発話（4–8 往復）。
本家の公開コードの既定に合わせ、4・5・6・7・8 往復を等確率で選ぶ。本家はミニバッチ単位で抽選する。こちらは画像 UID ごとに長さを再現でき、モデル間の比較で同じ長さを使える。
[論文 §3.3・付録 E](https://arxiv.org/html/2503.15633v1#S3.SS3)・[公開コードの長さ指定](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/generate.py#L305-L310)・[話者交替](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/multiturn_prompting.py#L181-L214)

1. 最初の質問から両役に同じ元画像を渡す。各発話の入力には、その時点までの両者の発話全文が順番どおり入る。初回の質問には画像選定時の種別を話題の一例として添える。旧対話の本文・未検証の個数や読み・画像タイトルは送信しない。
2. 画像に合ったタイミングで、自然に読める文字・見分けられる個数を扱う。普通の会話、両方を含む会話、根拠を添えた概数や部分読みも可能。話題の位置を事前に決める工程はなく、質問役には実際に読み取り・個数に触れた往復の番号も伝える。すでに答えが出た話題は掘り下げるか、別の話題に進む。
3. 回答役は発話と、その発話で扱った課題のメタデータを同じ応答で返す。公開された会話履歴には話者と発話本文だけを載せる。記録可能な各課題の実際の質問・回答位置、数値・原表記・読みは従来の `auxiliary_task` と `additional_tasks` に格納する。これはVLMの自己申告であり、最終的な正しさは人が画像で判断する。

| 課題 | 選べる回答 | 保存する情報 |
|---|---|---|
| カウント | 確定数・概数・見えている下限・不明 | 数値はそれぞれ `answer_count`・`estimated_count`・`minimum_count`。不明には数を割り当てない。 |
| OCR | 全部読める・一部だけ読める・判読できない | 全文は `recognized_text`、断片は `visible_fragment`。英数字を含む原表記と発話用の読みを別々に残す。不明には文字列を補わない。 |

確定以外では判断の理由を記録する。たとえば「手前に三台は見える。奥は重なっている」の三台は `minimum_count` に入る。読みの不一致、既出の質問、課題情報の欠落などは `review_warnings` に理由と発話番号（1始まり、対話全体は `null`）を保存して会話を最後まで生成する。人手比較用HTMLでは該当発話の横に注意を表示する。通常の会話を含む全候補が人手確認待ちとなり、正式な承認を経て利用できる。空の発話や課金状態が不明な送信は途中停止し、人手判断済み・変更済みの元データを保護する。

ローカルで次の Responses API リクエストを組み立てる例。下のコマンドは外部送信や台帳の更新を行わない。
リポジトリ直下で、対象 UID を指定する。出力には画像が含まれるので Git 管理外に保存する。

```bash
python3 -m ssvd.dialogue.preview_regeneration --uid "$PENDING_UID" > /tmp/jmv-next-request.json
```

再生時は初回の `identity` と、受信順の JSON 配列 `replies` を `--transcript` に渡す。最初の応答はユーザー発話、次は回答役の発話となる。結果は `request`、実行を続けられない場合の `hold`、完了時の `pending_review`。`--model` と `--pairs` で比較条件を揃えられる。

現在の方式は `visual_dialogue_v4`。4〜8往復あたり画像付き生成は8〜16回。v3の独立した事前観察にかかった生成1回分を省く。過去の部分応答は版ごとに区別し、v4の有料リクエストには流用しない。v3のオフライン再生には `--prompt-version grounded_roles_v3` を使う。[予算管理と人手判断の照合 #10](https://github.com/Yoshiki0418/j-moshivis/issues/10)

人手比較は同じ 20 画像（元ラベルで OCR 14、カウント 6）で行う。確定できる画像で過度に「分からない」と答えていないか、曖昧な画像で断定していないか、追質問や言い回しが自然か、全文を承認するまでの修正時間を記録する。生成後の通常・片方・両方の内訳と各回答種別、注意ラベルと実行未完了の件数も報告する。[少数試行 #12](https://github.com/Yoshiki0418/j-moshivis/issues/12)

### 20画像で試行する

リポジトリ直下で実行する。`prepare` は人手未確認の旧 Qwen 対話から抽出し、v3 の評価画像、固定の外部視覚評価・キャプション評価画像、試行内の画像と重複・類似する候補を除く。元の OCR／カウント分類や選定時の難しさは旧記録からの目印。元画像で実際に判別できる内容は各モデルが改めて観察し、人が確認する。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot prepare
.venv-data/bin/python -m ssvd.dialogue.pilot status
.venv-data/bin/python -m ssvd.dialogue.pilot review
```

`data/curation/gpt_pilot_issue12/visual_dialogue_v4/` に固定した画像リスト、SQLite台帳、匿名比較用の `review.html` を保存する。過去の台帳がある場合は同じ20枚・同じ往復数で比較し、生成時には最新の人手判断も確認する。v2 の集計は `status --legacy` と `recovery-status --legacy`、v3は `--archive grounded_roles_v3` で参照できる。候補生成後は、両モデルの全文と既存Qwen対話を原画像・448px画像で比較できる。正式な承認とTTSへの投入は既存の確認手順で行う。

有料の `run` は同期 Responses API で20画像を両モデルに投入する。`OPENAI_API_KEY` または権限600のキーファイルを使う。既定のファイルは `~/.config/j-moshivis/openai_api_key`、別パスは `--api-key-file` で指定できる。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot run
```

この試行の生成予算は旧方式の請求額・未確定予約も含めて累計2米ドル。送信前に画像・履歴を含む入力トークン数を公式APIで調べ、推論を含む最大出力分まで予約する。実績額は返された使用量とStandardの公表単価で算出する。課金状態が不明のまま終了した場合は、台帳を保持したまま同じコマンドで再開する。その呼び出しの予約は保持し、当該画像・モデルは保留として残りの組を処理する。照会に必要な送信時刻と `X-Client-Request-Id` は台帳の `calls` にある。[トークン数の確認](https://developers.openai.com/api/docs/guides/token-counting)・[Lunaの単価](https://developers.openai.com/api/docs/models/gpt-6-luna)・[Terraの単価](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### 過去の比較結果を開く

v2・v3には事前観察と予定した往復での整合性チェックがあり、初回版と追加比較版が保存されている。追加比較版の集計・ページは次のコマンドで参照できる。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-status --archive grounded_roles_v3
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-review --archive grounded_roles_v3
```

v4では品質上の注意が付いた候補も最後まで生成される。初回の確認は `review.html` で行う。

## テストの実行

```bash
PYTHONPATH=ssvd/dialogue .venv-data/bin/python -m unittest discover \
  -s ssvd/dialogue/tests -p 'test_*.py' -v

PYTHONPATH=ssvd/tts /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python \
  -m unittest discover -s ssvd/tts/tests -p 'test_*.py' -v
```
