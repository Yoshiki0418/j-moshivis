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

1. 最初に元画像を観察し、会話で自然に触れられる課題を 0〜2 種類選ぶ。元ラベルは参考情報として渡す。旧対話の本文・未検証の個数や読み・画像タイトルは渡さない。通常の会話だけでも、同じ画像の看板の文字と周囲の物の数を別の往復で尋ねてもよい。
2. 最初の質問から最後の回答まで、両役の各リクエストに同じ元画像を添付する。ユーザー役は画像と会話を見て質問を作る。予定した課題では尋ねる対象を伝え、個数・原表記・読みなどの正解メモは回答役にだけ渡す。回答役は写真を見直し、観察メモになかった形や色にも答えられる。数えきれない／表示を読めないときは、画像から判断できる範囲で返す。
3. 元画像のラベルは生成履歴の `identity.kind` に、実際の課題と発話位置は候補の `auxiliary_task` に保存する。2 種類なら後の課題が `additional_tasks` に入り、どちらの変更も review ID に反映される。通常対話だけの候補も人手確認待ちになる。

| 課題 | 選べる回答 | 保存する情報 |
|---|---|---|
| カウント | 確定数・概数・見えている下限・不明 | 数値はそれぞれ `answer_count`・`estimated_count`・`minimum_count`。不明には数を割り当てない。 |
| OCR | 全部読める・一部だけ読める・判読できない | 全文は `recognized_text`、断片は `visible_fragment`。英数字を含む原表記と発話用の読みを別々に残す。不明には文字列を補わない。 |

確定以外では判断の理由を記録する。たとえば「手前に三台は見える。奥は重なっている」の三台を、画面全体の正解数として扱わない。観察自体が成立しない画像や、観察と食い違う生成は `hold`。言い回しと画像の対応の最終判断は人が行う。

ローカルで次の Responses API リクエストを組み立てる例。下のコマンドは外部送信や台帳の更新を行わない。
リポジトリ直下で、対象 UID を指定する。出力には画像が含まれるので Git 管理外に保存する。

```bash
python3 -m ssvd.dialogue.preview_regeneration --uid "$PENDING_UID" > /tmp/jmv-next-request.json
```

再生時は初回の出力の `identity` と、同じ設定でそれまでに得た JSON オブジェクトの配列 `replies` を一つの JSON にして `--transcript` に渡す。応答順は画像の観察、ユーザー、アシスタントの繰り返し。結果は `request`、`hold`、`pending_review` のいずれか。完了時の `candidate` は確認用の別版。`--model` と `--pairs` で比較条件を揃えられる。

両役に画像を渡す方式は `grounded_roles_v3`。観察だけに画像を渡した v2 の部分応答は、v3 のリクエストに再利用しない。
4〜8 往復なら、観察 1 回と各発話を合わせて画像付きリクエストは 9〜17 回になる。各回の画像・テキスト入力と実際の出力を台帳で集計するため、v2 の実測単価をそのまま流用できない。[予算管理と人手判断の照合 #10](https://github.com/Yoshiki0418/j-moshivis/issues/10)

人手比較は同じ 20 画像（元ラベルで OCR 14、カウント 6）で行う。確定できる画像で過度に「分からない」と答えていないか、曖昧な画像で断定していないか、追質問や言い回しが自然か、全文を承認するまでの修正時間を記録する。生成後の通常・片方・両方の内訳と各回答種別、保留件数も報告する。[少数試行 #12](https://github.com/Yoshiki0418/j-moshivis/issues/12)

### 20画像で試行する

リポジトリ直下で実行する。`prepare` は人手未確認の旧 Qwen 対話から抽出し、v3 の評価画像、固定の外部視覚評価・キャプション評価画像、試行内の画像と重複・類似する候補を除く。元の OCR／カウント分類や選定時の難しさは旧記録からの目印。元画像で実際に判別できる内容は各モデルが改めて観察し、人が確認する。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot prepare
.venv-data/bin/python -m ssvd.dialogue.pilot status
.venv-data/bin/python -m ssvd.dialogue.pilot review
```

`data/curation/gpt_pilot_issue12/grounded_roles_v3/` に固定した画像リスト、SQLite台帳、匿名比較用の `review.html` を保存する。v2 の台帳がある場合は同じ20枚・同じ往復数で比較し、生成時には最新の人手判断も確認する。v2 の集計は `status --legacy` と `recovery-status --legacy` で参照できる。候補生成後は、両モデルの全文と既存Qwen対話を原画像・448px画像で比較できる。正式な承認とTTSへの投入は既存の確認手順で行う。

有料の `run` は同期 Responses API で20画像を両モデルに投入する。`OPENAI_API_KEY` または権限600のキーファイルを使う。既定のファイルは `~/.config/j-moshivis/openai_api_key`、別パスは `--api-key-file` で指定できる。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot run
```

この試行の生成予算は旧方式の請求額・未確定予約も含めて累計2米ドル。送信前に画像・履歴を含む入力トークン数を公式APIで調べ、推論を含む最大出力分まで予約する。実績額は返された使用量とStandardの公表単価で算出する。課金状態が不明のまま終了した場合は、台帳を保持したまま同じコマンドで再開する。その呼び出しの予約は保持し、当該画像・モデルは保留として残りの組を処理する。照会に必要な送信時刻と `X-Client-Request-Id` は台帳の `calls` にある。[トークン数の確認](https://developers.openai.com/api/docs/guides/token-counting)・[Lunaの単価](https://developers.openai.com/api/docs/models/gpt-6-luna)・[Terraの単価](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### 保留を再検査して、未生成の発話を補う

初回40件の結果を確定してから、同じ20枚・同じモデルの応答を台帳から再生する。`recover` は、例えば直前の返答で「白い看板」を紹介し、次の質問で「その看板」と呼んだ場合などに、自然な言い換えや省略を再検査する。画像観察に同じ種類の独立した課題が二つあり、それぞれ単独で検査を通る場合は、先頭の課題を会話に使う。元の観察と応答は初回の台帳に残る。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot recover --api-key-file /path/to/private-key
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-status
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-review
```

追加比較は、既に受け取った各発話のリクエスト指紋を照合してから、次の未生成の発話を要求する。初回に全文が揃った候補は引き継ぐ。対応する根拠が取れないものは保留が続く。推定費用の上限は初回と共通の2米ドルで、人手確認と採用の条件も共通。`review.html` は初回、`recovery/review.html` は追加比較で作成した匿名の確認画面。各ページでは両モデルの全文が揃った画像に直接移動できる。

## テストの実行

```bash
PYTHONPATH=ssvd/dialogue .venv-data/bin/python -m unittest discover \
  -s ssvd/dialogue/tests -p 'test_*.py' -v

PYTHONPATH=ssvd/tts /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python \
  -m unittest discover -s ssvd/tts/tests -p 'test_*.py' -v
```
