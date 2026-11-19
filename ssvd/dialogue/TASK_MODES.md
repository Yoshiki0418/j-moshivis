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
同じ UID に新方式を適用したときの質問位置の割当は、2 問目 6,353 件、3 問目 6,340 件、4 問目 6,404 件、5 問目 3,587 件、6 問目 1,544 件。生成前の計算であり、保留や評価画像の除外によって完成した対話の分布は変わる。

本家は会話履歴を共有する二つの話者が同じ画像説明を参照し、1 発話ずつ生成する。
論文本文は 8–16 往復、公開コードの既定は 8–16 発話（4–8 往復）。
今回は日本語の既存データの長さに合わせて 4–6 往復とし、初問の切り口と課題を尋ねる位置を画像ごとに変える。
「うん」「へえ」などの挿入回数を自然さの指標にしない。
[論文 §3.3・付録 E](https://arxiv.org/html/2503.15633v1#S3.SS3)・[公開コードの長さ指定](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/generate.py#L305-L310)・[話者交替](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/multiturn_prompting.py#L181-L214)

1. 画像を一度送り、会話に使える場面の説明と、元の課題に対応する根拠を作る。元対話の本文、数、読み、画像タイトルはプロンプトに含めない。対象を数え切れない場合や文字を判別できない場合は、その画像を同じ種別で保留する。
2. ユーザー役は会話で初めて聞いた事柄から次の問いを作る。アシスタント役は画像から得た根拠を参照して答える。OCR では画像上の原表記と音声対話用の読みを別々に記録する。
3. 所定の位置に元種別の問いと答えがあるかを機械的に確認し、根拠との不一致は保留する。この検査は画像に対する正答を保証しない。最終候補は新しい review ID を持ち、本人の承認対象になる。

ローカルで次の Responses API リクエストを組み立てる例。下のコマンドは外部送信や台帳の更新を行わない。
リポジトリ直下で、対象 UID を指定する。出力には画像が含まれるので Git 管理外に保存する。

```bash
python3 -m ssvd.dialogue.preview_regeneration --uid "$PENDING_UID" > /tmp/jmv-next-request.json
```

再生時は初回の出力の `identity` と、同じ設定でそれまでに得た JSON オブジェクトの配列 `replies` を一つの JSON にして `--transcript` に渡す。応答順は画像の観察、ユーザー、アシスタントの繰り返し。結果は `request`、`hold`、`pending_review` のいずれか。完了時の `candidate` は確認用の別版。`--model` と `--pairs` で比較条件を揃えられる。

本実行時の評価画像の除外、費用計上、再開、送信と保存時の人手判定の再照合は [#10](https://github.com/Yoshiki0418/j-moshivis/issues/10) の範囲。モデル選定では同じ 20 画像（OCR 14、カウント 6）について、指定した対象を実際に数えた・読んだか、前の返答に沿う自然な追質問か、別の発話にも根拠のない説明がないか、全文の修正時間を人が確認する。保留を含む全画像を分母として種別別の件数と費用を記録する。[少数試行 #12](https://github.com/Yoshiki0418/j-moshivis/issues/12)

## テストの実行

```bash
PYTHONPATH=ssvd/dialogue .venv-data/bin/python -m unittest discover \
  -s ssvd/dialogue/tests -p 'test_*.py' -v

PYTHONPATH=ssvd/tts /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python \
  -m unittest discover -s ssvd/tts/tests -p 'test_*.py' -v
```
