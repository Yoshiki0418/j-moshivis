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
Long JSSODa pages use the [dedicated OCR generator](generate_jssoda_ocr_dataset.py),
which checks passages against the supplied page transcription.

## 日本の日常画像の要確認対話を新規生成する場合

対象は同じ画像の元の要確認種別が OCR または counting で、人がまだ判断していない Qwen 対話。
2026-11-19 の対象候補 24,228 件のうち、旧対話の `auxiliary_task.question_turn_index` が 4（3 問目）のものは 24,183 件。
旧ラベルの内訳は OCR 16,806 件、カウント 7,422 件。これは元画像を抽出するための内訳で、再生成後の課題数は画像の内容によって決まる。

本家は会話履歴を共有する二つの話者が同じ画像説明を参照し、1 発話ずつ生成する。
論文本文は 8–16 往復、公開コードの既定は 8–16 発話（4–8 往復）。
本家の公開コードの既定に合わせ、4・5・6・7・8 往復を等確率で選ぶ。本家はミニバッチ単位で抽選する。こちらは画像 UID ごとに長さを再現でき、モデル間の比較で同じ長さを使える。
[論文 §3.3・付録 E](https://arxiv.org/html/2503.15633v1#S3.SS3)・[公開コードの長さ指定](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/generate.py#L305-L310)・[話者交替](https://github.com/kyutai-labs/moshivis/blob/main/ssvd/multiturn_prompting.py#L181-L214)

### 生成時に両役へ渡すもの

各発話で、同じ元画像1枚と、そこまでに実際に交わした全発話を順番に渡す。旧対話のOCR・カウントラベルは選定と集計に使う。画像の事前要約、切り抜き、旧対話の答え、回答役が記録した非公開の注釈は入力に含めない。[リクエストの組み立て](regeneration_roles.py)

| 話者 | 話す目的 | 次の発話を作るとき |
|---|---|---|
| ユーザー役（v11） | 一緒に見えているものをきっかけに話す | 知りたいことは尋ねる。相手の返事から好みや感想が浮かんだら、それだけで番を渡せる。日常の話に広げたり、視線を別の対象に移したりしてよい。 |
| アシスタント役（v11） | 直前に相手が伝えたことに応じる | 質問には見える範囲で具体的に答える。感想には相手が関心を持った点から話を続ける。一般的な話を、画像内で実際に起きたこととして語らない。 |

例えば「入口の青い看板、何て書いてある？」のように対象から話し始められる。会話の舞台や話者の居場所は設定しない。看板に貼られた写真など、写真自体が話題のときはその語も使える。一枚から確かめられない前後の出来事や、話者が近づいたり手に取ったりした体験は作らない。

両役の発話は声に出す日本語にする。英字はカタカナで、数字は用途に合う読みで伝える。例えば「CAFE」と読める看板についての回答は「カフェって書いてあるよ」となる。[v11のプロンプト全文](visual_dialogue_prompts_v11.py)・[v10のプロンプト全文](visual_dialogue_prompts.py)

回答役は `{"text":"カフェって書いてあるよ","tasks":["ocr"]}` のように、発話とその往復の確認用ラベルを返す。文字列・読み・個数・確度を別に生成させると、同じ誤読を二か所に記録してしまう。正しさは人が元画像と発話を見比べて判断する。

| 回答で扱ったこと | `tasks` |
|---|---|
| 文字や看板の数字を読む。直接聞かれて一部だけ読めた、または読めないと答えた場合も含む | `["ocr"]` |
| 見える人や物の数を答える。「三台くらい」「少なくとも二人」のような答えも含む | `["counting"]` |
| 同じ往復で文字を読み、実物の数も答える | `["ocr","counting"]` |
| 色や位置などの通常の回答、個数を尋ねられて「数えきれない」とだけ答えた場合 | `[]` |

ラベルは各往復の質問・回答位置とともに `auxiliary_task` に保存し、二つ目以降を `additional_tasks` に保存する。旧対話から引き継ぐ要確認条件は候補全体にかかる。ラベルがゼロでも、人が全文を確認して承認してから音声合成に進む。生成した発話やラベルに確認事項があれば `review_warnings` として表示する。空の発話、未確定の課金、人手判断済みの元データは、それぞれ既存の保護処理の対象になる。

この形式は日本の日常画像のGPT再生成に使う。JSSODaの転記文や、カウント専用画像に付属する個数は、専用データの生成・照合に引き続き使う。

ローカルで次の Responses API リクエストを組み立てる例。下のコマンドは外部送信や台帳の更新を行わない。
リポジトリ直下で、対象 UID を指定する。出力には画像が含まれるので Git 管理外に保存する。

```bash
python3 -m ssvd.dialogue.preview_regeneration --uid "$PENDING_UID" --model gpt-6.1-terra --prompt-version visual_dialogue_v11 > /tmp/jmv-next-request.json
```

再生時は初回の `identity` と、受信順の JSON 配列 `replies` を `--transcript` に渡す。最初の応答はユーザー発話、次は回答役の発話となる。結果は `request`、実行を続けられない場合の `hold`、完了時の `pending_review`。`--model` と `--pairs` で比較条件を揃えられる。

v6の20対話では、ユーザーの127発話で相手に情報を求めたものが0件だった。v7で質問する目的を与え、v8では相づちを任意とした。v9で見えている対象から話し始めるようにし、v10では確認用の記録を往復のラベルに絞った。v10の先行100対話と本番467対話では、ユーザーの3,380発話すべてが疑問符で終わった。[v6の試行結果](https://github.com/Yoshiki0418/j-moshivis/issues/12#issuecomment-7058420549)・[v10の停止記録](https://github.com/Yoshiki0418/j-moshivis/issues/13)

v11では、直近のユーザー発話が続けて質問らしい言い方なら、返事から感じたことだけを伝える選択肢を示す。逆に感想が続いたときは、気になることを尋ねる選択肢を示す。話題の配分は会話に応じて決まり、疑問符の数で棄却や再生成はしない。所要リクエスト数は4〜8往復につき8〜16回。過去の応答を照合するときは生成時の版を指定する。[会話方針の変更 #54](https://github.com/Yoshiki0418/j-moshivis/issues/54)

モデル選定には同じ20画像（元ラベルで OCR 14、カウント 6）の比較を用いた。確認の観点は、読める対象に答えられるか、曖昧な対象を断定していないか、追質問や言い回しが自然か。`generated_tasks` はラベルの組み合わせ別の対話数、`marked_turns_by_kind` は各ラベルが付いた往復数を示す。一つの往復に両方付けば、OCR・カウントにそれぞれ1件を計上する。[少数試行 #12](https://github.com/Yoshiki0418/j-moshivis/issues/12)

### OpenAI API 枠の進捗確認と再開（3,600枚）

本番は `gpt-6.1-terra` の Flex で画像ごとに4〜8往復を生成する。最初の固定キューは3,600画像。過去の生成と人手判断の続きを扱うため、元画像とQwenのJSONL、ビューアのカタログと人手判断履歴、固定評価データ、少数試行・先行100件・本番の各台帳がある環境で実行する。[本番生成 #13](https://github.com/Yoshiki0418/j-moshivis/issues/13)

稼働中でも、リポジトリ直下から次のコマンドで確認できる。`status` は課金明細・確認待ち件数・残額を読み取る。`verify` は画像と元対話が凍結時の版と一致するか、固定評価画像との重複や凍結後の人手判断があるかを調べる。どちらもAPI呼び出しは0回。

```bash
# 最初の3,600画像を固定したときのSHA-256
production_manifest_sha=295cdc50e643f3de07c36cf6112f1d2105df1bcd718bf1ecb478633c263f67b5
.venv-data/bin/python -m ssvd.dialogue.production status --manifest-sha "$production_manifest_sha"
.venv-data/bin/python -m ssvd.dialogue.production_cohort verify --sha256 "$production_manifest_sha"
```

本番は2026年11月21日に依頼を受けて一時停止した。再開する際の有料コマンドは次のとおり。初回の `--new-dialogue-prompt` で、同じキューの未着手分をv11に割り当てる。11月21日の停止時点では3,129画像が該当する。確定した発話がある途中の4対話はv10のまま別に残る。完成分と人手判断はそのまま確認できる。稼働中のビューアが更新前に起動した場合は、v11を表示する前に更新済みのコードで起動する。

```bash
.venv-data/bin/python -m ssvd.dialogue.production run --manifest-sha "$production_manifest_sha" --workers 4 --new-dialogue-prompt visual_dialogue_v11
```

- いったん切り替えると、再開時にフラグを省略しても、台帳に記録した割当を引き継ぐ。`status` の `deferred_previous_prompt` は、前の方式の途中まで受領済みで新規送信を見送った対話数。
- `status` の `user_turns_by_prompt` は、今回の本番キューの完成分を版ごとに集計する。ユーザー総発話数、末尾が `?` または `？` の発話数に加え、全発話が疑問符で終わった対話数と、一度も疑問符で終わらなかった対話数を確認できる。承認後も集計に残る。感想への返し方や話題が自然につながったかは、対話本文を読んで確認する。
- 受領済みの発話は台帳の応答から復元する。送信後に応答の受領を確認できなかったときは最大見積額を予約したまま、その対話を `hold` とする。明細との照合に使うリクエストIDは台帳の `calls.client_id` にある。
- Flexの混雑で明確に拒否された発話は、待機時間を延ばしながら最大6回再試行する。利用枠切れや認証エラーでは停止する。初回の70ドルには少数試行と先行100件の計5.8958804ドル（未確定予約込み）も含む。残額はプロジェクト台帳からの見積もり。今回の停止時刻は2026年12月10日 0時（日本時間）。
- 生成結果は `data/curation/gpt_regeneration_issue13/production_v10_terra/production.sqlite3` に発話単位で保存する。`summary.json` と `pending_candidates.jsonl` は確認用の出力。完成した対話はビューアの `/#annotate-terra` から確認し、本人が承認した版を音声合成に使う。途中で元の対話を承認・編集・却下した場合もその判断が優先される。[人手確認 #9](https://github.com/Yoshiki0418/j-moshivis/issues/9)

別の対象群を検討する際は `production_cohort dry-run --limit 枚数` で抽出数と除外理由を調べられる（モジュール名は上の `verify` と共通）。新しい保存先に `freeze --manifest パス --limit 枚数` を実行すると固定リストと照合用のSHA-256が得られる。選定と凍結にAPI料金は発生しない。上の `run` は初回クレジット用の本番保存先を使用する。

### Codex CLI 枠の進捗確認と再開（20,505枚）

2026年11月21日、API本番3,600枚と先行試行、人手判断済みの対話、固定評価用画像との重複・類似を除き、残りの20,505枚を固定した。旧対話の分類でカウント6,281枚、OCR 14,224枚。長さは計123,067往復。生成時のモデルはAPIと同じ `gpt-6.1-terra` で、両役にv11のプロンプト、元画像と直前までの全発話を渡す。認証にはChatGPTのCodex利用枠を使う。APIの70ドル枠と課金台帳は別に管理する。[CLI生成の運用 #57](https://github.com/Yoshiki0418/j-moshivis/issues/57)

進捗はリポジトリ直下で確認できる。初回起動では画像と元対話を全件照合するため、最初の発話を生成するまで数分かかる。

```bash
cli_manifest_sha=b76864e29c9444bf4948e7f2bcff6f4901609130721d0c27e50da78b210566a6
.venv-data/bin/python -m ssvd.dialogue.codex_production status --manifest-sha "$cli_manifest_sha"
```

| 項目 | 読み方 |
|---|---|
| `queue` / `remaining` | 固定した画像数と、処理結果がまだ確定していない画像数。途中まで会話がある画像も残りに含む。 |
| `received_turns` / `outstanding_turns` | 受領済みの発話数と、送信を記録して受領が確定していない発話数。一往復でユーザーとアシスタントの二発話。 |
| `outcomes.pending_review` | 全発話を生成して人手確認用に保存した対話の累計。実際に今から確認できる件数はビューアに表示される。 |
| `outcomes.protected` / `outcomes.hold` | 元データの変更・人手判断などで保護した件数と、受領未確定などで保留した件数。 |

開始・再開には同じ固定リストと台帳を使う。ChatGPTでログイン済みのサーバーユーザーから次を実行する。`--workers 2` は最大2画像を並行処理する指定で、各画像内では発話順を守る。`--limit` を省略すると全対象を処理する。

```bash
codex login status
.venv-data/bin/python -m ssvd.dialogue.codex_production run --manifest-sha "$cli_manifest_sha" --workers 2
```

サーバーの本番プロセスはリモート接続を切っても続くように独立したセッションで起動する。固定リストと同じディレクトリに、最初の5枚は `pilot5.pid` / `pilot5.log`、全件処理は `generation.pid` / `generation.log` を置く。ログは終了時の記録が中心で、途中の件数は上の `status` で確認する。手動停止にはフォアグラウンドで Ctrl+C、切り離したプロセスではPIDが該当の `ssvd.dialogue.codex_production run` と一致することを確認して SIGTERM を送る。送信中の処理を終えた後の状態は `operator_stop` になる。

一時的な混雑（429）ではすべてのワーカーが新規送信を待ち、待機時間を延ばして再試行する。再試行を使い切ると `rate_limited`、Codexの利用枠を使い切ると `usage_limit` で停止する。前者は混雑の解消後、後者はアカウントの枠の回復後に同じコマンドで手動再開する。完了した発話を台帳から復元し、途中の画像から順に続ける。

通信断などで受領を確定できない場合は `invocation_error` や未受領の記録を確認する。再起動時、受領が未確定の発話を含む対話は `hold` に残し、他の対象を続ける。固定リスト・契約・発話・確認候補は `data/curation/codex_regeneration_terra_v11/` の `manifest.json`、`run_contract.json`、`generation.sqlite3` に保存する。

完成候補は既存のAPI生成分と同じ「Terraの本番対話を確認」に表示される。「生成経路」が `Codex CLI` の対話は、元画像と対話全文を確認して承認・編集後承認・却下できる。人が承認した版を音声合成に使う。[アノテーション画面](http://127.0.0.1:8765/#annotate-terra)・[ビューアの操作手順](../../dataset_viewer/README.md)

#### 8並列の常駐サービス（#59）

本番サーバーでは `j-moshivis-terra.service` を使う。確認には上記の台帳の `status` と次のコマンドを使う。サービスのPIDはsystemdが管理する。

```bash
systemctl --user status j-moshivis-terra.service
journalctl --user -u j-moshivis-terra.service -n 20
```

起動ごとに1件を先行処理し、続いて最大8枚を並行して生成する。画像ごとの発話順は従来どおり。通信タイムアウト・一時的な403・サーバーエラー・異常終了からはsystemdが待機後に再起動する。再試行間隔は2分から最大1時間。未受領発話のある対話は保留する。明示された利用上限と全件完了では終了する。枠回復後は `systemctl --user start j-moshivis-terra.service` で再開する。データや設定の不整合はログを確認する。手動停止は `systemctl --user stop j-moshivis-terra.service`。サーバー起動時も停止状態を維持する場合は `disable --now` を使う。

設置時はサービス定義を `ssvd/dialogue/systemd/` から `~/.config/systemd/user/` に配置し、`~/.config/j-moshivis/terra-cli.env` にChatGPTログイン済みの `codex` を含む `PATH` を指定する。CLIの設置場所を変えたらこの設定を更新する。ユーザーのlingerを有効にし、`systemctl --user daemon-reload`、`systemctl --user enable --now j-moshivis-terra.service` の順で登録する。

### 少数試行で使ったコマンド（20画像）

リポジトリ直下で実行する。`prepare` は人手未確認の旧 Qwen 対話から抽出し、v3 の評価画像、固定の外部視覚評価・キャプション評価画像、試行内の画像と重複・類似する候補を除く。元の OCR／カウント分類や選定時の難しさは旧記録からの目印。元画像で実際に判別できる内容は各モデルが改めて観察し、人が確認する。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot prepare
.venv-data/bin/python -m ssvd.dialogue.pilot status
.venv-data/bin/python -m ssvd.dialogue.pilot review
```

`data/curation/gpt_pilot_issue12/visual_dialogue_v10/` に固定した画像リスト、SQLite台帳、匿名比較用の `review.html` を保存する。過去の台帳がある場合は同じ20枚・同じ往復数で比較し、生成時には最新の人手判断も確認する。v2 の集計は `status --legacy`、v3〜v9 の既存台帳は `status --archive` または `review --archive` に版名を指定して参照できる。候補生成後は、両モデルの全文と既存Qwen対話を原画像・448px画像で比較できる。正式な承認とTTSへの投入は既存の確認手順で行う。

有料の `run` は同期 Responses API を使用する。`--model gpt-6-luna` で対象の20画像をLunaだけで生成できる。モデルを省略した場合は両モデルを対象とする。`OPENAI_API_KEY` または権限600のキーファイルを使う。既定のファイルは `~/.config/j-moshivis/openai_api_key`、別パスは `--api-key-file` で指定できる。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot run --model gpt-6-luna
```

この試行の生成予算は旧方式の請求額・未確定予約も含めて累計2米ドル。送信前に画像・履歴を含む入力トークン数を公式APIで調べ、推論を含む最大出力分まで予約する。実績額は返された使用量とStandardの公表単価で算出する。課金状態が不明のまま終了した場合は、台帳を保持したまま同じコマンドで再開する。その呼び出しの予約は保持し、当該画像・モデルは保留として残りの組を処理する。照会に必要な送信時刻と `X-Client-Request-Id` は台帳の `calls` にある。[トークン数の確認](https://developers.openai.com/api/docs/guides/token-counting)・[Lunaの単価](https://developers.openai.com/api/docs/models/gpt-6-luna)・[Terraの単価](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### 過去の比較結果を開く

v2・v3には事前観察と予定した往復での整合性チェックがあり、初回版と追加比較版が保存されている。追加比較版の集計・ページは次のコマンドで参照できる。

```bash
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-status --archive grounded_roles_v3
.venv-data/bin/python -m ssvd.dialogue.pilot recovery-review --archive grounded_roles_v3
```

v4以降では品質上の注意が付いた候補も最後まで生成される。初回の確認は `review.html` で行う。

## テストの実行

```bash
PYTHONPATH=ssvd/dialogue .venv-data/bin/python -m unittest discover \
  -s ssvd/dialogue/tests -p 'test_*.py' -v

PYTHONPATH=ssvd/tts /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python \
  -m unittest discover -s ssvd/tts/tests -p 'test_*.py' -v
```
