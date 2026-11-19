# 既存音声への相槌追加パイロット

元の音声・画像・アライメントを変更せず、3件の画像対話に短い相槌を追加して試聴する。
本番の生成・承認パイプラインや学習対象には接続しない。

## 2026-11-17の試作結果

`eval_results/backchannel_pilot_20261117/index.html` に3件を生成。
追加した「うん。」は各対話1回、約0.534秒。追加区間はそれぞれ
34.924–35.458秒、27.011–27.545秒、67.804–68.338秒。
全件で11項目の保持・整合性検査に合格。既存の検証ツールでも21発話のSPM一致、
3対話の学習ローダー復元一致を確認（本番設定 `keep_and_shift=True`）。
5件の単体テストと、ページHTTP 200・全15メディアのHTTP Range 206を確認。
生成時点の機械記録は未試聴として保存。その後のユーザー試聴で「冷たく聞こえる」との指摘があり、
下記の声・抑揚の比較を追加した。自然さの正式な承認は付与していない。
生成物は元音声・画像へのリンクを除き約11.8MB。

## 生成

必要な画像・音声・アライメント・補助ASR時刻はローカルに用意する。
実データの発話やUIDを含む日付付き計画JSONと、生成された音声・HTMLはGitには含めない。
公開する設定例 `sample_test/backchannel_pilot_plan.example.json` は架空の入力構造であり、
そのまま実行できるデータではない。これを日付付きのローカル計画JSONへコピーし、
`source` の4ファイル、元メタデータからコピーした `segment`、その位置 `segment_index`、
各ファイルの実際の `source_sha256`、相槌素材の `voice_path` を設定する。
音声素材の隣には生成条件を記録した `audit.json` が必要。

```bash
.venv-train/bin/python sample_test/build_backchannel_pilot.py \
  --plan sample_test/backchannel_pilot_plan_20261117.json \
  --timings /tmp/jmoshivis_bc_word_timing.json \
  --out eval_results/backchannel_pilot_20261117
```

`--out` は未作成のディレクトリを指定する。完成後は `auxiliary_asr.json` に
補助ASR結果、`plan.json` に入力設定を保存するため、それらから再生成できる。
入力音声・画像・メタデータ・アライメントのSHA256が計画と異なる場合は停止する。
補助ASRの入力JSONには、`backchannel.transcript` と `backchannel.words`
（各要素の `text`・`start`・`end`）、および `users`
（各要素の `uid`・`words`、単語ごとの `text`・`absolute_end`）を保存する。
ここで示す `/tmp` の時刻ファイルは当時のローカル作業ファイルであり、Gitには含まれない。

## パイロットの選定方法

- カタログで利用可能かつ音声・アライメント作成済みの日本の日常画像対話から選択。
- ユーザーが感想・説明を続ける3発話について、意味の区切りを手動選定。
- そのユーザー発話のみ、既存のWhisper large-v3-turboをCPUで実行して単語時刻を推定。
  元の学習アライメントを上書きしない。全発話を使うオフラインASRであり、オンラインの反応タイミング推定ではない。
- 区切りまでの文字列がASRと一致する単語境界のみ利用。明らかなASR誤りは計画に
  `asr_anchor_prefix` として明示し、元の発話テキストは変更しない。
- 可聴開始を区切りの0.4秒後に設定し、素材の先頭無音分を補正。
  この遅延は試聴用設定であり、最適値との主張はしない。
- 相槌の採否を自動で判断する汎用モデルは未実装。今回は3件の機構検証。

## 素材

既存のローカルStyle-Bert-VITS2／ほのかで生成した
`j-moshi-rag/data/pilots/backchannel_prosody_20261111/gentle_longer_solo.flac`
を読み取り専用で再利用する。別プロジェクトのファイルは変更しない。
素材には音声の人手承認は付いていないため、本試作も未承認。
元の生成条件とハッシュを `voice_source_audit.json` にコピーする。
素材だけ5msの端点フェードを適用。既存音声への全体正規化・リサンプリングは行わない。

## 保持・検査

- 入力はPCM16、2チャンネル。左=アシスタント、右=ユーザー。
- アシスタント側が前後0.3秒を含め厳密にゼロの場所のみ追加。
- ユーザー発話の開始後1秒、終了前0.8秒の余裕を要求。
- 元ファイル不変、ユーザーPCM全体一致、追加区間外のアシスタントPCM一致、総サンプル数とレート一致。
- 追加版FLACの再読込後もPCM一致を検査。
- 既存の全アライメント項目・時刻を保持し、相槌のみSPM変換して追加。
- 実際の `Interleaver` でトークンID欠落なし、元のテキストトークンの配置フレーム不変を検査。
- 入力音声トークンのキャッシュは流用しない。本試作ではMimiの再エンコード・学習は実施しない。

元音声・画像はリンク参照。追加音声3件はブラウザで確実にシークできるよう
可逆圧縮FLACとして保存し、同時に差分を `overlay.json` に記録する。
大規模データの差分オンデマンド読込は未実装。

## 承認と学習除外

出力は `eval_results` の独立フォルダ。全件 `preview_only=true`、
`training_eligible=false`、`review_required=true`、`review_status=pending`。
既存の承認IDは流用せず、`review_id=null`、`human_certified=false` にする。
既存学習インデクサーが要求する `stereo_dialogue.wav` も作らない。
元のASR品質判定を新しい音声の判定としてコピーしない。

## 試聴

```bash
.venv-data/bin/python sample_test/serve_backchannel_pilot.py \
  --root eval_results/backchannel_pilot_20261117 --port 8766
```

`http://127.0.0.1:8766/` のA/Bで同じ時点を切り替えて試聴できる。
サーバーはローカル待受、読取専用、公開対象は比較ページと登録された画像・音声だけ。
管理画面の承認APIには接続しない。自然さの確認結果を元データの承認と混同しない。

確認する項目は、相槌の声・抑揚、入る位置、話を遮らないか、相槌後に聞く状態へ戻るか。
機械検査の合格は自然さの保証ではない。

## 相槌の声・抑揚だけを比較する

`synthesize_backchannel_variants.py` は、ローカルに準備済みのStyle-Bert-VITS2／ほのか、
BERT、Whisperを使うCPU専用の試聴素材生成スクリプト。モデルやキャッシュのダウンロードは行わない。
`--phase synth` で音声、`--phase asr` で補助ASRを作り、出力は `--out` で指定する。
必要なモデル・依存パッケージが入ったPython環境で実行する。

比較ページは、先に作ったパイロットの保存済み `plan.json` を利用できる。
元の画像・音声・メタデータ・アライメントがパイロット時点のハッシュと一致しない場合は停止する。

```bash
.venv-data/bin/python sample_test/build_backchannel_voice_comparison.py \
  --original-root eval_results/backchannel_pilot_20261117 \
  --assets eval_results/backchannel_voice_warmth_20261117/assets \
  --out eval_results/backchannel_voice_comparison_local
```

`--plan` で別のローカル計画JSONも指定可能。出力先に完成済みの比較ページがあれば上書きせず停止する。
新しい声の比較は試聴専用で、追加音声の学習アライメントを生成・承認するものではない。

## テスト

```bash
.venv-data/bin/python -m unittest discover -s sample_test/tests -p 'test_backchannel_pilot.py' -v
```
