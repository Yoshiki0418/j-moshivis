# 作業の進め方

- 作業開始時に [CONTRIBUTING.md](CONTRIBUTING.md) と対象Issueを確認する。今後のタスク・優先度・結果はIssueで管理する。
- Issue・PRタイトルは `prefix: タイトル`、実装ブランチは `prefix/issue番号/english-kebab-case`。
- コード・設定・文書の変更は、最新mainからIssue専用ブランチを作り、PR・レビュー・検証を経てmainへマージする。マージ済みブランチを再利用しない。
- 承認・実験・既存スクリプトの実行など、追跡ファイルの変更がない作業は `chore:` の運用Issueで管理する。空コミットや空PRを作らない。必要な実装は別Issueへ分離する。
- 各Issueに優先度（`priority:P0`〜`priority:P3`）、種別（`type:implementation` / `type:operation` / `type:tracking`）、状態ラベルを一つずつ付け、本文と一致させる。
- GitHub標準のTypeも設定する。コード・動作設定の編集を伴わない承認・実験・管理・文書作業は `Task`、コード・動作設定の追加や改善は `Feature`、不具合修正は `Bug`。文書のみの変更は `Task` と `type:implementation` を併用してPRを作る。
- PRでは完了する実装Issueだけを `Closes #番号` で閉じる。運用Issue・親Issueには `Refs #番号` を使う。完了時に状態ラベルと親の進捗を更新する。
- 公開IssueやPRへ画像・音声・対話本文・UID一覧・秘密情報を掲載しない。集計と検証結果を記録し、生成物は既存のGit管理外の保管方針に従う。
