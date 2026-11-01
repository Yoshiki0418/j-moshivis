# Claude Code CLI (Sonnet 5) による画像対話生成スクリプト
#
# 使い方:
#   .venv-data/bin/python ssvd/dialogue/generate_claude_cli.py \
#       --meta /gpu-server/yamamoto/j-moshivis/data/images/alfredplpl_jp/metadata.jsonl \
#       --out /gpu-server/yamamoto/j-moshivis/data/dialogues/claude_sonnet5.jsonl \
#       --max_samples 5
#
# - 1画像につき claude -p を1回呼び、マルチターン対話全体をJSONで得る
# - --allowedTools Read で画像読み込みのみ許可 (安全のため他ツールは無効)
# - レジューム可: 出力済みuidはスキップ
# - 並列実行: --workers で claude プロセスを並列化

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SSVD_DIR = Path(__file__).resolve().parents[1]
if str(SSVD_DIR) not in sys.path:
    sys.path.insert(0, str(SSVD_DIR))

from review_store import (  # noqa: E402
    image_file_sha256,
    requires_review,
    review_metadata,
)

from prompt_utils import (
    build_instruction,
    collect_done_keys,
    collect_done_uids,
    extract_auxiliary_task,
)
from task_profiles import (
    NORMAL_STRATEGIES,
    TASK_MODES,
    auxiliary_verification_status,
    load_task_entries,
    public_grounding,
    selected_auxiliary_grounding,
    validate_task_dialogue,
)

MODEL = "claude-sonnet-5"
# ヘッドレス実行専用の空ディレクトリ (プロジェクト外)。
# - プロジェクトのCLAUDE.md/メモリを読み込まない (毎回まっさらなコンテキスト)
# - セッションログが本作業と混ざらず、まとめて削除できる
HEADLESS_CWD = "/gpu-server/yamamoto/claude_headless_work"



# Older runs may already contain a failure log.  Honour it for backward
# compatibility, but do not add formatting/grounding misses to it: those are
# retryable under the bounded wrapper.
FAIL_SKIP_THRESHOLD = 2  # 同一UIDがこの回数失敗したら恒久スキップ (コンテンツ拒否等)


def _load_permanent_failures(path) -> set:
    from collections import Counter
    try:
        counts = Counter(l.strip() for l in open(path, encoding="utf-8") if l.strip())
    except FileNotFoundError:
        return set()
    return {u for u, c in counts.items() if c >= FAIL_SKIP_THRESHOLD}


def build_prompt(
    image_path: str,
    uid: str,
    task_mode: str = "normal",
    grounding: dict | None = None,
    normal_strategy: str = "standard",
    optional_groundings: dict | None = None,
) -> str:
    return (
        f"次の画像ファイルをReadツールで読み込んでください: {image_path}\n\n"
        f"{build_instruction(uid, task_mode, grounding, normal_strategy, optional_groundings)}"
    )


def parse_dialogue(text: str):
    """応答からJSON部分を抽出してパース"""
    text = text.strip()
    # コードブロックで囲まれている場合を除去
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
        dialogue = data.get("dialogue")
        if not isinstance(dialogue, list) or len(dialogue) < 4:
            return None
        for turn in dialogue:
            if not isinstance(turn, dict):
                return None
            if turn.get("speaker") not in ("ユーザー", "アシスタント"):
                return None
            if not turn.get("text"):
                return None
        return dialogue
    except json.JSONDecodeError:
        return None


def generate_one(entry: dict) -> dict | None:
    image_path = entry["absolute_image_path"]
    prompt = build_prompt(
        image_path,
        entry["uid"],
        entry["task_mode"],
        entry.get("grounding"),
        entry.get("normal_strategy", "standard"),
        entry.get("optional_groundings"),
    )
    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--model", MODEL,
                "--allowedTools", "Read",
                "--output-format", "json",
            ],
            capture_output=True, text=True, timeout=300,
            cwd=HEADLESS_CWD,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        print(f"⚠️ claude failed for {entry['uid'][:12]}: {result.stderr[:200]}")
        return None
    try:
        response = json.loads(result.stdout)
        answer = response.get("result", "")
        cost = response.get("total_cost_usd", 0)
    except json.JSONDecodeError:
        return None
    dialogue = parse_dialogue(answer)
    if dialogue is None:
        print(f"⚠️ parse failed for {entry['uid'][:12]}: {str(answer)[:80]}")
        # A formatting miss is retryable; the wrapper presents the same
        # canonical grounding again on its next bounded pass.
        return None
    auxiliary_task = None
    if entry.get("normal_strategy") == "vlm_adaptive":
        auxiliary_task = extract_auxiliary_task(answer)
    valid, reason = validate_task_dialogue(entry, dialogue, auxiliary_task)
    if not valid:
        print(f"⚠️ validation failed for {entry['uid'][:12]}: {reason}")
        # The canonical answer was already supplied in the prompt.  Treat a
        # malformed rendering as retryable instead of permanently excluding
        # the source image after two formatting mistakes.
        return None
    record = {
        "uid": entry["uid"],
        "image": entry["path"],
        "dialogue": dialogue,
        "generator": MODEL,
        "cost_usd": cost,
        "task_mode": entry["task_mode"],
    }
    if entry.get("normal_strategy") == "vlm_adaptive":
        record["normal_strategy"] = "vlm_adaptive"
        record["auxiliary_task"] = dict(auxiliary_task)
        verification_status = auxiliary_verification_status(auxiliary_task)
        if verification_status is not None:
            record["verification_status"] = verification_status
    grounding = public_grounding(entry)
    if grounding is None:
        grounding = selected_auxiliary_grounding(entry, auxiliary_task)
    if grounding is not None:
        record["grounding"] = grounding
    if entry.get("split_group_key") is not None:
        record["split_group_key"] = entry["split_group_key"]
    if requires_review(record):
        try:
            record["image_sha256"] = image_file_sha256(image_path)
        except OSError as error:
            print(
                f"⚠️ review image hash failed for {entry['uid'][:12]}: {error}"
            )
            return None
        record.update(review_metadata(record))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--task-mode",
        choices=TASK_MODES,
        default="normal",
        help="normal/counting/ocrを明示指定。autoはcapabilityから行ごとに推定",
    )
    parser.add_argument(
        "--normal-strategy",
        choices=NORMAL_STRATEGIES,
        default="standard",
        help="normal時の生成戦略。vlm_adaptiveは画像ごとに補助count/OCRを判断",
    )
    parser.add_argument("--abort_after_failures", type=int, default=20,
                        help="連続失敗がこの回数に達したら使用制限とみなして中断 (exit 2)")
    parser.add_argument("--registry_dir", default=None,
                        help="生成済みUIDを走査するディレクトリ (デフォルト: --out の親)")
    parser.add_argument("--registry_glob", default="*.jsonl",
                        help="レジストリ対象ファイルのパターン。例: 'claude_*.jsonl' で"
                             "Sonnet出力のみを生成済みとみなし、他モデル生成分を再生成(置き換え)できる")
    parser.add_argument(
        "--registry-scope",
        choices=("task", "uid"),
        default="task",
        help="taskは(uid, task_mode)、uidはタスクを問わず既存UIDを生成済みとみなす",
    )
    args = parser.parse_args()

    entries, selection_stats = load_task_entries(
        args.meta, args.task_mode, args.normal_strategy
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry = Path(args.registry_dir) if args.registry_dir else out_path.parent
    failed_log = str(out_path) + ".failed"
    perm_failed = _load_permanent_failures(failed_log)
    if args.registry_scope == "task":
        done = collect_done_keys(registry, args.registry_glob)
        entries = [
            entry
            for entry in entries
            if (entry["uid"], entry["task_mode"]) not in done
            and entry["uid"] not in perm_failed
        ]
    else:
        done = collect_done_uids(registry, args.registry_glob)
        entries = [
            entry
            for entry in entries
            if entry["uid"] not in done and entry["uid"] not in perm_failed
        ]
    if perm_failed:
        print(f"[info] skipping {len(perm_failed)} permanently-failed uids (refusals etc.)", flush=True)
    if args.max_samples is not None:
        entries = entries[:args.max_samples]
    print(f"[info] task selection: {json.dumps(selection_stats, ensure_ascii=False)}", flush=True)
    print(f"[info] generating for {len(entries)} images "
          f"({len(done)} uid/task pairs already done in registry {registry})", flush=True)
    if not entries:
        sys.exit(3)  # 全画像生成済み → ラッパーはここで停止する

    total_cost = 0.0
    ok = 0
    fail_streak = 0
    aborted = False
    with out_path.open("a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_one, e): e for e in entries}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as error:
                entry = futures[fut]
                print(f"⚠️ worker failed for {entry['uid'][:12]}: {error}", flush=True)
                rec = None
            if rec is None:
                fail_streak += 1
                if fail_streak >= args.abort_after_failures:
                    print(f"[abort] {fail_streak} consecutive failures — "
                          f"usage limit suspected, stopping this pass", flush=True)
                    aborted = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                continue
            fail_streak = 0
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            total_cost += rec.get("cost_usd") or 0
            print(f"✅ {ok}/{len(entries)} {rec['uid'][:12]} (${total_cost:.3f} total)", flush=True)

    print(f"\n Done: {ok}/{len(entries)} succeeded, total cost ${total_cost:.3f}", flush=True)
    if aborted:
        sys.exit(2)  # 制限疑い → ラッパーが待機後に再実行


if __name__ == "__main__":
    main()
