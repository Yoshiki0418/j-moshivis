# Claude Code CLI (Sonnet 5) による「非画像」日常対話生成スクリプト
#
# 目的:
#   1. 画像対話の前後に連結して「画像と無関係な話題では画像の話をしない」ことを学習させる
#      (MoshiVis 論文の p_concat 相当)
#   2. 画像対話データに欠けているフィラー(えっと/うーん等)を供給する
#
# 使い方:
#   .venv-data/bin/python ssvd/dialogue/generate_nonvisual_cli.py \
#       --out data/dialogues_nonvisual/claude_sonnet5_nonvisual.jsonl \
#       --num 20000 --workers 4
#
# - 画像を使わないため Read ツールは不要 (--allowedTools を渡さない = ツールなし)
# - uid は「題材x切り口x場面」の組み合わせインデックスから決定論的に生成 → レジューム可

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nonvisual_topics import pick
from prompt_utils import collect_done_uids

PROMPT_PATH = Path(__file__).parent / "prompt_nonvisual.md"
MODEL = "claude-sonnet-5"
# ヘッドレス実行専用の空ディレクトリ (プロジェクト外)。毎回まっさらなコンテキストで実行する
HEADLESS_CWD = "/gpu-server/yamamoto/claude_headless_work"

FAILED_LOG = None
FAIL_SKIP_THRESHOLD = 2


def _record_failure(uid: str) -> None:
    if FAILED_LOG is not None:
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(uid + "\n")


def _load_permanent_failures(path) -> set:
    from collections import Counter
    try:
        counts = Counter(l.strip() for l in open(path, encoding="utf-8") if l.strip())
    except FileNotFoundError:
        return set()
    return {u for u, c in counts.items() if c >= FAIL_SKIP_THRESHOLD}


def make_uid(index: int) -> str:
    """通し番号から決定論的な uid を作る (画像対話の uid と衝突しない接頭辞つき)"""
    return hashlib.sha256(f"nonvisual-{index}".encode()).hexdigest()


def build_prompt(uid: str) -> str:
    topic, angle, scene = pick(uid)
    instr = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        f"{instr}\n\n"
        f"## この会話の題材 (重要)\n"
        f"題材: {topic}\n"
        f"切り口: ユーザーがこの話題について{angle}ところから始める\n"
        f"場面: {scene}話している設定\n"
        f"題材や切り口の文言をそのまま台詞に流用せず、自然な言い回しにすること。\n"
        f"繰り返しになりますが、視覚に関する言及(画像・写真・見えているもの)は一切しないこと。"
    )


def parse_dialogue(text: str):
    """応答からJSON部分を抽出してパース"""
    text = text.strip()
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
            if turn.get("speaker") not in ("ユーザー", "アシスタント"):
                return None
            if not turn.get("text"):
                return None
        # 視覚言及が混入していないかの最終チェック
        joined = "".join(t["text"] for t in dialogue)
        for ng in ("画像", "写真", "写ってい", "見えてる", "見えている", "この景色"):
            if ng in joined:
                return None
        # 英字混入を弾く (SPM語彙に大文字英字が無く、TTSも読めないため)
        if any("a" <= c.lower() <= "z" for c in joined):
            return None
        return dialogue
    except json.JSONDecodeError:
        return None


def generate_one(uid: str) -> dict | None:
    try:
        result = subprocess.run(
            ["claude", "-p", build_prompt(uid), "--model", MODEL, "--output-format", "json"],
            capture_output=True, text=True, timeout=300, cwd=HEADLESS_CWD,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        print(f"⚠️ claude failed for {uid[:12]}: {result.stderr[:200]}")
        return None
    try:
        response = json.loads(result.stdout)
        answer = response.get("result", "")
        cost = response.get("total_cost_usd", 0)
    except json.JSONDecodeError:
        return None
    dialogue = parse_dialogue(answer)
    if dialogue is None:
        print(f"⚠️ parse failed for {uid[:12]}: {str(answer)[:80]}")
        _record_failure(uid)
        return None
    topic, angle, scene = pick(uid)
    return {
        "uid": uid,
        "dialogue": dialogue,
        "generator": MODEL,
        "kind": "nonvisual",
        "topic": topic,
        "angle": angle,
        "scene": scene,
        "cost_usd": cost,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=20000, help="生成する対話の総数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="この実行での上限")
    parser.add_argument("--abort_after_failures", type=int, default=20)
    parser.add_argument("--registry_dir", default=None)
    parser.add_argument("--registry_glob", default="*.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry = Path(args.registry_dir) if args.registry_dir else out_path.parent
    global FAILED_LOG
    FAILED_LOG = str(out_path) + ".failed"
    perm_failed = _load_permanent_failures(FAILED_LOG)
    done = collect_done_uids(registry, args.registry_glob)

    uids = [make_uid(i) for i in range(args.num)]
    todo = [u for u in uids if u not in done and u not in perm_failed]
    if perm_failed:
        print(f"[info] skipping {len(perm_failed)} permanently-failed uids", flush=True)
    if args.max_samples is not None:
        todo = todo[:args.max_samples]
    print(f"[info] generating {len(todo)} dialogues ({len(done)} already done in {registry})", flush=True)
    if not todo:
        sys.exit(3)

    total_cost = 0.0
    ok = 0
    fail_streak = 0
    aborted = False
    with out_path.open("a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_one, u): u for u in todo}
        for fut in as_completed(futures):
            rec = fut.result()
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
            print(f"✅ {ok}/{len(todo)} {rec['uid'][:12]} (${total_cost:.3f} total)", flush=True)

    print(f"\n Done: {ok}/{len(todo)} succeeded, total cost ${total_cost:.3f}", flush=True)
    if aborted:
        sys.exit(2)


if __name__ == "__main__":
    main()
