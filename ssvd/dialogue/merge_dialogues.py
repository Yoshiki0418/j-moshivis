# 対話JSONL群を「モデル優先順位つき」でマージするスクリプト
#
# 同じUID・同じtask modeの対話が複数モデルに存在する場合
# (例: Qwen生成分を後からSonnetで再生成した場合)、優先順位の高いモデルのものだけを
# 採用する。通常・counting・OCRは同じ画像でも別サンプルとして残す。
#
# 使い方:
#   .venv-data/bin/python ssvd/dialogue/merge_dialogues.py \
#       --inputs data/dialogues/*.jsonl \
#       --out data/dialogues_merged/dialogues_final.jsonl
#
# 優先順位 (上ほど優先):
#   1. claude-sonnet-5
#   2. その他 (Qwen等) — 同一優先度内では後に書かれたもの (=新しいもの) を採用

import argparse
import json
from pathlib import Path

PRIORITY = {
    "claude-sonnet-5": 0,   # 最優先
    # 未知のgeneratorは 100 (最低優先)
}


def prio(generator: str) -> int:
    return PRIORITY.get(generator, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    best = {}   # (uid, task_mode) -> record; legacy records are normal mode
    n_read = 0
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    rec = json.loads(l)
                    uid = str(rec["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
                n_read += 1
                key = (uid, str(rec.get("task_mode", "normal")))
                cur = best.get(key)
                # 優先度が高い(値が小さい)方を採用。同順位なら後勝ち(=新しい方)
                if cur is None or prio(rec.get("generator", "")) <= prio(cur.get("generator", "")):
                    best[key] = rec

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_gen = {}
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in best.values():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            g = rec.get("generator", "unknown")
            by_gen[g] = by_gen.get(g, 0) + 1

    print(f"read {n_read} records -> {len(best)} unique uid/task pairs "
          f"({n_read - len(best)} duplicates resolved)")
    for g, c in sorted(by_gen.items(), key=lambda x: -x[1]):
        print(f"  {c:6d}  {g}")


if __name__ == "__main__":
    main()
