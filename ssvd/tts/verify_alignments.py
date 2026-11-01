# 精緻化済みアライメントが学習時のSPMトークン列と一致しているかを全件検証する
# 使い方: .venv-train/bin/python ssvd/tts/verify_alignments.py --datadir data/speech/data_stereo
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
from jmoshivis.datasets.interleaver import Interleaver

SPM_PATH = "/gpu-server/yamamoto/j-moshivis/models/j-moshi/tokenizer_spm_32k_3.model"
FRAME_RATE = 12.5
PAD, EOT, ZERO = -2, -3, -1   # 検証用ダミーID (実語彙IDと衝突しない負値)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/speech/data_stereo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expected", type=int, default=None,
                    help="期待する検証済み対話数")
    ap.add_argument("--strict", action="store_true",
                    help="件数・形式・ID列・時刻・復元の不一致があれば非0終了")
    ap.add_argument("--require-noshift", action="store_true",
                    help="診断用keep_and_shift=Falseの復元一致もstrict判定に含める")
    args = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
    il = Interleaver(sp, FRAME_RATE, PAD, EOT, ZERO, keep_main_only=True, keep_and_shift=True, device="cpu")
    il_noshift = Interleaver(sp, FRAME_RATE, PAD, EOT, ZERO, keep_main_only=True, keep_and_shift=False, device="cpu")

    dirs = sorted(p for p in Path(args.datadir).iterdir() if (p / "assistant_dialogue.refined.json").exists())
    if args.limit:
        dirs = dirs[:args.limit]
    c = Counter(); delays = []; utt_total = utt_id_ok = 0; tok_total = 0; dens = []
    bad_examples = []
    for d in dirs:
        data = json.loads((d / "assistant_dialogue.refined.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "dialogue.json").read_text(encoding="utf-8"))
        if data.get("format") != "spm_pieces_v2":
            c["old_format"] += 1; continue
        ta = data["token_alignments"]
        a_entries = [a for a in ta if a[2] == "ASSISTANT"]
        # ① 発話ごとに id 列 == sp.encode(正規化テキスト)  (utterances の n_tokens で順に切り出す)
        pos = 0
        for u in data["utterances"]:
            if u["speaker"] != "ASSISTANT":
                continue
            utt_total += 1
            ids = [a[0][0] for a in a_entries[pos:pos + u["n_tokens"]]]; pos += u["n_tokens"]
            if ids == sp.encode(u["text_spm"]) and sp.unk_id() not in ids:
                utt_id_ok += 1
            elif len(bad_examples) < 3:
                bad_examples.append((d.name[:8], u["text_spm"][:30], sp.decode(ids)[:30]))
            c["normalized_utts"] += (u["text_spm"] != u["text"])
        tok_total += len(a_entries)
        # ② 単調性・正の長さ
        starts = [a[1][0] for a in a_entries]
        mono = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
        pos = all(a[1][1] > a[1][0] for a in a_entries)
        c["monotonic"] += mono; c["positive_dur"] += pos
        # ③ ローダー経由で復元一致 (keep_and_shift=True / False)
        al = [(a[0], tuple(a[1]), a[2]) for a in ta]
        orig = "".join(u["text_spm"] for u in data["utterances"] if u["speaker"] == "ASSISTANT").replace(" ", "")
        for name, inter in (("shift", il), ("noshift", il_noshift)):
            stream = inter.prepare_item(al, meta["duration"], pretokenized=True)[0, 0].tolist()
            ids = [t for t in stream if t not in (PAD, EOT, ZERO)]
            # 文字列比較だと SPM decode の NFKC 正規化 (全角！（）？→半角) で偽陰性になるため ID 列で比較
            c[f"decode_ok_{name}"] += (ids == [a[0][0] for a in a_entries])
            if name == "shift":
                # ④ keep_and_shift による遅延: 各トークンの本来の開始フレームと実配置フレームの差
                placed = [i for i, t in enumerate(stream) if t not in (PAD, EOT, ZERO)]
                orig_frames = [int(a[1][0] * FRAME_RATE) for a in sorted(a_entries, key=lambda x: x[1][0])]
                if len(placed) == len(orig_frames):
                    delays.extend((p - o) / FRAME_RATE for p, o in zip(placed, orig_frames))
                dens.append(len(ids) / max(1e-6, sum(s["end"] - s["start"] for s in meta["segments"] if s["speaker"] == "ASSISTANT")))
        c["dialogues"] += 1

    n = c["dialogues"]
    print(f"検証対話数: {n} (旧形式: {c['old_format']})")
    print(f"① 発話単位でトークンID列 == sp.encode(正規化原文) かつ <unk> なし: {utt_id_ok}/{utt_total} ({utt_id_ok/max(1,utt_total)*100:.2f}%)  総トークン {tok_total}  語彙外置換した発話 {c['normalized_utts']}")
    print(f"② 時刻の単調性: {c['monotonic']}/{n}   正の長さ: {c['positive_dur']}/{n}")
    print(f"③ ローダー出力を復元して原文一致: keep_and_shift=True {c['decode_ok_shift']}/{n} ({c['decode_ok_shift']/max(1,n)*100:.2f}%) | =False {c['decode_ok_noshift']}/{n} ({c['decode_ok_noshift']/max(1,n)*100:.2f}%)")
    if delays:
        d = np.array(delays)
        print(f"④ keep_and_shift の遅延: 平均 {d.mean():.3f}s, 中央値 {np.median(d):.3f}s, 95%ile {np.percentile(d,95):.3f}s, 最大 {d.max():.3f}s | 遅延0のトークン {np.mean(d<=0)*100:.1f}%")
        print(f"   アシスタント発話中のトークン密度: 平均 {np.mean(dens):.2f} tok/s (フレームレート {FRAME_RATE}/s)")
    for ex in bad_examples:
        print("   不一致例:", ex)

    if args.strict:
        failures = []
        if args.expected is not None and n != args.expected:
            failures.append(f"dialogues={n}, expected={args.expected}")
        if c["old_format"]:
            failures.append(f"old_format={c['old_format']}")
        if utt_id_ok != utt_total:
            failures.append(f"utterance_token_ids={utt_id_ok}/{utt_total}")
        if c["monotonic"] != n:
            failures.append(f"monotonic={c['monotonic']}/{n}")
        if c["positive_dur"] != n:
            failures.append(f"positive_dur={c['positive_dur']}/{n}")
        if c["decode_ok_shift"] != n:
            failures.append(f"decode_ok_shift={c['decode_ok_shift']}/{n}")
        # 学習ローダーは keep_and_shift=True を使う。Falseでは同一フレームに
        # 複数トークンが重なると後続トークンが失われるため、通常は診断表示だけにする。
        if args.require_noshift and c["decode_ok_noshift"] != n:
            failures.append(f"decode_ok_noshift={c['decode_ok_noshift']}/{n}")
        if failures:
            print("STRICT VERIFY FAILED: " + "; ".join(failures), file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
