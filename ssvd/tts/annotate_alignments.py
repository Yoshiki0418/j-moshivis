# アシスタントチャネル (ch0) を whisper_timestamped で単語アライメントし、
# assistant_dialogue.refined.json ({"alignments": [[word, [start, end], "ASSISTANT"], ...]}) を生成する。
# 併せて dialogue.json の発話区間ごとに ASR結果と原文の CER を計算して asr_check.json に保存 (品質フィルタ用)。
#
# 使い方 (シャード並列):
#   CUDA_VISIBLE_DEVICES=1 /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/annotate_alignments.py \
#       --datadir data/speech/data_stereo --shard 0 --num_shards 2 --model large-v3
import argparse
import hashlib
from itertools import zip_longest
import json
import os
import time
from pathlib import Path

import sys
import numpy as np
import soundfile as sf
import torch
from math import gcd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from refine_alignments import refine_dialogue, SPM_PATH  # noqa: E402

ASR_SR = 16000
SP = None


def completed_dialogues(roots, shard, num_shards, stable=False):
    """Interleave roots and partition completed files without live-index drift."""
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("shard must be in [0, num_shards)")
    groups = []
    for root in dict.fromkeys(Path(root).resolve() for root in roots):
        if not root.is_dir():
            continue
        groups.append(sorted(p for p in root.iterdir()
                             if (not stable or int.from_bytes(hashlib.sha256(p.name.encode()).digest()[:8], "big") % num_shards == shard)
                             and (p / "stereo_dialogue.wav").is_file()
                             and (p / "dialogue.json").is_file()))
    dirs = [p for row in zip_longest(*groups) for p in row if p is not None]
    if stable:
        return dirs
    return [p for i, p in enumerate(dirs) if i % num_shards == shard]


def cer(ref: str, hyp: str) -> float:
    ref = "".join(ref.split()); hyp = "".join(hyp.split())
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def annotate_one(d: Path, model, language: str) -> dict:
    """発話区間が既知なので、アシスタント発話ごとに ch0 を切り出して whisper に掛ける
    (VAD不要・無音区間での幻覚を防止・発話単位の CER が正確に取れる)"""
    import whisper_timestamped as whisper
    meta = json.loads((d / "dialogue.json").read_text(encoding="utf-8"))
    audio, sr = sf.read(d / "stereo_dialogue.wav", dtype="float32")
    ch0 = audio[:, 0]
    if sr != ASR_SR:
        g = gcd(ASR_SR, sr)
        ch0 = resample_poly(ch0, ASR_SR // g, sr // g).astype(np.float32)
    words, checks = [], []
    pad = 0.1
    for s in meta["segments"]:
        if s["speaker"] != "ASSISTANT":
            continue
        a, b = max(0.0, s["start"] - pad), s["end"] + pad
        clip = ch0[int(a * ASR_SR):int(b * ASR_SR)]
        if len(clip) < ASR_SR * 0.2:
            continue
        out = whisper.transcribe(model, clip, language=language, vad=False, beam_size=5, best_of=5,
                                 temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), verbose=None)
        seg_words = []
        for seg in out.get("segments", []):
            for w in seg.get("words", []):
                if w.get("text") and w["end"] > w["start"]:
                    seg_words.append([w["text"], [round(a + w["start"], 3), round(min(a + w["end"], b), 3)], "ASSISTANT"])
        words.extend(seg_words)
        hyp = "".join(w[0] for w in seg_words)
        checks.append({"start": s["start"], "end": s["end"], "ref": s["text"], "hyp": hyp,
                       "cer": round(cer(s["text"], hyp), 3)})
    user_entries = [[s["text"], [s["start"], s["end"]], "USER"] for s in meta["segments"] if s["speaker"] == "USER"]
    alignments = sorted(words + user_entries, key=lambda x: x[1][0])
    tmp = d / f"whisper_words.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps({"alignments": alignments}, ensure_ascii=False), encoding="utf-8")
    tmp.rename(d / "whisper_words.json")
    refine_dialogue(d, SP)  # → assistant_dialogue.refined.json (SPMトークン単位 + token id)
    mean_cer = float(np.mean([c["cer"] for c in checks])) if checks else 1.0
    asr_result = {
        "mean_cer": round(mean_cer, 3),
        "max_cer": round(max((c["cer"] for c in checks), default=1.0), 3),
        "n_words": len(words),
        "utterances": checks,
    }
    # 中断時に途中まで書かれたJSONを完了成果物として残さない。
    asr_tmp = d / f"asr_check.json.tmp.{os.getpid()}"
    asr_tmp.write_text(
        json.dumps(asr_result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    asr_tmp.replace(d / "asr_check.json")
    return {"mean_cer": mean_cer, "n_words": len(words)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/speech/data_stereo")
    ap.add_argument("--extra-datadir", action="append", default=[], help="追加の音声ルートを交互に処理")
    ap.add_argument("--stable-sharding", action="store_true", help="音声追加中もUIDの担当シャードを固定")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs = completed_dialogues([args.datadir, *args.extra_datadir], args.shard,
                              args.num_shards, args.stable_sharding)
    # whisper の atomic rename 後、refine/asr 書き込み前に中断される場合がある。
    # 3成果物すべてが揃ったUIDだけを完了扱いにし、部分完了は安全に再処理する。
    required_outputs = (
        "whisper_words.json",
        "assistant_dialogue.refined.json",
        "asr_check.json",
    )
    todo = [
        p for p in dirs
        if not all((p / name).exists() for name in required_outputs)
    ]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(dirs)} dirs, {len(todo)} to annotate", flush=True)
    if not todo:
        return
    import whisper_timestamped as whisper
    import sentencepiece as spm
    global SP
    SP = spm.SentencePieceProcessor(model_file=SPM_PATH)
    model = whisper.load_model(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time(); cers = []; bad = 0
    for i, d in enumerate(todo, 1):
        try:
            r = annotate_one(d, model, args.language)
            cers.append(r["mean_cer"])
        except Exception as e:
            bad += 1
            print(f"⚠️ {d.name[:12]}: {e}", flush=True)
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[shard {args.shard}] {i}/{len(todo)} mean_cer={np.mean(cers):.3f} errors={bad} {el/60:.1f}min ({el/i:.1f}s/dlg)", flush=True)
    print(f"[shard {args.shard}] done", flush=True)


if __name__ == "__main__":
    main()
