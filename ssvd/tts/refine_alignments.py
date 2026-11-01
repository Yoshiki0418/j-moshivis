# whisper の文字単位タイムスタンプ + 元テキスト → SPM トークン単位のアライメントに精緻化する
#
# 背景: 学習ローダーは alignments の各 word を個別に SPM で再トークナイズするため、whisper の区切り
# (「ホ」「ーム」等) では (1) 単語ごとに '▁' が付く (2) トークン境界が本来の tokenizer と一致しない。
# ここでは元テキストを sp.encode した「本来のトークン列」を正とし、各トークンの時刻を whisper の
# 文字時刻から補間して割り当てる。出力には token id 列も同梱し、ローダーは再トークナイズせずに使う。
#
# 使い方 (一括/移行): /gpu-server/yamamoto/miniforge3/envs/sbv2/bin/python ssvd/tts/refine_alignments.py --datadir data/speech/data_stereo
import argparse
import difflib
import json
import os
import re
import unicodedata
from pathlib import Path

import sentencepiece as spm

SPM_PATH = "/gpu-server/yamamoto/j-moshivis/models/j-moshi/tokenizer_spm_32k_3.model"
MIN_DUR = 0.02
FORMAT = "spm_pieces_v2"
_RUN_RE = re.compile(r"[A-Za-z0-9]+|.")  # ハイフン等は区切り扱い (Wi-Fi → ダブリューアイ + エフアイ)
_KANA_RE = re.compile(r"[\u30a0-\u30ff\u3040-\u309f\u30fcー]+")


def _has_unk(sp, s: str) -> bool:
    return sp.unk_id() in sp.encode(s)


def normalize_for_spm(sp, text: str) -> str:
    """語彙外 (<unk>) になる区間を、TTS(pyopenjtalk)と同じ読みのカナに置換する。
    例: 'JR東日本'→'ジェイアール東日本', 'LaLaport'→'エルエーエルエーピーオーアールティー', '籐'→'トー'
    英数字の連続は1語として扱う (LaLaport の L だけを置換しないため)。"""
    text = text.replace("\ufffd", "")  # 文字化け置換文字は除去
    if not _has_unk(sp, text):
        return text
    import pyopenjtalk
    out = []
    for m in _RUN_RE.finditer(text):
        run = m.group(0)
        if not _has_unk(sp, run):
            out.append(run)
            continue
        # 読み (カナ) を取得。pyopenjtalk が全角英字などを残す場合は NFKC→小文字化して語彙内に収める
        read = unicodedata.normalize("NFKC", pyopenjtalk.g2p(run, kana=True))
        conv = "".join(ch if _KANA_RE.match(ch) else ch.lower() for ch in read)
        conv = "".join(ch for ch in conv if ch.strip() and not _has_unk(sp, ch))
        if conv:
            out.append(conv)
            continue
        low = run.lower()  # 読みが取れない場合: 小文字化 (小文字は全て語彙内)
        out.append("".join(c for c in low if not _has_unk(sp, c)))
    return "".join(out)


def char_times_from_words(words: list) -> tuple[str, list]:
    """whisper 単語列 → (連結文字列, 各文字の (start, end))。単語内は線形補間"""
    chars, times = [], []
    for text, (s, e), _ in words:
        n = len(text)
        if n == 0:
            continue
        step = (e - s) / n
        for i, ch in enumerate(text):
            chars.append(ch)
            times.append((s + i * step, s + (i + 1) * step))
    return "".join(chars), times


def align_orig_to_times(orig: str, hyp: str, hyp_times: list, seg_start: float, seg_end: float) -> list:
    """元テキストの各文字に時刻を割り当てる。ASR誤りで一致しない文字は前後の一致文字から補間"""
    n = len(orig)
    t = [None] * n
    sm = difflib.SequenceMatcher(None, orig, hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                t[i1 + k] = hyp_times[j1 + k]
        elif tag == "replace" and (i2 - i1) > 0 and (j2 - j1) > 0:
            # 置換区間は hyp 側の時間幅を orig 側の文字数で等分
            s, e = hyp_times[j1][0], hyp_times[j2 - 1][1]
            step = (e - s) / (i2 - i1)
            for k in range(i2 - i1):
                t[i1 + k] = (s + k * step, s + (k + 1) * step)
    # 未割当 (削除された文字など) を前後から補間
    idx = [i for i, v in enumerate(t) if v is not None]
    if not idx:  # 全く一致しない → 区間を等分
        step = (seg_end - seg_start) / max(1, n)
        return [(seg_start + i * step, seg_start + (i + 1) * step) for i in range(n)]
    i = 0
    while i < n:
        if t[i] is not None:
            i += 1
            continue
        j = i
        while j < n and t[j] is None:
            j += 1
        s = t[i - 1][1] if i > 0 else seg_start
        e = t[j][0] if j < n else seg_end
        if e <= s:
            e = s + MIN_DUR * (j - i)
        step = (e - s) / (j - i)
        for k in range(j - i):
            t[i + k] = (s + k * step, s + (k + 1) * step)
        i = j
    return t


def refine_utterance(sp, orig: str, words: list, seg_start: float, seg_end: float) -> list:
    """→ [(piece_text, [s, e], token_id), ...]  (sp.encode(orig) と同じトークン列・順序)"""
    pieces = sp.encode(orig, out_type=str)
    ids = sp.encode(orig)
    hyp, hyp_times = char_times_from_words(words)
    ct = align_orig_to_times(orig, hyp, hyp_times, seg_start, seg_end)
    out, pos, last_end = [], 0, seg_start
    for piece, tid in zip(pieces, ids):
        core = piece[1:] if piece.startswith("▁") else piece
        if core and orig[pos:pos + len(core)] == core:
            s, e = ct[pos][0], ct[pos + len(core) - 1][1]
            pos += len(core)
        elif core:  # 正規化などで文字が一致しない場合は探索
            k = orig.find(core, pos)
            if k >= 0:
                s, e = ct[k][0], ct[k + len(core) - 1][1]
                pos = k + len(core)
            else:
                s, e = last_end, last_end + MIN_DUR
        else:  # '▁' 単独 (文頭ダミー): 次の文字の直前に置く
            nxt = ct[pos][0] if pos < len(ct) else last_end
            s, e = max(seg_start, nxt - MIN_DUR), nxt
        s = max(s, last_end)          # 単調性
        e = max(e, s + MIN_DUR)       # 正の長さ
        out.append((piece, [round(s, 3), round(e, 3)], tid))
        last_end = e
    return out


def refine_dialogue(d: Path, sp) -> dict:
    meta = json.loads((d / "dialogue.json").read_text(encoding="utf-8"))
    raw = json.loads((d / "whisper_words.json").read_text(encoding="utf-8"))["alignments"]
    aw = [w for w in raw if w[2] == "ASSISTANT"]
    alignments, token_alignments, utterances = [], [], []
    n_tok = 0
    for s in meta["segments"]:
        text_spm = normalize_for_spm(sp, s["text"])
        if s["speaker"] == "USER":
            alignments.append([text_spm, [s["start"], s["end"]], "USER"])
            token_alignments.append([sp.encode(text_spm), [s["start"], s["end"]], "USER"])
            utterances.append({"speaker": "USER", "text": s["text"], "text_spm": text_spm,
                               "start": s["start"], "end": s["end"], "n_tokens": 0})
            continue
        words = [w for w in aw if s["start"] - 0.25 <= w[1][0] < s["end"] + 0.25]
        refined = refine_utterance(sp, text_spm, words, s["start"], s["end"])
        for piece, ts, tid in refined:
            alignments.append([piece, ts, "ASSISTANT"])
            token_alignments.append([[tid], ts, "ASSISTANT"])
        n_tok += len(refined)
        utterances.append({"speaker": "ASSISTANT", "text": s["text"], "text_spm": text_spm,
                           "start": s["start"], "end": s["end"], "n_tokens": len(refined)})
    # 発話順 = 時刻順なので、順序を保ったまま出力 (発話ごとのトークン数で切り出せる)
    out = {"format": FORMAT, "tokenizer": os.path.basename(SPM_PATH),
           "alignments": alignments, "token_alignments": token_alignments, "utterances": utterances}
    tmp = d / f"assistant_dialogue.refined.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.rename(d / "assistant_dialogue.refined.json")
    return {"n_tokens": n_tok}


def migrate_raw(d: Path) -> bool:
    """旧出力 (whisper生単語が refined.json に入っている) を whisper_words.json へ退避"""
    r = d / "assistant_dialogue.refined.json"
    if (d / "whisper_words.json").exists() or not r.exists():
        return (d / "whisper_words.json").exists()
    data = json.loads(r.read_text(encoding="utf-8"))
    if str(data.get("format", "")).startswith("spm_pieces"):
        return False
    r.rename(d / "whisper_words.json")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/speech/data_stereo")
    ap.add_argument("--force", action="store_true", help="精緻化済みでも再生成")
    args = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
    n = 0
    for d in sorted(Path(args.datadir).iterdir()):
        if not (d / "dialogue.json").exists():
            continue
        migrate_raw(d)
        if not (d / "whisper_words.json").exists():
            continue
        r = d / "assistant_dialogue.refined.json"
        if r.exists() and not args.force:
            try:
                if json.loads(r.read_text(encoding="utf-8")).get("format") == FORMAT:
                    continue
            except json.JSONDecodeError:
                pass
        refine_dialogue(d, sp); n += 1
    print(f"refined {n} dialogues")


if __name__ == "__main__":
    main()
