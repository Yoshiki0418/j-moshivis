"""アノテーションスクリプト用のユーティリティ（主に後処理）"""

import re
from functools import lru_cache
from typing import Dict, List, Pattern, Sequence, Tuple

PIXELPROSE_TRIM_CANDIDATES = (
    "The image is",
    "This image is",
    "The background is",
    "The text is in",
    "The font is",
    "The style of the image is",
    "This is a photograph",
    # 日本語キャプション用の追加候補
    "この画像は",
    "この写真は",
    "背景は",
    "文字は",
)


def preprocess_pixelprose_captions(caption: str) -> Dict[str, str]:
    """PixelProse キャプションを前処理"""
    caption = caption.strip()
    if caption.startswith("This image displays"):
        caption = caption[len("This image displays:") :].strip()
    if caption.startswith("この画像は") or caption.startswith("この写真は"):
        caption = caption.split("は", 1)[-1].strip()

    if not caption:
        return {"caption": ""}

    # 先頭を大文字化（英語の場合）
    caption = caption[0].upper() + caption[1:] if caption[0].isalpha() else caption[0] + caption[1:]

    sentences = [s.strip().replace("\n", " ") for s in re.split(r"[。\.]", caption)]
    sentences = [x for x in sentences if len(x) > 0]
    if len(sentences) > 0:
        for idx, sentence in enumerate(sentences[2:], 2):
            if any(sentence.startswith(c) for c in PIXELPROSE_TRIM_CANDIDATES):
                sentences = sentences[:idx]
                break
        if not (sentences[-1].endswith("。") or sentences[-1].endswith(".")):
            sentences[-1] += "。"

    return {"caption": "。".join(sentences)}


def maybe_shorten_caption(caption: str, max_cap_len: int = 1500) -> str:
    """キャプションを指定文字数以下に短縮"""
    if len(caption) < max_cap_len:
        shortened_cap = caption
    else:
        shortened_cap = ""
        for sentence in re.split(r"[。\.]", caption):
            if len(shortened_cap) + len(sentence) < max_cap_len:
                shortened_cap += sentence + "。"
            else:
                break
            if shortened_cap.endswith("。。"):
                shortened_cap = shortened_cap[:-1]
        if not shortened_cap:
            shortened_cap = caption[:max_cap_len]
    return shortened_cap


@lru_cache
def compile_pattern(s: str) -> Pattern:
    """正規表現をキャッシュ付きでコンパイル"""
    return re.compile(s)


@lru_cache
def get_replace_pattern() -> Pattern:
    """LLM 出力の軽い後処理用パターン"""
    left_right_replace = r'([*\s"「」』『]?)+'  # 日本語の引用符も含める
    speaker_string = r"(Speaker [1-2]|Me|Question|Answer|質問|回答|話者[12]):(\s[(（].+[)）])?"
    pattern = re.compile(
        f'({left_right_replace + speaker_string + left_right_replace}|"$)'
    )
    return pattern


def get_strings_for_logging(
    s: List[Dict], length_q: int = 40, length_a: int = 160
) -> Tuple[str, str]:
    """ログ用に質問と回答を短縮"""
    q, a = "None", "None"

    if not s:
        return q, a

    if isinstance(s[0], dict):
        if "question" in s[0]:
            q, a = s[0]["question"], s[0].get("answer", "")
        elif "質問" in s[0]:
            q, a = s[0]["質問"], s[0].get("回答", "")
        elif "caption" in s[0]:
            q, a = s[0]["caption"], s[1]["caption"]
        elif "text" in s[0]:
            q, a = s[0]["text"], s[1]["text"]

    if isinstance(s[0], str):
        if len(s) > 1:
            q, a = s[0], s[1]
        else:
            q, a = s[0], ""

    def __extend_string__(s: str, length: int) -> str:
        if len(s) < length:
            return s + " " * (length - len(s))
        return s[: length - 3] + "..."

    return __extend_string__(q, length=length_q), __extend_string__(a, length=length_a)


def sanitize_line(s: str) -> str:
    """発話行をサニタイズ"""
    if not isinstance(s, str):
        raise ValueError
    s = s.replace("*", "").strip()
    if s and s[0] in ['"', "'", "「", "『"]:
        s = s[1:]
    if s and s[-1] in ['"', "'", "」", "』"]:
        s = s[:-1]
    return s.strip()


def postprocess_synth_annot(
    uid: str,
    res: Dict[str, str] | List[str],
    idx: Dict[str, int],
    min_num_turns: int = 3,
    trim_first_question: bool = False,
) -> Sequence:
    """合成アノテーションをDB保存用の形式に変換"""
    rows = []
    try:
        speaker = 1  # MTC は必ず質問者から開始

        for turn, it in enumerate(res):
            speaker = int((turn % 2) == 0)

            if turn == 0 and speaker == 1 and trim_first_question:
                pos = it.find("？")
                if pos == -1:
                    pos = it.find("?")
                if pos != -1:
                    it = it[: pos + 1]

            # 回答に質問が混じっている場合を削除
            if speaker == 0:
                pos = max(it.find("?"), it.find("？"))
                if 0 <= pos < len(it) - 10:
                    it = it[pos + 1 :]

            # 短すぎる応答は無視
            if len(it) < 2:
                break

            rows.append((uid, idx[uid], turn, speaker, sanitize_line(it)))

        if len(rows) < min_num_turns:
            raise KeyError

        idx[uid] += 1
    except (KeyError, ValueError):
        return []

    return rows
