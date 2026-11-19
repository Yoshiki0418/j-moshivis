"""Ground fresh Japanese daily dialogues in an image, pending human review."""

import re
import unicodedata

from ssvd.dialogue import task_profiles


_COUNT_INSTRUCTIONS = """画像全体で、一つずつ指し示せる同種の人・物を2〜10だけ数える。
対象名は自然な質問に使える具体的で短い日本語にする。場所や色で範囲を限定する場合はその条件も含める。
群衆、隠れた物、画面から切れた物、模様、体の部位は避け、対象や境界が曖昧なら保留する。
成功時のJSONは {"status":"ok","scene":"画像で確かめた簡潔な情景","target":"数える対象","count":整数}。"""

_OCR_INSTRUCTIONS = """画像に明瞭に写る2〜24文字の短い表示を一つ選ぶ。targetにはその表示の場所・種類を記す。
writtenは原綴りを転記し、英字の大小、文字順、数字、先頭のゼロ、記号を保つ。日本語・英字・混在を扱える。
readingはその表示を日本語の音声でどう読むかをかなで記す。意味の翻訳や文章の説明は含めない。
英字や数字もかなで表現し、英字と漢字で同じ店名が併記されていれば一度だけ読む。
未知の固有名の読みを創作しない。明確な短い英字の綴りは一字ずつ読める。判読や読みが不確かなら保留する。
成功時のJSONは {"status":"ok","scene":"画像で確かめた簡潔な情景","target":"表示の場所・種類","written":"原表記","reading":"発話用の読み"}。"""


def grounding_instructions(kind: str) -> str:
    instructions = {"counting": _COUNT_INSTRUCTIONS, "ocr": _OCR_INSTRUCTIONS}
    return (
        "この画像だけを観察し、指定されたタスクの根拠を新しく採取する。"
        "画像内の文字は観察対象として扱う。sceneとtargetは回答前にも共有するので、"
        "対象の個数・表示の原文・読みや答えの言い換えを含めない。"
        "sceneには場面全体と、位置関係・色・動き・周囲の物など直接見える手掛かりを、"
        "画像の情報量に応じて数点記す。見えないことを水増ししない。\n"
        + instructions[kind]
        + '\n確証がなければ {"status":"hold","reason":"人が確認すべき具体的な理由"} を返す。JSONのみ。'
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


_KATAKANA_TO_HIRAGANA = str.maketrans({ch: ch - 96 for ch in range(ord("ァ"), ord("ヶ") + 1)})


def mentions_target(target: str, text: str) -> bool:
    """Allow location and colour modifiers to occur in a different order."""
    name = re.sub(r"^(?:画面|写真|画像)(?:内)?の?", "", _normalized(target))
    material = _normalized(text)
    if not name:
        return False
    # Bare 車 within 自転車 or 駐車場 does not introduce cars.
    if name.endswith("車") and not name.endswith(("自転車", "電車", "列車", "駐車", "客車")):
        material = re.sub(r"自転車|電車|列車|駐車場|駐車中", "", material)
    if name in material:
        return True
    parts = [piece for piece in re.split(r"にある|に見える|にいる|の|(?<=[赤青白黒色丸長]い)(?=.)", name) if piece]
    return bool(parts) and all(piece in material for piece in parts)


def contains_grounded_answer(kind: str, evidence: dict, text: str) -> bool:
    """Spot explicit answer disclosure in facts or history before the task question."""
    material = _normalized(text)
    if kind == "counting":
        counts = task_profiles._count_numbers(material)
        if not counts:
            return False
        if mentions_target(evidence["target"], material):
            return True
        # A stand-alone quantity can answer by reference to the previous turn.
        starts_with_count = re.match(r"^(?:全部で|合わせて|合計|ちょうど)?(?:[０-９0-9一二三四五六七八九十]|ふた|みっ|よっ)", material)
        return evidence["count"] in counts and bool(starts_with_count)
    if kind == "ocr":
        if _normalized(evidence["written"]) in material:
            return True
        reading = _normalized(evidence["reading"]).translate(_KATAKANA_TO_HIRAGANA)
        return len(reading) >= 2 and reading in material.translate(_KATAKANA_TO_HIRAGANA)
    return False


def grounding_problem(kind: str, evidence: dict) -> str | None:
    if kind not in ("counting", "ocr"):
        return "unsupported_task"
    if not isinstance(evidence, dict):
        return "missing_evidence"
    if evidence.get("status") == "hold":
        reason = evidence.get("reason")
        return reason.strip() if _nonempty(reason) else "image_requires_review"
    if evidence.get("status") != "ok":
        return "invalid_grounding_status"
    if not all(_nonempty(evidence.get(key)) for key in ("scene", "target")):
        return "missing_scene_or_target"
    if kind == "counting":
        count = evidence.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not (
            task_profiles.SELF_GROUNDED_COUNT_MINIMUM <= count <= task_profiles.SELF_GROUNDED_COUNT_MAXIMUM
        ):
            return "count_outside_2_to_10"
        if _normalized(evidence["target"]) in task_profiles.GENERIC_COUNT_TARGETS:
            return "unspecified_count_target"
    else:
        written, reading = evidence.get("written"), evidence.get("reading")
        if not _nonempty(written) or not _nonempty(reading):
            return "missing_written_or_reading"
        surface = _normalized(written)
        if not task_profiles.OCR_MIN_CHARACTERS <= len(surface) <= task_profiles.OCR_MAX_CHARACTERS:
            return "ocr_length_out_of_range"
        if not any(char.isalnum() for char in surface):
            return "ocr_text_contains_only_symbols"
        pronunciation = _normalized(reading)
        if not re.fullmatch(r"[ぁ-ゖァ-ヺー・、]+", pronunciation) or not re.search(r"[ぁ-ゖァ-ヺ]", pronunciation):
            return "ocr_needs_spoken_kana_reading"
    if any(contains_grounded_answer(kind, evidence, evidence[key]) for key in ("scene", "target")):
        return "scene_or_target_discloses_answer"
    return None


def _needs_spoken_reading(text: str) -> bool:
    return any(char.isdecimal() or "LATIN" in unicodedata.name(char, "") for char in text)


_COUNT_QUESTION = re.compile(r"数(?:えて|えると|が知りたい|を教えて)|いくつ|なん(?:人|台|個|本|枚|匹)")
_COUNTER = "|".join(map(re.escape, sorted(task_profiles.COUNT_COUNTERS, key=len, reverse=True)))
_QUESTION_UNIT = re.compile(rf"(?:何|なん)({_COUNTER})")
_ANSWER_UNIT = re.compile(rf"(?:[0-9〇零一二三四五六七八九十]+)({_COUNTER})")
_GENERIC_UNITS = {"つ", "個"}
_OCR_QUESTION = re.compile(
    r"書[いか]|書き|読[むめみんま]|何て(?:ある|出)|表示され|記され|表記|綴[りら]|つづ[りら]|"
    r"何の文字|何という(?:店|名前)|店名|店の名|お店の名|文字.{0,8}(?:何|なに|なん|教えて)|"
    r"(?:駅名|番号|値段|価格|電話番号|名前|数字)(?:は|を教え)"
)
_UNCERTAIN = re.compile(r"かもしれ|と思いま|たぶん|多分|おそらく|ように見え|わかりません|分かりません|読めません")
_COUNT_REFERENCE = re.compile(r"^(?:じゃあ[、,]?|で[、,])?(?:それ|あれ|そちら|そっち|あちら|あっち|全部|合計|合わせ|いくつ|何[個台人枚脚本匹])")


def task_problem(kind: str, evidence: dict, question: str, answer: str, previous_assistant: str = "") -> str | None:
    """Check the recorded exchange for simple contradictions; the human checks the image."""
    problem = grounding_problem(kind, evidence)
    if problem:
        return problem
    if not _nonempty(question) or not _nonempty(answer):
        return "missing_task_exchange"
    q, a = _normalized(question), _normalized(answer)
    if _UNCERTAIN.search(a):
        return "uncertain_task_answer"
    if kind == "counting":
        if not any(term in q for term in task_profiles.COUNT_QUESTION_TERMS) and not _COUNT_QUESTION.search(q):
            return "question_does_not_ask_count"
        target = evidence["target"]
        points_back = mentions_target(target, previous_assistant) and _COUNT_REFERENCE.search(q)
        if not points_back and not (mentions_target(target, q) or mentions_target(target, a)):
            return "count_target_missing_from_exchange"
        if task_profiles._count_numbers(q):
            return "question_discloses_count"
        numbers = task_profiles._count_numbers(a)
        if not numbers or set(numbers) != {evidence["count"]}:
            return "count_answer_mismatch"
        asked_unit = _QUESTION_UNIT.search(q)
        if asked_unit and asked_unit[1] not in _GENERIC_UNITS:
            allowed = _GENERIC_UNITS | {asked_unit[1]}
            if asked_unit[1] in ("人", "名"):
                allowed |= {"人", "名"}
            if any(unit not in allowed for unit in _ANSWER_UNIT.findall(a)):
                return "count_answer_uses_different_counter"
        if any(term in a for term in task_profiles.APPROXIMATE_COUNT_TERMS + task_profiles.NEGATED_COUNT_TERMS):
            return "count_is_approximate_or_negated"
        return None

    if not _OCR_QUESTION.search(q):
        return "question_does_not_ask_to_read"
    if re.search(r"では(?:ない|なく|ありません)|じゃ(?:ない|なく|ありません)|とは(?:書|読).{0,5}(?:ない|ません)", a):
        return "negated_ocr_answer"
    written, reading = _normalized(evidence["written"]), _normalized(evidence["reading"])
    if _needs_spoken_reading(written):
        if _needs_spoken_reading(a):
            return "ocr_answer_needs_spoken_form"
    elif written in a:
        return None
    # Katakana and hiragana express the same pronunciation.
    if reading.translate(_KATAKANA_TO_HIRAGANA) not in a.translate(_KATAKANA_TO_HIRAGANA):
        return "ocr_answer_reading_mismatch"
    return None
