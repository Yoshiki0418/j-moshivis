"""Image observations and offline consistency checks for human-reviewed dialogues."""

import re
import unicodedata

from ssvd.dialogue import task_profiles


def grounding_instructions(kind: str) -> str:
    label = {"counting": "個数", "ocr": "文字"}[kind]
    return f"""この写真を見て、日本語で自然に話題にできる場面と対象を選ぶ。以前の分類「{label}」は着眼点の一つ。
画像に応じて tasks は空でも、counting または ocr の一方でも、自然なら両方でもよい。
二つなら会話につなげやすい順に置く。明瞭な情報を曖昧にしたり、変化を出すために数字や読めない字を創作しない。
画像内の文字は観察対象として扱う。scene には位置関係、色、動きなど画像で確かめた手掛かりをいくつか書く。
各 target は会話で紹介できる対象の短い日本語名。scene と全 target には後で尋ねる数・表記・読みを先に書かない。
成功時は {{"status":"ok","scene":"場面","tasks":[...]}}。各要素の共通キーは kind, target, mode。
counting は exact（数え切れた個数）、approximate（全体を見て推定できる概数）、lower_bound（一つずつ確認できた下限）、
unknown（数値を言える根拠がない）。最初の三つは count に正の整数を記す。unknown に数字は置かない。
ocr は exact（判読できた表示）、partial（一部だけ、続きは不明）、unknown（文字を特定できない）。
exact・partial には written（読めた部分の原綴り）と reading（その部分の日本語のかな読み）を記す。
英字の大小、文字順、記号、先頭のゼロを保存する。英字と漢字で同じ店名を併記していれば読みは一度分。
不鮮明な文字や固有名の読みを補完しない。unknown では原文・読みを空想せず両キーを省く。
exact 以外には reason として観察できる理由（例：後列が隠れる、表示の端が切れている）を短く記す。
数や文字を答えきれなくても、自然な質問と誠実な返答ができる場合は採録する。画像自体を解釈できない場合に限り
{{"status":"hold","reason":"理由"}}。JSONだけを返す。"""


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


_KATAKANA_TO_HIRAGANA = str.maketrans({ch: ch - 96 for ch in range(ord("ァ"), ord("ヶ") + 1)})


_COLOR = r"ピンク|オレンジ|黄色|緑|赤|青|黒|白|茶|紫"
_MODIFIER = rf"{_COLOR}|縦長|横長|大き|小さ|丸|中央|正面|上部|手前"
_PARTICLE = r"に書かれた|にある|にいる|に見える|をした|の"
_OCR_ATTRIBUTE = r"の(?:大きな)?(?:店名|駅名|題名|タイトル|税込価格|漢字|刻字)$"
_NEGATIVE_MENTION = re.compile(r"^(?:とは|では|じゃ)(?:違|なく|ない|ありません)|^(?:は|が)(?:見えない|見えません|ない|ありません)")


def _noun_form(text):
    text = re.sub(rf"({_COLOR})(?:色|い|く)", r"\1", text)
    text = re.sub(r"(大き|小さ|丸)[ないく]", r"\1", text)
    text = re.sub(r"縦(?=看板)", "縦長", text)
    return re.sub(r"上の(?:方|ほう)", "上部", text)


def _target_parts(name):
    # Preserve every noun and modifier. Split a few productive compounds so
    # 「料金所上」 can match 「料金所の上」 and 「縦長看板」 can match 「縦長の看板」.
    name = re.sub(r"([一-龯])上(?=の|$)", r"\1の上", name)
    name = re.sub(r"([一-龯])(?=ポスター)", r"\1の", name)
    target = re.split(rf"({_MODIFIER})|(?:{_PARTICLE})", _noun_form(name))
    return [word for word in target if word]


def _mentions_phrase(name, material, for_ocr):
    if for_ocr:
        # Introducing the sign suffices before asking for its name or price.
        # Keep selectors attached to the sign itself (colour, shape, position).
        name = re.sub(_OCR_ATTRIBUTE, "", name)
    parts = _target_parts(name)
    if not parts:
        return False
    # No cross-sentence or arbitrary-word match: e.g. a red car beside a blue
    # sign does not establish a red sign. Allow grammatical particles and
    # extra visual modifiers between consecutive anchors in the same phrase.
    extra = r"の|に|は|を|が|ある|した|書かれた|光る|な|く|、|,|色"
    modifiers = _MODIFIER
    if any(re.fullmatch(_COLOR, part) for part in parts):
        modifiers = r"縦長|横長|大き|小さ|丸|中央|正面|上部|手前"
    gap = rf"(?:(?:{extra}|{modifiers})){{0,6}}"
    pattern = gap.join(re.escape(word) for word in parts)
    material = _noun_form(material)
    for hit in re.finditer(pattern, material):
        if not _NEGATIVE_MENTION.match(material[hit.end():]):
            return True
    return False


def mentions_target(target: str, text: str, *, allow_paraphrase=False, for_ocr=False) -> bool:
    """Allow location and colour modifiers to occur in a different order."""
    name = re.sub(r"^(?:画面|写真|画像)(?:内)?の?", "", _normalized(target))
    material = _normalized(text)
    if not name:
        return False
    if name.endswith("車") and not name.endswith(("自転車", "電車", "列車", "駐車", "客車")):
        material = re.sub(r"自転車|電車|列車|駐車場|駐車中", "", material)
    if allow_paraphrase:
        return _mentions_phrase(name, material, for_ocr)
    if name in material:
        return True
    parts = [piece for piece in re.split(r"にある|に見える|にいる|の|(?<=[赤青白黒色丸長]い)(?=.)", name) if piece]
    return bool(parts) and all(piece in material for piece in parts)


def contains_grounded_answer(kind: str, evidence: dict, text: str) -> bool:
    """Check for a recorded answer appearing before its exchange."""
    material = _normalized(text)
    if kind == "counting":
        # Counts in an OCR reading or a different object should not spoil a later count.
        counts = _quantities(material)
        if evidence.get("count") not in counts:
            return False
        return mentions_target(evidence["target"], material) or bool(
            re.match(r"^(?:全部で|合わせて|合計|ちょうど)?[０-９0-9一二三四五六七八九十百千ふみよ]", material)
        )
    if kind == "ocr" and evidence.get("written"):
        written = _normalized(evidence["written"])
        reading = _normalized(evidence["reading"]).translate(_KATAKANA_TO_HIRAGANA)
        # A one-character fragment often occurs in an unrelated compound (駅, 口).
        # Only treat it as disclosed if the speaker actually identifies the writing.
        if len(written) == 1:
            return bool(re.search(rf"[「『]{re.escape(written)}[」』]|{re.escape(written)}(?:と|って)(?:書|読)", material))
        return written in material or len(reading) >= 2 and reading in material.translate(_KATAKANA_TO_HIRAGANA)
    return False


_MODES = {"counting": {"exact", "approximate", "lower_bound", "unknown"},
          "ocr": {"exact", "partial", "unknown"}}


def _evidence_problem(kind: str, task: dict) -> str | None:
    mode = task.get("mode")
    if kind not in _MODES or not isinstance(mode, str) or mode not in _MODES[kind]:
        return "invalid_task_mode"
    if not _nonempty(task.get("target")):
        return "missing_target"
    if mode != "exact" and not _nonempty(task.get("reason")):
        return "uncertainty_needs_visual_reason"
    values = set(task) - {"kind", "target", "mode", "reason"}
    if mode == "unknown":
        return "unknown_task_contains_answer" if values else None
    if kind == "counting":
        if values != {"count"} or type(task["count"]) is not int or task["count"] <= 0:
            return "count_must_be_positive_integer"
        if _normalized(task["target"]) in task_profiles.GENERIC_COUNT_TARGETS:
            return "unspecified_count_target"
    else:
        if values != {"written", "reading"} or not all(_nonempty(task[k]) for k in values):
            return "missing_written_or_reading"
        if not any(char.isalnum() for char in _normalized(task["written"])):
            return "ocr_text_contains_only_symbols"
        pronunciation = _normalized(task["reading"])
        if not re.fullmatch(r"[ぁ-ゖァ-ヺー・、]+", pronunciation) or not re.search(r"[ぁ-ゖァ-ヺ]", pronunciation):
            return "ocr_needs_spoken_kana_reading"
    return None


def grounding_problem(kind: str, evidence: dict) -> str | None:
    if kind not in _MODES or not isinstance(evidence, dict):
        return "missing_evidence"
    if evidence.get("status") == "hold":
        reason = evidence.get("reason")
        return reason.strip() if _nonempty(reason) else "image_requires_review"
    if evidence.get("status") != "ok" or not _nonempty(evidence.get("scene")):
        return "missing_observable_scene"
    tasks = evidence.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > len(_MODES):
        return "invalid_tasks"
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("kind"), str) or task["kind"] in seen:
            return "invalid_or_duplicate_task"
        seen.add(task["kind"])
        problem = _evidence_problem(task["kind"], task)
        if problem:
            return problem
    public = [evidence["scene"], *(task["target"] for task in tasks)]
    for task in tasks:
        if any(contains_grounded_answer(task["kind"], task, text) for text in public):
            return "scene_or_target_discloses_answer"
    return None


_COUNTER = "|".join(map(re.escape, sorted(task_profiles.COUNT_COUNTERS, key=len, reverse=True)))
_QUESTION_UNIT = re.compile(rf"(?:何|なん)({_COUNTER})")
_NUMBER = r"[0-9〇零一二三四五六七八九十百千万]+"
_NUMBER_WITH_UNIT = re.compile(rf"({_NUMBER})({_COUNTER})")
_COUNT_QUESTION = re.compile(r"数(?:えて|えると|が知りたい|を教えて)|いくつ(?!か)|何台|何人|(?:台数|人数|個数|本数)(?:は|って|を)")
_OCR_QUESTION = re.compile(
    r"書[いかき]|読[むめみんま]|表示され|記され|表記|綴[りら]|つづ[りら]|"
    r"何の文字|何という(?:店|名前)|店名|店の名|お店の名|文字.{0,8}(?:何|なに|なん|教えて)|"
    r"(?:駅名|番号|値段|価格|電話番号|名前|数字)(?:は|を教え)"
)
_OCR_EXPLICIT = re.compile(r"(?:何|なに|なん|どう).{0,5}(?:書い|書か|読[めむん]|表記)|(?:看板|表札|表示|文字).{0,7}(?:読[めん]|読み取|(?:電話番号|店名)(?:は|って|を))")
_COUNT_REFERENCE = re.compile(r"^(?:じゃあ[、,]?|で[、,])?(?:それ|あれ|そちら|そっち|あちら|あっち|全部|合計|合わせ|いくつ|何)")
_UNKNOWN = re.compile(
    r"(?:分|判|わ)か(?:らな|らず|りませ|んない|んね)|不(?:明|鮮明)|はっきりしな|はっきりとは?言え|"
    r"(?:読み取|読|数え|見え|判別|判断|区別|特定|確認|断定|言い切|見分け|見切|見通)"
    r"(?:.{0,8}?)(?:な[いく]|ず|ません|難し|無理)|難し|むずかし|判然としな|判別がつか"
)
_ESTIMATE = re.compile(r"くらい|ぐらい|ほど|前後|およそ|約|ざっと|ざっくり|だいたい|大体|たぶん|多分|おそらく|かな|程度|かも|ように見え|に見え|いそう|だと思")
_LOWER = re.compile(
    rf"少なくとも|少なくて[もは]|少なく見(?:積も|て)|最低|以上|(?:見え(?:てい?)?る|確認できる)(?:範囲|分|だけ|のは)|"
    rf"(?:{_COUNTER})は(?:確認でき|見え|い(?:る|ます)|あ(?:る|りま))"
)
_UPPER = re.compile(r"以下|未満|多く(?:て|と)も|最大")
_ASSERTED_TOTAL = re.compile(
    rf"ぴったり|きっかり|ちょうど|それで全部|これで全部|で間違い(?:ない|ありません)|"
    rf"(?:全部で|合計(?:で|は)?)({_NUMBER})(?:{_COUNTER})(?:です|だ[よね]|[。、！!]|$)"
)
_NEGATED_NUMBER = re.compile(r"(?:では|じゃ)(?:な[いく]|ありません)|はい(?:ない|ません)|は(?:ない|ありません)")
_DENIED_BOUND = re.compile(
    r"(?:以上|最低|少なくとも).{0,9}?(?:いそうにない|いなさそう|なさそう|見当た(?:らな|りませ)|"
    r"(?:い|あ)る(?:よう|感じ)(?:には|は)?(?:見えな|しな)|とは?言えな)"
)
_PARTIAL = re.compile(r"だけ|しか|まで|一部|途中|残り|続き|前半|後半|先頭|最初|最後|末尾|部分|冒頭")
_QUOTED = re.compile(r"[「『\"]([^」』\"]+)[」』\"]")


def explicit_question_kinds(question: str) -> set[str]:
    """Recognize clear task requests in turns scheduled for ordinary conversation."""
    q = _normalized(question)
    kinds = set()
    if (task_profiles.EXPLICIT_COUNT_QUESTION_RE.search(q)
            or re.search(r"数えて(?:くれ|もら|みて|ほし|[?？])|(?:台数|人数|個数|本数)(?:を教え|が知りたい|[はって]+[?？])", q)):
        kinds.add("counting")
    if _OCR_EXPLICIT.search(q):
        kinds.add("ocr")
    return kinds


def _number_value(surface: str) -> int:
    """Interpret Arabic and ordinary Japanese multiplicative numerals."""
    total = section = 0
    digits = ""
    for ch in surface:
        if ch.isdecimal():
            digits += str(unicodedata.decimal(ch))
        elif ch in task_profiles.KANJI_DIGITS:
            digits += str(task_profiles.KANJI_DIGITS[ch])
        elif ch == "万":
            total += (section + int(digits or "0") or 1) * 10000
            section, digits = 0, ""
        else:
            section += int(digits or "1") * {"十": 10, "百": 100, "千": 1000}[ch]
            digits = ""
    return total + section + int(digits or "0")


def _quantities(text: str) -> list[int]:
    values = [_number_value(m[1]) for m in _NUMBER_WITH_UNIT.finditer(text)]
    for surface, value in task_profiles.NATIVE_COUNT_FORMS.items():
        # Standalone とお is also the start of ordinary words like とおり.
        if surface == "とお":
            if re.search(r"とお(?=くらい|ぐらい|です|だ[よね]|[。、!?]|$)", text):
                values.append(value)
        elif surface in text:
            values.append(value)
    # A short reply may omit the counter, especially before a hedge or lower bound.
    for m in re.finditer(rf"({_NUMBER})(?=(?:くらい|ぐらい|以上|程度|前後|ほど|です|かな|[〜~。、!?]|$))", text):
        values.append(_number_value(m[1]))
    return values


def _short_count_reference(target, previous, question, answer):
    """Resolve '看板はいくつ？' to the one qualified sign in the preceding turn."""
    parts = _target_parts(_normalized(target))
    if len(parts) < 2 or not mentions_target(target, previous, allow_paraphrase=True):
        return False
    head = parts[-1]
    if previous.count(head) != 1:
        return False
    if not re.fullmatch(rf"(?:じゃあ[、,]?)?(?:その|あの)?{re.escape(head)}[はが、,]*"
                        rf"(?:全部で|合わせて)?(?:いくつ|(?:何|なん)(?:{_COUNTER}))"
                        r"(?:ある|あります|見え(?:る|ます)|います|いる)?(?:んです|です)?(?:か|かな|の)?[?？。]*",
                        question):
        return False
    prefix, named, quantity = answer.partition(head)
    # An explicitly different subset in the answer must not inherit the count.
    allowed = "|".join(map(re.escape, sorted(parts[:-1], key=len, reverse=True)))
    if named and not re.fullmatch(rf"(?:{allowed}|その|あの|に|は|の|な|、|,)*", _noun_form(prefix)):
        return False
    # The named object must itself be counted. Possessives (看板の写真が…)
    # and another subject (看板の横に犬が…) cannot intervene before the quantity.
    number = rf"{_NUMBER}(?:{_COUNTER})"
    subject = r"(?:は|が|なら)?[、,]?" if named else ""
    return bool(re.fullmatch(subject + rf"(?:全部で|合計で|少なくとも|約)?{number}"
                             r"(?:くらい|ぐらい|ほど|以上)?(?:は|が)?"
                             r"(?:見えます|見える|ある|あります|いる|います|だよ|です|だね)?"
                             r"(?:よ|ね|かな)?[。!！]?", quantity if named else answer))


def _count_problem(task, question, answer, previous_assistant, *, allow_paraphrase=False):
    if not (_QUESTION_UNIT.search(question) or _COUNT_QUESTION.search(question) or "数は" in question):
        return "question_does_not_ask_count"
    target = task["target"]
    points_back = mentions_target(target, previous_assistant, allow_paraphrase=allow_paraphrase) and _COUNT_REFERENCE.search(question)
    present = any(mentions_target(target, turn, allow_paraphrase=allow_paraphrase) for turn in (question, answer))
    if not present and not points_back and not (allow_paraphrase and _short_count_reference(target, previous_assistant, question, answer)):
        return "count_target_missing_from_exchange"
    mode = task["mode"]
    if mode == "unknown":
        # 「一人ひとりは数えられない」 describes the difficulty rather than a count of one.
        statement = re.sub(r"(?:一人|ひとり)(?:一人|ひとり|ずつ)|一つ(?:一つ|ひとつ|ずつ)", "", answer)
        if (_quantities(statement) or re.search(rf"[0-9]|{_NUMBER}数(?:{_COUNTER})", statement)
                or not _UNKNOWN.search(answer)):
            return "unknown_count_needs_non_numeric_answer"
        return None
    numbers = _quantities(answer)
    if set(numbers) != {task["count"]}:
        return "count_answer_mismatch"
    unit = _QUESTION_UNIT.search(question)
    if unit and unit[1] not in {"つ", "個"}:
        allowed = {"つ", "個", unit[1]}
        if unit[1] in ("人", "名"):
            allowed.update(("人", "名"))
        if any(m[2] not in allowed for m in _NUMBER_WITH_UNIT.finditer(answer)):
            return "count_answer_uses_different_counter"
    if _UPPER.search(answer) or _NEGATED_NUMBER.search(answer) or _DENIED_BOUND.search(answer):
        return "unsupported_or_negated_count"
    if mode != "exact" and _ASSERTED_TOTAL.search(answer):
        return "count_answer_claims_exact_total"
    # A visible count with an explicit unknown remainder is also a lower bound.
    lower = bool(_LOWER.search(answer) or (
        _UNKNOWN.search(answer) and re.search(r"(?:見え|確認でき)", answer)
    ))
    estimate = bool(_ESTIMATE.search(answer))
    if mode == "lower_bound" and not lower:
        return "count_answer_loses_lower_bound"
    if mode == "approximate" and (lower or not estimate):
        return "count_answer_loses_estimate"
    if mode == "exact" and (lower or estimate or _UNKNOWN.search(answer)):
        return "count_answer_changes_certainty"
    return None


_ROMAJI = {}
for _kana, _latin in (
    ("あいうえお", "a i u e o"), ("かきくけこ", "ka ki ku ke ko"), ("がぎぐげご", "ga gi gu ge go"),
    ("さしすせそ", "sa shi su se so"), ("ざじずぜぞ", "za ji zu ze zo"), ("たちつてと", "ta chi tsu te to"),
    ("だぢづでど", "da ji zu de do"), ("なにぬねの", "na ni nu ne no"), ("はひふへほ", "ha hi fu he ho"),
    ("ばびぶべぼ", "ba bi bu be bo"), ("ぱぴぷぺぽ", "pa pi pu pe po"), ("まみむめも", "ma mi mu me mo"),
    ("やゆよ", "ya yu yo"), ("らりるれろ", "ra ri ru re ro"), ("わをん", "wa wo n"),
):
    _ROMAJI.update(zip(_kana, _latin.split()))


def _bilingual_form(written, reading):
    """Keep the Japanese half of a sign when the Latin half spells the supplied kana.

    Only allow a complete plain Hepburn word. Mixed labels, product codes and
    translations with a different pronunciation still need the recorded reading.
    """
    parts = re.fullmatch(r"([一-龯々ぁ-ゖァ-ヺー]+)([A-Za-z]+)|([A-Za-z]+)([一-龯々ぁ-ゖァ-ヺー]+)", written)
    if not parts or not all(ch in _ROMAJI for ch in reading):
        return None
    japanese, latin = (parts[1], parts[2]) if parts[1] else (parts[4], parts[3])
    romanized = "".join(_ROMAJI[ch] for ch in reading)
    return japanese if latin.lower() == romanized else None


def _ocr_problem(task, question, answer, *, allow_paraphrase=False):
    if not _OCR_QUESTION.search(question):
        return "question_does_not_ask_to_read"
    # Explicitly asking about the right sign contradicts evidence for the left
    # sign. Short references like 「その看板」「何て書いてある？」 are fine.
    for side, noun in re.findall(r"([左右])(?:側|端)?の([一-龯ァ-ヺー]+)", task["target"]):
        other = "右" if side == "左" else "左"
        subject = rf"(?:側|端)?の{re.escape(noun)}(?=[にはをが、,？?])"
        if re.search(other + subject, question) and not re.search(side + subject, question):
            return "ocr_question_refers_to_different_target"
    mode = task["mode"]
    if mode == "unknown":
        illegible = _UNKNOWN.search(answer) or re.search(r"文字.{0,4}(?:つぶれ|潰れ|ぼやけ|かすれ)", answer)
        if (not illegible or _QUOTED.search(answer) or re.search(r"[A-Za-z0-9]", answer)
                or re.search(r"(?:と|って)(?:書いてあ|読め(?:る|ます)|読めそう(?!に(?:ない|もない|ありません)))|きっと|たぶん|多分|おそらく|かもしれ|だと思", answer)):
            return "unknown_ocr_needs_unreadable_answer"
        return None
    written = _normalized(task["written"])
    reading = _normalized(task["reading"]).translate(_KATAKANA_TO_HIRAGANA)
    uses_reading = any(ch.isdecimal() or "LATIN" in unicodedata.name(ch, "") for ch in written)
    if uses_reading and re.search(r"[A-Za-z0-9]", answer):
        return "ocr_answer_needs_spoken_form"
    japanese = None
    if allow_paraphrase and mode == "exact" and uses_reading:
        surface = _bilingual_form(written, reading)
        # Bind to the whole quoted string so a coincidental kanji substring
        # (e.g. a station name followed by an invented suffix) cannot pass.
        if surface and surface in _QUOTED.findall(answer):
            japanese = surface
    if not japanese and (uses_reading or written not in answer) and reading not in answer.translate(_KATAKANA_TO_HIRAGANA):
        return "ocr_answer_reading_mismatch"
    for quote in _QUOTED.findall(answer):
        if quote != japanese and quote != written and quote.translate(_KATAKANA_TO_HIRAGANA) != reading:
            return "ocr_answer_contains_unsupported_text"
    if mode == "partial":
        # Guard a positive, unquoted transcription that extends a recorded fragment.
        # A later caveat about the remainder does not support the invented suffix.
        for claim in re.findall(r"([^。、!?「」『』]+?)(?:と|って)(?:書いて|書かれ|読め|読み取)", answer):
            kana = claim.translate(_KATAKANA_TO_HIRAGANA)
            if ((written in claim or reading in kana)
                    and not (claim.endswith(written) or kana.endswith(reading))):
                return "ocr_answer_contains_unsupported_text"
    if re.search(r"では(?:ない|なく|ありません)|じゃ(?:ない|なく|ありません)|とは(?:書|読).{0,5}(?:ない|ません)", answer):
        return "negated_ocr_answer"
    if mode == "partial" and not (_PARTIAL.search(answer) or _UNKNOWN.search(answer)):
        return "ocr_answer_loses_partial_reading"
    if mode == "exact" and _UNKNOWN.search(answer):
        return "unconfirmed_ocr_answer"
    return None


def task_problem(kind: str, evidence: dict, question: str, answer: str, previous_assistant: str = "",
                 *, allow_paraphrase=False) -> str | None:
    """Check simple spoken/evidence contradictions; the reviewer verifies the image."""
    problem = _evidence_problem(kind, evidence)
    if problem:
        return problem
    if not _nonempty(question) or not _nonempty(answer):
        return "missing_task_exchange"
    q, a = _normalized(question), _normalized(answer)
    if kind == "counting":
        return _count_problem(evidence, q, a, previous_assistant, allow_paraphrase=allow_paraphrase)
    return _ocr_problem(evidence, q, a, allow_paraphrase=allow_paraphrase)
