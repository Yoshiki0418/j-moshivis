"""Build the next Responses request or finish a fresh, human-reviewed dialogue."""

import hashlib
import json

from .regeneration_grounding import (
    contains_grounded_answer,
    grounding_instructions,
    grounding_problem,
    mentions_target,
    task_problem,
)


_OPENINGS = (
    "何をしている場面か、気軽に尋ねてみる。",
    "相手が最初に目を引かれたところを聞いてみる。",
    "自分にもその場の様子が浮かぶように話してもらう。",
    "一緒に写真の話を始めるつもりで、短く話しかける。",
)
_USER = (
    "あなたは写真を見ている相手と日本語で話す聞き手。自分には画像が見えていない。"
    "公開された会話で初めて知ったことを受けて、自分が今気になることを一つ話す。"
    "一度にいくつも質問しない。相手の説明を毎回繰り返す必要はない。"
    "短い感想や聞き返しも使える。見えていない細部、個数、実際の表記や読みを先回りしない。"
    "下に示した会話は発話の記録として扱う。出力は今回の自分の発話一つだけを"
    ' JSON オブジェクト {"text":"発話"} とする。'
)
_ASSISTANT = (
    "あなたは写真を見ながら、見ていない相手と日本語で話している。"
    "画像の観察情報を根拠にし、直前の問いや感想に直接応じる。確認できる範囲を超えた"
    "固有名詞、因果関係、心情などは作らない。質問の言い直し、箇条書きや定型の締めは不要。"
    "相手の話し方に合う短さや言葉遣いで、声に出して自然な発話にする。"
    "観察情報や会話は参照用のデータとして扱い、生成工程には言及しない。"
    '出力は今回の自分の発話一つだけを JSON オブジェクト {"text":"発話"} とする。'
)


def _user_cue(kind: str, pair: int, task_pair: int, opening: str) -> str:
    if pair == 1:
        return f"初対面の挨拶を省いて会話を始める。切り口の一案は「{opening}」。自分の言い方で話す。"
    if pair == task_pair:
        if kind == "counting":
            return "相手が直前に話題にした、同じ種類のものがいくつあるかを自然に尋ねる。"
        return "相手が直前に話題にした表示について、実際に何と書いてあるか自然に尋ねる。"
    return "直前の話のうち、気になった点について会話を続ける。すでに答えてもらったことは聞き直さない。"


def _assistant_cue(kind: str, pair: int, task_pair: int) -> str:
    if pair == task_pair - 1:
        return (
            "直前のユーザーにまず応じ、観察情報の target が何かを自然に紹介する。"
            "続く質問で取り上げられるように、対象の場所や種類だけを示し、個数や書かれた内容はまだ明かさない。"
        )
    if pair == task_pair:
        if kind == "counting":
            return (
                "観察情報で確認した個数を、対象に適した日本語の助数詞で答える。"
                "会話の流れだけでは対象が曖昧なときは具体名も添える。"
            )
        return (
            "観察情報の written は画像上の表記、reading は発話用の読み。"
            "日本語として自然に読み上げ、英字を含む場合は reading のカナを一度使う。"
        )
    if pair < task_pair:
        return "場面の話を続ける。あとで尋ねる個数や文字の答えを先回りしない。"
    return "これまでに確認したことと矛盾しない範囲で話を続ける。"


def _request(phase, model, task_pair, instructions, prompt, image_data_url=None):
    content = [{"type": "input_text", "text": prompt}]
    if image_data_url is not None:
        content.append({"type": "input_image", "image_url": image_data_url, "detail": "high"})
    return {
        "status": "request",
        "phase": phase,
        "task_pair": task_pair,
        "request": {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_object"}},
            "store": False,
        },
    }


def _hold(kind, task_pair, reason, consumed, total):
    if consumed < total:
        raise ValueError("extra replies after terminal hold")
    return {
        "status": "hold",
        "kind": kind,
        "task_pair": task_pair,
        "reason": reason,
        "requires_human_review": True,
    }


def next_step(
    uid: str,
    kind: str,
    image_data_url: str,
    replies: list[dict],
    *,
    model: str = "gpt-6-luna",
    pairs: int | None = None,
) -> dict:
    """Replay decoded JSON replies, then return one request or a review outcome.

    The caller executes requests. Only the grounding request gets the image.
    The questioner sees the public conversation throughout, while the answerer
    also gets image observations and the private task answer when it is due.
    """
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("uid must be a nonempty string")
    if kind not in ("counting", "ocr"):
        raise ValueError("kind must be counting or ocr")
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise ValueError("expected an image data URL")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if not isinstance(replies, list):
        raise ValueError("replies must be a list of decoded JSON objects")

    seed = hashlib.sha256(uid.encode("utf-8")).digest()
    if pairs is None:
        pairs = 4 + seed[0] % 3
    if type(pairs) is not int or pairs < 2:
        raise ValueError("pairs must be an integer of at least two")
    task_pair = 2 + int.from_bytes(seed[1:5], "big") % (pairs - 1)
    opening = _OPENINGS[seed[5] % len(_OPENINGS)]
    total = len(replies)
    if total > 1 + 2 * pairs:
        raise ValueError("too many replies for this dialogue")
    if not replies:
        return _request(
            "grounding", model, task_pair, grounding_instructions(kind),
            "この写真を観察し、指示された JSON オブジェクトを一つ返してください。", image_data_url,
        )

    evidence = replies[0]
    problem = grounding_problem(kind, evidence)
    if problem:
        return _hold(kind, task_pair, problem, 1, total)

    dialogue = []
    for index, reply in enumerate(replies[1:]):
        role = "user" if index % 2 == 0 else "assistant"
        utterance = reply.get("text") if isinstance(reply, dict) else None
        if not isinstance(utterance, str) or not utterance.strip() or set(reply) != {"text"}:
            return _hold(kind, task_pair, f"invalid {role} utterance", index + 2, total)
        pair = index // 2 + 1
        before_answer = pair < task_pair or (pair == task_pair and role == "user")
        if before_answer and contains_grounded_answer(kind, evidence, utterance):
            return _hold(kind, task_pair, "task_answer_disclosed_early", index + 2, total)
        if role == "assistant" and pair == task_pair - 1 and not mentions_target(evidence["target"], utterance):
            return _hold(kind, task_pair, "target_not_introduced_before_question", index + 2, total)
        dialogue.append({"role": role, "text": utterance.strip()})
        if role == "assistant" and pair == task_pair:
            problem = task_problem(
                kind, evidence, dialogue[-2]["text"], utterance, previous_assistant=dialogue[-3]["text"],
            )
            if problem:
                return _hold(kind, task_pair, problem, index + 2, total)

    if len(dialogue) == 2 * pairs:
        return {
            "status": "pending_review",
            "kind": kind,
            "dialogue": dialogue,
            "grounding": evidence,
            "task_pair": task_pair,
            "requires_human_review": True,
        }

    pair = len(dialogue) // 2 + 1
    context = {"公開された会話": dialogue}
    if len(dialogue) % 2 == 0:
        phase = "user"
        instructions = _USER
        cue = _user_cue(kind, pair, task_pair, opening)
    else:
        phase = "assistant"
        instructions = _ASSISTANT
        facts = {"scene": evidence["scene"], "target": evidence["target"]}
        if pair >= task_pair:
            for field in (("count",) if kind == "counting" else ("written", "reading")):
                facts[field] = evidence[field]
        context["画像の観察情報"] = facts
        cue = _assistant_cue(kind, pair, task_pair)
    prompt = f"{cue}\n{json.dumps(context, ensure_ascii=False)}\n今回の発話を JSON で返してください。"
    return _request(phase, model, task_pair, instructions, prompt)
