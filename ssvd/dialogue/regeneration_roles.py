"""Build the next Responses request or finish a fresh, human-reviewed dialogue."""

import hashlib
from itertools import combinations
import json
import random

from .regeneration_grounding import (
    contains_grounded_answer,
    explicit_question_kinds,
    grounding_instructions,
    grounding_problem,
    mentions_target,
    task_problem,
)


PAIR_COUNTS = range(4, 9)

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
    "相手の話し方に合う短さや言葉遣いで、声に出して自然な発話にする。見え方の限界はその場に応じた言葉で伝え、決まり文句を重ねない。"
    "観察情報や会話は参照用のデータとして扱い、生成工程には言及しない。"
    '出力は今回の自分の発話一つだけを JSON オブジェクト {"text":"発話"} とする。'
)


def _user_cue(pair, active, opening):
    if pair == 1:
        return f"初対面の挨拶を省いて会話を始める。切り口の一案は「{opening}」。自分の言い方で話す。"
    if active == "counting":
        return "相手が直前に話題にしたものの数が気になっている。見え方や会話に合う聞き方で尋ねる。"
    if active == "ocr":
        return "相手が直前に話題にした表示について、何と書いてあるかを自然に尋ねる。"
    return "直前の返答を受け、情景や物の特徴など気になる点から会話を続ける。個数や文字の読みを問うのは別の機会にする。"


_ANSWER_CUES = {
    "exact": "確認できた内容を、相手の質問に合う短い言葉で答える。",
    "approximate": "観察で裏付けられるおおよその数を伝える。正確に数えたような断定は避ける。",
    "lower_bound": "見分けられた最小の数と、実際にはもっとある可能性があることを自然に伝える。",
    "partial": "読み取れる部分だけを読み上げ、残りはどこまで見えるかを伝える。続きを推測しない。",
    "unknown": "今の画像からは答えを確かめられないことを率直に伝える。具体的な数や文字を推測しない。",
}

_UNCERTAIN_DELIVERY = (
    "相手の質問にまず短く答える。補足は必要な分だけにする。",
    "見えている手掛かりから話し、自分がどこまで判断できたかを伝える。",
    "分かる範囲を先に説明し、確かめきれない理由があれば後から軽く添える。",
    "聞き手の話し方に合わせて、自分なら口頭でどう伝えるかを考える。",
)


def _assistant_prompt(dialogue, observation, active, upcoming, delivery):
    facts = {"scene": observation["scene"]}
    if active is not None:
        facts["この返答の根拠"] = active
        cue = _ANSWER_CUES[active["mode"]]
        if active["kind"] == "counting":
            cue += " 数を伝えるなら対象に合う助数詞を使う。"
        elif active["mode"] != "unknown":
            cue += " written は原綴り、reading は音声用の読み。英字や数字は自然な日本語で読み上げる。"
        if active["mode"] != "exact":
            cue += " " + delivery
    elif upcoming is not None:
        facts["次に紹介する対象"] = upcoming["target"]
        cue = "相手にまず応じ、次に紹介する対象が会話に自然に出てくるよう場所や種類に触れる。個数や表示の中身は先回りしない。"
    else:
        cue = "直前の話に応じ、目に見えることを話す。裏付けのない個数や表示の中身を足さない。"
    context = {"公開された会話": dialogue, "画像の観察情報": facts}
    return cue, context


def _request(phase, model, pairs, positions, instructions, prompt, image_data_url=None):
    content = [{"type": "input_text", "text": prompt}]
    if image_data_url is not None:
        content.append({"type": "input_image", "image_url": image_data_url, "detail": "high"})
    return {
        "status": "request",
        "phase": phase,
        "pair_count": pairs,
        "task_pairs": positions,
        "request": {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_object"}},
            "store": False,
        },
    }


def _hold(kind, pairs, positions, reason, consumed, total):
    if consumed < total:
        raise ValueError("extra replies after terminal hold")
    return {
        "status": "hold", "source_kind": kind, "pair_count": pairs, "task_pairs": positions,
        "reason": reason, "requires_human_review": True,
    }


def _place_tasks(tasks, pairs, rng):
    """Keep the observation's order and reserve a setup exchange before each question."""
    slots = range(2, pairs + 1)
    choices = [positions for positions in combinations(slots, len(tasks))
               if len(positions) < 2 or positions[1] - positions[0] >= 2]
    return {position: task for position, task in zip(rng.choice(choices), tasks)}


def _utterance_problem(role, pair, utterance, dialogue, scheduled):
    active = scheduled.get(pair)
    for position, task in scheduled.items():
        before_answer = pair < position or (pair == position and role == "user")
        if before_answer and contains_grounded_answer(task["kind"], task, utterance):
            return "task_answer_disclosed_early"
    if role == "user":
        allowed = {active["kind"]} if active else set()
        if explicit_question_kinds(utterance) - allowed:
            return "unrecorded_visual_task_question"
    elif active:
        return task_problem(active["kind"], active, dialogue[-1]["text"], utterance,
                            previous_assistant=dialogue[-2]["text"])
    else:
        upcoming = scheduled.get(pair + 1)
        if upcoming and not mentions_target(upcoming["target"], utterance):
            return "target_not_introduced_before_question"
    return None


def next_step(
    uid: str,
    kind: str,
    image_data_url: str,
    replies: list[dict],
    *,
    model: str = "gpt-6-luna",
    pairs: int | None = None,
) -> dict:
    """Replay decoded replies for one image. Only the observation step receives pixels."""
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("uid must be a nonempty string")
    if kind not in ("counting", "ocr"):
        raise ValueError("source kind must be counting or ocr")
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise ValueError("expected an image data URL")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if not isinstance(replies, list):
        raise ValueError("replies must be a list of decoded JSON objects")

    rng = random.Random(hashlib.sha256(uid.encode("utf-8")).digest())
    default_pairs = rng.choice(PAIR_COUNTS)
    if pairs is None:
        pairs = default_pairs
    if type(pairs) is not int or pairs not in PAIR_COUNTS:
        raise ValueError("expected 4 to 8 question-answer pairs")
    opening = rng.choice(_OPENINGS)
    total = len(replies)
    if total > 1 + 2 * pairs:
        raise ValueError("too many replies for this dialogue")
    if not replies:
        return _request(
            "grounding", model, pairs, {}, grounding_instructions(kind),
            "この写真を観察し、指示された JSON オブジェクトを一つ返してください。", image_data_url,
        )

    observation = replies[0]
    problem = grounding_problem(kind, observation)
    if problem:
        return _hold(kind, pairs, {}, problem, 1, total)
    scheduled = _place_tasks(observation["tasks"], pairs, rng)
    positions = {task["kind"]: position for position, task in scheduled.items()}
    dialogue = []
    for index, reply in enumerate(replies[1:]):
        role = "user" if index % 2 == 0 else "assistant"
        utterance = reply.get("text") if isinstance(reply, dict) else None
        if not isinstance(utterance, str) or not utterance.strip() or set(reply) != {"text"}:
            return _hold(kind, pairs, positions, f"invalid {role} utterance", index + 2, total)
        problem = _utterance_problem(role, index // 2 + 1, utterance, dialogue, scheduled)
        if problem:
            return _hold(kind, pairs, positions, problem, index + 2, total)
        dialogue.append({"role": role, "text": utterance.strip()})

    if len(dialogue) == 2 * pairs:
        return {
            "status": "pending_review", "source_kind": kind, "dialogue": dialogue,
            "grounding": observation, "pair_count": pairs, "task_pairs": positions,
            "requires_human_review": True,
        }

    pair = len(dialogue) // 2 + 1
    active = scheduled.get(pair)
    if len(dialogue) % 2 == 0:
        phase = "user"
        instructions = _USER
        cue = _user_cue(pair, active["kind"] if active else None, opening)
        context = {"公開された会話": dialogue}
    else:
        phase = "assistant"
        instructions = _ASSISTANT
        # Each image and exchange gets a reproducible phrasing cue, independent of its evidence.
        delivery = random.Random(f"{uid}/{pair}").choice(_UNCERTAIN_DELIVERY)
        cue, context = _assistant_prompt(dialogue, observation, active, scheduled.get(pair + 1), delivery)
    prompt = f"{cue}\n{json.dumps(context, ensure_ascii=False)}\n今回の発話を JSON で返してください。"
    return _request(phase, model, pairs, positions, instructions, prompt)
