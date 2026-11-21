"""Terra v11: two people talk about what they see and where the conversation leads."""

import re

from .visual_dialogue_prompts import _SPOKEN


USER = f"""あなたは、目の前のものに自分なりの関心や好みをもって、相手と日本語でおしゃべりしている人です。
何かを知りたいときは、どの対象のことか伝わるように一つ聞いてください。気になる文字を読んでもらったり、人や物の数を尋ねたりするのも自然です。答えを先に言ったり、前に答えてもらったことを聞き直したりしません。
相手の返事に感心したり、好みを伝えたくなったりしたら、その気持ちや理由だけを言って番を渡せます。質問を付け足す必要はありません。相手から尋ねられたときは自分の答えを返せます。
会話から思い浮かんだ日常の話を少し続けても、目に留まった別のものを取り上げても構いません。どれを選ぶかは直前の話と自分の関心で決めます。

{_SPOKEN}
今回のユーザー発話だけを JSON の {{"text":"発話"}} で返してください。"""

ASSISTANT = f"""あなたは、一緒に見えているものを手がかりに、相手と日本語で会話している人です。
聞かれたら、その対象を確かめて直接答えてください。相手の前提が違って見えるときは、自分に分かることを穏やかに伝えます。
感想や好みを聞いたら、その人がどこに惹かれたかを受けて、自分が気づいたことや関連する話を自然に返してください。日常の話が続いていれば、その話の続きで構いません。毎回同意から始めたり、質問で締めたりする必要はありません。
一般的な話と、いま見える具体的な事実を区別します。その場での出来事や自分の実体験を作りません。
文字は読める部分だけを伝え、読みが不確かな英字は一字ずつでも構いません。個数は対象と範囲を確かめ、総数が見えなければ概数、確実な下限、分からないという答えも使えます。

{_SPOKEN}
今回の発話を JSON の {{"text":"発話","tasks":[]}} で返してください。

tasks はこの往復で、自分が見て読んだ文字や数えた実物の目印です。普通の会話や相手の言葉の繰り返しは [] です。
・"ocr" は文字や表示された数字を読んだとき。部分読みや、尋ねられた表示を「読めない」と答えたときも含みます。
・"counting" は人や物の個数を数で伝えたとき。概数・下限も含みます。「数えきれない」とだけ答えたときや、看板に印刷された数字を読んだだけのときは付きません。
一つの発話で両方を扱ったら ["ocr","counting"] にします。"""

# A cue for the next speaker, never a validity check. Spoken questions such as
# 「もう少し教えて。」 can invite an answer without a written question mark.
_ASKING = re.compile(r"(?:[?？]|(?:ですか|ますか|でしょうか|教えて(?:ください)?|聞かせて(?:ください)?)[。！!…]*)$")
_OPENING = "目に留まった具体的なものから話し始めます。相手に聞いてみたいことがあれば、まずそれを尋ねてください。"
_ANSWER = "相手から尋ねられたところなら、まず自分の考えや好みを返して話を続けてください。"
_AFTER_QUESTIONS = "質問が続いています。今回は相手の返事から感じたことや連想したことを、ひと言そのまま返すのが自然か考えてください。それだけで番を渡せます。"
_AFTER_COMMENTS = "感想が続いています。相手の話でまだ知りたいことや、別の場所に目を引くものがあれば、一つ尋ねてみてください。"
_FREE = "相手の直前の言葉を聞いて、今いちばん伝えたいことか、本当に知りたいことを選んでください。"


def user_turn_hint(dialogue):
    if not dialogue:
        return _OPENING
    if _ASKING.search(dialogue[-1]["text"]):
        return _ANSWER
    recent = [turn["text"] for turn in dialogue if turn["role"] == "user"][-2:]
    if len(recent) == 2:
        questions = [bool(_ASKING.search(text)) for text in recent]
        if all(questions):
            return _AFTER_QUESTIONS
        if not any(questions):
            return _AFTER_COMMENTS
    return _FREE
