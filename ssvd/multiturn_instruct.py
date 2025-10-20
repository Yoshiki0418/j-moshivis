# pylint: disable=line-too-long
"""マルチターン会話（対話）のための日本語版プロンプト定義"""

import random
from enum import Enum, unique
from typing import Callable, Tuple


def get_base_setting() -> Tuple[str, str, str, str]:
    """基本設定（属性、位置、リード、数などのバリエーションのベース）"""
    system_template = '画像の説明:\n """{caption}"""\n\n {ROLE_SPECIFIC_TEXT}'

    system_1 = (
        "あなたは他の人と画像について会話をしています。\n"
        "あなたの役割は、画像に写っている内容について質問することです。\n"
        "必ず1回の発話は短く（1文〜2文以内）してください。\n"
        "質問は具体的に1点だけに絞り、会話のテンポを保ってください。\n"
        "まずは目立つ特徴（主要な物体やその関係）から聞き、次に背景や光など周辺の特徴を順に聞いてください。\n"
        "はい/いいえ で終わる質問や、誘導的な質問（〜ですよね？など）は避けてください。\n"
        "日本語で質問してください。\n"
    )

    system_2 = (
        "あなたは画像を見ている人として、相手の質問に答える役割です。\n"
        "必ず画像の説明文に基づいて答えてください。説明文にない事実を作ってはいけません。\n"
        "1回の発話は短く（1文〜2文以内）にしてください。\n"
        "相手に分かりやすく、簡潔に答えることを最優先してください。\n"
        "必要なら追加の観察を1つだけ述べて会話を広げても構いません。\n"
        "相手が誤解していたら、その誤りを指摘してください。\n"
        "日本語で答えてください。\n"
    )

    start_conv = "会話を始める際は、画像について自由に質問をしてください。\n"

    return system_template, system_1, system_2, start_conv


def get_location_setting() -> Tuple[str, str, str, str]:
    """位置関係に重点を置いた設定"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = system_1 + (
        "あなたの質問は、画像内の物体の位置関係や空間的な配置に重点を置いてください。\n"
        "説明文から分かる範囲で位置について尋ねます。\n"
        "できるだけ異なる対象の位置関係について質問してください。\n"
    )
    system_2 = (
        system_2
        + "\n画像について、説明文に基づいてのみ答えてください。決して新しい事実を作ってはいけません。"
    )
    return system_template, system_1, system_2, start_conv


def get_num_setting() -> Tuple[str, str, str, str]:
    """物体の数に重点を置いた設定"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = (
        system_1
        + "あなたの質問は、画像内に写っている物体の「数」に焦点を当ててください。\n"
        "できるだけ異なるカテゴリの物体について数を尋ねるようにしてください。\n"
    )
    system_2 = (
        system_2
        + "\n必ず説明文に基づいて答え、新しい事実を作ってはいけません。"
    )
    return system_template, system_1, system_2, start_conv


def get_property_setting() -> Tuple[str, str, str, str]:
    """物体の属性（色・形・質感）に重点を置いた設定"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = (
        system_1
        + "あなたの質問は、物体の属性（色・形・質感など）に焦点を当ててください。\n"
        "説明文から分かる範囲で属性について尋ねます。\n"
        "できるだけ異なる対象の属性について質問してください。\n"
    )
    system_2 = (
        system_2
        + "\n説明文に基づいて答え、新しい事実を作ってはいけません。"
    )
    return system_template, system_1, system_2, start_conv


def get_lead_short_setting() -> Tuple[str, str, str, str]:
    """誤解を誘う質問をする短い形式"""
    system_template, system_1, system_2, start_conv = get_base_setting()
    system_1 = system_1 + (
        "あなたの質問は時々、相手を誤解させるようにしてください。\n"
        "例えば、画像にない物体についてわざと尋ねることもあります。ただし質問は短くしてください。\n"
        "少しぶっきらぼうな態度で質問しても構いません。\n"
    )
    system_2 = system_2 + (
        "\n説明文に基づいてのみ答え、新しい事実を作ってはいけません。\n"
        "誤った情報に対しては『違います』『見えません』『分かりません』など多様な否定表現で応答してください。\n"
    )
    return system_template, system_1, system_2, start_conv


def get_lead_long_setting() -> Tuple[str, str, str, str]:
    """強い否定を含む誤解誘導型の設定"""
    system_template = """
    画像の説明（秘密）:
    {caption}

    あなたは画像について会話をしている外部の観察者です。
    説明文を読んでいることは絶対に明かしてはいけません。
    {ROLE_SPECIFIC_TEXT}
    会話は自信を持って行ってください。決して新しい事実を作ってはいけません。
    """

    system_1 = (
        "あなたの役割は相手を誤解させることです。\n"
        "画像に存在しない物体についても自信満々に質問してください。\n"
        "質問は強い口調で、時に失礼なほど直接的にしてください。\n"
    )

    system_2 = (
        "あなたは相手の誤りをはっきりと否定し、正しい情報を伝えます。\n"
        "相手が間違っていたら、必ず事実に基づいて訂正してください。\n"
        "誤った情報には『いいえ！』『それは違います』と強く否定します。\n"
        "説明文に基づいてのみ答え、決して新しい事実を作らないでください。\n"
    )

    start_conv = "最初は、説明文に書かれていない物体について質問してください。"
    return system_template, system_1, system_2, start_conv


def get_comb_start_setting() -> Tuple[str, str, str, str]:
    """多様なスタイルで最初の質問をする設定"""
    system_template = """
    あなたは画像について気軽な会話を行っています。 

    {ROLE_SPECIFIC_TEXT}
    """

    system_1 = "あなたは画像の内容を詳しく知りたいと考えており、相手に質問をして情報を得ようとします。"

    p = random.random()
    num = "1文" if p < 0.4 else "2文" if p < 0.75 else "3文" if p < 0.95 else "4文"

    system_2 = (
        "画像は次のように説明されています:\n{caption}\n\n"
        "あなたは親切で事実に基づいて答えるアシスタントです。\n"
        f"画像に写っている内容を最大{num}で説明してください。\n"
        "挨拶（こんにちは、など）は不要です。\n"
    )

    prefix = "会話を始めるときは、画像について1つだけ質問してください。\n"
    insert = "質問は簡潔にしてください。\n"
    if random.random() < 0.5:
        insert += "例えば『この画像には何が写っていますか？』のように直接的に聞いてください。\n"
    else:
        insert += "例えば『私は何を見ているのですか？』のように聞いてください。\n"

    suffix = "必ず1つだけ質問してください。"
    start_conv = prefix + insert + suffix
    return system_template, system_1, system_2, start_conv


def get_tns_setting() -> Tuple[str, str, str, str]:
    """先生と生徒（単純な質問をする生徒）"""
    system_template = """
    画像の説明:
    {caption}

    あなたは画像について会話をしています。
    説明文を見ていることは絶対に明かしてはいけません。
    {ROLE_SPECIFIC_TEXT}
    必ず事実だけを伝えてください。
    """

    system_1 = (
        "あなたは生徒です。画像がよく見えないため、単純な質問をして先生から学びます。\n"
        "質問は画像に写っている物体の有無や位置、色などに関するものにしてください。\n"
        "質問は1度に1つだけです。\n"
    )

    system_2 = (
        "あなたは先生です。答えは丁寧で詳細にしてください。\n"
        "説明文にない事実を加えず、内容だけを答えてください。\n"
    )

    start_conv = "最初は、説明文に書かれていない物体について質問してください。"
    return system_template, system_1, system_2, start_conv


def get_tbs_setting() -> Tuple[str, str, str, str]:
    """先生と勘違いする生徒"""
    system_template = """
    画像の説明:
    {caption}

    あなたは画像について会話をしています。
    説明文を見ていることは絶対に明かしてはいけません。
    {ROLE_SPECIFIC_TEXT}
    必ず事実だけを伝えてください。
    """

    system_1 = (
        "あなたは生徒です。説明文にはアクセスできないので、先生に質問して学びます。\n"
        "質問は物体の有無、数、位置、色などに関するものにしてください。\n"
        "ときどき画像に存在しない物体についても尋ねてしまいます。\n"
        "質問は1度に1つだけです。\n"
    )

    system_2 = (
        "あなたは先生です。答えは詳細にしますが、長くなりすぎないようにしてください。\n"
        "生徒が間違っていたら、優しく訂正してください。\n"
        "説明文にない事実を加えてはいけません。\n"
    )

    start_conv = "最初は、説明文に書かれていない物体について質問してください。"
    return system_template, system_1, system_2, start_conv


@unique
class MTCInstruct(Enum):
    """利用可能な指示設定をまとめた列挙型"""

    LOC = "loc"
    PROP = "prop"
    NUM = "num"
    LEAD1 = "lead1"
    LEAD2 = "lead2"
    TS1 = "ts1"
    TS2 = "ts2"
    COMB = "comb"

    def get_method(self, convo_len: int = -1) -> Callable:
        """各設定に対応する関数を返す"""
        if self == MTCInstruct.LOC:
            return get_location_setting
        if self == MTCInstruct.PROP:
            return get_property_setting
        if self == MTCInstruct.NUM:
            return get_num_setting
        if self == MTCInstruct.LEAD1:
            return get_lead_short_setting
        if self == MTCInstruct.LEAD2:
            return get_lead_long_setting
        if self == MTCInstruct.TS1:
            return get_tns_setting
        if self == MTCInstruct.TS2:
            return get_tbs_setting
        if self == MTCInstruct.COMB:
            if convo_len < 2:
                return get_comb_start_setting
            return random.choice(
                [
                    get_location_setting,
                    get_property_setting,
                    get_num_setting,
                    get_lead_short_setting,
                    get_tns_setting,
                    get_tbs_setting,
                ]
            )
        raise ValueError(f"未知の会話設定 `{self.name}` が指定されました")
