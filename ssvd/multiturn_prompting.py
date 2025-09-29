"""対話生成のメインパイプライン（日本語版対応）"""

import json
from copy import copy
from random import random
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import rich
import torch
from multiturn_instruct import MTCInstruct
from transformers import Pipeline
from utils import (
    compile_pattern,
    get_replace_pattern,
    get_strings_for_logging,
    maybe_shorten_caption,
)


def list_to_prompt(
    convo_list: List[str],
    img_caption: str,
    pipe: Pipeline,
    setting: str,
) -> List[Dict]:
    """
    会話リストをチャット形式のプロンプトに変換する

    :param convo_list: 会話の履歴（文字列リスト）
    :param img_caption: 対応する画像キャプション
    :param pipe: Transformers パイプライン
    :param setting: instruct の種類
    """
    try:
        setting_obj = MTCInstruct(setting)
        system_template, speaker1_template, speaker2_template, start_conv = (
            setting_obj.get_method(len(convo_list))()
        )
    except ValueError as e:
        raise NotImplementedError("未知の MTCInstruct 設定", setting) from e

    convo_list = copy(convo_list)
    if len(convo_list) % 2 == 0:
        convo_list = [
            system_template.format(
                ROLE_SPECIFIC_TEXT=speaker1_template.format(caption=img_caption),
                caption=img_caption,
            ),
            start_conv,
        ] + convo_list
    else:
        convo_list = [
            system_template.format(
                ROLE_SPECIFIC_TEXT=speaker2_template.format(caption=img_caption),
                caption=img_caption,
            )
        ] + convo_list

    def speaker_iter() -> Iterator:
        yield "system"
        while True:
            yield "user"
            yield "assistant"

    def prefix_iter() -> Iterator:
        yield ""
        while True:
            yield "質問: "
            yield "回答: "

    chat = [
        {"role": speaker, "content": prefix + c}
        for c, speaker, prefix in zip(convo_list, speaker_iter(), prefix_iter())
    ]
    tok = pipe.tokenizer
    return tok.apply_chat_template(
        chat,
        tokenize=False,
        continue_final_message=False,
    )


def postprocess_mtc(
    s: str,
    drop_probs: Optional[Dict[str, Dict]] = None,
    default_prob: float = 0.8,
    setting: Optional[str] = None,
) -> str:
    """生成テキストの後処理
    - 「説明文によると」「記述では」などを削除
    - LLM的な役割表現を削除
    - 不要な定型句を省略
    """
    pattern = get_replace_pattern()
    s = pattern.sub("", s)
    if drop_probs is None:
        drop_probs = {
            r"へえ[,、]": dict(p=default_prob, replace_by=""),
            r"そうですね[。]?": dict(p=default_prob, replace_by="。"),
            r"まあ[,、]": dict(p=default_prob, replace_by=""),
            r"とても印象的": dict(p=0.5, replace_by="印象的"),
            r"正直": dict(p=0.3, replace_by=""),
            r"よく分かりません": dict(p=default_prob, replace_by=""),
            # 役割表現の削除
            r"先生[:：]": dict(p=1.0, replace_by=""),
            r"アシスタント": dict(p=1.0, replace_by=""),
            r"あなた[:：]": dict(p=1.0, replace_by=""),
            r"話者1": dict(p=1.0, replace_by=""),
            r"話者2": dict(p=1.0, replace_by=""),
            # 「説明文に〜」系の言い回しを修正
            r"説明文": dict(p=1.0, replace_by="画像"),
            r"記述されていない": dict(p=1.0, replace_by="写っていない"),
            r"記述": dict(p=1.0, replace_by="画像"),
            r"示されていない": dict(p=1.0, replace_by="見えない"),
        }
    if setting is not None and setting not in {"cap", "cap2", "rnd"}:
        drop_probs[r"説明"] = dict(p=1.0, replace_by="画像")
    for drop_s, drop_kwargs in drop_probs.items():
        pattern = compile_pattern(drop_s)
        p = drop_kwargs["p"]
        r = drop_kwargs["replace_by"]
        if random() < p:
            s = pattern.sub(r, s).strip()
            try:
                if s and drop_s[0].isupper():
                    s = s[0].upper() + s[1:]
            except IndexError:
                pass
    s = s.strip()
    if not s.startswith('"'):
        s = '"' + s
    if not s.endswith('"'):
        s += '"'
    return s


class ConvoIter:
    """会話の逐次構築を行うクラス"""

    def __init__(
        self,
        convo_length: int = 4,
        batch_size: int = 64,
        pipe: Optional[Pipeline] = None,
        setting: str = "mtc",
    ) -> None:
        self.convos: Dict[str, List[str]] = {}
        self.convo_length = convo_length
        self.batch_size = batch_size
        self.pipe = pipe
        self.setting = setting
        self.last_updated: Optional[List[str]] = None

    def add_to_convos(self, uid: str, answer: str) -> None:
        """会話に新しい発話を追加"""
        if uid not in self.convos:
            self.convos[uid] = []
        self.convos[uid].append(answer)
        self.last_updated = self.convos[uid]

    def make_iter(self, captions: Sequence[str], img_ids: Sequence[str]) -> Iterator:
        """会話のイテレータを生成"""
        convo_ids_within_loop = []
        captions_within_loop = []
        for count, (uid, img_caption) in enumerate(zip(img_ids, captions)):
            img_caption = maybe_shorten_caption(img_caption, max_cap_len=1000)
            convo_ids_within_loop.append(uid)
            captions_within_loop.append(img_caption)
            return_value = list_to_prompt(
                convo_list=[],
                img_caption=img_caption,
                pipe=self.pipe,
                setting=self.setting,
            )
            yield return_value
            if ((count + 1) % self.batch_size) == 0:
                for _ in range(self.convo_length - 1):
                    for uid, img_caption in zip(
                        convo_ids_within_loop, captions_within_loop
                    ):
                        return_value = list_to_prompt(
                            self.convos[uid],
                            img_caption=img_caption,
                            pipe=self.pipe,
                            setting=self.setting,
                        )
                        yield return_value

                convo_ids_within_loop = []
                captions_within_loop = []


@torch.no_grad()
def run_multiturn_pipeline(
    pipe: Pipeline,
    captions: Sequence[str],
    img_ids: Sequence[str],
    out_file: str,
    batch_size: int = 64,
    convo_length: int = 6,
    setting: str = "mtc",
    temperature: float = 0.0,
    max_new_tokens: int = 80,
) -> None:
    """マルチターン対話を生成するメイン関数"""
    assert len(captions) == len(img_ids)
    count = 0

    def uid_iter(img_ids: Sequence[str]) -> Iterator:
        """UID を batch_size ごとに繰り返す"""
        nonlocal convo_length
        ids = np.array(
            list(img_ids) + [None] * (batch_size - len(img_ids) % batch_size)
        ).reshape(-1, batch_size)

        for batch_ids in ids:
            for _ in range(convo_length):
                yield from batch_ids

    convo_iter = ConvoIter(
        convo_length=convo_length, batch_size=batch_size, pipe=pipe, setting=setting
    )
    data_iter = convo_iter.make_iter(captions, img_ids)
    total = len(captions) * convo_length
    try:
        for uid, out in zip(
            uid_iter(img_ids),
            pipe(
                data_iter,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                add_special_tokens=False,
                batch_size=batch_size,
                do_sample=temperature > 0,
                temperature=temperature,
            ),
        ):
            answer = postprocess_mtc(out[0]["generated_text"], setting=setting)
            convo_iter.add_to_convos(uid=uid, answer=answer)
            count += 1
            if (count % (batch_size * convo_length)) == 0:
                try:
                    assert convo_iter.last_updated is not None
                    q, a = get_strings_for_logging(
                        [
                            dict(
                                zip(
                                    ["質問", "回答"], convo_iter.last_updated[-2:]
                                )
                            )
                        ]
                    )
                    print(
                        f"{count+1:>8d}/{total:8d} ({100*(count+1)/total:6.2f}%)\t質問: {q} \t回答: {a}",
                        flush=True,
                    )

                except Exception as e:  # pylint: disable=W0718
                    rich.print(
                        "[red]警告:[/red] ログ出力時にエラーが発生しました。",
                        flush=True,
                    )
                    print(f"結果: {convo_iter.convos[uid]}", flush=True)
                    print(e, flush=True)

    except Exception as e:  # pylint: disable = W0718
        rich.print(
            "[red]警告:[/red] パイプライン実行中にエラーが発生しました。"
            " 既存の結果を保存して終了します。",
            flush=True,
        )
        print(e, flush=True)

    print(flush=True)
    with open(out_file, "w") as f:
        for uid, res in convo_iter.convos.items():
            json.dump({"uid": uid, "res": res}, f, ensure_ascii=False)
            f.write("\n")
