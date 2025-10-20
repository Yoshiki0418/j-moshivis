# python filter_jsonl.py --input input.jsonl --output filtered.jsonl
import json
from langdetect import detect
import re
from pathlib import Path
from typing import List

def is_japanese(text: str) -> bool:
    """
    日本語かどうかを判定する関数（中国語との区別強化版）。
    """
    try:
        lang = detect(text)
        if lang == "ja":
            return True
        if lang in ["zh-cn", "zh-tw", "zh"]:
            return False
    except:
        pass

    # ひらがな or カタカナを含んでいれば日本語とみなす
    if re.search(r"[ぁ-んァ-ン]", text):
        return True

    # 漢字のみで構成される場合は中国語の可能性が高いので False
    if re.match(r"^[一-龥0-9\s]+$", text):
        return False

    return False


def filter_dialogue(dialogue: List[dict]) -> List[dict]:
    """
    ユーザー＋アシスタントのセット単位でフィルタリング。
    日本語でない発話を含むペアは除外。
    """
    filtered = []
    # 2発話ずつ（ユーザー→アシスタント）で処理
    for i in range(0, len(dialogue), 2):
        if i + 1 >= len(dialogue):
            break
        user_turn = dialogue[i]
        assistant_turn = dialogue[i + 1]

        if is_japanese(user_turn["text"]) and is_japanese(assistant_turn["text"]):
            filtered.extend([user_turn, assistant_turn])
    return filtered


def process_jsonl(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            uid = sample["uid"]
            dialogue = sample["dialogue"]

            filtered_dialogue = filter_dialogue(dialogue)
            if filtered_dialogue:
                fout.write(json.dumps({
                    "uid": uid,
                    "dialogue": filtered_dialogue
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter JSONL dialogues to only Japanese.")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    process_jsonl(args.input, args.output)
