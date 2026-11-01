# 対話生成プロンプトの共通部品 (Claude CLI版 / ローカルVLM版で共用)
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from task_profiles import NORMAL_STRATEGIES

PROMPT_PATHS = {
    "normal": Path(__file__).parent / "prompt_dialogue.md",
    "counting": Path(__file__).parent / "prompt_counting.md",
    "ocr": Path(__file__).parent / "prompt_ocr.md",
}
ADAPTIVE_NORMAL_APPENDIX_PATH = (
    Path(__file__).parent / "prompt_normal_vlm_adaptive_appendix.md"
)
ADAPTIVE_PROMPT_VERSION = "count_balance_v2"

# 会話の切り出しスタイル (uidハッシュで決定論的に選択 → 同一画像は常に同じスタイル)
# 全スタイル共通: 最初の発話は雑談ではなく「画像についての質問」。切り口だけを変える
OPENING_STYLES = [
    "何が写っているか・何が見えるかをストレートに尋ねる",
    "どこの場所なのか(どんなところか)を尋ねる",
    "何をしている場面なのか、なにが起きているのかを尋ねる",
    "人や生き物が写っているかどうかを尋ねる",
    "色合いや見た目の印象がどんな感じかを尋ねる",
    "どんな雰囲気の場所なのか(賑やか・静か・落ち着く等)を尋ねる",
    "屋内なのか屋外なのか、どんな環境なのかを尋ねる",
    "一番目立っているもの・真っ先に目に入るものは何かを尋ねる",
    "目の前にあるもの・見ているものの正体が何なのかを尋ねる",
    "画像の様子から天気や時間帯がどんな感じかを尋ねる",
    "相手が見ている写真がどんな写真なのかを尋ねる",
    "興味や驚きを短くひとこと示してから、すぐに何が写っているかを尋ねる",
]


def _stable_style_index(uid: str) -> int:
    """Keep the legacy hex mapping and support descriptive targeted-data UIDs."""
    try:
        value = int(uid[:8], 16)
    except ValueError:
        value = int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big")
    return value % len(OPENING_STYLES)


def build_instruction(
    uid: str,
    task_mode: str = "normal",
    grounding: Mapping[str, Any] | None = None,
    normal_strategy: str = "standard",
    optional_groundings: Mapping[str, Any] | None = None,
) -> str:
    """Build the shared prompt and inject only canonical task grounding."""
    if task_mode not in PROMPT_PATHS:
        raise ValueError(f"unsupported prompt task mode: {task_mode}")
    if normal_strategy not in NORMAL_STRATEGIES:
        raise ValueError(f"unsupported normal strategy: {normal_strategy}")
    if task_mode != "normal" and normal_strategy != "standard":
        raise ValueError("vlm_adaptive normal strategy requires normal task mode")
    instr = PROMPT_PATHS[task_mode].read_text(encoding="utf-8")
    if task_mode != "normal":
        if not isinstance(grounding, Mapping):
            raise ValueError(f"{task_mode} mode requires grounding")
        # Give the generator only the facts it needs to phrase the dialogue.
        # Provenance, selection policy and split metadata remain in the output
        # record, but must not distract or leak into the spoken conversation.
        if task_mode == "counting":
            required = ("question_en", "answer_count")
        else:
            required = ("text",)
        missing = [key for key in required if key not in grounding]
        if missing:
            raise ValueError(f"{task_mode} grounding lacks: {', '.join(missing)}")
        prompt_grounding = {key: grounding[key] for key in required}
        payload = json.dumps(prompt_grounding, ensure_ascii=False, sort_keys=True)
        return (
            f"{instr}\n\n"
            "## 今回の確認済み正解情報 (最重要)\n"
            f"{payload}\n\n"
            "この情報を正解として先に受け取り、答えを推測し直さず、"
            "自然な日本語対話にしてください。"
        )

    style = OPENING_STYLES[_stable_style_index(uid)]
    normal_instruction = (
        f"{instr}\n\n"
        f"## この会話の切り出し方 (重要)\n"
        f"最初のユーザー発話は、雑談や近況の前置きなしで、画像についての質問から始めること。\n"
        f"質問の切り口: {style}。\n"
        f"ただしユーザーは画像が見えていない設定を厳守すること。写っているもの"
        f"(場所の種類・物の名前など)をユーザーが先に言い当ててはいけない。"
        f"悪い例: 「美術館っぽい雰囲気だけど」「なんかポスターっぽいけど」のように"
        f"見えていないはずの内容を推測で口にするのは禁止。"
        f"切り口に沿って、内容をまだ知らない人としての素朴な聞き方にする。\n"
        f"「今どんな景色見てるの?」「今どんな写真見てるの?」のような定型文は禁止。"
        f"スタイル説明の文言をそのまま流用せず、自然な言い回しにすること。"
    )
    if normal_strategy == "standard":
        return normal_instruction

    canonical_candidates: dict[str, dict[str, Any]] = {}
    if isinstance(optional_groundings, Mapping):
        count_candidate = optional_groundings.get("counting")
        if isinstance(count_candidate, Mapping):
            required = ("question_en", "answer_count")
            if all(key in count_candidate for key in required):
                canonical_candidates["counting"] = {
                    key: count_candidate[key] for key in required
                }
        ocr_candidate = optional_groundings.get("ocr")
        if isinstance(ocr_candidate, Mapping) and "text" in ocr_candidate:
            canonical_candidates["ocr"] = {"text": ocr_candidate["text"]}
    candidates_json = json.dumps(
        canonical_candidates, ensure_ascii=False, sort_keys=True
    )
    appendix = ADAPTIVE_NORMAL_APPENDIX_PATH.read_text(encoding="utf-8")
    if not canonical_candidates:
        grounding_rule = (
            "今回は確認済み正解候補がありません。画像から数や文字を判断する場合、"
            "grounding_source は必ず vlm_visual にしてください。canonical は使えません。"
            "判断に確信がなければ kind と grounding_source を none にします。"
        )
    else:
        allowed = "、".join(sorted(canonical_candidates))
        grounding_rule = (
            f"canonical を使える種類は {allowed} だけです。下の候補にない対象を"
            "画像から判断する場合は vlm_visual にしてください。"
        )
    canonical_examples = []
    for kind, candidate in canonical_candidates.items():
        metadata = {
            "kind": kind, "grounding_source": "canonical",
            "question_turn_index": 4, "answer_turn_index": 5,
        }
        metadata.update(
            {"answer_count": candidate["answer_count"]} if kind == "counting"
            else {"recognized_text": candidate["text"]}
        )
        canonical_examples.append(json.dumps(metadata, ensure_ascii=False))
    examples_text = (
        "\n候補を使う場合の auxiliary_task:\n" + "\n".join(canonical_examples)
        if canonical_examples else ""
    )
    return (
        f"{normal_instruction}\n\n{appendix}\n\n"
        "## 利用可能な確認済み正解候補\n"
        f"{candidates_json}\n\n{grounding_rule}{examples_text}\n\n"
        "候補は会話に必ず使う指示ではありません。画像を見たうえで、通常対話の流れに"
        "自然に合い、かつ条件を明瞭に確認できる場合だけ利用してください。"
    )


def extract_auxiliary_task(text: str) -> Mapping[str, Any] | None:
    """Extract adaptive metadata without changing legacy dialogue parsers."""
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        payload = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None
    auxiliary_task = payload.get("auxiliary_task")
    return auxiliary_task if isinstance(auxiliary_task, Mapping) else None


def collect_done_uids(registry_dir: Path, pattern: str = "*.jsonl") -> set:
    """レジストリディレクトリ内のJSONL群から生成済みUIDを収集。
    デフォルトは全ファイル対象 = どのモデルで生成済みでも再生成しない。
    pattern='claude_*.jsonl' 等に絞ると、他モデル生成分を再生成(置き換え)できる。"""
    done = set()
    for jf in registry_dir.glob(pattern):
        with jf.open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try:
                        done.add(str(json.loads(l)["uid"]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
    return done


def collect_done_keys(registry_dir: Path, pattern: str = "*.jsonl") -> set[tuple[str, str]]:
    """Collect ``(uid, task_mode)`` while treating legacy rows as normal mode."""
    done: set[tuple[str, str]] = set()
    for jsonl_path in registry_dir.glob(pattern):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    done.add((str(record["uid"]), str(record.get("task_mode", "normal"))))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return done
