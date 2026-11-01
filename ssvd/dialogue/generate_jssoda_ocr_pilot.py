"""Generate a small, grounded dialogue pilot for long JSSODa pages.

The production ``ocr`` profile is intentionally limited to short scene text.
JSSODa contains long synthetic pages, so this pilot asks the VLM to select a
few short, contiguous spans from the canonical transcription and use them in
natural spoken dialogue.  Every selected span is checked against the source
transcription before the record is accepted.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


DEFAULT_MODEL = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
LOCATION_QUESTION_TERMS = (
    "冒頭",
    "最初",
    "段落",
    "行目",
    "文目",
    "見出し",
    "結び",
    "最後",
    "次の文",
    "前の文",
)
INCOMPLETE_SPAN_ENDINGS = (
    "おらず",
    "ており",
    "であり",
    "となり",
    "により",
    "すべきは",
    "一方で",
)


def compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def build_span_candidates(text: str, limit: int = 12) -> list[dict[str, str]]:
    """Extract short canonical clauses distributed across the document."""
    spans: list[dict[str, str]] = []
    seen: set[str] = set()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs, 1):
        sentences = [
            part.strip()
            for part in re.findall(r"[^。！？]+[。！？]?", paragraph)
            if part.strip()
        ]
        for sentence_index, sentence in enumerate(sentences, 1):
            clauses = [part.strip() for part in re.split(r"[、，,]", sentence)]
            for clause in clauses:
                span = clause.strip(" \t\r\n。！？")
                normalized = compact(span)
                if not 4 <= len(normalized) <= 24 or normalized in seen:
                    continue
                seen.add(normalized)
                spans.append(
                    {
                        "exact_text": span,
                        "location_hint": (
                            f"第{paragraph_index}段落の第{sentence_index}文付近"
                        ),
                    }
                )
    if len(spans) <= limit:
        return spans
    # Keep coverage from the beginning through the end rather than only the
    # first paragraph.
    positions = [round(i * (len(spans) - 1) / (limit - 1)) for i in range(limit)]
    return [spans[position] for position in positions]


def exact_transcription(record: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    annotations = record.get("reference_annotations")
    if not isinstance(annotations, list):
        raise ValueError("reference_annotations is missing")
    for annotation in annotations:
        if (
            isinstance(annotation, Mapping)
            and annotation.get("kind") == "exact_transcription"
            and isinstance(annotation.get("text"), str)
            and annotation["text"].strip()
        ):
            return annotation["text"].strip(), annotation
    raise ValueError("exact transcription is missing")


def load_candidates(metadata: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with metadata.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                text, annotation = exact_transcription(record)
            except ValueError:
                continue
            image_path = Path(str(record.get("path", "")))
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            if not image_path.is_file():
                continue
            candidate = dict(record)
            candidate["_reference_text"] = text
            candidate["_annotation"] = dict(annotation)
            candidate["_image_path"] = image_path
            candidate["_ocr_candidates"] = build_span_candidates(text)
            if len(candidate["_ocr_candidates"]) < 2:
                continue
            candidates.append(candidate)
    return candidates


def select_representative(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose deterministic layout/length-diverse examples."""
    if count <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            str(item["_annotation"].get("orientation", "")),
            int(item["_annotation"].get("num_columns", 0) or 0),
            len(compact(item["_reference_text"])),
            str(item.get("candidate_uid", "")),
        ),
    )
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in ordered:
        key = (
            str(item["_annotation"].get("orientation", "unknown")),
            int(item["_annotation"].get("num_columns", 0) or 0),
        )
        buckets.setdefault(key, []).append(item)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for key in sorted(buckets):
        bucket = buckets[key]
        item = bucket[len(bucket) // 2]
        uid = str(item.get("candidate_uid"))
        if uid not in used:
            selected.append(item)
            used.add(uid)
        if len(selected) == count:
            return selected

    # Fill remaining slots at evenly spaced text-length quantiles.
    by_length = sorted(
        candidates,
        key=lambda item: (
            len(compact(item["_reference_text"])),
            str(item.get("candidate_uid", "")),
        ),
    )
    remaining = count - len(selected)
    for index in range(remaining):
        position = round((index + 1) * (len(by_length) - 1) / (remaining + 1))
        for offset in range(len(by_length)):
            item = by_length[(position + offset) % len(by_length)]
            uid = str(item.get("candidate_uid"))
            if uid not in used:
                selected.append(item)
                used.add(uid)
                break
    return selected[:count]


def build_prompt(entry: Mapping[str, Any]) -> str:
    annotation = entry["_annotation"]
    reference = entry["_reference_text"]
    reference_json = json.dumps(reference, ensure_ascii=False)
    candidate_json = json.dumps(entry["_ocr_candidates"], ensure_ascii=False)
    return f"""あなたは日本語の視覚音声対話データセットを作成する担当者です。
画像は、複数段または複数列からなる日本語の長文ページです。画像を実際に確認し、
下記の確認済み全文転記を正解として、自然な2人会話を1件作成してください。

設定:
- ユーザーは画像を直接見られず、アシスタントだけが画像を見ています。
- dialogue配列の要素数は必ず10です。ユーザー5発話とアシスタント5発話を、
  ユーザーから始めて交互に並べます。10往復や20要素にはしません。
- 各発話は自然な話し言葉の1〜2文、音声でおおむね3〜8秒です。
- 最初はユーザーが画像の内容を知らない聞き方にし、アシスタントが文書画像だと説明します。
- その後、画像中の文字を短く正確に読む質問応答を2〜3組だけ自然に含めます。
- OCR対象は後述の「確認済み短文候補」から2〜3個を選びます。
- exact_textは候補の文字列を一字も変更せず、そのままコピーしてください。
- 「冒頭」「結び」「第2段落」など、画像を見れば対象箇所が一意になる質問にします。
- ユーザーのOCR質問は位置だけを尋ねます。exact_text、その一部、内容の要約、
  「桶の有用性が失われていないところ」のような答えを示す手掛かりを先に言いません。
- OCR以外の質問でも、それまでアシスタントが話していない候補文字や固有表現を
  ユーザー側から先に出してはいけません。
- 良い質問例は「第2段落の最初の文を読んでください」です。
- 悪い質問例は「医療について書かれた文を読んでください」です。
- OCR回答には選んだ exact_text を一字も変更せず含めます。
- 全文の読み上げ、長い引用、箇条書き、書き言葉、正解情報を渡されたという説明は禁止です。
- 画像や全文転記にない内容を断定しません。

レイアウト情報:
- 方向: {annotation.get('orientation', 'unknown')}
- 列数: {annotation.get('num_columns', 'unknown')}

確認済み全文転記:
{reference_json}

確認済み短文候補:
{candidate_json}

次のJSONだけを出力してください。位置情報はコード側で確認するため、対話だけを出力します。

{{"dialogue":[{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}}]}}
"""


def parse_payload(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def derive_targets(
    dialogue: Any, candidates: list[Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Bind canonical spans to the assistant turns that actually contain them."""
    if not isinstance(dialogue, list):
        return []
    candidate_rows = sorted(
        candidates,
        key=lambda item: len(compact(str(item["exact_text"]))),
        reverse=True,
    )
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer_index in range(1, len(dialogue), 2):
        turn = dialogue[answer_index]
        if not isinstance(turn, Mapping) or not isinstance(turn.get("text"), str):
            continue
        question = dialogue[answer_index - 1]
        if (
            not isinstance(question, Mapping)
            or not isinstance(question.get("text"), str)
            or not any(
                term in question["text"] for term in LOCATION_QUESTION_TERMS
            )
        ):
            # The opening summary may naturally contain a canonical phrase,
            # but it is not an OCR response unless the user asked for a
            # concrete text position.
            continue
        answer = compact(turn["text"])
        for candidate in candidate_rows:
            exact_text = str(candidate["exact_text"])
            normalized = compact(exact_text)
            if normalized in seen or normalized not in answer:
                continue
            targets.append(
                {
                    "question_turn_index": answer_index - 1,
                    "answer_turn_index": answer_index,
                    "exact_text": exact_text,
                }
            )
            seen.add(normalized)
            break
        if len(targets) == 3:
            break
    return targets


def validate_payload(
    payload: Mapping[str, Any], reference_text: str, candidates: list[Mapping[str, str]]
) -> tuple[bool, str]:
    dialogue = payload.get("dialogue")
    targets = payload.get("ocr_targets")
    if not isinstance(dialogue, list) or not 8 <= len(dialogue) <= 12:
        return False, "dialogue must contain 8-12 turns"
    for index, turn in enumerate(dialogue):
        expected_speaker = "ユーザー" if index % 2 == 0 else "アシスタント"
        if (
            not isinstance(turn, Mapping)
            or turn.get("speaker") != expected_speaker
            or not isinstance(turn.get("text"), str)
            or not turn["text"].strip()
        ):
            return False, f"invalid dialogue turn {index}"
    if not isinstance(targets, list) or not 2 <= len(targets) <= 3:
        return False, "ocr_targets must contain 2-3 items"

    reference_compact = compact(reference_text)
    candidate_compact = {
        compact(str(candidate["exact_text"])) for candidate in candidates
    }
    # Reject long source-specific phrases introduced by the user before the
    # assistant has said them.  This catches answer leakage even in turns that
    # were not selected as canonical OCR targets.
    prior_assistant = ""
    for turn_index, turn in enumerate(dialogue):
        turn_text = compact(str(turn["text"]))
        if turn_index % 2 == 1:
            prior_assistant += turn_text
            continue
        for candidate_text in candidate_compact:
            match = difflib.SequenceMatcher(
                None, candidate_text, turn_text
            ).find_longest_match()
            if match.size < 6:
                continue
            shared = candidate_text[match.a : match.a + match.size]
            if shared not in prior_assistant:
                return False, f"user turn {turn_index} leaks unseen OCR content"
    seen: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            return False, f"target {index} is not an object"
        question_index = target.get("question_turn_index")
        answer_index = target.get("answer_turn_index")
        exact_text = target.get("exact_text")
        if (
            not isinstance(question_index, int)
            or not isinstance(answer_index, int)
            or answer_index != question_index + 1
            or question_index < 0
            or answer_index >= len(dialogue)
            or question_index % 2 != 0
        ):
            return False, f"target {index} has invalid turn indices"
        if not isinstance(exact_text, str):
            return False, f"target {index} has no exact_text"
        normalized = compact(exact_text)
        if not 2 <= len(normalized) <= 24:
            return False, f"target {index} is outside the 2-24 character limit"
        if normalized not in reference_compact:
            return False, f"target {index} is not in the canonical transcription"
        if normalized not in candidate_compact:
            return False, f"target {index} is not one of the supplied candidates"
        if normalized not in compact(str(dialogue[answer_index]["text"])):
            return False, f"target {index} is not in the declared answer"
        answer_surface = compact(str(dialogue[answer_index]["text"])).strip(
            "「」『』。！？"
        )
        if (
            any(normalized.endswith(ending) for ending in INCOMPLETE_SPAN_ENDINGS)
            and answer_surface.endswith(normalized)
        ):
            return False, f"target {index} answer ends with an incomplete clause"
        question_text = str(dialogue[question_index]["text"])
        if not any(term in question_text for term in LOCATION_QUESTION_TERMS):
            return False, f"target {index} question does not identify a text position"
        question_for_overlap = compact(question_text)
        for term in LOCATION_QUESTION_TERMS:
            question_for_overlap = question_for_overlap.replace(compact(term), "")
        overlap = difflib.SequenceMatcher(
            None, normalized, question_for_overlap
        ).find_longest_match().size
        if overlap >= 4:
            return False, f"target {index} text is leaked in the user question"
        if normalized in seen:
            return False, f"target {index} duplicates another OCR target"
        seen.add(normalized)
    return True, "ok"


def public_record(
    entry: Mapping[str, Any], payload: Mapping[str, Any], model: str
) -> dict[str, Any]:
    annotation = entry["_annotation"]
    return {
        "uid": str(entry["candidate_uid"]),
        "image": str(entry["path"]),
        "dialogue": payload["dialogue"],
        "generator": model,
        "task_mode": "ocr_document_pilot",
        "grounding": {
            "kind": "ocr_document",
            "targets": payload["ocr_targets"],
            "orientation": annotation.get("orientation"),
            "num_columns": annotation.get("num_columns"),
            "canonical_transcription_characters": len(
                compact(entry["_reference_text"])
            ),
        },
        "split_group_key": entry.get("split_group_key"),
        "source": entry.get("source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/images/targeted_v4/image_pool_v1/ocr_images.jsonl"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    candidates = load_candidates(args.meta)
    pool_size = min(len(candidates), args.max_samples + max(3, args.max_samples // 2))
    selected = select_representative(candidates, pool_size)
    print(
        f"[info] loaded={len(candidates)} selected_pool={len(selected)} "
        f"target={args.max_samples}",
        flush=True,
    )
    if len(selected) < args.max_samples:
        raise SystemExit("not enough eligible JSSODa samples")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
    )
    sampling = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=1800,
        seed=42,
    )

    accepted: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    pending = selected
    for attempt in range(1, args.max_attempts + 1):
        if not pending or len(accepted) >= args.max_samples:
            break
        conversations = []
        valid_entries = []
        for entry in pending:
            try:
                image = Image.open(entry["_image_path"]).convert("RGB")
            except OSError as error:
                failures[str(entry["candidate_uid"])] = f"image error: {error}"
                continue
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": image},
                            {"type": "text", "text": build_prompt(entry)},
                        ],
                    }
                ]
            )
            valid_entries.append(entry)
        outputs = llm.chat(conversations, sampling)
        retry: list[dict[str, Any]] = []
        for entry, output in zip(valid_entries, outputs):
            uid = str(entry["candidate_uid"])
            payload = parse_payload(output.outputs[0].text)
            if payload is None:
                failures[uid] = "JSON parse failed"
                retry.append(entry)
                continue
            payload["ocr_targets"] = derive_targets(
                payload.get("dialogue"), entry["_ocr_candidates"]
            )
            valid, reason = validate_payload(
                payload,
                entry["_reference_text"],
                entry["_ocr_candidates"],
            )
            if not valid:
                failures[uid] = reason
                retry.append(entry)
                print(f"[retry {attempt}] {uid}: {reason}", flush=True)
                continue
            accepted[uid] = public_record(entry, payload, args.model)
            failures.pop(uid, None)
            print(f"[accepted] {len(accepted)}/{args.max_samples} {uid}", flush=True)
        pending = retry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in selected:
            uid = str(entry["candidate_uid"])
            if uid in accepted:
                handle.write(json.dumps(accepted[uid], ensure_ascii=False) + "\n")
                written += 1
                if written == args.max_samples:
                    break
    failure_path = args.out.with_suffix(".failures.json")
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Done: {written}/{args.max_samples} written "
        f"({len(accepted)}/{len(selected)} pool rows accepted); output={args.out}",
        flush=True,
    )
    if written != args.max_samples:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
