"""Generate the full grounded JSSODa OCR dialogue dataset with resume support.

Each accepted dialogue has one opening exchange followed by four OCR exchanges.
Every OCR answer must be an exact full sentence from the canonical JSSODa
transcription, and every OCR exchange is represented in ``grounding.targets``.
The output is appended after every batch so an interrupted worker can resume.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from ssvd.dialogue import generate_jssoda_ocr_pilot as pilot


READ_REQUEST_TERMS = ("読んで", "読み上げ", "何と", "教えて", "文は", "どこ")


def strict_span_candidates(text: str, limit: int = 16) -> list[dict[str, str]]:
    """Return 8-24 character clauses spread over the whole document."""
    spans = [
        row
        for row in pilot.build_span_candidates(text, limit=1000)
        if 8 <= len(pilot.compact(row["exact_text"])) <= 24
    ]
    if len(spans) <= limit:
        return spans
    positions = [round(i * (len(spans) - 1) / (limit - 1)) for i in range(limit)]
    return [spans[position] for position in positions]


def load_entries(metadata: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    with metadata.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record.get("candidate_uid") or f"line:{line_number}")
            try:
                reference, annotation = pilot.exact_transcription(record)
            except ValueError as error:
                rejected[uid] = str(error)
                continue
            image_path = Path(str(record.get("path", "")))
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            if not image_path.is_file():
                rejected[uid] = f"image is missing: {image_path}"
                continue
            candidates = strict_span_candidates(reference)
            if len(candidates) < 4:
                rejected[uid] = f"only {len(candidates)} strict OCR candidates"
                continue
            entry = dict(record)
            entry["_reference_text"] = reference
            entry["_annotation"] = dict(annotation)
            entry["_image_path"] = image_path
            entry["_ocr_candidates"] = candidates
            entries.append(entry)
    entries.sort(key=lambda row: str(row["candidate_uid"]))
    return entries, rejected


def build_prompt(entry: Mapping[str, Any]) -> str:
    annotation = entry["_annotation"]
    reference = json.dumps(entry["_reference_text"], ensure_ascii=False)
    candidates = json.dumps(entry["_ocr_candidates"], ensure_ascii=False)
    return f"""あなたは日本語の視覚音声対話データセットを作成する担当者です。
画像と確認済み全文転記を使い、自然な2人会話を1件作成してください。

必須条件:
- dialogueは必ず10要素です。ユーザーとアシスタントを交互に5発話ずつ並べます。
- 最初の2発話は、ユーザーが画像の概要を尋ね、アシスタントが日本語の文書画像だと短く説明します。
- 残る4組はすべてOCR質問応答です。
- OCR質問は「第2段落の最初の文を読んでください」のように、文字内容を漏らさず位置だけを指定します。
- 「読みます」「読みますか」は使わず、「読んでください」など自然な依頼にします。
- 各OCR回答には、確認済み短文候補から異なる1個を一字も変更せず含めます。
- 候補だけを単独で返さず、確認済み全文転記から候補を含む完全な一文を正確に読みます。
- 全文転記にない文字、要約、補足、推測をOCR回答へ混ぜません。
- ユーザー側から、まだアシスタントが話していない固有表現や正解文字列を先に出しません。
- 各発話は音声で自然に聞こえる1〜2文にします。
- 箇条書き、メタ説明、正解情報を与えられたという説明は禁止です。

レイアウト情報:
- 方向: {annotation.get('orientation', 'unknown')}
- 列数: {annotation.get('num_columns', 'unknown')}

確認済み全文転記:
{reference}

確認済み短文候補:
{candidates}

次の形式のJSONだけを出力してください。
{{"dialogue":[{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}},{{"speaker":"ユーザー","text":"..."}},{{"speaker":"アシスタント","text":"..."}}]}}
"""


def derive_all_targets(
    dialogue: Any, candidates: list[Mapping[str, str]]
) -> list[dict[str, Any]]:
    if not isinstance(dialogue, list):
        return []
    ordered = sorted(
        candidates,
        key=lambda row: len(pilot.compact(str(row["exact_text"]))),
        reverse=True,
    )
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer_index in range(3, len(dialogue), 2):
        answer_turn = dialogue[answer_index]
        if not isinstance(answer_turn, Mapping):
            continue
        answer = pilot.compact(str(answer_turn.get("text", "")))
        for candidate in ordered:
            exact = str(candidate["exact_text"])
            normalized = pilot.compact(exact)
            if normalized in seen or normalized not in answer:
                continue
            targets.append(
                {
                    "question_turn_index": answer_index - 1,
                    "answer_turn_index": answer_index,
                    "exact_text": exact,
                }
            )
            seen.add(normalized)
            break
    return targets


def validate(
    dialogue: Any,
    targets: list[Mapping[str, Any]],
    reference: str,
    candidates: list[Mapping[str, str]],
) -> tuple[bool, str]:
    if not isinstance(dialogue, list) or len(dialogue) != 10:
        return False, "dialogue must contain exactly 10 turns"
    for index, turn in enumerate(dialogue):
        expected = "ユーザー" if index % 2 == 0 else "アシスタント"
        if (
            not isinstance(turn, Mapping)
            or turn.get("speaker") != expected
            or not isinstance(turn.get("text"), str)
            or not turn["text"].strip()
        ):
            return False, f"invalid dialogue turn {index}"
    if len(targets) != 4:
        return False, f"expected 4 grounded OCR targets, got {len(targets)}"
    if {int(row["question_turn_index"]) for row in targets} != {2, 4, 6, 8}:
        return False, "not every OCR question is grounded"

    reference_compact = pilot.compact(reference)
    allowed = {pilot.compact(str(row["exact_text"])) for row in candidates}
    prior_assistant = ""
    for turn_index, turn in enumerate(dialogue):
        surface = pilot.compact(str(turn["text"]))
        if turn_index % 2 == 1:
            prior_assistant += surface
            continue
        for candidate in allowed:
            match = difflib.SequenceMatcher(None, candidate, surface).find_longest_match()
            if match.size >= 6 and candidate[match.a : match.a + match.size] not in prior_assistant:
                return False, f"user turn {turn_index} leaks unseen OCR content"

    seen: set[str] = set()
    for target in targets:
        question_index = int(target["question_turn_index"])
        answer_index = int(target["answer_turn_index"])
        exact = pilot.compact(str(target["exact_text"]))
        question = str(dialogue[question_index]["text"])
        answer = pilot.compact(str(dialogue[answer_index]["text"]))
        answer_without_wrappers = answer.strip("「」『』。！？")
        if answer_index != question_index + 1:
            return False, "target indices are not adjacent"
        if exact in seen:
            return False, "OCR target is duplicated"
        if exact not in allowed or exact not in reference_compact or exact not in answer:
            return False, "OCR target is not grounded in the canonical transcription"
        if not 8 <= len(exact) <= 24:
            return False, "OCR target is outside the 8-24 character limit"
        if len(answer_without_wrappers) < len(exact) + 4:
            return False, "OCR answer is not a complete sentence around the target"
        if answer_without_wrappers not in reference_compact:
            return False, "full OCR answer is not in the canonical transcription"
        if "読みます" in question:
            return False, "OCR question uses an unnatural reading declaration"
        if not any(term in question for term in pilot.LOCATION_QUESTION_TERMS):
            return False, "OCR question does not identify a text position"
        if not any(term in question for term in READ_REQUEST_TERMS):
            return False, "OCR question is not a natural reading request"
        question_without_location = pilot.compact(question)
        for term in pilot.LOCATION_QUESTION_TERMS:
            question_without_location = question_without_location.replace(
                pilot.compact(term), ""
            )
        overlap = difflib.SequenceMatcher(
            None, exact, question_without_location
        ).find_longest_match().size
        if overlap >= 4:
            return False, "OCR answer text is leaked in the user question"
        seen.add(exact)
    return True, "ok"


def public_record(
    entry: Mapping[str, Any], dialogue: list[Mapping[str, str]], targets: list[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    reference = str(entry["_reference_text"])
    annotation = entry["_annotation"]
    return {
        "uid": str(entry["candidate_uid"]),
        "image": str(entry["path"]),
        "dialogue": dialogue,
        "generator": model,
        "task_mode": "ocr_document",
        "grounding": {
            "kind": "ocr_document",
            "targets": targets,
            "all_ocr_turns_grounded": True,
            "reference_annotation_kind": "exact_transcription",
            "canonical_transcription_sha256": hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest(),
            "canonical_transcription_characters": len(pilot.compact(reference)),
            "orientation": annotation.get("orientation"),
            "num_columns": annotation.get("num_columns"),
        },
        "split_group_key": entry.get("split_group_key"),
        "source": entry.get("source"),
    }


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"corrupt output at {path}:{line_number}: {error}") from error
            completed.add(str(row["uid"]))
    return completed


def save_failures(path: Path, failures: Mapping[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(sorted(failures.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def batched(rows: list[dict[str, Any]], size: int):
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/images/targeted_v4/image_pool_v1/ocr_images.jsonl"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=pilot.DEFAULT_MODEL)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")

    entries, preflight_rejections = load_entries(args.meta)
    assigned = [
        row for index, row in enumerate(entries) if index % args.num_shards == args.shard_id
    ]
    completed = load_completed(args.out)
    pending = [row for row in assigned if str(row["candidate_uid"]) not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.out.with_suffix(".failures.json")
    failures = {
        uid: reason
        for index, (uid, reason) in enumerate(sorted(preflight_rejections.items()))
        if index % args.num_shards == args.shard_id
    }
    print(
        f"[startup] shard={args.shard_id}/{args.num_shards} eligible_total={len(entries)} "
        f"assigned={len(assigned)} completed={len(completed)} pending={len(pending)} "
        f"preflight_rejected={len(failures)}",
        flush=True,
    )

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
        seed=42 + args.shard_id,
    )

    start = time.monotonic()
    accepted_this_run = 0
    processed_this_run = 0
    with args.out.open("a", encoding="utf-8", buffering=1) as output_handle:
        for batch_number, batch in enumerate(batched(pending, args.batch_size), 1):
            retry = batch
            accepted_batch: dict[str, dict[str, Any]] = {}
            for attempt in range(1, args.max_attempts + 1):
                if not retry:
                    break
                conversations = []
                valid_entries = []
                for entry in retry:
                    uid = str(entry["candidate_uid"])
                    try:
                        with Image.open(entry["_image_path"]) as source_image:
                            image = source_image.convert("RGB")
                    except OSError as error:
                        failures[uid] = f"image error: {error}"
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
                outputs = llm.chat(conversations, sampling) if conversations else []
                next_retry = []
                for entry, result in zip(valid_entries, outputs):
                    uid = str(entry["candidate_uid"])
                    payload = pilot.parse_payload(result.outputs[0].text)
                    if payload is None:
                        failures[uid] = "JSON parse failed"
                        next_retry.append(entry)
                        continue
                    dialogue = payload.get("dialogue")
                    targets = derive_all_targets(dialogue, entry["_ocr_candidates"])
                    valid, reason = validate(
                        dialogue,
                        targets,
                        entry["_reference_text"],
                        entry["_ocr_candidates"],
                    )
                    if not valid:
                        failures[uid] = reason
                        next_retry.append(entry)
                        continue
                    accepted_batch[uid] = public_record(
                        entry, dialogue, targets, args.model
                    )
                    failures.pop(uid, None)
                retry = next_retry

            for entry in batch:
                uid = str(entry["candidate_uid"])
                record = accepted_batch.get(uid)
                if record is not None:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    accepted_this_run += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
            processed_this_run += len(batch)
            save_failures(failure_path, failures)
            elapsed = max(time.monotonic() - start, 0.001)
            print(
                f"[progress] shard={args.shard_id} batch={batch_number} "
                f"processed={processed_this_run}/{len(pending)} "
                f"accepted={accepted_this_run} failed={processed_this_run - accepted_this_run} "
                f"rate={processed_this_run / elapsed:.3f} samples/s",
                flush=True,
            )

    save_failures(failure_path, failures)
    print(
        f"[done] shard={args.shard_id} processed={processed_this_run} "
        f"accepted={accepted_this_run} failures={len(failures)} output={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
