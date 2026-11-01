"""非画像対話を p_concat 用TTS入力へ決定論的に選別する。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SPEAKERS = ("ユーザー", "アシスタント")
VISUAL_TERMS = ("画像", "写真", "写ってい", "見えてる", "見えている", "この景色")


def rejection_reasons(record: dict) -> list[str]:
    reasons: list[str] = []
    if record.get("kind") != "nonvisual":
        reasons.append("kind_not_nonvisual")

    dialogue = record.get("dialogue")
    if not isinstance(dialogue, list):
        return reasons + ["dialogue_not_list"]
    if not 4 <= len(dialogue) <= 6:
        reasons.append("turn_count_not_4_to_6")

    texts: list[str] = []
    for index, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            reasons.append("turn_not_object")
            continue
        expected = SPEAKERS[index % 2]
        if turn.get("speaker") != expected:
            reasons.append("speaker_not_alternating")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            reasons.append("empty_text")
        else:
            texts.append(text)
            if len(text) > 300:
                reasons.append("text_over_300_chars")

    joined = "".join(texts)
    if any(term in joined for term in VISUAL_TERMS):
        reasons.append("visual_reference")
    if any("a" <= char.lower() <= "z" for char in joined):
        reasons.append("ascii_letter")
    return sorted(set(reasons))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    report_path = Path(args.report)
    for path in (accepted_path, rejected_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    reason_counts: Counter[str] = Counter()
    total = 0

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                reasons = ["invalid_json"]
                rejected.append({
                    "line_number": line_number,
                    "reasons": reasons,
                    "error": str(error),
                })
                reason_counts.update(reasons)
                continue

            uid = record.get("uid")
            reasons = rejection_reasons(record)
            if not isinstance(uid, str) or not uid:
                reasons.append("missing_uid")
            elif uid in seen:
                reasons.append("duplicate_uid")
            else:
                seen.add(uid)

            reasons = sorted(set(reasons))
            if reasons:
                rejected.append({"line_number": line_number, "reasons": reasons, "record": record})
                reason_counts.update(reasons)
            else:
                accepted.append(record)

    accepted_tmp = accepted_path.with_suffix(accepted_path.suffix + ".tmp")
    rejected_tmp = rejected_path.with_suffix(rejected_path.suffix + ".tmp")
    with accepted_tmp.open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with rejected_tmp.open("w", encoding="utf-8") as handle:
        for record in rejected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    accepted_tmp.replace(accepted_path)
    rejected_tmp.replace(rejected_path)

    report = {
        "schema_version": "jmoshivis.nonvisual-tts-qc.v1",
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "accepted_path": str(accepted_path.resolve()),
        "accepted_sha256": sha256(accepted_path),
        "rejected_path": str(rejected_path.resolve()),
        "rejected_sha256": sha256(rejected_path),
    }
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_tmp.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
