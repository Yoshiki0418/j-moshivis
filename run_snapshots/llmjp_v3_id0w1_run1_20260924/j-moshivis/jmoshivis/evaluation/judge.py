"""Prepare and score the frozen JAMMEval short-answer LLM judge protocol.

``prepare`` writes OpenAI Batch API requests without making a network call.
After the batch finishes, ``score`` parses its output and reports both the
official-compatible accuracy (ungraded rows excluded) and a conservative
all-target accuracy (failures count as wrong).  A paper result should only be
accepted when both model and grader failure counts are zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import ManifestError, read_jsonl, read_manifest, sha256_file
from .score import read_predictions, validate_prediction_run
from .suite import DEFAULT_SUITE_LOCK, verify_suite_manifest


OFFICIAL_SOURCE = (
    "https://github.com/llm-jp/simple-evals-mm/blob/"
    "979e030f867c0dea8da093078965b3057fdd2416/src/simple_evals_mm/common.py"
)
DEFAULT_JUDGE_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_N_REPEATS = 3
WORKSPACE = Path(__file__).resolve().parents[3]
JUDGE_IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("score.py"),
    Path(__file__).with_name("schema.py"),
    Path(__file__).with_name("suite.py"),
)
SHORT_ANSWER_BENCHMARKS = {
    "jammeval-cc-ocr-ja-refined",
    "jammeval-heron-bench-refined",
    "jammeval-ja-vlm-bench-refined",
}

# Verbatim protocol from the pinned official source above (Apache-2.0).
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

When judging equivalence, allow variations in script or notation that convey the same meaning
(e.g., '2羽' and '二羽' should be considered equivalent).

Treat the following cases as correct:
- The extracted answer includes additional context (e.g., series name, author name, location, broader category) while still containing the correct_answer exactly or as its unambiguous, specific instance.
  (For example, "富嶽三十六景 江戸日本橋" is correct if the correct_answer is "江戸日本橋".)
- The extracted answer is more specific than the [correct_answer] while remaining consistent with it.
- The extracted answer is an alternate name, synonymous phrasing, or another commonly accepted way to refer to the same concept, object, place, or artwork.
- The extracted answer omits information that is not essential to the correctness of the question.
- Allow minor variations in spacing, capitalization, or script, as long as the core correct_answer is unambiguously present.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

_VERDICT_RE = re.compile(r"correct\s*:\s*(yes|no)", flags=re.IGNORECASE)


def build_grader_prompt(question: str, correct_answer: str, response: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )


def _judge_implementation_hashes() -> dict[str, str]:
    return {
        str(path.resolve().relative_to(WORKSPACE.resolve())): sha256_file(path)
        for path in JUDGE_IMPLEMENTATION_FILES
    }


def _verify_judge_implementation(suite_contract: dict[str, Any] | None) -> dict[str, str]:
    actual = _judge_implementation_hashes()
    if suite_contract is not None and (
        actual != suite_contract["judge_implementation_file_sha256"]
    ):
        raise ManifestError("judge implementation does not match the frozen suite lock")
    return actual


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_batch(
    manifest_path: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    n_repeats: int = DEFAULT_N_REPEATS,
    require_run_metadata: bool = False,
    suite_lock_path: Path | None = None,
    allow_nonstandard_protocol: bool = False,
) -> dict[str, Any]:
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if not allow_nonstandard_protocol and (
        model != DEFAULT_JUDGE_MODEL or n_repeats != DEFAULT_N_REPEATS
    ):
        raise ValueError(
            "non-standard judge model/repeats require allow_nonstandard_protocol=True"
        )
    examples = read_manifest(manifest_path)
    suite_contract = (
        None
        if suite_lock_path is None
        else verify_suite_manifest(suite_lock_path, manifest_path)
    )
    if suite_contract is not None and (
        model != suite_contract["judge_model"]
        or n_repeats != suite_contract["judge_n_repeats"]
        or hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest()
        != suite_contract["judge_template_sha256"]
    ):
        raise ManifestError("judge configuration does not match the frozen suite lock")
    judge_implementation = _verify_judge_implementation(suite_contract)
    prediction_run = validate_prediction_run(
        manifest_path, predictions_path, required=require_run_metadata
    )
    if suite_contract is not None and prediction_run is not None:
        if prediction_run.get("suite_lock_sha256") != suite_contract["suite_lock_sha256"]:
            raise ManifestError("prediction run and judge use different suite locks")
    predictions = read_predictions(predictions_path)
    targets = [example for example in examples if example.benchmark in SHORT_ANSWER_BENCHMARKS]
    missing_or_failed = 0
    request_map: dict[str, dict[str, Any]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for example in targets:
            prediction = predictions.get(example.sample_id)
            if prediction is None or prediction.get("error"):
                missing_or_failed += 1
                continue
            response = str(prediction.get("answer_text", prediction.get("generated_text", "")))
            prompt = build_grader_prompt(example.rendered_prompt, example.answers[0], response)
            body = {
                    "model": model,
                    "input": [
                        {
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        },
                    ],
                    "max_output_tokens": 8192,
                    "reasoning": {"effort": "none"},
            }
            body_sha256 = hashlib.sha256(
                json.dumps(
                    body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            for repeat in range(1, n_repeats + 1):
                custom_id = "jv1-" + hashlib.sha256(
                    f"{example.sample_id}\0{repeat}\0{body_sha256}".encode("utf-8")
                ).hexdigest()[:40]
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
                request_map[custom_id] = {
                    "sample_id": example.sample_id,
                    "repeat": repeat,
                    "body_sha256": body_sha256,
                }
                count += 1
    temporary.replace(output_path)
    template_sha256 = hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": "jmoshivis.jammeval-judge-batch.v2",
        "official_source": OFFICIAL_SOURCE,
        "grader_template_sha256": template_sha256,
        "model": model,
        "n_repeats": n_repeats,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "num_targets": len(targets),
        "num_requests": count,
        "num_model_missing_or_failed": missing_or_failed,
        "batch_requests_sha256": sha256_file(output_path),
        "prediction_run_sidecar_sha256": (
            None if prediction_run is None else prediction_run.get("_sidecar_sha256")
        ),
        "suite_contract": suite_contract,
        "judge_implementation_file_sha256": judge_implementation,
        "request_map": request_map,
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".meta.json"), metadata)
    return metadata


def _response_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return str(body["output_text"])
    parts: list[str] = []
    for output in body.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(parts)


def read_batch_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for line_number, row in read_jsonl(path):
        sample_id = row.get("custom_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ManifestError(f"{path}:{line_number}: missing custom_id")
        if sample_id in results:
            raise ManifestError(f"{path}:{line_number}: duplicate custom_id {sample_id!r}")
        response = row.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}
        raw = _response_text(body) if status_code == 200 else ""
        matches = {value.lower() for value in _VERDICT_RE.findall(raw)}
        results[sample_id] = {
            "status_code": status_code,
            "raw_response": raw,
            # Conflicting verdict fields are an invalid judge response rather
            # than a first/last-match choice that could silently change a score.
            "verdict": next(iter(matches)) if len(matches) == 1 else None,
            "model": body.get("model"),
            "request_error": row.get("error"),
        }
    return results


def _load_batch_contract(
    batch_requests_path: Path,
    *,
    manifest_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata_path = batch_requests_path.with_suffix(batch_requests_path.suffix + ".meta.json")
    if not metadata_path.is_file():
        raise ManifestError(f"judge batch metadata is required: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ManifestError(f"invalid judge batch metadata {metadata_path}: {exc}") from exc
    if metadata.get("schema_version") != "jmoshivis.jammeval-judge-batch.v2":
        raise ManifestError(f"unsupported judge batch schema: {metadata.get('schema_version')!r}")
    checks = {
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "batch_requests_sha256": sha256_file(batch_requests_path),
        "grader_template_sha256": hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest(),
    }
    mismatched = [key for key, expected in checks.items() if metadata.get(key) != expected]
    if mismatched:
        raise ManifestError(
            "judge batch contract mismatch: "
            + ", ".join(f"{key}" for key in mismatched)
        )
    request_map = metadata.get("request_map")
    if not isinstance(request_map, dict) or len(request_map) != metadata.get("num_requests"):
        raise ManifestError("judge batch metadata has an invalid request_map")
    request_ids: set[str] = set()
    for line_number, request in read_jsonl(batch_requests_path):
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in request_map:
            raise ManifestError(
                f"{batch_requests_path}:{line_number}: request is absent from request_map"
            )
        if custom_id in request_ids:
            raise ManifestError(f"{batch_requests_path}:{line_number}: duplicate custom_id")
        request_ids.add(custom_id)
        body_sha256 = hashlib.sha256(
            json.dumps(
                request.get("body"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if request_map[custom_id].get("body_sha256") != body_sha256:
            raise ManifestError(f"{batch_requests_path}:{line_number}: request body mismatch")
    if request_ids != set(request_map):
        raise ManifestError("judge batch request IDs do not match request_map")
    metadata["_metadata_path"] = str(metadata_path.resolve())
    metadata["_metadata_sha256"] = sha256_file(metadata_path)
    return metadata, request_map


def score_batch(
    manifest_path: Path,
    predictions_path: Path,
    judge_results_path: Path,
    batch_requests_path: Path,
    *,
    require_run_metadata: bool = False,
    suite_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = read_manifest(manifest_path)
    suite_contract = (
        None
        if suite_lock_path is None
        else verify_suite_manifest(suite_lock_path, manifest_path)
    )
    prediction_run = validate_prediction_run(
        manifest_path, predictions_path, required=require_run_metadata
    )
    if suite_contract is not None:
        if prediction_run is not None and (
            prediction_run.get("suite_lock_sha256") != suite_contract["suite_lock_sha256"]
        ):
            raise ManifestError("prediction run and judge use different suite locks")
    judge_implementation = _verify_judge_implementation(suite_contract)
    predictions = read_predictions(predictions_path)
    contract, request_map = _load_batch_contract(
        batch_requests_path,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
    )
    if contract.get("judge_implementation_file_sha256") != judge_implementation:
        raise ManifestError(
            "judge batch was prepared with a different judge implementation"
        )
    if suite_contract is not None:
        contract_suite = contract.get("suite_contract")
        if not isinstance(contract_suite, dict) or (
            contract_suite.get("suite_lock_sha256") != suite_contract["suite_lock_sha256"]
        ):
            raise ManifestError("judge batch and score use different suite locks")
    judge_results = read_batch_results(judge_results_path)
    targets = [example for example in examples if example.benchmark in SHORT_ANSWER_BENCHMARKS]
    target_ids = {example.sample_id for example in targets}
    extras = sorted(set(judge_results) - set(request_map))
    n_repeats = int(contract["n_repeats"])
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    requests_by_target: dict[tuple[str, int], str] = {}
    for custom_id, mapping in request_map.items():
        sample_id = str(mapping.get("sample_id", ""))
        repeat = int(mapping.get("repeat", 0))
        if sample_id not in target_ids or not 1 <= repeat <= n_repeats:
            raise ManifestError(f"invalid request_map entry for {custom_id}")
        key = (sample_id, repeat)
        if key in requests_by_target:
            raise ManifestError(f"duplicate judge request for {sample_id}, repeat {repeat}")
        requests_by_target[key] = custom_id

    for example in targets:
        prediction = predictions.get(example.sample_id)
        model_error = "missing_prediction" if prediction is None else prediction.get("error")
        for repeat in range(1, n_repeats + 1):
            custom_id = requests_by_target.get((example.sample_id, repeat))
            if model_error:
                if custom_id is not None:
                    raise ManifestError(
                        f"model-failed sample unexpectedly has a judge request: {example.sample_id}"
                    )
                judged = None
            else:
                if custom_id is None:
                    raise ManifestError(
                        f"judge batch is missing request for {example.sample_id}, repeat {repeat}"
                    )
                judged = judge_results.get(custom_id)
            verdict = None if judged is None or model_error else judged["verdict"]
            score = None if verdict is None else float(verdict == "yes")
            row = {
                "sample_id": example.sample_id,
                "benchmark": example.benchmark,
                "repeat": repeat,
                "judge_custom_id": custom_id,
                "model_error": model_error,
                "judge_missing": judged is None and not model_error,
                "judge_status_code": None if judged is None else judged["status_code"],
                "judge_verdict": verdict,
                "judge_model": None if judged is None else judged["model"],
                "score": score,
                "raw_judge_response": None if judged is None else judged["raw_response"],
            }
            rows.append(row)
            grouped[example.benchmark].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        per_repeat: list[dict[str, Any]] = []
        for repeat in range(1, n_repeats + 1):
            repeat_items = [row for row in items if row["repeat"] == repeat]
            graded = [row for row in repeat_items if row["score"] is not None]
            correct = sum(float(row["score"]) for row in graded)
            per_repeat.append(
                {
                    "repeat": repeat,
                    "num_targets": len(repeat_items),
                    "num_graded": len(graded),
                    "num_correct": int(correct),
                    "num_model_failed": sum(bool(row["model_error"]) for row in repeat_items),
                    "num_judge_failed_or_missing": sum(
                        not row["model_error"] and row["score"] is None for row in repeat_items
                    ),
                    "accuracy_official_compatible": (
                        None if not graded else correct / len(graded)
                    ),
                    "accuracy_all_targets_failures_wrong": (
                        correct / len(repeat_items) if repeat_items else None
                    ),
                }
            )
        official_scores = [
            float(item["accuracy_official_compatible"])
            for item in per_repeat
            if item["accuracy_official_compatible"] is not None
        ]
        conservative_scores = [
            float(item["accuracy_all_targets_failures_wrong"])
            for item in per_repeat
            if item["accuracy_all_targets_failures_wrong"] is not None
        ]
        return {
            "num_targets": len(items) // n_repeats,
            "n_repeats": n_repeats,
            "per_repeat": per_repeat,
            "scores_official_compatible": official_scores,
            "mean_accuracy_official_compatible": (
                statistics.fmean(official_scores) if official_scores else None
            ),
            "std_accuracy_official_compatible": (
                statistics.stdev(official_scores) if len(official_scores) > 1 else None
            ),
            "scores_all_targets_failures_wrong": conservative_scores,
            "mean_accuracy_all_targets_failures_wrong": (
                statistics.fmean(conservative_scores) if conservative_scores else None
            ),
            "total_model_failed": sum(bool(row["model_error"]) for row in items),
            "total_judge_failed_or_missing": sum(
                not row["model_error"] and row["score"] is None for row in items
            ),
        }

    summary = {
        "schema_version": "jmoshivis.jammeval-judge-score.v2",
        "official_source": OFFICIAL_SOURCE,
        "grader_template_sha256": hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest(),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "judge_results_sha256": sha256_file(judge_results_path),
        "batch_requests": str(batch_requests_path.resolve()),
        "batch_requests_sha256": sha256_file(batch_requests_path),
        "batch_metadata": contract["_metadata_path"],
        "batch_metadata_sha256": contract["_metadata_sha256"],
        "judge_model_requested": contract["model"],
        "n_repeats": n_repeats,
        "overall": aggregate(rows),
        "by_benchmark": {name: aggregate(items) for name, items in sorted(grouped.items())},
        "judge_models": sorted(
            {str(row["judge_model"]) for row in rows if row["judge_model"]}
        ),
        "num_extra_judge_results": len(extras),
        "extra_judge_result_ids": extras[:100],
        "paper_guard": "Do not report the official-compatible score unless model and judge failure counts are both zero.",
        "judge_implementation_file_sha256": judge_implementation,
    }
    protocol_ready = bool(
        contract.get("model") == DEFAULT_JUDGE_MODEL
        and n_repeats == DEFAULT_N_REPEATS
        and set(summary["judge_models"]) == {DEFAULT_JUDGE_MODEL}
        and all(
            row["judge_model"] == DEFAULT_JUDGE_MODEL
            for row in rows
            if not row["model_error"]
        )
        and contract.get("grader_template_sha256")
        == hashlib.sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest()
    )
    summary["protocol_ready"] = protocol_ready
    summary["suite_contract"] = suite_contract
    summary["ready_for_jammeval_judge"] = bool(
        suite_contract is not None
        and prediction_run is not None
        and prediction_run.get("suite_lock_sha256") == suite_contract["suite_lock_sha256"]
        and prediction_run.get("suite_id") == suite_contract["suite_id"]
        and prediction_run.get("status") == "complete"
        and prediction_run.get("generation") == "greedy"
        and prediction_run.get("generation") == suite_contract["paper_generation"]
        and prediction_run.get("response_seconds")
        == suite_contract["response_tail_seconds"]
        and prediction_run.get("response_overlap_seconds")
        == suite_contract["response_overlap_seconds"]
        and not prediction_run.get("diagnostic")
        and not prediction_run.get("checkpoint_hash_skipped")
        and bool(prediction_run.get("checkpoint_sha256"))
        and protocol_ready
        and summary["overall"]["total_model_failed"] == 0
        and summary["overall"]["total_judge_failed_or_missing"] == 0
        and not extras
    )
    summary["result_scope"] = "jammeval_short_answer_judge_only"
    summary["paper_guard"] = (
        "JAMMEval judge metrics are ready; local JA-VG/CVQA metrics must still be checked separately."
        if summary["ready_for_jammeval_judge"]
        else "Do not report as a paper result until the frozen suite, inference run, "
             "judge implementation/model/repeats, and generation window all match, "
             "with zero model/judge failures or extra results."
    )
    return rows, summary


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Write OpenAI Batch API request JSONL.")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--suite-lock", type=Path, default=DEFAULT_SUITE_LOCK)
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    prepare.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    prepare.add_argument(
        "--allow-nonstandard-protocol", action="store_true",
        help="Diagnostic only: permit a judge model/repeat count outside the frozen protocol.",
    )
    prepare.add_argument(
        "--allow-untracked-predictions", action="store_true",
        help="Diagnostic only: allow predictions without a finalized run sidecar.",
    )
    score = subparsers.add_parser("score", help="Parse batch output and aggregate accuracy.")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--suite-lock", type=Path, default=DEFAULT_SUITE_LOCK)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--judge-results", type=Path, required=True)
    score.add_argument("--batch-requests", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--summary", type=Path, required=True)
    score.add_argument(
        "--allow-untracked-predictions", action="store_true",
        help="Diagnostic only: allow predictions without a finalized run sidecar.",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        metadata = prepare_batch(
            args.manifest,
            args.predictions,
            args.output,
            model=args.model,
            n_repeats=args.n_repeats,
            require_run_metadata=not args.allow_untracked_predictions,
            suite_lock_path=args.suite_lock,
            allow_nonstandard_protocol=args.allow_nonstandard_protocol,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    rows, summary = score_batch(
        args.manifest,
        args.predictions,
        args.judge_results,
        args.batch_requests,
        require_run_metadata=not args.allow_untracked_predictions,
        suite_lock_path=args.suite_lock,
    )
    _write_rows(args.output, rows)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
