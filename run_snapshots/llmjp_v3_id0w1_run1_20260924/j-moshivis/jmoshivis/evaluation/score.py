"""Score cached J-MoshiVis predictions without loading a model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import SampleScore, aggregate, score_one, score_to_dict
from .schema import ManifestError, read_jsonl, read_manifest, sha256_file
from .suite import DEFAULT_SUITE_LOCK, verify_suite_manifest


PACKAGE_PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
SCORING_IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("metrics.py"),
    Path(__file__).with_name("official_rouge_ja.py"),
    Path(__file__).with_name("schema.py"),
    Path(__file__).with_name("suite.py"),
)
SCORING_DEPENDENCIES = (
    "rouge-score",
    "fugashi",
    "neologdn",
    "unidic-lite",
    "emoji",
    "fasttext-wheel",
    "numpy",
)


def _scoring_provenance() -> dict[str, Any]:
    from .official_rouge_ja import BOOTSTRAP_SEED

    versions: dict[str, str | None] = {}
    for distribution in SCORING_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "implementation_file_sha256": {
            str(path.resolve().relative_to(WORKSPACE.resolve())): sha256_file(path)
            for path in SCORING_IMPLEMENTATION_FILES
        },
        "dependency_versions": versions,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    successful: set[str] = set()
    for line_number, value in read_jsonl(path):
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ManifestError(f"{path}:{line_number}: prediction has no sample_id")
        # A failed attempt is intentionally retried by run_inference.  Keep
        # the latest attempt, but never accept two successful rows for one
        # sample because that would make result selection ambiguous.
        if sample_id in successful:
            raise ManifestError(
                f"{path}:{line_number}: row appears after a successful prediction "
                f"for {sample_id!r}"
            )
        predictions[sample_id] = value
        if not value.get("error"):
            successful.add(sample_id)
    return predictions


def validate_prediction_run(
    manifest_path: Path,
    predictions_path: Path,
    *,
    required: bool,
) -> dict[str, Any] | None:
    """Bind predictions to their finalized inference provenance sidecar."""

    run_path = predictions_path.with_suffix(predictions_path.suffix + ".run.json")
    if not run_path.is_file():
        if required:
            raise ManifestError(f"prediction run metadata is required: {run_path}")
        return None
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ManifestError(f"invalid prediction run metadata {run_path}: {exc}") from exc
    if run.get("schema_version") != "jmoshivis.image-eval.run.v2":
        raise ManifestError(f"unsupported prediction run schema: {run.get('schema_version')!r}")
    manifest_sha256 = sha256_file(manifest_path)
    if run.get("manifest_sha256") != manifest_sha256:
        raise ManifestError("prediction run was produced from a different manifest")
    if run.get("status") == "running":
        raise ManifestError("prediction run is still marked running; resume or finalize it first")
    declared_predictions_sha256 = run.get("predictions_sha256")
    if not declared_predictions_sha256:
        raise ManifestError("prediction run metadata has no finalized predictions_sha256")
    if declared_predictions_sha256 != sha256_file(predictions_path):
        raise ManifestError("prediction file checksum does not match its run metadata")
    fingerprint = run.get("run_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ManifestError("prediction run metadata has no run_fingerprint")
    for line_number, value in read_jsonl(predictions_path):
        if value.get("run_fingerprint") != fingerprint:
            raise ManifestError(
                f"{predictions_path}:{line_number}: prediction belongs to a different run"
            )
    manifest_ids = {example.sample_id for example in read_manifest(manifest_path)}
    prediction_records = read_predictions(predictions_path)
    extra_ids = sorted(set(prediction_records) - manifest_ids)
    if extra_ids:
        raise ManifestError(
            "prediction run contains sample IDs outside its manifest: "
            + ", ".join(extra_ids[:5])
        )
    successful_ids = {
        sample_id
        for sample_id, record in prediction_records.items()
        if not record.get("error")
    }
    if run.get("status") == "complete" and successful_ids != manifest_ids:
        missing_ids = sorted(manifest_ids - successful_ids)
        raise ManifestError(
            "prediction run is marked complete but lacks successful rows: "
            + ", ".join(missing_ids[:5])
        )
    if run.get("num_manifest_examples") is not None and (
        int(run["num_manifest_examples"]) != len(manifest_ids)
    ):
        raise ManifestError("prediction run num_manifest_examples is inconsistent")
    if run.get("num_successful_total") is not None and (
        int(run["num_successful_total"]) != len(successful_ids)
    ):
        raise ManifestError("prediction run num_successful_total is inconsistent")
    run["_sidecar_path"] = str(run_path.resolve())
    run["_sidecar_sha256"] = sha256_file(run_path)
    return run


def evaluate_files(
    manifest_path: Path,
    predictions_path: Path,
    *,
    include_official_rouge_ja: bool = False,
    language_id_model: Path | None = None,
    require_run_metadata: bool = False,
    suite_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = read_manifest(manifest_path)
    suite_contract = (
        None
        if suite_lock_path is None
        else verify_suite_manifest(suite_lock_path, manifest_path)
    )
    if suite_contract is not None and len(examples) != suite_contract["frozen_num_questions"]:
        raise ManifestError("manifest question count does not match the frozen suite lock")
    if language_id_model is not None and suite_contract is not None:
        actual_lid_sha256 = sha256_file(language_id_model)
        if actual_lid_sha256 != suite_contract["language_id_model_sha256"]:
            raise ManifestError("language-ID model does not match the frozen suite lock")
    scoring_provenance = _scoring_provenance()
    scoring_protocol_verified = False
    if suite_contract is not None and include_official_rouge_ja:
        expected_scoring = suite_contract["scoring_protocol"]
        if scoring_provenance != expected_scoring:
            raise ManifestError(
                "installed scorer code/dependencies do not match the frozen suite lock"
            )
        scoring_protocol_verified = True
    prediction_run = validate_prediction_run(
        manifest_path, predictions_path, required=require_run_metadata
    )
    predictions = read_predictions(predictions_path)
    manifest_ids = {example.sample_id for example in examples}
    extra = sorted(set(predictions) - manifest_ids)

    scored_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[SampleScore]] = defaultdict(list)
    grouped_question: dict[tuple[str, str], list[SampleScore]] = defaultdict(list)
    all_scores: list[SampleScore] = []
    missing = 0
    failed = 0

    for example in examples:
        prediction_record = predictions.get(example.sample_id)
        if prediction_record is None:
            missing += 1
            generated_text = ""
            inference_error = "missing_prediction"
        else:
            generated_text = str(
                prediction_record.get("answer_text", prediction_record.get("generated_text", ""))
            )
            inference_error = prediction_record.get("error")
            if inference_error:
                failed += 1
                generated_text = ""

        score = score_one(
            example.sample_id,
            generated_text,
            example.answers,
            task=example.task,
            choices=example.choices,
            choice_labels=example.choice_labels,
            correct_choice=example.correct_choice,
        )
        all_scores.append(score)
        grouped[(example.domain, example.benchmark)].append(score)
        if example.question_type:
            grouped_question[(example.benchmark, example.question_type)].append(score)
        scored_rows.append(
            {
                "sample_id": example.sample_id,
                "benchmark": example.benchmark,
                "domain": example.domain,
                "task": example.task,
                "question_type": example.question_type,
                "question": example.question,
                "references": example.answers,
                "prediction": generated_text,
                "inference_error": inference_error,
                "metrics": score_to_dict(score),
            }
        )

    by_benchmark: dict[str, Any] = {}
    domain_datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (domain, benchmark), scores in sorted(grouped.items()):
        result = aggregate(scores)
        result["domain"] = domain
        by_benchmark[benchmark] = result
        domain_datasets[domain].append(result)

    # Dataset-macro keeps a 4,000-question set from drowning out a 50-question
    # set.  Raw pooled scores remain available as ``overall_micro``.
    by_domain: dict[str, Any] = {}
    metric_names = (
        "exact_match", "answer_in_response", "char_f1",
        "normalized_edit_similarity", "choice_accuracy",
    )
    for domain, datasets in sorted(domain_datasets.items()):
        result: dict[str, Any] = {
            "num_datasets": len(datasets),
            "num_samples": sum(item["num_samples"] for item in datasets),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in datasets if metric in item]
            if values:
                result[f"macro_{metric}"] = sum(values) / len(values)
        by_domain[domain] = result

    by_question_type = {
        f"{benchmark}/{question_type}": aggregate(scores)
        for (benchmark, question_type), scores in sorted(grouped_question.items())
    }
    if include_official_rouge_ja:
        from .official_rouge_ja import sakana_rouge_scores

        for benchmark in by_benchmark:
            if benchmark != "ja-vg-vqa-500":
                continue
            benchmark_rows = [row for row in scored_rows if row["benchmark"] == benchmark]
            # Public JA-VG uses one reference per question.  For any future
            # multi-reference set, use the first canonical reference and keep
            # the deterministic max-over-reference metrics alongside it.
            references = [str(row["references"][0]) for row in benchmark_rows]
            generated = [str(row["prediction"]) for row in benchmark_rows]
            by_benchmark[benchmark]["sakana_mecab_rouge_percent"] = sakana_rouge_scores(
                references, generated, language_id_model=language_id_model
            )
    summary = {
        "schema_version": "jmoshivis.image-eval.score.v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "overall_micro": aggregate(all_scores),
        "by_domain_dataset_macro": by_domain,
        "by_benchmark": by_benchmark,
        "by_question_type": by_question_type,
        "num_missing_predictions": missing,
        "num_failed_predictions": failed,
        "num_extra_predictions": len(extra),
        "extra_prediction_ids": extra[:100],
        "metric_notes": {
            "primary_short_answer": "exact_match (or the benchmark's frozen official metric/judge)",
            "official_ja_vg_comparison": "MeCab ROUGE-L when --official-rouge-ja is enabled; clean-subset scores are not directly comparable to the public 500-item score",
            "answer_in_response": "diagnostic only: NFKC-normalized reference substring with common Japanese negation guard",
            "exact_match": "primary dependency-free short-answer metric: strict normalized full-string match",
            "choice_accuracy": "strict full-response choice-label accuracy; CVQA requires one numeric label 1-4 and does not credit option-text fallback",
            "confidence_intervals": "Wilson 95% interval for binary metrics",
        },
        "scoring_provenance": scoring_provenance,
        "scoring_protocol_verified": scoring_protocol_verified,
    }
    if prediction_run is None:
        summary["prediction_run"] = None
        summary["ready_for_local_metrics"] = False
        summary["paper_guard"] = "Missing prediction run metadata."
    else:
        summary["prediction_run"] = {
            "path": prediction_run["_sidecar_path"],
            "sha256": prediction_run["_sidecar_sha256"],
            "status": prediction_run.get("status"),
            "run_fingerprint": prediction_run.get("run_fingerprint"),
            "checkpoint": prediction_run.get("checkpoint"),
            "checkpoint_sha256": prediction_run.get("checkpoint_sha256"),
            "generation": prediction_run.get("generation"),
            "diagnostic": prediction_run.get("diagnostic"),
            "checkpoint_hash_skipped": prediction_run.get("checkpoint_hash_skipped"),
        }
        local_metrics_ready = bool(
            suite_contract is not None
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
            and prediction_run.get("checkpoint_sha256")
            and missing == 0
            and failed == 0
            and len(extra) == 0
            and include_official_rouge_ja
            and language_id_model is not None
            and scoring_protocol_verified
        )
        summary["ready_for_local_metrics"] = local_metrics_ready
        summary["result_scope"] = "local_metrics_only"
        summary["paper_guard"] = (
            "Local metrics are ready; JAMMEval judge readiness must still be checked separately."
            if local_metrics_ready
            else "Do not report as a paper result until the frozen suite and scorer "
                 "protocol match, the run is complete/greedy/fully hashed/non-diagnostic, "
                 "and missing/failed/extra prediction counts are all zero."
        )
    summary.setdefault("result_scope", "local_metrics_only")
    summary["suite_contract"] = suite_contract
    if language_id_model is not None:
        summary["language_id_model"] = {
            "path": str(language_id_model.resolve()),
            "sha256": sha256_file(language_id_model),
        }
    return scored_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-lock", type=Path, default=DEFAULT_SUITE_LOCK)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--official-rouge-ja",
        action="store_true",
        help="Also reproduce SakanaAI's MeCab-based ROUGE (optional dependencies).",
    )
    parser.add_argument(
        "--lid176-model", type=Path,
        help="Optional lid.176.ftz path; adds SakanaAI's language-filtered *_ja metrics.",
    )
    parser.add_argument(
        "--allow-untracked-predictions", action="store_true",
        help="Diagnostic only: score without a finalized .run.json provenance sidecar.",
    )
    args = parser.parse_args()
    if args.lid176_model is not None and not args.lid176_model.is_file():
        raise FileNotFoundError(args.lid176_model)
    if args.lid176_model is not None and not args.official_rouge_ja:
        raise ValueError("--lid176-model requires --official-rouge-ja")

    rows, summary = evaluate_files(
        args.manifest,
        args.predictions,
        include_official_rouge_ja=args.official_rouge_ja,
        language_id_model=args.lid176_model,
        require_run_metadata=not args.allow_untracked_predictions,
        suite_lock_path=args.suite_lock,
    )
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.scores.with_suffix(args.scores.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.scores)
    _atomic_json(args.summary, summary)
    print(
        f"scored {len(rows)} samples | "
        f"exact={summary['overall_micro']['exact_match']:.3f} | "
        f"answer-in-response(diagnostic)={summary['overall_micro']['answer_in_response']:.3f}"
    )
    print(f"scores:  {args.scores}")
    print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
