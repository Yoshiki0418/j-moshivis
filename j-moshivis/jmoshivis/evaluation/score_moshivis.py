"""Score cached predictions with the three metrics reported by MoshiVis.

This is intentionally separate from :mod:`jmoshivis.evaluation.score`.  The
existing scorer implements the frozen Japanese ``vision_v1`` protocol, while
this module implements the classical benchmarks used by MoshiVis/Babillage:

* OCR-VQA: case-insensitive exact-answer accuracy;
* VQAv2: the official ten-annotator consensus accuracy;
* COCO Captions: CIDEr after the COCO PTB tokenization pipeline.

The model is never loaded here.  Predictions produced by ``run_inference`` are
scored from their synchronized text stream (``answer_text``), so no output ASR
is involved.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cider_ja import japanese_cider_score
from .metrics_moshivis import (
    ocr_vqa_exact_match,
    official_coco_cider_score,
    vqav2_accuracy,
)
from .schema import EvalExample, ManifestError, read_manifest, sha256_file
from .score import read_predictions, validate_prediction_run
from .suite_moshivis import verify_evaluation_suite


SCHEMA_VERSION = "jmoshivis.moshivis-eval.score.v1"
VALID_METRICS = {"ocr_vqa_accuracy", "vqav2_accuracy", "cider", "cider_ja"}
_LEGACY_METRICS = {"vqa_accuracy": "vqav2_accuracy"}
PACKAGE_PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
SCORING_IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("cider_ja.py"),
    Path(__file__).with_name("metrics_moshivis.py"),
    Path(__file__).with_name("schema.py"),
    Path(__file__).with_name("score.py"),
    Path(__file__).with_name("suite.py"),
    Path(__file__).with_name("suite_moshivis.py"),
)

# Names used by the upstream datasets and by the conversational Babillage
# variants.  A manifest may avoid name inference entirely by setting
# ``metadata.moshivis_metric`` on every row.
_BENCHMARK_ALIASES = {
    "ocrvqa": "ocr_vqa_accuracy",
    "coocrvqa": "ocr_vqa_accuracy",
    "vqav2": "vqav2_accuracy",
    "covqav2": "vqav2_accuracy",
    "coco": "cider",
    "cococaptions": "cider",
    "cococaptions2014": "cider",
    "cococo": "cider",
}


def metric_for_example(example: EvalExample) -> str:
    explicit = example.metadata.get("moshivis_metric")
    if explicit is not None:
        explicit = _LEGACY_METRICS.get(str(explicit), str(explicit))
        if explicit not in VALID_METRICS:
            raise ManifestError(
                f"{example.sample_id}: unsupported metadata.moshivis_metric {explicit!r}"
            )
        return str(explicit)
    canonical = re.sub(r"[^a-z0-9]", "", example.benchmark.lower())
    metric = _BENCHMARK_ALIASES.get(canonical)
    if metric is None:
        raise ManifestError(
            f"{example.sample_id}: cannot infer a MoshiVis metric from benchmark "
            f"{example.benchmark!r}; set metadata.moshivis_metric explicitly"
        )
    return metric


def _compute_cider(
    predictions: dict[str, str],
    references: dict[str, list[str]],
) -> tuple[float, dict[str, float]]:
    """Run the shared canonical COCO PTBTokenizer + CIDEr implementation."""

    ids = list(predictions)
    result = official_coco_cider_score(
        [predictions[sample_id] for sample_id in ids],
        [references[sample_id] for sample_id in ids],
    )
    return result.score, {
        sample_id: float(score)
        for sample_id, score in zip(ids, result.per_example, strict=True)
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dependency_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _java_version() -> str | None:
    """Return the exact Java runtime banner used by COCO's PTB tokenizer."""

    try:
        process = subprocess.run(
            ["java", "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    banner = "\n".join(
        value.strip() for value in (process.stderr, process.stdout) if value.strip()
    )
    return banner or None


def _ptb_tokenizer_jar_sha256() -> str | None:
    """Hash the Stanford tokenizer jar bundled by pycocoevalcap."""

    try:
        specification = importlib.util.find_spec("pycocoevalcap")
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if specification is None or not specification.submodule_search_locations:
        return None
    for package_root in specification.submodule_search_locations:
        jar = Path(package_root) / "tokenizer" / "stanford-corenlp-3.4.1.jar"
        if jar.is_file():
            return sha256_file(jar)
    return None


def scoring_provenance() -> dict[str, Any]:
    """Return the exact scorer code, package, Java, and PTB-jar identity."""

    return {
        "implementation_file_sha256": {
            str(path.resolve().relative_to(WORKSPACE.resolve())): sha256_file(path)
            for path in SCORING_IMPLEMENTATION_FILES
        },
        "dependency_versions": {
            "pycocoevalcap": _dependency_version("pycocoevalcap"),
            "fugashi": _dependency_version("fugashi"),
            "unidic-lite": _dependency_version("unidic-lite"),
            "java": _java_version(),
        },
        "dependency_artifact_sha256": {
            "stanford-corenlp-3.4.1.jar": _ptb_tokenizer_jar_sha256(),
        },
    }


def evaluate_files(
    manifest_path: Path,
    predictions_path: Path,
    *,
    require_run_metadata: bool = False,
    suite_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = read_manifest(manifest_path)
    suite_contract = (
        None
        if suite_lock_path is None
        else verify_evaluation_suite(suite_lock_path, manifest_path)
    )
    if suite_contract is not None and (
        len(examples) != suite_contract["frozen_num_questions"]
    ):
        raise ManifestError("manifest question count does not match the frozen suite lock")
    prediction_run = validate_prediction_run(
        manifest_path, predictions_path, required=require_run_metadata
    )
    predictions = read_predictions(predictions_path)
    manifest_ids = {example.sample_id for example in examples}
    extra_ids = sorted(set(predictions) - manifest_ids)

    missing = 0
    failed = 0
    rows: list[dict[str, Any]] = []
    row_by_id: dict[str, dict[str, Any]] = {}
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    cider_predictions: dict[str, dict[str, str]] = defaultdict(dict)
    cider_references: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for example in examples:
        metric = metric_for_example(example)
        record = predictions.get(example.sample_id)
        if record is None:
            missing += 1
            prediction = ""
            inference_error: Any = "missing_prediction"
        else:
            inference_error = record.get("error")
            if inference_error:
                failed += 1
                prediction = ""
            else:
                prediction = str(
                    record.get("answer_text", record.get("generated_text", ""))
                )

        if not example.answers:
            raise ManifestError(
                f"{example.sample_id}: {metric} requires at least one reference"
            )
        if metric == "vqav2_accuracy" and len(example.answers) != 10:
            raise ManifestError(
                f"{example.sample_id}: official VQAv2 scoring requires exactly "
                f"10 annotator answers, found {len(example.answers)}"
            )
        if metric in {"cider", "cider_ja"} and len(example.answers) < 2:
            raise ManifestError(
                f"{example.sample_id}: COCO CIDEr requires multiple reference "
                "captions per evaluation unit; collapse duplicate image rows into "
                "one manifest row with all captions in answers"
            )

        if metric == "ocr_vqa_accuracy":
            sample_score = ocr_vqa_exact_match(prediction, example.answers)
        elif metric == "vqav2_accuracy":
            sample_score = vqav2_accuracy(prediction, example.answers)
        else:
            sample_score = None
            cider_predictions[example.benchmark][example.sample_id] = prediction
            cider_references[example.benchmark][example.sample_id] = example.answers

        row = {
            "sample_id": example.sample_id,
            "benchmark": example.benchmark,
            "split": example.split,
            "task": example.task,
            "metric": metric,
            "question": example.question,
            "references": example.answers,
            "prediction": prediction,
            "inference_error": inference_error,
            "score": sample_score,
        }
        rows.append(row)
        row_by_id[example.sample_id] = row
        grouped_rows[(example.benchmark, metric)].append(row)

    cider_corpus_scores: dict[str, float] = {}
    for (benchmark, metric), benchmark_rows in grouped_rows.items():
        if metric not in {"cider", "cider_ja"}:
            continue
        sample_ids = [row["sample_id"] for row in benchmark_rows]
        benchmark_predictions = {
            sample_id: cider_predictions[benchmark][sample_id] for sample_id in sample_ids
        }
        benchmark_references = {
            sample_id: cider_references[benchmark][sample_id] for sample_id in sample_ids
        }
        if metric == "cider_ja":
            result = japanese_cider_score(
                list(benchmark_predictions.values()), list(benchmark_references.values())
            )
            corpus_score = result.score
            cider_sample_scores = dict(zip(sample_ids, result.per_example, strict=True))
        else:
            corpus_score, cider_sample_scores = _compute_cider(
                benchmark_predictions, benchmark_references
            )
        cider_corpus_scores[benchmark] = corpus_score
        for sample_id, sample_score in cider_sample_scores.items():
            row_by_id[sample_id]["score"] = sample_score

    by_benchmark: dict[str, Any] = {}
    for (benchmark, metric), benchmark_rows in sorted(grouped_rows.items()):
        values = [float(row["score"]) for row in benchmark_rows]
        if metric in {"cider", "cider_ja"}:
            # CIDEr is a corpus metric: use the scorer's corpus result rather
            # than the mean of diagnostic per-image scores.
            raw_score = cider_corpus_scores[benchmark]
        else:
            raw_score = sum(values) / len(values)
        by_benchmark[benchmark] = {
            "metric": metric,
            "num_samples": len(benchmark_rows),
            "score": raw_score,
            # MoshiVis plots accuracy and CIDEr in the familiar 0--100 scale.
            "paper_score": raw_score * 100.0,
        }

    current_scoring_provenance = scoring_provenance()
    scoring_protocol_verified = bool(
        suite_contract is not None
        and suite_contract.get("scoring_protocol") == current_scoring_provenance
    )
    run_complete = bool(
        prediction_run is not None
        and prediction_run.get("status") == "complete"
        and missing == 0
        and failed == 0
        and not extra_ids
    )
    allowed_modalities: set[str] = set()
    if suite_contract is not None:
        configured_modalities = suite_contract.get("paper_input_modalities")
        if isinstance(configured_modalities, list):
            allowed_modalities.update(str(value) for value in configured_modalities)
        configured_modality = suite_contract.get("paper_input_modality")
        if configured_modality is not None:
            allowed_modalities.add(str(configured_modality))
    text_generation_limit_verified = bool(
        suite_contract is not None
        and prediction_run is not None
        and (
            prediction_run.get("input_modality") != "text"
            or (
                suite_contract.get("text_max_new_tokens") is not None
                and prediction_run.get("max_new_tokens")
                == suite_contract["text_max_new_tokens"]
            )
        )
    )
    asset_inventory_verified = False
    if suite_contract is not None and prediction_run is not None:
        frozen_inventories = suite_contract.get("frozen_asset_inventories")
        if frozen_inventories is None:
            # Legacy v1 predates modality-specific frozen inventories.
            asset_inventory_verified = True
        elif isinstance(frozen_inventories, dict):
            expected_inventory = frozen_inventories.get(
                prediction_run.get("input_modality")
            )
            actual_inventory = {
                "asset_inventory_sha256": prediction_run.get(
                    "asset_inventory_sha256"
                ),
                "num_unique_images": prediction_run.get("num_unique_images"),
                "num_unique_audio": prediction_run.get("num_unique_audio"),
            }
            asset_inventory_verified = (
                isinstance(expected_inventory, dict)
                and actual_inventory == expected_inventory
            )
    paper_ready = bool(
        run_complete
        and suite_contract is not None
        and prediction_run is not None
        and prediction_run.get("suite_lock_sha256")
        == suite_contract["suite_lock_sha256"]
        and prediction_run.get("suite_id") == suite_contract["suite_id"]
        and prediction_run.get("generation") == "greedy"
        and prediction_run.get("generation") == suite_contract["paper_generation"]
        and (
            not allowed_modalities
            or prediction_run.get("input_modality") in allowed_modalities
        )
        and prediction_run.get("response_seconds")
        == suite_contract["response_tail_seconds"]
        and prediction_run.get("response_overlap_seconds")
        == suite_contract["response_overlap_seconds"]
        and text_generation_limit_verified
        and asset_inventory_verified
        and not prediction_run.get("diagnostic")
        and not prediction_run.get("checkpoint_hash_skipped")
        and bool(prediction_run.get("checkpoint_sha256"))
        and scoring_protocol_verified
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "MoshiVis/Babillage classical vision benchmark metrics",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "by_benchmark": by_benchmark,
        "num_samples": len(examples),
        "num_missing_predictions": missing,
        "num_failed_predictions": failed,
        "num_extra_predictions": len(extra_ids),
        "extra_prediction_ids": extra_ids[:100],
        "ready_for_scoring": run_complete,
        "paper_ready": paper_ready,
        "paper_guard": (
            "Frozen suite, asset inventory, scorer, complete greedy run, modality, "
            "response window, and checkpoint provenance all match."
            if paper_ready
            else "Diagnostic result only: direct comparison requires --suite-lock and "
                 "a complete non-diagnostic tracked greedy run whose suite, modality, "
                 "response window, asset inventory, checkpoint hash, scorer code, and "
                 "dependencies match."
        ),
        "metric_notes": {
            "ocr_vqa_accuracy": "lowercase-only full-string exact match; whitespace and punctuation retained",
            "vqav2_accuracy": "official ten-annotator leave-one-out VQAv2 consensus accuracy",
            "cider": "pycocoevalcap PTBTokenizer + CIDEr; score is raw, paper_score is x100",
            "cider_ja": "NFKC + fugashi/UniDic Japanese words, punctuation removed + pycocoevalcap CIDEr; adapted metric",
            "prediction_source": "synchronized model text tokens; output audio is not transcribed",
        },
        "scoring_provenance": current_scoring_provenance,
        "scoring_protocol_verified": scoring_protocol_verified,
        "text_generation_limit_verified": text_generation_limit_verified,
        "asset_inventory_verified": asset_inventory_verified,
        "suite_contract": suite_contract,
    }
    if prediction_run is None:
        summary["prediction_run"] = None
    else:
        summary["prediction_run"] = {
            "path": prediction_run["_sidecar_path"],
            "sha256": prediction_run["_sidecar_sha256"],
            "status": prediction_run.get("status"),
            "run_fingerprint": prediction_run.get("run_fingerprint"),
            "generation": prediction_run.get("generation"),
            "input_modality": prediction_run.get("input_modality"),
            "max_new_tokens": prediction_run.get("max_new_tokens"),
            "checkpoint": prediction_run.get("checkpoint"),
            "checkpoint_sha256": prediction_run.get("checkpoint_sha256"),
            "asset_inventory_sha256": prediction_run.get(
                "asset_inventory_sha256"
            ),
            "num_unique_images": prediction_run.get("num_unique_images"),
            "num_unique_audio": prediction_run.get("num_unique_audio"),
        }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--suite-lock",
        type=Path,
        help="Frozen MoshiVis/Babillage suite lock; omitted results are diagnostic only.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--allow-untracked-predictions",
        action="store_true",
        help="Diagnostic only: score without a finalized .run.json provenance sidecar.",
    )
    args = parser.parse_args()

    rows, summary = evaluate_files(
        args.manifest,
        args.predictions,
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
    for benchmark, result in summary["by_benchmark"].items():
        print(
            f"{benchmark}: {result['metric']}="
            f"{result['paper_score']:.2f} ({result['num_samples']} samples)"
        )
    print(f"scores:  {args.scores}")
    print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
