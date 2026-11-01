"""Load and enforce a frozen image-evaluation suite contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ManifestError, sha256_file


PACKAGE_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_LOCK = PACKAGE_PROJECT / "evaluation/suite_v1.lock.json"
SUITE_LOCK_SCHEMA = "jmoshivis.image-eval.suite-lock.v1"


def verify_suite_manifest(lock_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise ManifestError(f"suite lock not found: {lock_path}")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ManifestError(f"invalid suite lock {lock_path}: {exc}") from exc
    if lock.get("schema_version") != SUITE_LOCK_SCHEMA:
        raise ManifestError(f"unsupported suite lock schema: {lock.get('schema_version')!r}")
    try:
        frozen = lock["manifests"]["fixed_audio"]
        expected_manifest_sha256 = str(frozen["sha256"])
        expected_questions = int(frozen["questions"])
        language_id_sha256 = str(
            lock["scoring_artifacts"]["fasttext_language_id"]["sha256"]
        )
        judge_model = str(lock["judge_protocol"]["model"])
        judge_n_repeats = int(lock["judge_protocol"]["n_repeats"])
        judge_template_sha256 = str(lock["judge_protocol"]["grader_template_sha256"])
        paper_generation = str(lock["scope"]["paper_generation"])
        response_tail_seconds = float(lock["scope"]["response_tail_seconds"])
        response_overlap_seconds = float(lock["scope"]["response_overlap_seconds"])
        scoring_protocol = lock["scoring_protocol"]
        judge_implementation = lock["judge_protocol"]["implementation_file_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("suite lock has no valid manifests.fixed_audio contract") from exc
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ManifestError(
            "manifest is not the frozen suite artifact: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    return {
        "suite_id": str(lock.get("suite_id", "")),
        "suite_lock": str(lock_path.resolve()),
        "suite_lock_sha256": sha256_file(lock_path),
        "frozen_manifest_sha256": expected_manifest_sha256,
        "frozen_num_questions": expected_questions,
        "language_id_model_sha256": language_id_sha256,
        "judge_model": judge_model,
        "judge_n_repeats": judge_n_repeats,
        "judge_template_sha256": judge_template_sha256,
        "paper_generation": paper_generation,
        "response_tail_seconds": response_tail_seconds,
        "response_overlap_seconds": response_overlap_seconds,
        "scoring_protocol": scoring_protocol,
        "judge_implementation_file_sha256": judge_implementation,
    }
