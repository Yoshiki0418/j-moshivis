"""Version-dispatched suite locks for MoshiVis/Babillage evaluation.

The historical :mod:`suite` module is byte-frozen by ``vision_v1`` and must
not change.  This module preserves that contract by delegating v1 locks to the
legacy verifier and implements the reusable v2 contract used by classical
MoshiVis benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .audio_protocol import (
    ASSISTANT_PRELUDE_PROTOCOL,
    BARE_QUESTION_PROTOCOL,
)
from .schema import (
    EvalExample,
    ManifestError,
    read_manifest,
    sha256_file,
    validate_question_audio_provenance,
)
from .suite import SUITE_LOCK_SCHEMA as LEGACY_SUITE_LOCK_SCHEMA
from .suite import verify_suite_manifest as verify_legacy_suite_manifest


SUITE_LOCK_SCHEMA_V2 = "jmoshivis.image-eval.suite-lock.v2"
SUPPORTED_SUITE_LOCK_SCHEMAS = (
    LEGACY_SUITE_LOCK_SCHEMA,
    SUITE_LOCK_SCHEMA_V2,
)
PAPER_INPUT_MODALITIES = frozenset(("audio", "text"))
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def load_suite_lock(lock_path: Path) -> dict[str, Any]:
    """Load a suite lock as a JSON object with a supported schema marker."""

    if not lock_path.is_file():
        raise ManifestError(f"suite lock not found: {lock_path}")
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ManifestError(f"invalid suite lock {lock_path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"suite lock must be a JSON object: {lock_path}")
    schema = value.get("schema_version")
    if schema not in SUPPORTED_SUITE_LOCK_SCHEMAS:
        raise ManifestError(f"unsupported suite lock schema: {schema!r}")
    return value


def benchmark_composition(examples: Iterable[EvalExample]) -> dict[str, Any]:
    """Return deterministic benchmark/task/split counts for a manifest."""

    benchmark_counts: Counter[str] = Counter()
    task_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        benchmark_counts[example.benchmark] += 1
        task_counts[example.benchmark][example.task] += 1
        split_counts[example.benchmark][example.split] += 1
    return {
        benchmark: {
            "questions": benchmark_counts[benchmark],
            "tasks": dict(sorted(task_counts[benchmark].items())),
            "splits": dict(sorted(split_counts[benchmark].items())),
        }
        for benchmark in sorted(benchmark_counts)
    }


def verified_asset_inventories(
    examples: Iterable[EvalExample],
    manifest_path: Path,
    *,
    modalities: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Verify assets once and return one deterministic inventory per modality.

    A manifest digest alone does not freeze files referenced by its paths.  The
    returned digests bind every sample ID to the actual image bytes and, for
    audio input, the prompt-bound question-audio bytes.  Image files are hashed
    once even when both audio and text protocols are frozen.
    """

    selected = tuple(dict.fromkeys(str(value) for value in modalities))
    if not selected or set(selected) - PAPER_INPUT_MODALITIES:
        raise ValueError("modalities must contain only 'audio' and/or 'text'")
    image_hashes: dict[Path, str] = {}
    audio_hashes: dict[Path, str] = {}
    inventories = {modality: hashlib.sha256() for modality in selected}
    for example in examples:
        image = example.resolve_image(manifest_path).resolve()
        if not image.is_file():
            raise ManifestError(f"{example.sample_id}: image not found: {image}")
        if image not in image_hashes:
            image_hashes[image] = sha256_file(image)
        image_digest = image_hashes[image]
        declared_image = example.source.get("image_sha256")
        if not declared_image:
            raise ManifestError(
                f"{example.sample_id}: source.image_sha256 is required"
            )
        if str(declared_image) != image_digest:
            raise ManifestError(
                f"{example.sample_id}: image checksum mismatch: {image}"
            )

        audio_digest: str | None = None
        if "audio" in inventories:
            audio = example.resolve_audio(manifest_path)
            if audio is None:
                raise ManifestError(f"{example.sample_id}: audio_path is required")
            audio = audio.resolve()
            if not audio.is_file():
                raise ManifestError(f"{example.sample_id}: audio not found: {audio}")
            validate_question_audio_provenance(example)
            if audio not in audio_hashes:
                audio_hashes[audio] = sha256_file(audio)
            audio_digest = audio_hashes[audio]
            declared_audio = example.metadata.get("question_audio", {}).get(
                "wav_sha256"
            )
            if not declared_audio:
                raise ManifestError(
                    f"{example.sample_id}: "
                    "metadata.question_audio.wav_sha256 is required"
                )
            if str(declared_audio) != audio_digest:
                raise ManifestError(
                    f"{example.sample_id}: audio checksum mismatch: {audio}"
                )
        for modality, inventory in inventories.items():
            modality_audio_digest = (
                audio_digest if modality == "audio" else "text-input"
            )
            inventory.update(
                f"{example.sample_id}\0{image_digest}\0{modality_audio_digest}\n".encode(
                    "utf-8"
                )
            )
    return {
        modality: {
            "asset_inventory_sha256": inventory.hexdigest(),
            "num_unique_images": len(image_hashes),
            "num_unique_audio": len(audio_hashes) if modality == "audio" else 0,
        }
        for modality, inventory in inventories.items()
    }


def verified_asset_inventory(
    examples: Iterable[EvalExample],
    manifest_path: Path,
    *,
    require_audio: bool,
) -> dict[str, Any]:
    """Compatibility wrapper returning one verified modality inventory."""

    modality = "audio" if require_audio else "text"
    return verified_asset_inventories(
        examples,
        manifest_path,
        modalities=(modality,),
    )[modality]


def _require_sha256(value: Any, field: str) -> str:
    digest = str(value)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _verify_v2_suite(
    lock: dict[str, Any], lock_path: Path, manifest_path: Path
) -> dict[str, Any]:
    try:
        suite_id = str(lock["suite_id"])
        frozen_manifest = lock["manifest"]
        expected_manifest_sha256 = _require_sha256(
            frozen_manifest["sha256"], "manifest.sha256"
        )
        expected_questions = int(frozen_manifest["questions"])
        expected_composition = frozen_manifest["benchmark_composition"]
        raw_asset_inventories = frozen_manifest["frozen_asset_inventories"]
        scope = lock["scope"]
        paper_generation = str(scope["paper_generation"])
        response_tail_seconds = float(scope["response_tail_seconds"])
        response_overlap_seconds = float(scope["response_overlap_seconds"])
        text_max_new_tokens = int(scope["text_max_new_tokens"])
        raw_modalities = scope["paper_input_modalities"]
        audio_protocol = str(scope["audio_dialogue_protocol"])
        scoring_protocol = lock["scoring_protocol"]
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("suite v2 lock has an invalid required contract") from error

    if not suite_id:
        raise ManifestError("suite v2 suite_id must not be empty")
    if expected_questions < 1:
        raise ManifestError("manifest.questions must be positive")
    if not isinstance(expected_composition, dict) or not expected_composition:
        raise ManifestError("manifest.benchmark_composition must be a non-empty object")
    if paper_generation != "greedy":
        raise ManifestError("suite v2 paper_generation must be 'greedy'")
    if response_tail_seconds <= 0:
        raise ManifestError("scope.response_tail_seconds must be positive")
    if not 0 <= response_overlap_seconds < response_tail_seconds:
        raise ManifestError(
            "scope.response_overlap_seconds must be non-negative and below the tail"
        )
    if text_max_new_tokens < 1:
        raise ManifestError("scope.text_max_new_tokens must be positive")
    if not isinstance(raw_modalities, list) or not raw_modalities:
        raise ManifestError("scope.paper_input_modalities must be a non-empty list")
    modalities = [str(value) for value in raw_modalities]
    if len(set(modalities)) != len(modalities):
        raise ManifestError("scope.paper_input_modalities contains duplicates")
    unsupported = sorted(set(modalities) - PAPER_INPUT_MODALITIES)
    if unsupported:
        raise ManifestError(
            "unsupported paper input modalities: " + ", ".join(unsupported)
        )
    if not isinstance(raw_asset_inventories, dict) or set(
        raw_asset_inventories
    ) != set(modalities):
        raise ManifestError(
            "manifest.frozen_asset_inventories must contain exactly the frozen "
            "paper input modalities"
        )
    asset_inventories: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        raw_inventory = raw_asset_inventories[modality]
        if not isinstance(raw_inventory, dict):
            raise ManifestError(
                f"manifest.frozen_asset_inventories.{modality} must be an object"
            )
        try:
            digest = _require_sha256(
                raw_inventory["asset_inventory_sha256"],
                "manifest.frozen_asset_inventories."
                f"{modality}.asset_inventory_sha256",
            )
            unique_images = int(raw_inventory["num_unique_images"])
            unique_audio = int(raw_inventory["num_unique_audio"])
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(
                f"manifest.frozen_asset_inventories.{modality} is invalid"
            ) from error
        if unique_images < 1 or unique_audio < 0:
            raise ManifestError(
                f"manifest.frozen_asset_inventories.{modality} counts are invalid"
            )
        if modality == "text" and unique_audio != 0:
            raise ManifestError(
                "manifest.frozen_asset_inventories.text must not include audio"
            )
        if modality == "audio" and unique_audio < 1:
            raise ManifestError(
                "manifest.frozen_asset_inventories.audio must include audio"
            )
        asset_inventories[modality] = {
            "asset_inventory_sha256": digest,
            "num_unique_images": unique_images,
            "num_unique_audio": unique_audio,
        }
    if audio_protocol != BARE_QUESTION_PROTOCOL:
        if audio_protocol == ASSISTANT_PRELUDE_PROTOCOL:
            detail = (
                "assistant audio is injected without teacher-forcing its synchronized "
                "text; use it only with run_inference --diagnostic against a "
                f"{BARE_QUESTION_PROTOCOL} suite"
            )
        else:
            detail = f"unsupported protocol {audio_protocol!r}"
        raise ManifestError(
            f"suite v2 paper audio protocol must be {BARE_QUESTION_PROTOCOL}: {detail}"
        )
    if not isinstance(scoring_protocol, dict):
        raise ManifestError("scoring_protocol must be an object")
    for field in (
        "implementation_file_sha256",
        "dependency_versions",
        "dependency_artifact_sha256",
    ):
        if not isinstance(scoring_protocol.get(field), dict):
            raise ManifestError(f"scoring_protocol.{field} must be an object")

    assistant = scope.get("assistant_prelude")
    assistant_digest: str | None = None
    if assistant is not None:
        raise ManifestError(
            f"scope.assistant_prelude is invalid for {BARE_QUESTION_PROTOCOL}"
        )

    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ManifestError(
            "manifest is not the frozen suite artifact: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    examples = read_manifest(manifest_path)
    if len(examples) != expected_questions:
        raise ManifestError(
            "manifest question count does not match suite v2: "
            f"expected {expected_questions}, got {len(examples)}"
        )
    actual_composition = benchmark_composition(examples)
    if actual_composition != expected_composition:
        raise ManifestError("manifest benchmark composition does not match suite v2")
    return {
        "suite_id": suite_id,
        "suite_lock_schema": SUITE_LOCK_SCHEMA_V2,
        "suite_lock": str(lock_path.resolve()),
        "suite_lock_sha256": sha256_file(lock_path),
        "frozen_manifest_sha256": expected_manifest_sha256,
        "frozen_num_questions": expected_questions,
        "frozen_benchmark_composition": expected_composition,
        "frozen_asset_inventories": asset_inventories,
        "paper_generation": paper_generation,
        "paper_input_modalities": modalities,
        "response_tail_seconds": response_tail_seconds,
        "response_overlap_seconds": response_overlap_seconds,
        "text_max_new_tokens": text_max_new_tokens,
        "frozen_audio_dialogue_protocol": audio_protocol,
        "frozen_assistant_prelude_wav_sha256": assistant_digest,
        "scoring_protocol": scoring_protocol,
    }


def verify_evaluation_suite(
    lock_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Verify either a frozen legacy v1 lock or a MoshiVis suite-lock v2."""

    lock = load_suite_lock(lock_path)
    if lock["schema_version"] == LEGACY_SUITE_LOCK_SCHEMA:
        contract = verify_legacy_suite_manifest(lock_path, manifest_path)
        # v1 froze only audio input.  Adding these common dispatch fields does
        # not mutate the historical lock or its byte-pinned verifier.
        return {
            **contract,
            "suite_lock_schema": LEGACY_SUITE_LOCK_SCHEMA,
            "paper_input_modalities": ["audio"],
        }
    return _verify_v2_suite(lock, lock_path, manifest_path)


def suite_allows_paper_input(
    suite_contract: dict[str, Any], input_modality: str
) -> bool:
    """Return whether a non-diagnostic run may use ``input_modality``."""

    modalities = suite_contract.get("paper_input_modalities", ())
    return input_modality in modalities
