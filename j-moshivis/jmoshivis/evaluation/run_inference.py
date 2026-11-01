"""Run resumable audio-prompt image evaluation with J-MoshiVis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .audio_protocol import (
    ASSISTANT_PRELUDE_PROTOCOL,
    AUDIO_DIALOGUE_PROTOCOLS,
    BARE_QUESTION_PROTOCOL,
    audio_protocol_contract_from_lock,
    suite_freezes_audio_protocol,
)
from .response import EXTRACTION_VERSION
from .schema import (
    ManifestError,
    read_manifest,
    sha256_file,
    validate_question_audio_provenance,
)
from .suite import DEFAULT_SUITE_LOCK
from .suite_moshivis import suite_allows_paper_input, verify_evaluation_suite


# The moshi package has its own lazy torch.compile wrapper.  Set both guards
# before importing runtime/model modules; this environment intentionally runs
# the benchmark in eager mode.
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


PACKAGE_PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
# Hash the full local package rather than maintaining a brittle hand-written
# import list.  Model loading reaches config, image preprocessing, utility and
# streaming modules indirectly; a missed file would otherwise permit a resume
# with different executable code under the same run fingerprint.
IMPLEMENTATION_FILES = tuple(
    sorted((PACKAGE_PROJECT / "jmoshivis").rglob("*.py"), key=lambda path: str(path))
)
DEPENDENCY_DISTRIBUTIONS = (
    "torch", "moshi", "numpy", "soundfile", "sentencepiece",
    "safetensors", "huggingface-hub", "transformers", "pillow",
    "torchvision", "PyYAML", "einops",
)
CUDA_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "TORCH_CUDNN_V8_API_DISABLED",
)
RUN_INVARIANT_KEYS = (
    "manifest_sha256", "checkpoint", "checkpoint_sha256", "checkpoint_size",
    "suite_id", "suite_lock_sha256", "frozen_manifest_sha256", "frozen_num_questions",
    "config_sha256", "tokenizer_sha256", "mimi_weight", "device",
    "mimi_weight_sha256", "asset_inventory_sha256", "num_unique_images",
    "num_unique_audio",
    "input_modality", "max_new_tokens",
    "audio_dialogue_protocol", "assistant_prelude_wav",
    "assistant_prelude_wav_sha256", "assistant_prelude_wav_size",
    "response_seconds", "response_overlap_seconds", "generation", "seed",
    "sampling_config", "diagnostic", "checkpoint_hash_skipped",
    "runtime_version", "extraction_version", "implementation_file_sha256",
    "dependency_versions", "python", "python_executable", "python_implementation",
    "platform", "runtime_environment", "no_torch_compile", "torchdynamo_disable",
)


@dataclass(frozen=True)
class _PredictionScan:
    """Validated prediction prefix plus an optional recoverable final fragment."""

    completed: frozenset[str]
    attempts: dict[str, int]
    rows: tuple[tuple[int, dict[str, Any]], ...]
    observed_size: int
    observed_sha256: str | None
    committed_size: int
    tail_action: Literal["truncate", "append_newline"] | None = None
    tail_size: int = 0


def _implementation_hashes() -> dict[str, str]:
    return {
        str(path.resolve().relative_to(WORKSPACE.resolve())): sha256_file(path)
        for path in IMPLEMENTATION_FILES
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _nvidia_smi_inventory() -> dict[str, Any]:
    """Read stable GPU/driver identity without creating a CUDA context here."""

    fields = "index,uuid,name,driver_version,memory.total"
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    stdout = process.stdout.strip()
    stderr = process.stderr.strip()
    return {
        "available": process.returncode == 0,
        "query_fields": fields.split(","),
        "returncode": process.returncode,
        "gpus": (
            [line.strip() for line in stdout.splitlines() if line.strip()]
            if process.returncode == 0
            else []
        ),
        "error": "\n".join(value for value in (stderr, stdout) if value)
        if process.returncode != 0
        else "",
    }


def _runtime_environment() -> dict[str, Any]:
    """Capture runtime identity while avoiding CUDA device API calls.

    Importing torch is already required by ``runtime`` before this function is
    called.  We inspect build metadata only; GPU identity comes from the
    out-of-process ``nvidia-smi`` query, so this function does not select a
    device or create a CUDA context in the evaluator process.
    """

    try:
        import torch

        torch_config = torch.__config__.show()
        torch_build: dict[str, Any] = {
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "cuda_build": getattr(torch.version, "cuda", None),
            "hip_build": getattr(torch.version, "hip", None),
            "build_config": torch_config,
            "build_config_sha256": hashlib.sha256(torch_config.encode("utf-8")).hexdigest(),
        }
    except Exception as exc:  # Metadata should remain inspectable on CPU-only hosts.
        torch_build = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "torch_build": torch_build,
        "cuda_environment": {key: os.environ.get(key) for key in CUDA_ENVIRONMENT_KEYS},
        "nvidia_smi": _nvidia_smi_inventory(),
    }


def _run_fingerprint(run_config: dict[str, Any]) -> str:
    invariant_payload = {key: run_config[key] for key in RUN_INVARIANT_KEYS}
    return hashlib.sha256(
        json.dumps(
            invariant_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_nonfinite_json_constant)


def _decode_prediction_row(path: Path, line_number: int, raw: bytes) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{path}:{line_number}: invalid UTF-8: {exc}") from exc
    try:
        value = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"{path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path}:{line_number}: expected a JSON object")
    return value


def _scan_predictions(
    path: Path,
    *,
    allow_incomplete_final_row: bool = False,
) -> _PredictionScan:
    """Strictly validate JSONL, optionally quarantining only a torn final row.

    A serializer-produced row always starts with ``{`` and never contains a
    literal newline.  Consequently, bytes after the final newline can be
    recovered without weakening validation of any earlier line.  A complete
    JSON object missing only its newline is retained; malformed interior rows
    are never repaired or skipped.
    """

    if not path.exists():
        return _PredictionScan(frozenset(), {}, (), 0, None, 0)

    data = path.read_bytes()
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if data.endswith(b"\n"):
        prefix = data
        tail = b""
    else:
        boundary = data.rfind(b"\n") + 1
        prefix = data[:boundary]
        tail = data[boundary:]

    rows: list[tuple[int, dict[str, Any]]] = []
    # ``prefix`` either ends in a newline or is empty.  Splitting explicitly
    # on LF matches the writer and gives stable physical line numbers.
    for line_number, raw in enumerate(prefix.split(b"\n")[:-1], 1):
        value = _decode_prediction_row(path, line_number, raw)
        if value is None:
            raise ManifestError(f"{path}:{line_number}: empty prediction row")
        rows.append((line_number, value))

    tail_action: Literal["truncate", "append_newline"] | None = None
    committed_size = len(prefix)
    if tail:
        tail_line = prefix.count(b"\n") + 1
        try:
            value = _decode_prediction_row(path, tail_line, tail)
            if value is None:
                raise ManifestError(f"{path}:{tail_line}: empty prediction row")
        except ManifestError:
            # Do not turn arbitrary trailing corruption into a retry.  Only a
            # prefix that could have come from our object serializer is a
            # recoverable, interrupted append.
            if not allow_incomplete_final_row or not tail.lstrip().startswith(b"{"):
                raise
            tail_action = "truncate"
        else:
            rows.append((tail_line, value))
            # The object itself is durable; normalize the missing delimiter
            # rather than discarding and rerunning it.  For a finalized run,
            # the caller first verifies the sidecar's complete-file hash.
            tail_action = "append_newline"
            committed_size = len(data)

    completed: set[str] = set()
    attempts: dict[str, int] = {}
    for line_number, value in rows:
        sample_id = value.get("sample_id")
        if not sample_id:
            raise ValueError(f"{path}:{line_number}: prediction has no sample_id")
        sample_id = str(sample_id)
        if sample_id in completed:
            raise ValueError(
                f"{path}:{line_number}: prediction row appears after a successful "
                f"attempt for {sample_id!r}"
            )
        attempts[sample_id] = attempts.get(sample_id, 0) + 1
        if not value.get("error"):
            completed.add(sample_id)
    return _PredictionScan(
        frozenset(completed),
        attempts,
        tuple(rows),
        len(data),
        observed_sha256,
        committed_size,
        tail_action,
        len(tail),
    )


def _prediction_state(path: Path) -> tuple[set[str], dict[str, int]]:
    scan = _scan_predictions(path)
    return set(scan.completed), scan.attempts


def _fsync_directory(path: Path) -> None:
    """Persist a create/replace where the filesystem supports directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some network filesystems reject directory fsync even though file
        # fsync is supported.  The data file itself has already been synced.
        pass
    finally:
        os.close(descriptor)


def _repair_prediction_tail(path: Path, scan: _PredictionScan) -> str | None:
    """Apply a previously validated tail repair, refusing concurrent changes."""

    if scan.tail_action is None:
        return None
    if not path.is_file():
        raise RuntimeError(f"prediction file disappeared before tail recovery: {path}")
    current = path.read_bytes()
    if (
        len(current) != scan.observed_size
        or hashlib.sha256(current).hexdigest() != scan.observed_sha256
    ):
        raise RuntimeError("prediction file changed while preparing tail recovery")

    with path.open("r+b", buffering=0) as handle:
        if scan.tail_action == "truncate":
            handle.truncate(scan.committed_size)
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return scan.tail_action


def _validate_prediction_rows_fingerprint(
    path: Path,
    rows: tuple[tuple[int, dict[str, Any]], ...],
    expected: str,
) -> None:
    for line_number, value in rows:
        if value.get("run_fingerprint") != expected:
            raise ValueError(
                f"{path}:{line_number}: prediction run_fingerprint does not match "
                "the current run metadata"
            )


def _validate_prediction_fingerprints(path: Path, expected: str) -> None:
    if not path.exists():
        return
    _validate_prediction_rows_fingerprint(path, _scan_predictions(path).rows, expected)


def _asset_inventory(
    examples,
    manifest_path: Path,
    *,
    require_audio: bool = True,
) -> dict[str, object]:
    """Hash and verify all immutable inputs used by this run.

    Hashing the manifest alone is insufficient: its paths may still point to
    mutable image/WAV files.  The aggregate digests are stored in run metadata
    and therefore also guard ``--resume``.
    """

    image_hashes: dict[Path, str] = {}
    audio_hashes: dict[Path, str] = {}
    inventory = hashlib.sha256()
    for example in examples:
        image = example.resolve_image(manifest_path).resolve()
        if image not in image_hashes:
            image_hashes[image] = sha256_file(image)
        image_digest = image_hashes[image]
        declared_image = example.source.get("image_sha256")
        if not declared_image:
            raise ValueError(f"{example.sample_id}: source.image_sha256 is required")
        if str(declared_image) != image_digest:
            raise ValueError(f"{example.sample_id}: image checksum mismatch: {image}")

        audio_digest = "text-input"
        if require_audio:
            audio = example.resolve_audio(manifest_path)
            if audio is None:
                raise ValueError(f"{example.sample_id}: audio_path is required")
            validate_question_audio_provenance(example)
            audio = audio.resolve()
            if audio not in audio_hashes:
                audio_hashes[audio] = sha256_file(audio)
            audio_digest = audio_hashes[audio]
            declared_audio = example.metadata.get("question_audio", {}).get("wav_sha256")
            if not declared_audio:
                raise ValueError(f"{example.sample_id}: metadata.question_audio.wav_sha256 is required")
            if str(declared_audio) != audio_digest:
                raise ValueError(f"{example.sample_id}: audio checksum mismatch: {audio}")
        inventory.update(
            f"{example.sample_id}\0{image_digest}\0{audio_digest}\n".encode("utf-8")
        )
    return {
        "asset_inventory_sha256": inventory.hexdigest(),
        "num_unique_images": len(image_hashes),
        "num_unique_audio": len(audio_hashes),
    }


def _verify_frozen_asset_inventory(
    suite_contract: dict[str, Any],
    input_modality: str,
    actual: dict[str, object],
) -> None:
    """Bind the one inference-time asset scan to its frozen v2 inventory."""

    inventories = suite_contract.get("frozen_asset_inventories")
    # Legacy v1 predates modality-specific inventories.  Its existing runtime
    # checksum validation remains unchanged.
    if inventories is None:
        return
    if not isinstance(inventories, dict):
        raise ValueError("frozen_asset_inventories must be an object")
    expected = inventories.get(input_modality)
    # A diagnostic modality need not be among the paper modalities.
    if expected is None:
        return
    if actual != expected:
        raise ValueError(
            f"{input_modality} asset inventory does not match the frozen suite"
        )


def _write_run_metadata(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _append_prediction_row(path: Path, row: dict[str, Any]) -> None:
    """Append and durably commit one JSONL row.

    If the process or host dies during ``os.write``, at worst the last line is
    incomplete.  Resume recovery validates every earlier line and can truncate
    only that suffix.  An exception is propagated immediately, so the process
    never continues writing after a short/failed append.
    """

    payload = (
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    existed = path.exists()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(
                f"short prediction append: wrote {written} of {len(payload)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _finalize_run_metadata(
    run_config: dict,
    run_path: Path,
    output_path: Path,
    manifest_ids: set[str],
    *,
    succeeded: int,
    failed: int,
    aborted_early: bool,
) -> None:
    """Atomically seal output provenance, including after a crash at the last row."""

    final_completed, final_attempts = _prediction_state(output_path)
    run_config["finished_unix"] = time.time()
    run_config["num_succeeded_this_invocation"] = succeeded
    run_config["num_failed_this_invocation"] = failed
    run_config["aborted_early"] = aborted_early
    run_config["predictions_sha256"] = sha256_file(output_path)
    run_config["num_prediction_rows"] = sum(final_attempts.values())
    run_config["num_successful_total"] = len(final_completed)
    run_config["num_manifest_examples"] = len(manifest_ids)
    run_config["status"] = (
        "complete" if manifest_ids.issubset(final_completed) else "incomplete"
    )
    _write_run_metadata(run_path, run_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-lock", type=Path, default=DEFAULT_SUITE_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_PROJECT / "configs/j-moshi-vis.yaml")
    parser.add_argument("--tokenizer", type=Path, default=WORKSPACE / "models/j-moshi/tokenizer_spm_32k_3.model")
    parser.add_argument(
        "--mimi-weight",
        default=str(WORKSPACE / "models/moshika-vis-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"),
        help="Pinned local Mimi checkpoint file (its bytes are hashed).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input-modality",
        choices=("audio", "text"),
        default="audio",
        help="Use the frozen question WAV or the exact rendered prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Text-mode continuation limit (default: 96); ignored in audio mode.",
    )
    parser.add_argument(
        "--audio-dialogue-protocol",
        choices=AUDIO_DIALOGUE_PROTOCOLS,
        default=BARE_QUESTION_PROTOCOL,
        help=(
            "Audio stream layout. vision_v1 remains bare_question_v1; the "
            "assistant-prelude protocol is diagnostic-only because its synchronized "
            "assistant text is not teacher-forced."
        ),
    )
    parser.add_argument(
        "--assistant-prelude-wav",
        type=Path,
        help=(
            "Pinned 24-kHz assistant greeting WAV for "
            "assistant_prelude_audio_only_v2."
        ),
    )
    parser.add_argument("--response-seconds", type=float, default=10.0)
    parser.add_argument("--response-overlap-seconds", type=float, default=0.32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", action="store_true", help="Enable sampling; paper comparisons should keep greedy default.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Mark a non-paper diagnostic run (required with --skip-checkpoint-hash).",
    )
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="Abort early after this many consecutive sample failures (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_contract = verify_evaluation_suite(args.suite_lock, args.manifest)
    suite_contract.update(audio_protocol_contract_from_lock(args.suite_lock))
    all_examples = read_manifest(
        args.manifest,
        check_files=True,
        require_audio=args.input_modality == "audio",
    )
    if len(all_examples) != suite_contract["frozen_num_questions"]:
        raise ValueError(
            f"suite question count mismatch: {len(all_examples)} != "
            f"{suite_contract['frozen_num_questions']}"
        )
    examples = all_examples
    if args.limit is not None:
        examples = examples[: args.limit]
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.skip_checkpoint_hash and not args.diagnostic:
        raise ValueError("--skip-checkpoint-hash is unsafe and requires --diagnostic")
    assistant_prelude_path: Path | None = None
    assistant_prelude_sha256: str | None = None
    assistant_prelude_size: int | None = None
    if args.input_modality == "text":
        if (
            args.audio_dialogue_protocol != BARE_QUESTION_PROTOCOL
            or args.assistant_prelude_wav is not None
        ):
            raise ValueError(
                "audio dialogue protocol options cannot be used with text input"
            )
    elif args.audio_dialogue_protocol == BARE_QUESTION_PROTOCOL:
        if args.assistant_prelude_wav is not None:
            raise ValueError(
                "--assistant-prelude-wav requires "
                f"--audio-dialogue-protocol {ASSISTANT_PRELUDE_PROTOCOL}"
            )
    else:
        if args.assistant_prelude_wav is None:
            raise ValueError(
                f"{ASSISTANT_PRELUDE_PROTOCOL} requires --assistant-prelude-wav"
            )
        if not args.assistant_prelude_wav.is_file():
            raise FileNotFoundError(
                f"assistant prelude WAV not found: {args.assistant_prelude_wav}"
            )
        assistant_prelude_path = args.assistant_prelude_wav.resolve()
        assistant_prelude_sha256 = sha256_file(assistant_prelude_path)
        assistant_prelude_size = assistant_prelude_path.stat().st_size
    if args.input_modality == "audio" and not args.diagnostic and not (
        suite_freezes_audio_protocol(
            suite_contract,
            protocol=args.audio_dialogue_protocol,
            assistant_prelude_sha256=assistant_prelude_sha256,
        )
    ):
        raise ValueError(
            "the selected audio dialogue protocol/prelude WAV is not frozen by "
            "this suite lock; use --diagnostic or create a new suite lock that "
            "pins its protocol identifier and WAV SHA-256"
        )
    if not args.diagnostic and not suite_allows_paper_input(
        suite_contract, args.input_modality
    ):
        raise ValueError(
            f"input modality {args.input_modality!r} is not frozen for paper use by "
            "this suite lock; use --diagnostic or a suite-lock v2 that lists it "
            "in scope.paper_input_modalities"
        )
    if (
        args.input_modality == "text"
        and not args.diagnostic
        and args.max_new_tokens != suite_contract.get("text_max_new_tokens")
    ):
        raise ValueError(
            "non-diagnostic text inference must match the frozen suite "
            "max_new_tokens "
            f"({suite_contract.get('text_max_new_tokens')})"
        )
    requested_generation = "sampling" if args.sampling else "greedy"
    if not args.diagnostic and (
        requested_generation != suite_contract["paper_generation"]
        or args.response_seconds != suite_contract["response_tail_seconds"]
        or args.response_overlap_seconds != suite_contract["response_overlap_seconds"]
    ):
        raise ValueError(
            "non-diagnostic inference must match the frozen suite generation/window "
            f"({suite_contract['paper_generation']}, "
            f"tail={suite_contract['response_tail_seconds']}, "
            f"overlap={suite_contract['response_overlap_seconds']})"
        )
    run_path = args.output.with_suffix(args.output.suffix + ".run.json")
    previous_run: dict[str, Any] | None = None
    resume_scan: _PredictionScan | None = None
    if args.resume:
        if not run_path.is_file():
            raise FileNotFoundError(f"cannot safely resume without run metadata: {run_path}")
        previous_value = _strict_json_loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(previous_value, dict):
            raise ValueError(f"run metadata must be a JSON object: {run_path}")
        previous_run = previous_value
        # Destructive recovery of a malformed tail is restricted to a process
        # that demonstrably died while its sidecar still said ``running``.  A
        # complete row missing only LF can later be normalized after checking
        # a finalized sidecar's complete-file hash.
        resume_scan = _scan_predictions(
            args.output,
            allow_incomplete_final_row=previous_run.get("status") == "running",
        )
        completed, attempts = set(resume_scan.completed), resume_scan.attempts
    else:
        completed, attempts = set(), {}
    manifest_ids = {example.sample_id for example in all_examples}
    unknown_prediction_ids = sorted(set(attempts) - manifest_ids)
    if unknown_prediction_ids:
        raise ValueError(
            "prediction output contains IDs outside the manifest: "
            + ", ".join(unknown_prediction_ids[:5])
        )
    pending = [example for example in examples if example.sample_id not in completed]
    print(
        f"manifest={len(examples)} selected, completed={len(completed)}, pending={len(pending)} | "
        f"modality={args.input_modality}, mode={'sampling' if args.sampling else 'greedy'}"
    )
    if args.dry_run:
        for example in pending[:5]:
            print(f"  {example.sample_id}: {example.question}")
        return 0
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume or choose a new path: {args.output}")

    # Import only after dry-run so data/schema checks never initialise CUDA.
    from .runtime import (
        JmoshiVisAudioRuntime,
        RUNTIME_VERSION as AUDIO_RUNTIME_VERSION,
        SAMPLING_CONFIG,
        RuntimeConfig,
    )
    if args.input_modality == "text":
        from .text_runtime import (
            JmoshiVisTextRuntime,
            TEXT_RUNTIME_VERSION,
            TextRuntimeConfig,
        )
        runtime_version = TEXT_RUNTIME_VERSION
    else:
        runtime_version = AUDIO_RUNTIME_VERSION

    checkpoint_sha256 = None if args.skip_checkpoint_hash else sha256_file(args.checkpoint)
    # Hash the full manifest even for a small --limit smoke run, so the same
    # output can safely continue over the complete set with --resume.
    assets = _asset_inventory(
        all_examples,
        args.manifest,
        require_audio=args.input_modality == "audio",
    )
    _verify_frozen_asset_inventory(suite_contract, args.input_modality, assets)
    mimi_weight: str | None = None
    mimi_sha256: str | None = None
    if args.input_modality == "audio":
        mimi_path = Path(args.mimi_weight)
        if not mimi_path.is_file():
            raise FileNotFoundError(
                "--mimi-weight must be a pinned local file so it can be hashed: "
                f"{args.mimi_weight}"
            )
        mimi_weight = args.mimi_weight
        mimi_sha256 = sha256_file(mimi_path)
    run_config = {
        "schema_version": "jmoshivis.image-eval.run.v2",
        "created_unix": time.time(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        **suite_contract,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": args.checkpoint.stat().st_size,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "mimi_weight": mimi_weight,
        "mimi_weight_sha256": mimi_sha256,
        "device": args.device,
        "input_modality": args.input_modality,
        "max_new_tokens": args.max_new_tokens if args.input_modality == "text" else None,
        "audio_dialogue_protocol": (
            args.audio_dialogue_protocol if args.input_modality == "audio" else None
        ),
        "assistant_prelude_wav": (
            str(assistant_prelude_path) if assistant_prelude_path is not None else None
        ),
        "assistant_prelude_wav_sha256": assistant_prelude_sha256,
        "assistant_prelude_wav_size": assistant_prelude_size,
        "response_seconds": args.response_seconds,
        "response_overlap_seconds": args.response_overlap_seconds,
        "generation": requested_generation,
        "sampling_config": SAMPLING_CONFIG,
        "seed": args.seed,
        "diagnostic": args.diagnostic,
        "checkpoint_hash_skipped": args.skip_checkpoint_hash,
        "runtime_version": runtime_version,
        "extraction_version": EXTRACTION_VERSION,
        "implementation_file_sha256": _implementation_hashes(),
        "dependency_versions": _dependency_versions(),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "runtime_environment": _runtime_environment(),
        "no_torch_compile": os.environ.get("NO_TORCH_COMPILE"),
        "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"),
        **assets,
    }
    run_fingerprint = _run_fingerprint(run_config)
    run_config["run_fingerprint"] = run_fingerprint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert previous_run is not None
        assert resume_scan is not None
        previous = previous_run
        changed = [
            key for key in RUN_INVARIANT_KEYS
            if previous.get(key) != run_config.get(key)
        ]
        if changed:
            details = ", ".join(
                f"{key}: {previous.get(key)!r} -> {run_config.get(key)!r}" for key in changed
            )
            raise RuntimeError(f"resume configuration mismatch ({details})")
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("resume run_fingerprint mismatch")
        # Validate the complete prefix before changing even the recoverable
        # suffix.  This prevents tail repair from obscuring an independent
        # provenance failure in an earlier row.
        _validate_prediction_rows_fingerprint(
            args.output,
            resume_scan.rows,
            run_fingerprint,
        )
        previous_predictions_sha256 = previous.get("predictions_sha256")
        if previous.get("status") != "running":
            if not previous_predictions_sha256:
                raise RuntimeError(
                    "cannot safely resume finalized predictions without their checksum"
                )
            if resume_scan.observed_sha256 != previous_predictions_sha256:
                raise RuntimeError("prediction file changed after the previous invocation finished")
        repair = _repair_prediction_tail(args.output, resume_scan)
        if repair == "truncate":
            print(
                f"recovered interrupted final prediction append by removing "
                f"{resume_scan.tail_size} uncommitted bytes",
                file=sys.stderr,
                flush=True,
            )
        elif repair == "append_newline":
            print(
                "recovered complete final prediction row by restoring its JSONL newline",
                file=sys.stderr,
                flush=True,
            )
        # Re-read strictly after mutation.  This also detects any concurrent
        # modification that happened after the guarded recovery operation.
        completed, attempts = _prediction_state(args.output)
        _validate_prediction_fingerprints(args.output, run_fingerprint)
        unknown_prediction_ids = sorted(set(attempts) - manifest_ids)
        if unknown_prediction_ids:
            raise ValueError(
                "prediction output contains IDs outside the manifest: "
                + ", ".join(unknown_prediction_ids[:5])
            )
        pending = [example for example in examples if example.sample_id not in completed]
        run_config["created_unix"] = previous.get("created_unix", run_config["created_unix"])
    if not pending:
        # A process may have died after flushing the final prediction row but
        # before sealing the sidecar.  Re-finalize instead of leaving a valid
        # complete result permanently marked as ``running``.
        _finalize_run_metadata(
            run_config,
            run_path,
            args.output,
            manifest_ids,
            succeeded=0,
            failed=0,
            aborted_early=False,
        )
        print("nothing to do; existing predictions verified and run metadata finalized")
        return 0
    run_config["status"] = "running"
    _write_run_metadata(run_path, run_config)

    if args.input_modality == "audio":
        runtime = JmoshiVisAudioRuntime(
            RuntimeConfig(
                checkpoint=args.checkpoint,
                model_config=args.config,
                tokenizer=args.tokenizer,
                mimi_weight=args.mimi_weight,
                audio_dialogue_protocol=args.audio_dialogue_protocol,
                assistant_prelude_wav=assistant_prelude_path,
                device=args.device,
                response_seconds=args.response_seconds,
                response_overlap_seconds=args.response_overlap_seconds,
                use_sampling=args.sampling,
                seed=args.seed,
            )
        )
    else:
        runtime = JmoshiVisTextRuntime(  # type: ignore[assignment, possibly-undefined]
            TextRuntimeConfig(  # type: ignore[possibly-undefined]
                checkpoint=args.checkpoint,
                model_config=args.config,
                tokenizer=args.tokenizer,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                use_sampling=args.sampling,
                seed=args.seed,
            )
        )

    succeeded = 0
    failed = 0
    consecutive_failures = 0
    aborted_early = False
    for index, example in enumerate(pending, 1):
        started = time.time()
        try:
            if args.input_modality == "audio":
                result = runtime.infer(
                    example.resolve_image(args.manifest),
                    example.resolve_audio(args.manifest),  # type: ignore[arg-type]
                    speech_end_sec=example.metadata.get("question_audio", {}).get("speech_end_sec"),
                    sample_id=example.sample_id,
                )
            else:
                result = runtime.infer(
                    example.resolve_image(args.manifest),
                    example.rendered_prompt,
                    sample_id=example.sample_id,
                )
            error = None
            succeeded += 1
            consecutive_failures = 0
        except Exception as exc:  # Keep failures in the denominator.
            result = {"answer_text": "", "generated_text_all": ""}
            error = f"{type(exc).__name__}: {exc}"
            failed += 1
            consecutive_failures += 1
        attempt = attempts.get(example.sample_id, 0) + 1
        attempts[example.sample_id] = attempt
        row = {
            "schema_version": "jmoshivis.image-eval.prediction.v2",
            "sample_id": example.sample_id,
            "benchmark": example.benchmark,
            "runtime_version": runtime_version,
            "input_modality": args.input_modality,
            "extraction_version": EXTRACTION_VERSION,
            "run_fingerprint": run_fingerprint,
            "answer_text": result.pop("answer_text"),
            "error": error,
            "attempt": attempt,
            "wall_time_sec": time.time() - started,
            **result,
        }
        _append_prediction_row(args.output, row)
        print(
            f"[{index}/{len(pending)}] {example.sample_id} "
            f"{'ERROR ' + error if error else row['answer_text'][:80]}",
            flush=True,
        )
        if consecutive_failures >= args.max_consecutive_failures:
            aborted_early = True
            print(
                f"aborting after {consecutive_failures} consecutive failures; "
                "fix the cause and rerun with --resume",
                file=sys.stderr,
                flush=True,
            )
            break
    _finalize_run_metadata(
        run_config,
        run_path,
        args.output,
        manifest_ids,
        succeeded=succeeded,
        failed=failed,
        aborted_early=aborted_early,
    )
    print(f"done: success={succeeded}, failed={failed}, output={args.output}")
    # Any error means the result is incomplete.  A subsequent --resume retries
    # only failed/missing samples while retaining successful ones.
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
