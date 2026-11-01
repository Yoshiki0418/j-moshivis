import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import accelerate
import accelerate.checkpointing as accelerate_checkpointing
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator

from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.loss import compute_loss_with_mask
from jmoshivis.conditioners.base import ConditionAttributes, TensorCondition
from jmoshivis.datasets.args import TrainerArgs
from jmoshivis.tools import WandBMetricsWriter
from jmoshivis.models.image_projection import ImageProjection


TRAINING_CHECKPOINT_SCHEMA_VERSION = "jmoshivis.training-checkpoint.v2"
TRAINING_CHECKPOINT_COMPLETE = "COMPLETE.json"
TRAINER_STATE_NAME = "trainer_state.json"
ASSIGNED_DEVICE_RNG_SCHEMA_VERSION = "jmoshivis.assigned-device-rng.v1"
ASSIGNED_DEVICE_RNG_ACCELERATE_VERSION = "1.14.0"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_normalize(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(path.rglob("*")):
        if artifact.name == TRAINING_CHECKPOINT_COMPLETE or artifact.is_dir():
            continue
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"unsupported checkpoint artifact: {artifact}")
        inventory.append(
            {
                "path": artifact.relative_to(path).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )
    return inventory


def _validate_checkpoint_inventory(
    checkpoint: Path,
    marker: dict[str, Any],
    state: dict[str, Any],
) -> None:
    inventory = marker.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"training checkpoint has no artifact inventory: {checkpoint}")
    expected: dict[str, dict[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError(f"invalid checkpoint inventory entry in {checkpoint}")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"invalid checkpoint inventory path in {checkpoint}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe checkpoint inventory path {relative!r}")
        if relative in expected:
            raise ValueError(f"duplicate checkpoint inventory path {relative!r}")
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
            raise ValueError(f"invalid checkpoint inventory size for {relative!r}")
        if not isinstance(item.get("sha256"), str) or re.fullmatch(
            r"[0-9a-f]{64}", item["sha256"]
        ) is None:
            raise ValueError(f"invalid checkpoint inventory hash for {relative!r}")
        expected[relative] = item

    actual_paths = {
        artifact.relative_to(checkpoint).as_posix()
        for artifact in checkpoint.rglob("*")
        if artifact.is_file() and artifact.name != TRAINING_CHECKPOINT_COMPLETE
    }
    if set(expected) != actual_paths:
        missing = sorted(set(expected) - actual_paths)
        untracked = sorted(actual_paths - set(expected))
        raise ValueError(
            f"checkpoint artifact inventory mismatch in {checkpoint}: "
            f"missing={missing}, untracked={untracked}"
        )
    for relative, item in expected.items():
        artifact = checkpoint / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"checkpoint artifact is not a regular file: {artifact}")
        actual_size = artifact.stat().st_size
        if actual_size != item["bytes"]:
            raise ValueError(
                f"checkpoint artifact size mismatch for {relative}: "
                f"expected {item['bytes']}, got {actual_size}"
            )
        actual_sha = _sha256_file(artifact)
        if actual_sha != item["sha256"]:
            raise ValueError(
                f"checkpoint artifact hash mismatch for {relative}: "
                f"expected {item['sha256']}, got {actual_sha}"
            )

    world_size = state.get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"invalid checkpoint world_size in {checkpoint}")
    expected_rng = {f"random_states_{rank}.pkl" for rank in range(world_size)}
    actual_rng = {
        relative
        for relative in expected
        if re.fullmatch(r"random_states_\d+\.pkl", relative)
    }
    if actual_rng != expected_rng:
        raise ValueError(
            f"checkpoint RNG inventory mismatch in {checkpoint}: "
            f"expected {sorted(expected_rng)}, got {sorted(actual_rng)}"
        )

    accelerate_state = state.get("accelerate_state")
    if not isinstance(accelerate_state, dict):
        raise ValueError(f"checkpoint Accelerate expectations missing in {checkpoint}")
    model_files = {
        relative
        for relative in expected
        if re.fullmatch(r"model(?:_\d+)?\.safetensors", relative)
    }
    optimizer_files = {
        relative
        for relative in expected
        if re.fullmatch(r"optimizer(?:_\d+)?\.bin", relative)
    }
    if len(model_files) != accelerate_state.get("models"):
        raise ValueError(f"checkpoint model inventory mismatch in {checkpoint}")
    if len(optimizer_files) != accelerate_state.get("optimizers"):
        raise ValueError(f"checkpoint optimizer inventory mismatch in {checkpoint}")
    if accelerate_state.get("custom_scheduler") and "custom_checkpoint_0.pkl" not in expected:
        raise ValueError(f"checkpoint scheduler state is missing in {checkpoint}")
    if accelerate_state.get("scaler") and "scaler.pt" not in expected:
        raise ValueError(f"checkpoint scaler state is missing in {checkpoint}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    """Flush checkpoint contents before the directory is published."""

    for child in path.rglob("*"):
        if child.is_file():
            descriptor = os.open(child, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directories = [path, *(child for child in path.rglob("*") if child.is_dir())]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _save_accelerate_state_without_global_cuda_rng(
    accelerator: Accelerator, path: Path
) -> None:
    """Save Accelerate state without querying every visible CUDA device.

    Accelerate 1.14.0 calls ``torch.cuda.get_rng_state_all()`` from
    ``save_accelerator_state``.  In a one-process-per-GPU job that initializes
    CUDA contexts on devices not assigned to the process.  Patch the exact
    version's module-level availability check only for the duration of the
    save; the assigned-device RNG is written strictly by
    :func:`_save_rank_rng_state` immediately afterwards.
    """

    if not torch.cuda.is_available():
        accelerator.save_state(str(path), safe_serialization=True)
        return
    if accelerate.__version__ != ASSIGNED_DEVICE_RNG_ACCELERATE_VERSION:
        raise RuntimeError(
            "assigned-device CUDA RNG checkpointing is pinned to Accelerate "
            f"{ASSIGNED_DEVICE_RNG_ACCELERATE_VERSION}, got {accelerate.__version__}; "
            "refusing to risk all-device CUDA context creation"
        )
    original_is_cuda_available = accelerate_checkpointing.is_cuda_available
    accelerate_checkpointing.is_cuda_available = lambda: False
    try:
        accelerator.save_state(str(path), safe_serialization=True)
    finally:
        accelerate_checkpointing.is_cuda_available = original_is_cuda_available


def _load_accelerate_state_without_global_cuda_rng(
    accelerator: Accelerator, path: Path
) -> None:
    """Load Accelerate state without touching unassigned CUDA RNGs.

    The custom rank payload contains an empty ``torch_cuda_manual_seed`` list
    solely because Accelerate 1.14.0 expects that key.  Suppress its global
    setter on CUDA jobs; :func:`_restore_rank_rng_state` restores exactly the
    assigned device immediately after this function returns.
    """

    if not torch.cuda.is_available():
        accelerator.load_state(str(path))
        return
    if accelerate.__version__ != ASSIGNED_DEVICE_RNG_ACCELERATE_VERSION:
        raise RuntimeError(
            "assigned-device CUDA RNG checkpointing is pinned to Accelerate "
            f"{ASSIGNED_DEVICE_RNG_ACCELERATE_VERSION}, got {accelerate.__version__}; "
            "refusing to risk all-device CUDA RNG restoration"
        )
    original_set_rng_state_all = torch.cuda.set_rng_state_all
    torch.cuda.set_rng_state_all = lambda states: None
    try:
        accelerator.load_state(str(path))
    finally:
        torch.cuda.set_rng_state_all = original_set_rng_state_all


def _save_rank_rng_state(path: Path, accelerator: Accelerator) -> None:
    """Atomically write CPU and assigned-device RNG state for one rank."""

    state: dict[str, Any] = {
        "schema_version": ASSIGNED_DEVICE_RNG_SCHEMA_VERSION,
        "process_index": accelerator.process_index,
        "device_type": accelerator.device.type,
        "step": accelerator.step,
        "random_state": random.getstate(),
        "numpy_random_seed": np.random.get_state(),
        "torch_manual_seed": torch.get_rng_state(),
        # Accelerate 1.14.0 unconditionally reads this key and passes it to
        # torch.cuda.set_rng_state_all() while loading, even though save-time
        # CUDA availability was deliberately disabled above.  An empty list is
        # a true no-op (the PyTorch implementation only iterates this value),
        # prevents Accelerate's misleading "Could not load random states"
        # message, and leaves the assigned-device restore below authoritative.
        "torch_cuda_manual_seed": [],
    }
    if accelerator.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Accelerator is assigned {accelerator.device}, but CUDA is unavailable"
            )
        state["assigned_cuda_rng_state"] = torch.cuda.get_rng_state(
            accelerator.device
        )
    output = path / f"random_states_{accelerator.process_index}.pkl"
    temporary = path / f".random_states_{accelerator.process_index}.incomplete"
    torch.save(state, temporary)
    os.replace(temporary, output)


def _read_rank_rng_state(path: Path, accelerator: Accelerator) -> dict[str, Any]:
    """Read and structurally validate this rank's inventoried RNG payload."""

    rng_path = path / f"random_states_{accelerator.process_index}.pkl"
    try:
        state = torch.load(rng_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load rank RNG state {rng_path}: {error}") from error
    if not isinstance(state, dict):
        raise ValueError(f"rank RNG state must be a mapping: {rng_path}")
    if state.get("schema_version") != ASSIGNED_DEVICE_RNG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported rank RNG schema in {rng_path}: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("process_index") != accelerator.process_index:
        raise ValueError(
            f"rank RNG process index mismatch in {rng_path}: "
            f"{state.get('process_index')!r} != {accelerator.process_index}"
        )
    if state.get("device_type") != accelerator.device.type:
        raise ValueError(
            f"rank RNG device type mismatch in {rng_path}: "
            f"{state.get('device_type')!r} != {accelerator.device.type!r}"
        )
    if not isinstance(state.get("step"), int) or state["step"] < 0:
        raise ValueError(f"invalid rank RNG step in {rng_path}")
    if not isinstance(state.get("random_state"), tuple):
        raise ValueError(f"invalid Python RNG state in {rng_path}")
    if not isinstance(state.get("numpy_random_seed"), tuple):
        raise ValueError(f"invalid NumPy RNG state in {rng_path}")
    if not isinstance(state.get("torch_manual_seed"), torch.Tensor):
        raise ValueError(f"invalid Torch CPU RNG state in {rng_path}")
    cuda_state = state.get("assigned_cuda_rng_state")
    if accelerator.device.type == "cuda" and not isinstance(cuda_state, torch.Tensor):
        raise ValueError(f"assigned CUDA RNG state is missing in {rng_path}")
    if accelerator.device.type != "cuda" and cuda_state is not None:
        raise ValueError(f"unexpected CUDA RNG state in {rng_path}")
    return state


def _restore_rank_rng_state(
    state: dict[str, Any], accelerator: Accelerator
) -> None:
    """Strictly restore CPU and only this rank's assigned CUDA RNG state."""

    try:
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_seed"])
        torch.set_rng_state(state["torch_manual_seed"])
        if accelerator.device.type == "cuda":
            torch.cuda.set_rng_state(
                state["assigned_cuda_rng_state"], device=accelerator.device
            )
        accelerator.step = state["step"]
    except Exception as error:
        raise RuntimeError(
            f"failed to restore strict rank {accelerator.process_index} RNG state"
        ) from error


def read_training_checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate an atomically completed training checkpoint without loading it."""

    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise ValueError(f"training checkpoint is not a directory: {checkpoint}")
    state_path = checkpoint / TRAINER_STATE_NAME
    marker_path = checkpoint / TRAINING_CHECKPOINT_COMPLETE
    if not marker_path.is_file():
        raise ValueError(
            f"training checkpoint is incomplete (missing {TRAINING_CHECKPOINT_COMPLETE}): "
            f"{checkpoint}"
        )
    if not state_path.is_file():
        raise ValueError(
            f"training checkpoint is incomplete (missing {TRAINER_STATE_NAME}): {checkpoint}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training checkpoint metadata in {checkpoint}: {error}") from error
    if not isinstance(marker, dict) or not isinstance(state, dict):
        raise ValueError(f"training checkpoint metadata must be JSON objects: {checkpoint}")
    for label, value in (("completion marker", marker), ("trainer state", state)):
        if value.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported {label} schema in {checkpoint}: "
                f"{value.get('schema_version')!r}"
            )
    expected_state_sha = marker.get("trainer_state_sha256")
    actual_state_sha = _sha256_file(state_path)
    if expected_state_sha != actual_state_sha:
        raise ValueError(
            f"training checkpoint metadata hash mismatch in {checkpoint}: "
            f"expected {expected_state_sha!r}, got {actual_state_sha}"
        )
    for key in ("global_step", "run_fingerprint", "world_size"):
        if marker.get(key) != state.get(key):
            raise ValueError(
                f"training checkpoint {key} mismatch between marker and state in {checkpoint}"
            )
    if not isinstance(state.get("global_step"), int) or state["global_step"] < 0:
        raise ValueError(f"invalid global_step in training checkpoint {checkpoint}")
    if not isinstance(state.get("micro_step"), int) or state["micro_step"] < 0:
        raise ValueError(f"invalid micro_step in training checkpoint {checkpoint}")
    if not isinstance(state.get("epoch"), int) or state["epoch"] < 0:
        raise ValueError(f"invalid epoch in training checkpoint {checkpoint}")
    if (
        not isinstance(state.get("accumulation_steps"), int)
        or state["accumulation_steps"] < 1
    ):
        raise ValueError(
            f"invalid accumulation_steps in training checkpoint {checkpoint}"
        )
    if not state.get("run_fingerprint"):
        raise ValueError(f"training checkpoint has no run fingerprint: {checkpoint}")
    if not isinstance(state.get("run_metadata"), dict):
        raise ValueError(f"training checkpoint has no run metadata: {checkpoint}")
    _validate_checkpoint_inventory(checkpoint, marker, state)
    return state


def gradient_accumulation_steps(
    target_batch_size: int,
    physical_batch_size: int,
    num_processes: int,
) -> int:
    """Return an exact accumulation count for the requested effective batch."""

    denominator = physical_batch_size * num_processes
    if target_batch_size <= 0 or denominator <= 0:
        raise ValueError("batch sizes and num_processes must be positive")
    if target_batch_size % denominator:
        raise ValueError(
            "target_batch_size must be divisible by physical_batch_size * num_processes; "
            f"got {target_batch_size} vs {physical_batch_size} * {num_processes}"
        )
    return target_batch_size // denominator


def audio_forward_plan(
    *, depformer_trainable: bool, speechless_batch: bool, dep_q: int
) -> tuple[int, bool]:
    """Choose a DDP-safe Depformer compute width and whether to run it."""

    if dep_q < 2:
        raise ValueError(f"training requires at least two audio codebooks, got {dep_q}")
    if depformer_trainable:
        # Retain all graph branches so per-codebook trainable parameters are
        # marked used by DDP; the loss still slices supervision to codebooks 0-1.
        return dep_q, True
    return 2, not speechless_batch


def compute_text_token_metric_counts(
    pred_text: torch.Tensor,
    text_target: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    text_padding_token_id: int,
    end_of_text_padding_id: int,
) -> dict[str, torch.Tensor]:
    """Return additive sufficient statistics for disjoint text metrics.

    Moshi uses two different padding-like targets in the synchronized text
    stream.  ``text_padding_token_id`` (currently ID 3) fills ordinary silent
    frames, while ``end_of_text_padding_id`` (currently ID 0) is also used by
    the data pipeline for end/zero padding.  Treating only ID 3 as padding
    incorrectly counts the often-dominant ID 0 tail as lexical content.

    The three target classes returned here are mutually exclusive and cover
    every valid target position:

    * ``end_of_text_padding``: target equals ``end_of_text_padding_id``;
    * ``text_padding``: target equals ``text_padding_token_id``;
    * ``content``: target equals neither padding ID.

    Counts, rather than per-microbatch ratios, make exact accumulation across
    microbatches and distributed ranks possible.
    """
    if pred_text.shape != text_target.shape or text_mask.shape != text_target.shape:
        raise ValueError(
            "pred_text, text_target, and text_mask must have identical shapes; "
            f"got {pred_text.shape}, {text_target.shape}, and {text_mask.shape}"
        )
    if text_padding_token_id == end_of_text_padding_id:
        raise ValueError("text padding and end-of-text padding IDs must be distinct")

    valid_mask = text_mask.bool()
    end_padding_mask = (text_target == end_of_text_padding_id) & valid_mask
    text_padding_mask = (text_target == text_padding_token_id) & valid_mask
    content_mask = valid_mask & ~end_padding_mask & ~text_padding_mask
    predicted_end_padding = (pred_text == end_of_text_padding_id) & valid_mask
    predicted_text_padding = (pred_text == text_padding_token_id) & valid_mask
    predicted_content = valid_mask & ~predicted_end_padding & ~predicted_text_padding
    correct = pred_text == text_target

    return {
        "text_valid": valid_mask.sum(),
        "text_correct": (correct & valid_mask).sum(),
        "content_targets": content_mask.sum(),
        "content_correct": (correct & content_mask).sum(),
        "text_padding_targets": text_padding_mask.sum(),
        "text_padding_correct": (correct & text_padding_mask).sum(),
        "end_padding_targets": end_padding_mask.sum(),
        "end_padding_correct": (correct & end_padding_mask).sum(),
        "content_predictions": predicted_content.sum(),
        "text_padding_predictions": predicted_text_padding.sum(),
        "end_padding_predictions": predicted_end_padding.sum(),
        "content_to_text_padding": (content_mask & predicted_text_padding).sum(),
        "content_to_end_padding": (content_mask & predicted_end_padding).sum(),
    }


def _safe_metric_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    denominator = denominator.to(dtype=torch.float64)
    value = numerator.to(dtype=torch.float64) / denominator.clamp(min=1)
    return torch.where(denominator > 0, value, torch.full_like(value, float("nan")))


def text_metrics_from_counts(counts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = counts["text_valid"]
    content = counts["content_targets"]
    text_padding = counts["text_padding_targets"]
    end_padding = counts["end_padding_targets"]
    return {
        "acc_global": _safe_metric_ratio(counts["text_correct"], valid),
        "acc_content": _safe_metric_ratio(counts["content_correct"], content),
        "acc_text_padding": _safe_metric_ratio(
            counts["text_padding_correct"], text_padding
        ),
        "acc_end_of_text_padding": _safe_metric_ratio(
            counts["end_padding_correct"], end_padding
        ),
        "target_ratio_content": _safe_metric_ratio(content, valid),
        "target_ratio_text_padding": _safe_metric_ratio(text_padding, valid),
        "target_ratio_end_of_text_padding": _safe_metric_ratio(end_padding, valid),
        "prediction_ratio_content": _safe_metric_ratio(
            counts["content_predictions"], valid
        ),
        "prediction_ratio_text_padding": _safe_metric_ratio(
            counts["text_padding_predictions"], valid
        ),
        "prediction_ratio_end_of_text_padding": _safe_metric_ratio(
            counts["end_padding_predictions"], valid
        ),
        "content_to_text_padding_rate": _safe_metric_ratio(
            counts["content_to_text_padding"], content
        ),
        "content_to_end_of_text_padding_rate": _safe_metric_ratio(
            counts["content_to_end_padding"], content
        ),
    }


def compute_text_token_metrics(
    pred_text: torch.Tensor,
    text_target: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    text_padding_token_id: int,
    end_of_text_padding_id: int,
) -> dict[str, torch.Tensor]:
    """Compute text metrics for one tensor; accumulation uses the count API."""

    counts = compute_text_token_metric_counts(
        pred_text,
        text_target,
        text_mask,
        text_padding_token_id=text_padding_token_id,
        end_of_text_padding_id=end_of_text_padding_id,
    )
    return text_metrics_from_counts(counts)


EFFECTIVE_BATCH_STAT_KEYS = (
    "loss_sum",
    "text_loss_sum",
    "audio_loss_sum",
    "microbatches",
    "text_valid",
    "text_correct",
    "content_targets",
    "content_correct",
    "text_padding_targets",
    "text_padding_correct",
    "end_padding_targets",
    "end_padding_correct",
    "content_predictions",
    "text_padding_predictions",
    "end_padding_predictions",
    "content_to_text_padding",
    "content_to_end_padding",
    "audio_valid",
    "audio_correct",
    "audio_cb0_valid",
    "audio_cb0_correct",
    "gate_sum",
    "gate_observations",
)


def new_effective_batch_statistics(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros((), dtype=torch.float64, device=device)
        for key in EFFECTIVE_BATCH_STAT_KEYS
    }


def finalize_effective_batch_statistics(
    statistics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    text_metrics = text_metrics_from_counts(statistics)
    microbatches = statistics["microbatches"]
    return {
        "loss": _safe_metric_ratio(statistics["loss_sum"], microbatches),
        "text_loss": _safe_metric_ratio(statistics["text_loss_sum"], microbatches),
        "audio_loss": _safe_metric_ratio(statistics["audio_loss_sum"], microbatches),
        **text_metrics,
        "audio_acc_global": _safe_metric_ratio(
            statistics["audio_correct"], statistics["audio_valid"]
        ),
        "audio_acc_codebook0": _safe_metric_ratio(
            statistics["audio_cb0_correct"], statistics["audio_cb0_valid"]
        ),
        "gate_mean": _safe_metric_ratio(
            statistics["gate_sum"], statistics["gate_observations"]
        ),
    }


def _normalized_audio_mask(
    audio_mask: torch.Tensor, audio_target: torch.Tensor
) -> torch.Tensor:
    if audio_mask.dim() == 2:
        return audio_mask.unsqueeze(1).expand_as(audio_target).bool()
    if audio_mask.dim() == 3:
        if audio_mask.size(1) < audio_target.size(1):
            raise ValueError("audio mask has fewer codebooks than the audio target")
        return audio_mask[:, : audio_target.size(1), :].bool()
    if audio_mask.dim() == 4 and audio_mask.size(1) == 1:
        squeezed = audio_mask.squeeze(1)
        if squeezed.size(1) < audio_target.size(1):
            raise ValueError("audio mask has fewer codebooks than the audio target")
        return squeezed[:, : audio_target.size(1), :].bool()
    raise ValueError(f"unsupported audio mask shape: {tuple(audio_mask.shape)}")


def update_effective_batch_statistics(
    statistics: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    text_loss: torch.Tensor,
    audio_loss: torch.Tensor,
    pred_text: torch.Tensor,
    text_target: torch.Tensor,
    text_mask: torch.Tensor,
    text_padding_token_id: int,
    end_of_text_padding_id: int,
    pred_audio: torch.Tensor | None,
    audio_target: torch.Tensor,
    audio_mask: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> None:
    """Add one microbatch to an effective-batch sufficient-statistic buffer."""

    statistics["loss_sum"] += loss.detach().to(dtype=torch.float64)
    statistics["text_loss_sum"] += text_loss.detach().to(dtype=torch.float64)
    statistics["audio_loss_sum"] += audio_loss.detach().to(dtype=torch.float64)
    statistics["microbatches"] += 1
    if gate is not None:
        if gate.numel() != 1:
            raise ValueError(f"gate metric must be scalar, got {tuple(gate.shape)}")
        statistics["gate_sum"] += gate.detach().to(dtype=torch.float64)
        statistics["gate_observations"] += 1
    text_counts = compute_text_token_metric_counts(
        pred_text,
        text_target,
        text_mask,
        text_padding_token_id=text_padding_token_id,
        end_of_text_padding_id=end_of_text_padding_id,
    )
    for key, value in text_counts.items():
        statistics[key] += value.detach().to(dtype=torch.float64)

    if pred_audio is None or audio_target.size(1) == 0:
        return
    real_audio_mask = _normalized_audio_mask(audio_mask, audio_target)
    correct_audio = (pred_audio == audio_target) & real_audio_mask
    statistics["audio_valid"] += real_audio_mask.sum().to(dtype=torch.float64)
    statistics["audio_correct"] += correct_audio.sum().to(dtype=torch.float64)
    cb0_mask = real_audio_mask[:, 0]
    statistics["audio_cb0_valid"] += cb0_mask.sum().to(dtype=torch.float64)
    statistics["audio_cb0_correct"] += (
        (pred_audio[:, 0] == audio_target[:, 0]) & cb0_mask
    ).sum().to(dtype=torch.float64)


def require_batched_image_conditions(
    condition_attributes: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a complete image batch or fail before the visual forward pass."""

    if not isinstance(condition_attributes, list):
        raise RuntimeError(
            "visual training requires condition_attributes to be a list; "
            f"got {type(condition_attributes).__name__}"
        )
    if len(condition_attributes) != batch_size:
        raise RuntimeError(
            "visual condition batch length mismatch: "
            f"got {len(condition_attributes)}, expected {batch_size}"
        )
    tensors: list[torch.Tensor] = []
    for sample_index, attributes in enumerate(condition_attributes):
        if not isinstance(attributes, ConditionAttributes):
            raise RuntimeError(
                "visual training requires ConditionAttributes for every sample; "
                f"sample {sample_index} has {type(attributes).__name__}"
            )
        image = attributes.tensor.get("image")
        if not isinstance(image, TensorCondition):
            raise RuntimeError(
                "visual training requires an image TensorCondition for every sample; "
                f"sample {sample_index} is missing it"
            )
        if not isinstance(image.tensor, torch.Tensor) or image.tensor.ndim < 1:
            raise RuntimeError(
                f"sample {sample_index} has an invalid image condition tensor"
            )
        if image.tensor.shape[0] != 1:
            raise RuntimeError(
                "each per-sample image TensorCondition must have batch dimension 1; "
                f"sample {sample_index} has shape {tuple(image.tensor.shape)}"
            )
        tensors.append(image.tensor.to(device))
    image_batch = torch.cat(tensors, dim=0)
    if image_batch.shape[0] != batch_size:
        raise RuntimeError(
            "concatenated image batch does not match code batch: "
            f"got {image_batch.shape[0]}, expected {batch_size}"
        )
    return image_batch


def register_nan_hooks(model):
    """
    モデル内の全レイヤーにフックを仕掛け、出力がNaNになった瞬間に
    そのレイヤー名を表示して停止させるデバッグ関数
    """
    def hook_fn(module, inputs, output):
        # 出力がTensorの場合
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__}")
                print(f"   Shape: {output.shape}")
                raise RuntimeError(f"NaN found in {module.__class__.__name__}")

        # 出力がタプルやリストの場合
        elif isinstance(output, (tuple, list)):
            for i, x in enumerate(output):
                if isinstance(x, torch.Tensor) and torch.isnan(x).any():
                    print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__} (Output index {i})")
                    raise RuntimeError(f"NaN found in {module.__class__.__name__}")
        
        # 出力が辞書の場合
        elif isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__} (Key: {k})")
                    raise RuntimeError(f"NaN found in {module.__class__.__name__}")

    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn)
    
    print(f"👀 NaN Hunter hooks registered for {model.__class__.__name__}")
# ==========================================


class MoshiVisBundle(nn.Module):
    """MoshiVis + ImageProjection をまとめて safetensors.save_model で保存する用"""

    def __init__(self, moshi_vis: MoshiVis, image_embedder: nn.Module):
        super().__init__()
        self.moshi_vis = moshi_vis
        self.image_embedder = image_embedder

    def state_dict(self, *args, **kwargs):
        # まず MoshiVis 本体
        sd = self.moshi_vis.state_dict(*args, **kwargs)

        # ImageProjection のパラメータに image_prefix. を付けて結合
        img_sd = self.image_embedder.state_dict(*args, **kwargs)
        for k, v in img_sd.items():
            sd[f"image_prefix.{k}"] = v

        return sd


class JmoshiVisTrainer:
    def __init__(
        self,
        model: MoshiVis,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args: TrainerArgs,
        accelerator: Accelerator,
        image_embedder: ImageProjection,
        use_amp: bool = False,
        writer: WandBMetricsWriter | None = None,
        tokenizer: Any = None,
        scheduler=None,
        run_metadata: dict[str, Any] | None = None,
    ):
        """
        MoshiVis Trainer
        - MoshiVis forward_text() を用いて学習
        - condition_attributes.image_embed を cross_attention_src として使用
        """
        self.accelerator = accelerator
        self.model = model
        # DDP でラップされると属性 (audio_offset 等) に届かないため、素のモデルへの参照も保持
        self.raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.use_amp = use_amp and torch.cuda.is_available()
        self.writer = writer
        self.global_step = 0
        self.micro_step = 0
        self.epoch = 0
        self.image_embedder = image_embedder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.run_metadata = _json_normalize(run_metadata or {})
        self.run_fingerprint = str(self.run_metadata.get("run_fingerprint", ""))
        self._last_complete_checkpoint: Path | None = None
        model_audit = getattr(self.raw_model, "visual_adapter_init_audit", None)
        raw_image_embedder = accelerator.unwrap_model(image_embedder)
        image_audit = getattr(raw_image_embedder, "visual_adapter_init_audit", None)
        if model_audit is not None and image_audit is not None and model_audit != image_audit:
            raise ValueError("model and image embedder visual-adapter audits disagree")
        current_audit = model_audit if model_audit is not None else image_audit
        if current_audit is not None:
            current_audit = _json_normalize(current_audit)
        metadata_audit = self.run_metadata.get("visual_adapter_init_audit")
        if current_audit is not None and metadata_audit != current_audit:
            raise ValueError(
                "run metadata does not match the model's visual-adapter initialization audit"
            )
        self.visual_adapter_init_audit = metadata_audit
        # 累積バッチ/停止/保存の設定 (config.train から渡す。未指定時は従来値)
        _get = (lambda k, d: args.get(k, d)) if hasattr(args, "get") else (lambda k, d: getattr(args, k, d))
        self.target_batch_size = int(_get("target_batch_size", 32))
        self.max_steps = int(_get("max_steps", 0))          # 0 = 無制限
        self.save_every = int(_get("save_every", 5000))
        self.keep_last = int(_get("keep_last", 3))
        self.checkpoint_dir = str(_get("checkpoint_dir", "./checkpoints"))
        self.log_interval = int(_get("log_interval", 1))
        self.accumulation_steps = gradient_accumulation_steps(
            self.target_batch_size,
            int(_get("batch_size", 1)),
            self.accelerator.num_processes,
        )
        self.accelerator.gradient_accumulation_steps = self.accumulation_steps
        depformer_parameter_names = (
            "depformer",
            "audio_linears",
        )
        self.depformer_trainable = any(
            parameter.requires_grad
            for name, parameter in self.raw_model.named_parameters()
            if name.startswith(depformer_parameter_names)
        )
        self.stop_requested = False
        # if self.accelerator.is_main_process:
        #     print("🕵️‍♀️ Registering NaN Hunter Hooks...")
        #     register_nan_hooks(self.model)
        #     register_nan_hooks(self.image_embedder)

    def train_epoch(self, dataloader, epoch: int, log_interval: int = 1, DEBUG: bool = False):
        self.epoch = epoch
        log_interval = self.log_interval
        accumulation_steps = self.accumulation_steps

        if self.accelerator.is_main_process:
            print(f"🚀 Epoch {epoch} Start | Grad Accum: {accumulation_steps} steps")

        total_loss = 0.0
        completed_optimizer_steps = 0
        
        # --- 移動平均用ウィンドウ ---
        loss_window = deque(maxlen=100)
        # Text
        text_acc_global_window = deque(maxlen=100)  # 全体
        text_acc_content_window = deque(maxlen=100) # 意味のある文字のみ (重要!)
        text_acc_pad_window = deque(maxlen=100)     # PADのみ
        # Audio
        audio_acc_global_window = deque(maxlen=100) # 全体
        audio_acc_cb0_window = deque(maxlen=100)    # Codebook 0のみ (重要!)
        effective_batch_statistics = new_effective_batch_statistics(self.device)

        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)

        self.optimizer.zero_grad(set_to_none=True) # ループ前に初期化

        for step, batch in pbar:
            self.micro_step += 1
            # -----------------------------------------------------
            # ★ 修正: accumulate コンテキストで囲む
            # -----------------------------------------------------
            with self.accelerator.accumulate(self.model, self.image_embedder):
                # --- Extract inputs ---
                codes = batch.codes.to(self.device)

                if DEBUG:
                    print("\n" + "="*60)
                    print(f"🔍 [CHECK 1] Input Codes (Step {step})")
                    print(f"   Shape: {codes.shape} (Batch, Codebooks, Time)")
                    print(f"   Values: Min={codes.min().item()}, Max={codes.max().item()}")
                    # パディング(-1や3)だらけになっていないか確認
                    unique, counts = torch.unique(codes, return_counts=True)
                    print(f"   Top 5 tokens: {list(zip(unique[:5].tolist(), counts[:5].tolist()))}")
                    print("="*60)

                # --- 画像入力の準備 ---
                image_input = require_batched_image_conditions(
                    batch.condition_attributes,
                    batch_size=codes.shape[0],
                    device=self.device,
                )

                if DEBUG:
                    print(f"🔍 [CHECK 2] Image Input")
                    if image_input is not None:
                        print(f"   Shape: {image_input.shape}")
                        print(f"   Stats: Mean={image_input.mean().item():.3f}, Std={image_input.std().item():.3f}")
                    else:
                        print("   ⚠️ WARNING: No Image Input found in this batch!")

                # --- Forward pass ---
                with self.accelerator.autocast():
                    embedder_out = self.image_embedder(image_input)
                    cross_attention_src = embedder_out["cross_attention_src"]

                    if DEBUG:
                        print(f"🔍 [CHECK 3] Embedder Output (Cross Attention Src)")
                        print(f"   Shape: {cross_attention_src.shape}")
                        print(f"   Stats: Mean={cross_attention_src.mean().item():.3f}, Std={cross_attention_src.std().item():.3f}")
                        print(f"   Max Val: {cross_attention_src.max().item():.3f}") # 爆発していないか(±20以内か)

                    if step == 0:
                        print(f"DEBUG: Accelerator Mixed Precision: {self.accelerator.mixed_precision}")
                        # ダミーテンソルを作って型を確認
                        with self.accelerator.autocast():
                            dummy = torch.tensor([1.0], device=self.device)
                            print(f"DEBUG: Real dtype inside autocast: {dummy.dtype}")

                    num_audio_codebooks_for_loss = 2
                    supervised_audio = codes[
                        :,
                        self.raw_model.audio_offset : self.raw_model.audio_offset
                        + num_audio_codebooks_for_loss,
                    ]
                    speechless_batch = not bool(
                        (supervised_audio != self.raw_model.zero_token_id).any().item()
                    )
                    # Skipping the Depformer is objective-equivalent for an
                    # all-speechless batch only while every Depformer parameter
                    # is frozen.  With trainable Depformer parameters, keeping
                    # the forward avoids changing DDP's used-parameter graph.
                    num_audio_codebooks_to_compute, compute_audio = audio_forward_plan(
                        depformer_trainable=self.depformer_trainable,
                        speechless_batch=speechless_batch,
                        dep_q=self.raw_model.dep_q,
                    )
                    outputs = self.model(
                        input_ids=codes,
                        cross_attention_src=cross_attention_src,
                        num_audio_codebooks=num_audio_codebooks_to_compute,
                        compute_audio=compute_audio,
                    )

                    text_logits = outputs["text_logits"]
                    audio_logits_full = outputs["audio_logits"]
                    audio_logits = (
                        None
                        if audio_logits_full is None
                        else audio_logits_full[
                            :, :num_audio_codebooks_for_loss
                        ].contiguous()
                    )

                    if DEBUG:
                        print(f"🔍 [CHECK 4] Outputs & Masks")
                        print(f"   Text Logits: {text_logits.shape}")
                        print(
                            "   Audio Logits: "
                            + ("skipped" if audio_logits is None else str(audio_logits.shape))
                        )
                        
                        # NaNチェック
                        if torch.isnan(text_logits).any():
                            print("   🚨 ERROR: Text Logits contain NaN!")
                        
                        # AudioはマスクがTrue(有効)な場所にあるNaNだけをエラーとする
                        real_audio_nan = (
                            torch.tensor(False, device=self.device)
                            if audio_logits is None
                            else (
                                torch.isnan(audio_logits)
                                & outputs["logits_mask"].unsqueeze(-1)
                            ).any()
                        )
                        
                        if real_audio_nan:
                            print("   🚨 FATAL ERROR: Audio Logits contain REAL NaN inside the mask!")
                            # ここで詳細を出して止める
                            raise RuntimeError("Audio logits exploded within the valid mask!")
                        else:
                            # 正常なNaN(パディング)はスルー
                            pass
                        
                        # Maskの確認: 全てFalseになっていないか？
                        t_mask = outputs["text_logits_mask"]
                        a_mask = outputs["logits_mask"]
                        print(f"   Text Mask Valid Ratio: {t_mask.float().mean().item():.2%}")
                        print(f"   Audio Mask Valid Ratio: {a_mask.float().mean().item():.2%}")
                        
                        if t_mask.sum() == 0:
                            print("   🚨 FATAL: Text Mask is empty! Loss will be 0.")
                        if a_mask.sum() == 0:
                            print("   🚨 FATAL: Audio Mask is empty! Loss will be 0.")

                    # --- Loss Calculation ---
                    text_target = codes[:, :self.raw_model.audio_offset]
                    # 🚀 MoshiVis style: Use only first 2 codebooks for audio loss (faster training)
                    # Original uses all dep_q (8), but first 2 carry most of the semantic information
                    audio_target = codes[:, self.raw_model.audio_offset:self.raw_model.audio_offset + num_audio_codebooks_for_loss]

                    # =========================================================================
                    # ▼ デバッグ用コード: ここから ▼
                    # =========================================================================
                    if DEBUG or (self.global_step % 100 == 0): # 毎回出すと重いので100ステップ毎などに制限
                        with torch.no_grad():
                            print(f"\n[Step {self.global_step}] Debug Inspection -------------------------")
                            
                            # 1. シェイプの確認
                            # text_logits: [B, 1, T, Vocab] 想定
                            # text_target: [B, 1, T] 想定
                            B, _, T, _ = text_logits.shape
                            
                            # 2. 予測トークン（ID）を取得 (Argmax)
                            # 確率が最大のIDを取得
                            pred_ids = torch.argmax(text_logits, dim=-1) # [B, 1, T]
                            
                            # 3. バッチ内の最初のサンプルの内容を表示
                            sample_idx = 0
                            
                            # ターゲット（正解）のID列
                            target_sample = text_target[sample_idx, 0, :].cpu().numpy()
                            # モデル予測のID列
                            pred_sample = pred_ids[sample_idx, 0, :].cpu().numpy()
                            # マスク（学習対象かどうか）
                            mask_sample = outputs["text_logits_mask"][sample_idx, 0, :].cpu().numpy()

                            print(f"Target IDs (First 50): {target_sample[:50]}")
                            print(f"Pred   IDs (First 50): {pred_sample[:50]}")
                            # print(f"Mask       (First 50): {mask_sample[:50]}")

                            # 4. (もしtokenizerを持っていれば) デコードして文字で確認
                            # self.tokenizer が Trainerにあると仮定しています
                            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                                # Maskされている部分（Paddingなど）を除外してデコードしてみる
                                valid_target = target_sample[mask_sample.astype(bool)]
                                valid_pred = pred_sample[mask_sample.astype(bool)]
                                
                                try:
                                    print(f"Target Text: {self.tokenizer.decode(valid_target)}")
                                    print(f"Pred   Text: {self.tokenizer.decode(valid_pred)}")
                                except Exception as e:
                                    print(f"Decode failed: {e}")
                            
                            print("----------------------------------------------------------\n")

                    text_loss = compute_loss_with_mask(
                        text_logits, text_target, outputs["text_logits_mask"],
                        mode="text",
                        text_padding_weight=self.args.text_padding_weight,
                        text_padding_ids={self.raw_model.text_padding_token_id, self.raw_model.end_of_text_padding_id},
                    )
                    audio_mask = outputs["logits_mask"]
                    if audio_mask.size(1) > num_audio_codebooks_for_loss:
                        audio_mask = audio_mask[:, :num_audio_codebooks_for_loss]
                    if audio_logits is None:
                        audio_loss = text_logits.new_zeros(())
                    else:
                        audio_loss = compute_loss_with_mask(
                            audio_logits, audio_target, audio_mask,
                            mode="audio",
                            first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
                        )

                    loss = 2.0 * text_loss + audio_loss
                    # Accelerate 1.14 divides inside Accelerator.backward().
                    # Passing an already divided loss would scale gradients twice.
                    with torch.no_grad():
                        update_effective_batch_statistics(
                            effective_batch_statistics,
                            loss=loss,
                            text_loss=text_loss,
                            audio_loss=audio_loss,
                            pred_text=text_logits.argmax(-1),
                            text_target=text_target,
                            text_mask=outputs["text_logits_mask"],
                            text_padding_token_id=self.raw_model.text_padding_token_id,
                            end_of_text_padding_id=self.raw_model.end_of_text_padding_id,
                            pred_audio=(
                                None if audio_logits is None else audio_logits.argmax(-1)
                            ),
                            audio_target=audio_target,
                            audio_mask=audio_mask,
                            gate=outputs.get("gate"),
                        )

                    if DEBUG:
                        print(f"🔍 [CHECK 5] Loss Values")
                        print(f"   Text Loss: {text_loss.item():.4f}")
                        print(f"   Audio Loss: {audio_loss.item():.4f}")
                        print(f"   Total Loss: {loss.item():.4f}")
                        print("="*60 + "\n")

                    # =========================================================
                    # 🔍 [CHECK 6] Alignment & Prediction Preview (Step 0のみ)
                    # =========================================================
                    if DEBUG:
                        print("\n" + "="*60)
                        print("👀 LOGITS vs TARGET ALIGNMENT CHECK")
                        
                        # --- 1. Text Alignment ---
                        # Batch 0, Channel 0, 最初の10トークンを表示
                        t_tgt = text_target[0, 0, :15].cpu().tolist()
                        t_pred = text_logits.argmax(dim=-1)[0, 0, :15].cpu().tolist()
                        
                        print(f" [Text] Target (GT): {t_tgt}")
                        print(f" [Text] Pred (Argmax): {t_pred}")
                        
                        # --- 2. Audio Alignment ---
                        # Batch 0, Channel 0, 最初の10トークン
                        a_tgt = audio_target[0, 0, :15].cpu().tolist()
                        a_pred = (
                            []
                            if audio_logits is None
                            else audio_logits.argmax(dim=-1)[0, 0, :15].cpu().tolist()
                        )
                        
                        print(f" [Audio] Target (GT): {a_tgt}")
                        print(f" [Audio] Pred (Argmax): {a_pred}")

                        # --- 3. Data Leakage Check ---
                        # もし初期状態で「Pred」が「Target」と完全に一致していたら、
                        # 「カンニング（未来のトークンが見えている）」バグです。
                        # 逆に、ランダムな予測になっていれば正常です。
                        text_match_rate = (text_logits.argmax(-1) == text_target).float().mean().item()
                        print(f" [Check] Initial Text Accuracy: {text_match_rate:.2%} (Should be low/random, NOT 100%)")
                        print("="*60 + "\n")

                # --- Backprop (蓄積される) ---
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # =========================================================
                    # 🔬 勾配ロギング（MHA学習確認用）
                    # =========================================================
                    if self.global_step % 50 == 0 and self.accelerator.is_main_process:
                        print(f"\n🔬 [Gradient Check] Step {self.global_step}")
                        mha_grad_found = False
                        gate_grad_found = False
                        
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                continue
                            
                            # MHA勾配チェック (layer 0のみ)
                            if "cross_attention.mha" in name and "layers.0" in name and not mha_grad_found:
                                grad_norm = param.grad.norm().item()
                                grad_mean = param.grad.abs().mean().item()
                                print(f"   MHA ({name[:50]}...): norm={grad_norm:.8f}, mean={grad_mean:.8f}")
                                mha_grad_found = True
                            
                            # Gate勾配チェック (layer 0のみ)
                            if "gate" in name and "layers.0" in name and not gate_grad_found:
                                grad_norm = param.grad.norm().item()
                                grad_mean = param.grad.abs().mean().item()
                                print(f"   Gate ({name[:50]}...): norm={grad_norm:.8f}, mean={grad_mean:.8f}")
                                gate_grad_found = True
                            
                            if mha_grad_found and gate_grad_found:
                                break
                        
                        if not mha_grad_found:
                            print("   ⚠️ WARNING: No MHA gradient found!")
                        if not gate_grad_found:
                            print("   ⚠️ WARNING: No Gate gradient found!")

                        # 🔬 MHA重みの変化を確認（optimizer.step前に記録）
                        mha_weight_before = None
                        mha_param = None
                        for name, param in self.model.named_parameters():
                            if "cross_attention.mha.in_proj_weight" in name and "layers.0" in name:
                                mha_weight_before = param.data.clone()
                                mha_param = param
                                break
                    
                    # 勾配クリッピング
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.accelerator.clip_grad_norm_(self.image_embedder.parameters(), 1.0)
                    
                    self.optimizer.step()
                    step_was_skipped = self.accelerator.optimizer_step_was_skipped
                    if self.scheduler is not None and not step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if step_was_skipped:
                        # A GradScaler overflow did not produce an optimizer
                        # update, so neither global_step nor its metrics/schedule
                        # may advance. Discard this attempted accumulation window.
                        effective_batch_statistics = new_effective_batch_statistics(
                            self.device
                        )
                        continue
                    self.global_step += 1  # 更新した回数だけカウントアップ

                    packed_statistics = torch.stack(
                        [effective_batch_statistics[key] for key in EFFECTIVE_BATCH_STAT_KEYS]
                    )
                    packed_statistics = self.accelerator.reduce(
                        packed_statistics, reduction="sum"
                    )
                    reduced_statistics = dict(
                        zip(EFFECTIVE_BATCH_STAT_KEYS, packed_statistics.unbind())
                    )
                    effective_metrics = finalize_effective_batch_statistics(
                        reduced_statistics
                    )
                    effective_batch_statistics = new_effective_batch_statistics(self.device)
                    current_loss_val = float(effective_metrics["loss"])
                    total_loss += current_loss_val
                    completed_optimizer_steps += 1

                    # 🔬 optimizer.step後のMHA重みを確認
                    if self.global_step % 50 == 1 and self.accelerator.is_main_process:
                        if mha_weight_before is not None:
                            after = mha_param.data
                            diff = (after - mha_weight_before).abs().mean().item()
                            print(f"   📊 MHA weight change after step {self.global_step}:")
                            print(f"      Before mean: {mha_weight_before.mean().item():.10f}")
                            print(f"      After mean:  {after.mean().item():.10f}")
                            print(f"      Abs diff mean: {diff:.12f}")
                            if diff < 1e-12:
                                print(f"      ❌ CRITICAL: MHA weights did NOT change!")

                    # These are exact count-based metrics for the full accumulation
                    # window and all ranks, not values from the final microbatch.
                    def metric_float(name: str) -> float:
                        return float(effective_metrics[name])

                    def metric_or_none(name: str) -> float | None:
                        value = metric_float(name)
                        return None if value != value else value

                    loss_window.append(current_loss_val)
                    text_acc_global_window.append(metric_float("acc_global"))
                    text_acc_content_window.append(metric_float("acc_content"))
                    text_acc_pad_window.append(metric_float("acc_text_padding"))
                    audio_acc_global_window.append(metric_float("audio_acc_global"))
                    audio_acc_cb0_window.append(metric_float("audio_acc_codebook0"))

                    if self.global_step % log_interval == 0:
                        def nanmean(values):
                            valid = [value for value in values if value == value]
                            return sum(valid) / len(valid) if valid else 0.0

                        pbar.set_postfix(
                            {
                                "L": f"{nanmean(loss_window):.3f}",
                                "TxCt": f"{nanmean(text_acc_content_window):.1%}",
                                "AuC0": f"{nanmean(audio_acc_cb0_window):.1%}",
                                "AuGl": f"{nanmean(audio_acc_global_window):.1%}",
                                "TxGl": f"{nanmean(text_acc_global_window):.1%}",
                            }
                        )
                        if self.writer is not None:
                            self.writer.log_step(
                                step=self.global_step,
                                loss=current_loss_val,
                                text_loss=metric_float("text_loss"),
                                audio_loss=metric_float("audio_loss"),
                                text_acc_global=metric_or_none("acc_global"),
                                text_acc_content=metric_or_none("acc_content"),
                                text_acc_pad=metric_or_none("acc_text_padding"),
                                text_acc_end_of_text_padding=metric_or_none(
                                    "acc_end_of_text_padding"
                                ),
                                text_target_ratio_content=metric_or_none(
                                    "target_ratio_content"
                                ),
                                text_target_ratio_pad=metric_or_none(
                                    "target_ratio_text_padding"
                                ),
                                text_target_ratio_end_of_text_padding=metric_or_none(
                                    "target_ratio_end_of_text_padding"
                                ),
                                text_prediction_ratio_content=metric_or_none(
                                    "prediction_ratio_content"
                                ),
                                text_prediction_ratio_pad=metric_or_none(
                                    "prediction_ratio_text_padding"
                                ),
                                text_prediction_ratio_end_of_text_padding=metric_or_none(
                                    "prediction_ratio_end_of_text_padding"
                                ),
                                text_content_to_pad_rate=metric_or_none(
                                    "content_to_text_padding_rate"
                                ),
                                text_content_to_end_of_text_padding_rate=metric_or_none(
                                    "content_to_end_of_text_padding_rate"
                                ),
                                audio_acc_global=metric_or_none("audio_acc_global"),
                                audio_acc_codebook0=metric_or_none(
                                    "audio_acc_codebook0"
                                ),
                                gate_mean=metric_or_none("gate_mean"),
                            )

                    # --- Save Checkpoint ---
                    if (
                        self.save_every > 0
                        and self.global_step > 0
                        and self.global_step % self.save_every == 0
                    ):
                        self.save_training_checkpoint()

                    # --- max_steps で停止 (データイテレータは無限なのでここで打ち切る) ---
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        self.stop_requested = True
                        break

        # End of Epoch
        print(f"✅ Epoch {epoch} finished.")
        return total_loss / max(1, completed_optimizer_steps)

    # def train_epoch(self, dataloader, epoch: int, log_interval: int = 1):
    #     total_loss = 0.0
    #     total_text_acc = 0.0
    #     total_audio_acc = 0.0

    #     pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}")

    #     for step, batch in pbar:
    #         global_step = self.global_step
    #         self.global_step += 1

    #         self.optimizer.zero_grad(set_to_none=True)

    #         # --- Extract inputs ---
    #         # batch は jmoshivis.datasets.interleaver.Batch
    #         # → batch.codes: torch.Size([B, D, T])
    #         # → batch.condition_attributes: Optional[ConditionAttributes]
    #         codes = batch.codes.to(self.device)

    #         # --- 画像入力の準備 ---
    #         image_input = None

    #         if isinstance(batch.condition_attributes, list):
    #             tensors = []
    #             for ca in batch.condition_attributes:
    #                 if hasattr(ca, "tensor") and "image" in ca.tensor:
    #                     tensors.append(ca.tensor["image"].tensor.to(self.device))
    #             if tensors:
    #                 image_input = torch.cat(tensors, dim=0)

    #                 # 最初のステップ、または一定間隔で確認
    #                 if self.global_step == 1 or self.global_step % 100 == 0:
    #                     print(f"\n🔍 [DEBUG Step {self.global_step}] Image Input Check:")
    #                     print(f"   - Shape: {image_input.shape}") # [B, 3, H, W] になっているか？ (例: [B, 3, 448, 448])
    #                     print(f"   - Min: {image_input.min().item():.3f}, Max: {image_input.max().item():.3f}") # -1.0 ~ 1.0 付近か？
    #                     print(f"   - Mean: {image_input.mean().item():.3f}, Std: {image_input.std().item():.3f}")

    #                     # 画像として保存して目視確認 (最初の1枚だけ)
    #                     try:
    #                         import os
    #                         import torchvision
    #                         os.makedirs("debug_images", exist_ok=True)
                            
    #                         # 正規化を戻す (mean=0.5, std=0.5 を仮定: [-1, 1] -> [0, 1])
    #                         # ※ ImageProcessorの設定に合わせて調整してください
    #                         img_to_save = image_input[0].clone().detach().float().cpu()
    #                         img_to_save = img_to_save * 0.5 + 0.5
    #                         img_to_save = torch.clamp(img_to_save, 0, 1)
                            
    #                         save_path = f"debug_images/step_{self.global_step}.png"
    #                         torchvision.utils.save_image(img_to_save, save_path)
    #                         print(f"   - Saved debug image to: {save_path}")
    #                     except Exception as e:
    #                         print(f"   - Failed to save debug image: {e}")

    #         if image_input is None and batch.condition_attributes is not None:
    #             # Case 1: moshi standard format (image_embed attribute)
    #             if hasattr(batch.condition_attributes, "image_embed"):
    #                 pass

    #         # --- Forward pass ---
    #         with self.accelerator.autocast():
    #             cross_attention_src = None
    #             if image_input is not None:
    #                 # ここでプロジェクション層 (proj_xa) に勾配が流れる！
    #                 # forward戻り値: {"cross_attention_src": ..., "cross_attention_mask": ...}
    #                 embedder_out = self.image_embedder(image_input)

    #                 cross_attention_src = embedder_out["cross_attention_src"]
    #                 # 必要であればマスクも取得 (Pixtralなど画像サイズ可変の場合)
    #                 # cross_attention_mask = embedder_out.get("cross_attention_mask", None)

    #             # MoshiVis forward_text は Text + Audioトークンの埋め込み系列を入力とする
    #             # _, text_logits, _ = self.model.forward_text(
    #             #     input_ids=codes,
    #             #     cross_attention_src=cross_attention_src,
    #             # )
    #             outputs = self.model.forward_speech(
    #                 input_ids=codes,
    #                 cross_attention_src=cross_attention_src,
    #                 # cross_attention_mask=cross_attention_mask # 必要なら追加
    #             )

    #             text_logits = outputs["text_logits"]             # [B,1,T_text,V]
    #             audio_logits = outputs["audio_logits"]           # [B,dep_q,T_audio,V]

    #             with torch.no_grad():
    #                 # shape
    #                 # text_logits: [B, 1, T, V]
    #                 # text_target: [B, 1, T]
    #                 # mask: [B, 1, T]
    #                 text_mask = outputs["text_logits_mask"]
    #                 text_target = codes[:, :self.raw_model.audio_offset]  # [B,1,T]

    #                 pred = text_logits.argmax(-1)   # [B,1,T]
    #                 correct = (pred == text_target) & text_mask       # boolean

    #                 # avoid ZeroDivision
    #                 if text_mask.sum() > 0:
    #                     step_text_acc = correct.sum() / text_mask.sum()
    #                 else:
    #                     step_text_acc = torch.tensor(0.0)

    #             # ロギング用
    #             total_text_acc += float(step_text_acc)

    #             text_target = codes[:, :self.raw_model.audio_offset]  # [B,T_text]
    #             audio_target = codes[:,self.model.audio_offset:self.model.audio_offset + self.model.dep_q]

    #             # decoded = self.tokenizer.decode(codes[0,0].tolist())
    #             # print(decoded)

    #             # print("DEBUG: text_target[0, 0, :50]", codes[0, 0])

    #             # text_logits: [B, 1, T, vocab_size]
    #             # 教師信号: text部分（= input_ids[:, 0, :]）を参照
    #             text_loss = compute_loss_with_mask(
    #                 text_logits,
    #                 text_target,
    #                 outputs["text_logits_mask"],
    #                 mode="text",
    #                 text_padding_weight=self.args.text_padding_weight,
    #                 text_padding_ids={
    #                     self.model.text_padding_token_id,
    #                     self.model.end_of_text_padding_id,
    #                 },
    #             )
    #             audio_loss = compute_loss_with_mask(
    #                 audio_logits,
    #                 audio_target,
    #                 outputs["logits_mask"],
    #                 mode="audio",
    #                 first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
    #             )

    #             loss = text_loss*2 + audio_loss

    #         # --- Backprop ---
    #         self.accelerator.backward(loss)
    #         if self.accelerator.sync_gradients:
    #             self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
    #             # もし image_embedder も学習対象ならそちらもクリップ対象にするのが安全ですが、
    #             # 通常は model 側に含めるか、params リストに対して行います。
    #             # image_embedder のパラメータも optimizer に含まれているので、
    #             # ここで image_embedder.parameters() もクリップするのがベストです。
    #             self.accelerator.clip_grad_norm_(self.image_embedder.parameters(), 1.0)
    #         self.optimizer.step()
    #         self.optimizer.zero_grad(set_to_none=True)

    #         total_loss += loss.item()
    #         avg_loss = total_loss / (step + 1)
    #         avg_text_acc = total_text_acc / (step + 1)
    #         avg_audio_acc = total_audio_acc / (step + 1)

    #         # ============================
    #         #   tqdm update
    #         # ============================
    #         if step % log_interval == 0:
    #             pbar.set_postfix({
    #                 "loss": f"{avg_loss:.4f}",
    #                 "text_acc": f"{avg_text_acc:.3f}",
    #                 "audio_acc": f"{avg_audio_acc:.3f}",
    #             })

    #         # --- WandB Logging (STEP) ---
    #         print(f"writer:{self.writer is not None}")
    #         if self.writer is not None:
    #             self.writer.log_step(
    #                 step=global_step,
    #                 loss=loss.item(),
    #                 text_loss=text_loss.item(),
    #                 audio_loss=audio_loss.item(),
    #             )

    #         # --- Step-based Checkpoint Saving ---
    #         import os
    #         from safetensors.torch import save_model

    #         if global_step % 2000 == 0 and global_step > 0:
    #             if self.accelerator.is_main_process:
    #                 ckpt_path = f"./checkpoints/step_{global_step}.safetensors"
    #                 os.makedirs("./checkpoints", exist_ok=True)

    #                 # unwrap MoshiVis 本体
    #                 model_to_save = self.model
    #                 if hasattr(model_to_save, "module"):
    #                     model_to_save = model_to_save.module

    #                 # MoshiVis + ImageProjection をまとめたバンドルを作成
    #                 bundle = MoshiVisBundle(model_to_save, self.image_embedder)

    #                 # shared weight 付きでも安全に保存できる
    #                 save_model(bundle, ckpt_path)

    #                 print(f"💾 Saved FULL safetensor checkpoint at step {global_step}: {ckpt_path}")



    #     # ==================================================
    #     #   Epoch Summary
    #     # ==================================================
    #     avg_loss = total_loss / len(dataloader)
    #     avg_text_acc = total_text_acc / len(dataloader)
    #     avg_audio_acc = total_audio_acc / len(dataloader)

    #     print(f"✅ Epoch {epoch} | Loss={avg_loss:.4f} | TextAcc={avg_text_acc:.3f} | AudioAcc={avg_audio_acc:.3f}")
    #     return avg_loss

    def _trainer_state(self) -> dict[str, Any]:
        if not self.run_fingerprint:
            raise ValueError("refusing to save training state without a run fingerprint")
        return {
            "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "accumulation_steps": self.accumulation_steps,
            "world_size": self.accelerator.num_processes,
            "accelerate_state": {
                "models": len(self.accelerator._models),
                "optimizers": len(self.accelerator._optimizers),
                "custom_scheduler": self.scheduler is not None,
                "scaler": self.accelerator.scaler is not None,
            },
            "run_fingerprint": self.run_fingerprint,
            "visual_adapter_init_audit": self.visual_adapter_init_audit,
            "run_metadata": self.run_metadata,
            "created_unix": time.time(),
        }

    def _validate_resume_metadata(
        self, checkpoint: Path, state: dict[str, Any]
    ) -> None:
        if state["run_fingerprint"] != self.run_fingerprint:
            raise ValueError(
                "refusing to resume an incompatible run: checkpoint fingerprint "
                f"{state['run_fingerprint']} != current fingerprint {self.run_fingerprint}"
            )
        if state.get("run_metadata") != self.run_metadata:
            raise ValueError(
                f"refusing to resume {checkpoint}: run metadata differs despite its fingerprint"
            )
        if state.get("visual_adapter_init_audit") != self.visual_adapter_init_audit:
            raise ValueError(
                f"refusing to resume {checkpoint}: visual-adapter initialization audit differs"
            )
        if state.get("accumulation_steps") != self.accumulation_steps:
            raise ValueError(
                f"refusing to resume {checkpoint}: accumulation_steps changed from "
                f"{state.get('accumulation_steps')} to {self.accumulation_steps}"
            )
        if state.get("world_size") != self.accelerator.num_processes:
            raise ValueError(
                f"refusing to resume {checkpoint}: world_size changed from "
                f"{state.get('world_size')} to {self.accelerator.num_processes}"
            )
        if state["micro_step"] % self.accumulation_steps:
            raise ValueError(
                f"refusing to resume {checkpoint}: micro_step={state['micro_step']} "
                "is not at an optimizer-window boundary"
            )
        if state["micro_step"] < state["global_step"] * self.accumulation_steps:
            raise ValueError(
                f"refusing to resume {checkpoint}: micro_step={state['micro_step']} "
                f"cannot contain global_step={state['global_step']}"
            )
        step_match = re.fullmatch(r"step_(\d+)", checkpoint.name)
        if step_match is not None and int(step_match.group(1)) != state["global_step"]:
            raise ValueError(
                f"checkpoint directory name and global_step disagree: {checkpoint}"
            )

    def inspect_training_checkpoint(
        self, path: str | os.PathLike[str]
    ) -> dict[str, Any]:
        checkpoint = Path(path)
        state = read_training_checkpoint_metadata(checkpoint)
        self._validate_resume_metadata(checkpoint, state)
        return state

    def load_training_checkpoint(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Restore a complete Accelerate checkpoint after strict metadata checks."""

        checkpoint = Path(path)
        state = self.inspect_training_checkpoint(checkpoint)
        # Validate the custom payload before Accelerate mutates model or
        # optimizer state.  Accelerate 1.14 silently ignores RNG load errors,
        # so restoration is repeated strictly below.
        rank_rng_state = _read_rank_rng_state(checkpoint, self.accelerator)
        _load_accelerate_state_without_global_cuda_rng(
            self.accelerator, checkpoint
        )
        _restore_rank_rng_state(rank_rng_state, self.accelerator)
        self.global_step = state["global_step"]
        self.micro_step = state["micro_step"]
        self.epoch = state["epoch"]
        self.stop_requested = self.max_steps > 0 and self.global_step >= self.max_steps
        self._last_complete_checkpoint = checkpoint.resolve()
        return state

    def save_training_checkpoint(self) -> Path:
        """Atomically save model, optimizer, scheduler, scaler, and RNG state.

        Accelerate writes into an unpublished temporary directory.  The main
        process writes metadata and the completion marker only after every rank
        has finished, then atomically renames the directory.  Resume ignores
        any temporary directory left by a crash.
        """

        root = Path(self.checkpoint_dir)
        checkpoint = root / f"step_{self.global_step}"
        temporary = root / f".step_{self.global_step}.incomplete"

        if (
            self._last_complete_checkpoint == checkpoint.resolve()
            and checkpoint.is_dir()
        ):
            return checkpoint
        if checkpoint.exists():
            state = self.inspect_training_checkpoint(checkpoint)
            if state["global_step"] == self.global_step:
                return checkpoint

        if self.accelerator.is_main_process:
            root.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                if not temporary.is_dir():
                    raise ValueError(
                        f"checkpoint temporary path is not a directory: {temporary}"
                    )
                shutil.rmtree(temporary)
            temporary.mkdir()
        self.accelerator.wait_for_everyone()

        _save_accelerate_state_without_global_cuda_rng(self.accelerator, temporary)
        _save_rank_rng_state(temporary, self.accelerator)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            state_path = temporary / TRAINER_STATE_NAME
            _atomic_write_json(state_path, self._trainer_state())
            inventory = _checkpoint_inventory(temporary)
            marker = {
                "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "global_step": self.global_step,
                "run_fingerprint": self.run_fingerprint,
                "world_size": self.accelerator.num_processes,
                "trainer_state_sha256": _sha256_file(state_path),
                "inventory": inventory,
                "completed_unix": time.time(),
            }
            _atomic_write_json(temporary / TRAINING_CHECKPOINT_COMPLETE, marker)
            _fsync_tree(temporary)
            os.replace(temporary, checkpoint)
            _fsync_directory(root)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._rotate_checkpoints()
        self.accelerator.wait_for_everyone()
        self._last_complete_checkpoint = checkpoint.resolve()
        return checkpoint

    def _rotate_checkpoints(self) -> None:
        """Retain only the newest atomically completed training checkpoints."""

        if self.keep_last <= 0:
            return
        root = Path(self.checkpoint_dir)
        completed: list[tuple[int, Path]] = []
        if not root.is_dir():
            return
        for child in root.iterdir():
            match = re.fullmatch(r"step_(\d+)", child.name)
            if (
                match is not None
                and child.is_dir()
                and (child / TRAINING_CHECKPOINT_COMPLETE).is_file()
            ):
                try:
                    marker = json.loads(
                        (child / TRAINING_CHECKPOINT_COMPLETE).read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, json.JSONDecodeError) as error:
                    print(f"[WARN] retaining unreadable checkpoint {child}: {error}")
                    continue
                if (
                    isinstance(marker, dict)
                    and marker.get("schema_version")
                    == TRAINING_CHECKPOINT_SCHEMA_VERSION
                    and marker.get("run_fingerprint") == self.run_fingerprint
                ):
                    completed.append((int(match.group(1)), child))
        completed.sort()
        for _, old_checkpoint in completed[:-self.keep_last]:
            shutil.rmtree(old_checkpoint)
            print(f"🗑️ removed old training checkpoint {old_checkpoint}")
        _fsync_directory(root)

    def save_checkpoint(self, path: str) -> None:
        """Atomically export model-only safetensors for inference.

        This intentionally does not contain optimizer/scheduler/RNG state and
        is therefore not accepted by :meth:`load_training_checkpoint`.
        """

        from safetensors.torch import save_model

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.incomplete")
        if temporary.exists():
            temporary.unlink()
        model_to_save = self.accelerator.unwrap_model(self.model)
        image_to_save = self.accelerator.unwrap_model(self.image_embedder)
        bundle = MoshiVisBundle(model_to_save, image_to_save)
        temporary_is_ours = True
        try:
            save_model(bundle, str(temporary))
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, output)
            temporary_is_ours = False
            _fsync_directory(output.parent)
        except BaseException:
            # save_model can leave a partially written file (for example on
            # ENOSPC).  The fixed temporary name was cleared by this call
            # immediately above, so any file now present belongs to this
            # export attempt.  Never touch the previously published output.
            if temporary_is_ours:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        print(f"💾 Saved model-only inference export: {output}")
