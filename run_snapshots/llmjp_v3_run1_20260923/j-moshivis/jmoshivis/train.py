import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from huggingface_hub import hf_hub_download
import sentencepiece as spm
from moshi.models.loaders import get_mimi
from .models.loaders import get_moshi_vis_train
from .datasets.interleaver import InterleavedTokenizer, Interleaver
from .datasets.data_loader import build_data_loader
from .datasets.asset_manifest import AssetManifestError, verify_asset_manifest
from .distributed import get_rank, get_world_size
from .config.kyuteye_config import KyuteyeConfig
from .trainer import (
    JmoshiVisTrainer,
    TRAINING_CHECKPOINT_COMPLETE,
    gradient_accumulation_steps,
    read_training_checkpoint_metadata,
)
from jmoshivis.tools import WandBMetricsWriter


RUN_METADATA_SCHEMA_VERSION = "jmoshivis.training-run.v2"
RUN_METADATA_NAME = "run.json"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_fingerprint(path: str | os.PathLike[str], *, count_lines: bool) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    records = 0
    with resolved.open("rb") as handle:
        if count_lines:
            for line in handle:
                digest.update(line)
                if line.strip():
                    records += 1
        else:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
    }
    if count_lines:
        result["records"] = records
    return result


def initialize_training_seed(
    seed: int, device: torch.device | None = None
) -> dict[str, Any]:
    """Seed Python, NumPy, Torch, and CUDA identically on every rank."""

    # torch.manual_seed() delegates to cuda.manual_seed_all() when CUDA is
    # available, creating peer-device contexts in one-process-per-GPU jobs.
    # Seed the CPU and exactly one assigned CUDA generator directly instead.
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device is not None and device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA training device must have an explicit rank-local index")
        # Lazily initialize only the explicitly assigned device before
        # accessing its generator object.
        torch.cuda.get_rng_state(device)
        if device.index >= len(torch.cuda.default_generators):
            raise RuntimeError(f"no CUDA generator exists for assigned device {device}")
        torch.cuda.default_generators[device.index].manual_seed(seed)
    torch_state = torch.get_rng_state().cpu().numpy().tobytes()
    cuda_state = None
    if device is not None and device.type == "cuda":
        # Query only this rank's assigned device. get_rng_state_all() would
        # create CUDA contexts on peer GPUs and waste scarce training memory.
        cuda_state = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return {
        "seed": seed,
        "device_specific": False,
        "python_state_sha256": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
        "numpy_state_sha256": hashlib.sha256(
            repr(np.random.get_state()).encode("utf-8")
        ).hexdigest(),
        "torch_state_sha256": hashlib.sha256(torch_state).hexdigest(),
        "cuda_state_sha256": (
            None if cuda_state is None else hashlib.sha256(cuda_state).hexdigest()
        ),
    }


def build_runtime_fingerprint(accelerator: Accelerator) -> dict[str, Any]:
    """Capture software and assigned-device identity on every training rank."""

    package_versions: dict[str, str] = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "sphn",
        "sentencepiece",
        "safetensors",
        "accelerate",
        "moshi",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    software = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": package_versions,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    device = getattr(accelerator, "device", torch.device("cpu"))
    rank = int(getattr(accelerator, "process_index", 0))
    device_record: dict[str, Any] = {
        "rank": rank,
        "type": device.type,
        "index": device.index,
        "name": "cpu",
        "capability": None,
    }
    if device.type == "cuda":
        device_record["name"] = torch.cuda.get_device_name(device)
        device_record["capability"] = list(torch.cuda.get_device_capability(device))
    gathered = gather_object([{"software": software, "device": device_record}])
    if not isinstance(gathered, list) or not gathered:
        raise RuntimeError("failed to gather runtime fingerprints")
    canonical_software = _canonical_json_bytes(gathered[0]["software"])
    if any(
        _canonical_json_bytes(item["software"]) != canonical_software
        for item in gathered[1:]
    ):
        raise RuntimeError("software runtime differs across distributed ranks")
    return {
        **software,
        "devices": sorted(
            (item["device"] for item in gathered), key=lambda item: item["rank"]
        ),
    }


def validate_training_paths(args: DictConfig) -> None:
    required_files = {
        "paths.kyuteye_config": args.paths.kyuteye_config,
        "paths.tokenizer": args.paths.tokenizer,
        "paths.moshivis_weight": args.paths.moshivis_weight,
        "data.train_data": args.data.train_data,
        "data.eval_data": args.data.eval_data,
    }
    problems = [
        f"{label} is not a file: {path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if problems:
        raise FileNotFoundError("invalid training inputs:\n- " + "\n- ".join(problems))


def verify_training_asset_manifest(
    args: DictConfig, accelerator: Accelerator
) -> dict[str, Any]:
    """Verify v3 teacher assets on rank 0 and broadcast the signed result."""

    result: dict[str, Any] | None = None
    error_message: str | None = None
    if accelerator.is_main_process:
        try:
            manifest_value = args.data.get("asset_manifest", None)
            if manifest_value is None or not str(manifest_value).strip():
                raise AssetManifestError(
                    "data.asset_manifest is required for paper-quality v3 training; "
                    "generate it before launching the run"
                )
            mode = str(args.data.get("asset_manifest_verification", "")).strip()
            if not mode:
                raise AssetManifestError(
                    "data.asset_manifest_verification must be explicitly set to "
                    "'fast' or 'full'"
                )
            result = verify_asset_manifest(
                str(manifest_value),
                {
                    "train": str(args.data.train_data),
                    "eval": str(args.data.eval_data),
                },
                mode=mode,
            )
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
    shared: list[Any] = [result, error_message]
    broadcast_object_list(shared)
    if shared[1] is not None:
        raise AssetManifestError(
            f"training asset manifest verification failed: {shared[1]}"
        )
    if not isinstance(shared[0], dict) or shared[0].get("status") != "verified":
        raise AssetManifestError("training asset manifest did not return a verified result")
    return shared[0]


def build_run_metadata(
    args: DictConfig,
    accelerator: Accelerator,
    visual_adapter_init_audit: dict[str, Any],
    *,
    mimi_weight: str | os.PathLike[str],
    seed_state: dict[str, Any],
    asset_manifest_verification: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint the resolved run configuration and immutable data inputs."""

    runtime = build_runtime_fingerprint(accelerator)
    payload: dict[str, Any] | None = None
    if accelerator.is_main_process:
        resolved_config = OmegaConf.to_container(args, resolve=True)
        assert isinstance(resolved_config, dict)
        train_config = resolved_config.get("train")
        if isinstance(train_config, dict):
            # The opt-in checkpoint location selects state; it does not define
            # the mathematical run and may change from an explicit path to latest.
            train_config.pop("resume_from", None)
        implementation_root = Path(__file__).resolve().parent
        implementation_files = sorted(implementation_root.rglob("*.py"))
        train_dataset = _file_fingerprint(args.data.train_data, count_lines=True)
        eval_dataset = _file_fingerprint(args.data.eval_data, count_lines=True)
        train_drop_count = train_dataset["records"] % accelerator.num_processes
        payload = {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "resolved_config": resolved_config,
            "distributed": {
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
                "accelerate_version": accelerate.__version__,
            },
            "datasets": {
                "train": train_dataset,
                "eval": eval_dataset,
            },
            "runtime": runtime,
            "asset_manifest_verification": asset_manifest_verification,
            "tokenizer": _file_fingerprint(args.paths.tokenizer, count_lines=False),
            "model_config": _file_fingerprint(
                args.paths.kyuteye_config, count_lines=False
            ),
            "base_model": _file_fingerprint(
                args.paths.moshivis_weight, count_lines=False
            ),
            "mimi_model": _file_fingerprint(mimi_weight, count_lines=False),
            "implementation": {
                source.relative_to(implementation_root).as_posix(): _file_fingerprint(
                    source, count_lines=False
                )
                for source in implementation_files
            },
            "seed_state": seed_state,
            "data_resume_policy": {
                "name": "exact_rank_stream_from_micro_step",
                "cursor": "checkpoint.micro_step",
                "rank_local_sample_offset": "micro_step * train.batch_size",
                "schedule_replay": "source_and_modality_from_seed",
                "row_seek": "per_source_per_modality_jsonl_cursor",
            },
            "data_sampling_policy": {
                "sample_unit": "one_jsonl_row_one_sample",
                "train_row_sharding": "shuffle_then_drop_remainder_rank_stride",
                "eval_row_sharding": "no_drop_rank_stride",
                "train_rows_dropped_per_epoch": train_drop_count,
                "train_rows_kept_per_epoch": train_dataset["records"]
                - train_drop_count,
                "world_size": accelerator.num_processes,
                "modality_schedule": "rank_balanced_per_optimizer_window",
                "source_schedule": "common_source_schedule_across_ranks",
                "long_row_crop_policy": str(
                    args.data.get("speech_chunk_policy", "one_random_crop_per_row")
                ),
            },
            "visual_adapter_init_audit": visual_adapter_init_audit,
        }
        # Normalize tuples and other JSON-compatible containers now so the
        # in-memory metadata compares exactly with its later JSON round-trip.
        payload = json.loads(_canonical_json_bytes(payload))
        payload["run_fingerprint"] = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
    shared: list[dict[str, Any] | None] = [payload]
    broadcast_object_list(shared)
    if shared[0] is None:
        raise RuntimeError("main process failed to construct run metadata")
    return shared[0]


def _write_run_record(
    checkpoint_dir: Path, metadata: dict[str, Any], *, status: str
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output = checkpoint_dir / RUN_METADATA_NAME
    temporary = checkpoint_dir / f".{RUN_METADATA_NAME}.incomplete"
    record = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "status": status,
        "updated_unix": time.time(),
        "metadata": metadata,
    }
    with temporary.open("wb") as handle:
        handle.write(_canonical_json_bytes(record))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    descriptor = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_run_record(checkpoint_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_dir / RUN_METADATA_NAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run metadata {path}: {error}") from error
    if record.get("schema_version") != RUN_METADATA_SCHEMA_VERSION:
        raise ValueError(f"unsupported run metadata schema in {path}")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError(f"run metadata payload is missing in {path}")
    return record


def _completed_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    completed: list[tuple[int, Path]] = []
    if not checkpoint_dir.is_dir():
        return completed
    for child in checkpoint_dir.iterdir():
        match = re.fullmatch(r"step_(\d+)", child.name)
        if (
            match is not None
            and child.is_dir()
            and (child / TRAINING_CHECKPOINT_COMPLETE).is_file()
        ):
            completed.append((int(match.group(1)), child))
    return sorted(completed)


def resolve_resume_path(
    checkpoint_dir: Path, resume_from: str | None
) -> Path | None:
    if resume_from is None or not str(resume_from).strip():
        return None
    if str(resume_from).strip().lower() == "latest":
        completed = _completed_checkpoints(checkpoint_dir)
        if not completed:
            raise ValueError(
                f"resume_from=latest requested but no complete checkpoint exists in {checkpoint_dir}"
            )
        return completed[-1][1]
    return Path(str(resume_from)).expanduser().resolve()


def rank_local_resume_sample_index(
    checkpoint_state: dict[str, Any] | None,
    *,
    local_batch_size: int,
    accumulation_steps: int,
) -> int:
    """Translate a checkpoint cursor into samples consumed by each DDP rank.

    ``micro_step`` counts yielded rank-local batches, including complete AMP
    overflow windows that did not advance ``global_step``.  Checkpoints are
    published only at optimizer-window boundaries, so this cursor can restore
    the data stream exactly without serializing generator objects.
    """

    if checkpoint_state is None:
        return 0
    if local_batch_size < 1:
        raise ValueError(
            f"local_batch_size must be positive, got {local_batch_size}"
        )
    if accumulation_steps < 1:
        raise ValueError(
            f"accumulation_steps must be positive, got {accumulation_steps}"
        )
    micro_step = checkpoint_state.get("micro_step")
    global_step = checkpoint_state.get("global_step")
    stored_accumulation = checkpoint_state.get("accumulation_steps")
    if not isinstance(micro_step, int) or micro_step < 0:
        raise ValueError(f"invalid checkpoint micro_step: {micro_step!r}")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError(f"invalid checkpoint global_step: {global_step!r}")
    if stored_accumulation != accumulation_steps:
        raise ValueError(
            "checkpoint accumulation_steps differs from the current run: "
            f"{stored_accumulation!r} != {accumulation_steps}"
        )
    if micro_step % accumulation_steps:
        raise ValueError(
            f"checkpoint micro_step={micro_step} is not at an optimizer-window "
            f"boundary of {accumulation_steps} micro-steps"
        )
    minimum_micro_steps = global_step * accumulation_steps
    if micro_step < minimum_micro_steps:
        raise ValueError(
            f"checkpoint micro_step={micro_step} cannot contain global_step="
            f"{global_step} with accumulation_steps={accumulation_steps}"
        )
    return micro_step * local_batch_size


def validate_run_directory(
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    resume_path: Path | None,
) -> dict[str, Any] | None:
    if resume_path is not None and resume_path.parent.resolve() != checkpoint_dir.resolve():
        raise ValueError(
            f"resume checkpoint {resume_path} is outside configured checkpoint_dir "
            f"{checkpoint_dir}"
        )
    record = _read_run_record(checkpoint_dir)
    completed = _completed_checkpoints(checkpoint_dir)
    if record is not None and record["metadata"] != metadata:
        raise ValueError(
            f"checkpoint directory belongs to an incompatible run: {checkpoint_dir}"
        )
    if resume_path is None and completed:
        raise ValueError(
            f"complete checkpoints already exist in {checkpoint_dir}; set train.resume_from "
            "explicitly (for example, 'latest') or choose a new checkpoint_dir"
        )
    if resume_path is None and record is not None and record.get("status") == "complete":
        raise ValueError(
            f"run {checkpoint_dir} is already complete; choose a new checkpoint_dir"
        )
    if resume_path is None and checkpoint_dir.is_dir():
        legacy_exports = [
            child.name
            for child in checkpoint_dir.iterdir()
            if child.is_file()
            and (
                re.fullmatch(r"step_\d+\.safetensors", child.name)
                or re.fullmatch(r"final_step_\d+\.safetensors", child.name)
            )
        ]
        if legacy_exports:
            raise ValueError(
                f"checkpoint_dir contains existing model exports ({legacy_exports}); "
                "choose a new checkpoint_dir instead of silently reusing it"
            )
    if resume_path is not None and record is None:
        raise ValueError(
            f"refusing resume without {RUN_METADATA_NAME} in {checkpoint_dir}"
        )
    state: dict[str, Any] | None = None
    if resume_path is not None:
        state = read_training_checkpoint_metadata(resume_path)
        if state.get("run_fingerprint") != metadata.get("run_fingerprint"):
            raise ValueError(
                f"refusing incompatible resume checkpoint {resume_path}: run fingerprint differs"
            )
        if state.get("run_metadata") != metadata:
            raise ValueError(
                f"refusing incompatible resume checkpoint {resume_path}: run metadata differs"
            )
    return state


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig):
    accelerator = Accelerator(mixed_precision="bf16")
    # DDP では各 rank が自分の GPU だけを使うよう accelerator.device を用いる (素の "cuda" だと rank1 も GPU0 にコンテキストを作る)
    device = accelerator.device
    validate_training_paths(args)
    asset_manifest_verification = verify_training_asset_manifest(args, accelerator)
    seed_state = initialize_training_seed(int(args.train.seed), device=device)
    # trainer.args は損失重みも参照するため、resolved config/fingerprint の
    # 作成前に trainer セクションの互換設定を train へ正規化する。
    for key in ("text_padding_weight", "first_codebook_weight_multiplier"):
        if key not in args.train and "trainer" in args and key in args.trainer:
            with open_dict(args.train):
                args.train[key] = args.trainer[key]
    accumulation_steps = gradient_accumulation_steps(
        int(args.train.target_batch_size),
        int(args.train.batch_size),
        accelerator.num_processes,
    )
    accelerator.gradient_accumulation_steps = accumulation_steps
    kyuteye_config = KyuteyeConfig.from_yml(args.paths.kyuteye_config)

    # --- Tokenizer / Processor ---
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.paths.tokenizer)

    if accelerator.is_main_process:
        print("Loading Mimi and MoshiVis...")
    mimi_weight = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.mimi_name,
    )
    # moshi 0.2.x で第2引数が mimi_config に変わったため device はキーワード指定
    mimi = get_mimi(mimi_weight, device=device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # Config からDepformer/Gate/CrossAttention設定を取得
    depformer_config = dict(args.get("depformer", {})) if "depformer" in args else None
    gate_config = dict(args.get("gate", {})) if "gate" in args else None
    cross_attention_config = (
        dict(args.get("cross_attention", {})) if "cross_attention" in args else None
    )

    print("Start get_moshi_vis_train")
    print(f"  Depformer config: {depformer_config}")
    print(f"  Gate config: {gate_config}")
    print(f"  CrossAttention config: {cross_attention_config}")

    moshi_vis, image_embedder = get_moshi_vis_train(
        kyuteye_config=kyuteye_config,
        moshivis_weight=args.paths.moshivis_weight,
        device=device,
        dtype=torch.bfloat16,
        strict=False,
        freeze_backbone=True,
        depformer_config=depformer_config,
        gate_config=gate_config,
        cross_attention_config=cross_attention_config,
    )

    visual_adapter_init_audit = getattr(
        moshi_vis, "visual_adapter_init_audit", None
    )
    image_adapter_init_audit = getattr(
        image_embedder, "visual_adapter_init_audit", None
    )
    if not isinstance(visual_adapter_init_audit, dict):
        raise RuntimeError("model did not provide visual_adapter_init_audit")
    if visual_adapter_init_audit != image_adapter_init_audit:
        raise RuntimeError("model and image embedder visual_adapter_init_audit disagree")
    run_metadata = build_run_metadata(
        args,
        accelerator,
        visual_adapter_init_audit,
        mimi_weight=mimi_weight,
        seed_state=seed_state,
        asset_manifest_verification=asset_manifest_verification,
    )
    checkpoint_dir = Path(str(args.train.checkpoint_dir)).expanduser().resolve()
    resume_value = args.train.get("resume_from", None)
    resume_path = resolve_resume_path(
        checkpoint_dir,
        None if resume_value is None else str(resume_value),
    )
    resume_state = validate_run_directory(checkpoint_dir, run_metadata, resume_path)
    resume_sample_index = rank_local_resume_sample_index(
        resume_state,
        local_batch_size=int(args.train.batch_size),
        accumulation_steps=accumulation_steps,
    )
    if accelerator.is_main_process:
        _write_run_record(checkpoint_dir, run_metadata, status="running")
    accelerator.wait_for_everyone()

    interleaver = Interleaver(
        tokenizer,
        mimi.frame_rate,
        moshi_vis.text_padding_token_id,
        moshi_vis.end_of_text_padding_id,
        moshi_vis.zero_token_id,
        keep_main_only=True,
        # 日本語SPMトークンは1フレーム(80ms)に複数載ることが多く、既定(False)だと同一フレームの
        # 先行トークンが捨てられる(「は」「、」等が欠落)。True でキューに積んで1フレーム1トークンずつ出力する
        keep_and_shift=True,
        device=device,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    """
    ───────────────────────────────────────────────
    🧠 DataLoader 構成概要（J-MoshiVis）
    ───────────────────────────────────────────────
    ・本セクションでは、Interleaver（音声・テキスト統合トークナイザ）を介して、
    MoshiVis が期待するフォーマットの学習データを読み込みます。

    ・出力は `Batch` オブジェクトであり、
    以下の2つの主構成を持ちます：

        Batch(
            codes: torch.Tensor,                # 量子化コード列 [B, D, T]
            condition_attributes: Optional[...] # 補助条件（例: 画像特徴やアライメント情報）
        )

    ───────────────────────────────────────────────
    📦 Shape specification
    ───────────────────────────────────────────────
    - codes: Tensor of shape [B, D, T]
        B : Batch size
            └─ 各ステップで並列処理するサンプル数
        D : Depth axis
            └─ Residual Quantizer（RQ）層の数
            例: Mimiでは17層（=17コードブック）を使用
        T : Time axis
            └─ 音声を一定フレーム幅で分割した系列長
            例: 約125ステップ ≒ 10秒前後の音声長

    例:
        >>> batch.codes.shape
        torch.Size([2, 17, 125])
        → 2サンプル / 各サンプル17層 / 125フレームのトークン列
    """
    target_len = int(mimi.frame_rate * args.duration_sec)
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.train.batch_size,
        seed=args.train.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
        image_root=args.data.image_root,
        image_embedder=image_embedder,
        device=device,
        mode="mixed",
        text_tokenizer=tokenizer,
        target_len=target_len,
        # The mixed sampler stratifies over exactly one optimizer update.
        modality_window_size=int(args.train.target_batch_size),
        start_sample_index=resume_sample_index,
    )

    # # --- 1. データローダ生成後に1バッチだけ取り出す ---
    # data_iter = iter(data_loader)
    # first_batch = next(data_iter)

    # print("=== Batch object ===")
    # print(type(first_batch))
    # print(first_batch)

    # # --- 2. 中身を要素ごとに確認 ---
    # if hasattr(first_batch, "codes"):
    #     print("\n[Shape] codes:", first_batch.codes.shape)
    #     if first_batch.condition_attributes:
    #         print("[Type] condition_attributes:", type(first_batch.condition_attributes))
    #         print("[Count] len(condition_attributes):", len(first_batch.condition_attributes))
    #         print("[Sample 0] condition_attributes[0]:", first_batch.condition_attributes[0])
    # else:
    #     # もしBatchクラスでなくlist形式のままならこちら
    #     print("\nFirst element sample keys:", first_batch[0].keys())
    #     print("First element sample detail:\n", first_batch[0])

    cross_attn_params = []
    gate_params = []
    depformer_params = []
    # other_params は今回空になるのが理想ですが、念のため残します
    other_params = []

    for name, p in moshi_vis.named_parameters():
        if not p.requires_grad:
            continue
        # Gate パラメータ (実際の名前は cross_attention.gate.alpha.*)
        if "gate" in name and "alpha" in name:
            gate_params.append(p)
        # Depformer パラメータ
        elif "depformer" in name or "audio_linears" in name:
            depformer_params.append(p)
        # Cross-Attention パラメータ

        # 【修正2】 norm_cross も CrossAttention グループに含める
        elif "cross_attention" in name or "xa" in name or "norm_cross" in name:
            cross_attn_params.append(p)

        else:
            # ここに出るものがなければOK
            print("[WARN] Unexpected trainable param:", name)
            other_params.append(p)

    embedder_params = [p for p in image_embedder.parameters() if p.requires_grad]

    if other_params:
        other_parameter_ids = {id(parameter) for parameter in other_params}
        unexpected = [
            name
            for name, parameter in moshi_vis.named_parameters()
            if parameter.requires_grad and id(parameter) in other_parameter_ids
        ]
        raise RuntimeError(
            "trainable model parameters are not assigned to an optimizer group: "
            + ", ".join(unexpected)
        )
    grouped_parameters = [
        *cross_attn_params,
        *gate_params,
        *depformer_params,
        *embedder_params,
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("a trainable parameter appears in multiple optimizer groups")
    expected_parameters = [
        *[parameter for parameter in moshi_vis.parameters() if parameter.requires_grad],
        *[
            parameter
            for parameter in image_embedder.parameters()
            if parameter.requires_grad
        ],
    ]
    expected_ids = {id(parameter) for parameter in expected_parameters}
    if set(grouped_ids) != expected_ids:
        raise RuntimeError(
            "optimizer parameter groups do not exactly cover all unique trainable parameters"
        )

    print(f"Trainable params: CrossAttn={len(cross_attn_params)}, Gate={len(gate_params)}, Depformer={len(depformer_params)}, Embedder={len(embedder_params)}")

    if accelerator.is_main_process:
        def print_trainable_parameters(model, model_name="Model"):
            print(f"\n=== Trainable Parameters in {model_name} ===")
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    total_params += num_params
                    print(f"{name}: {num_params:,} params | Shape: {list(param.shape)}")
            print(f"--- Total Trainable Params in {model_name}: {total_params:,} ---\n")
            return total_params

        # MoshiVis本体の学習対象パラメータを表示
        moshi_params = print_trainable_parameters(moshi_vis, "MoshiVis (Adapters)")

        # ImageEmbedderの学習対象パラメータを表示
        embedder_params_count = print_trainable_parameters(image_embedder, "ImageEmbedder (Projection)")

        print(f"🔥 Grand Total Trainable Parameters: {moshi_params + embedder_params_count:,}")

        # もし other_params に何か残っていたら、それも学習対象に加えるべきですが、
        # 上記の修正で norm_cross は CrossAttn に入るため、基本的には空になるはずです。

    # オプティマイザのパラメータグループを構築
    base_lr = float(args.train.get("learning_rate", 1e-4))
    weight_decay = float(args.train.get("weight_decay", 0.0))
    param_groups = [
        {"params": cross_attn_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": embedder_params,   "lr": base_lr, "weight_decay": weight_decay},
    ]

    # Gate パラメータ (凍結されていなければ追加)
    if gate_params:
        gate_lr = gate_config.get("learning_rate", 1e-5) if gate_config else 1e-5
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})
        print(f"🔧 Gate params added to optimizer (lr={gate_lr})")

    # Depformer パラメータ (設定されていれば追加)
    if depformer_params:
        depformer_lr = depformer_config.get("learning_rate", 5e-6) if depformer_config else 5e-6
        param_groups.append({"params": depformer_params, "lr": depformer_lr, "weight_decay": 0.0})
        print(f"🔧 Depformer params added to optimizer (lr={depformer_lr})")

    optimizer = torch.optim.AdamW(param_groups, fused=True)

    # 線形 warmup → cosine 減衰 (min_lr_ratio まで)。optimizer step 単位で進める
    max_steps = int(args.train.get("max_steps", 0))
    warmup_steps = int(args.train.get("warmup_steps", 0))
    min_lr_ratio = float(args.train.get("min_lr_ratio", 0.1))
    scheduler = None
    if max_steps > 0:
        import math
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"🔧 LR schedule: base={base_lr}, warmup={warmup_steps}, max_steps={max_steps}, min_ratio={min_lr_ratio}")
        # Do not pass this scheduler to Accelerator.prepare(): with the default
        # split_batches=False, AcceleratedScheduler advances once per process.
        # Our loader is already rank-sharded and max_steps counts optimizer
        # updates, so the plain scheduler must advance exactly once here.
        accelerator.register_for_checkpointing(scheduler)

    moshi_vis.forward = moshi_vis.forward_speech

    # DDP準備. The scheduler deliberately remains unwrapped (see above), while
    # its underlying optimizer param_groups are the same ones wrapped below.
    moshi_vis, image_embedder, optimizer, data_loader = accelerator.prepare(
        moshi_vis, image_embedder, optimizer, data_loader
    )

    writer = None

    # メインプロセスのみ WandB を起動し、writer に代入
    if accelerator.is_main_process:
        writer = WandBMetricsWriter(
            project_name="J-MoshiVis-Training",
            model_name="j-moshivis"
        )

    # --- Trainer Setup ---
    trainer = JmoshiVisTrainer(
        moshi_vis,
        optimizer,
        device,
        args.train,
        accelerator,
        image_embedder=image_embedder,
        writer=writer,
        tokenizer=tokenizer,
        scheduler=scheduler,
        run_metadata=run_metadata,
    )

    if resume_path is not None:
        # Validate metadata before mutating model/optimizer state.  The loader
        # has already been parameterized with the checkpoint's rank-local
        # micro-step cursor; it replays source/modality choices on CPU and seeks
        # each underlying JSONL iterator without re-encoding consumed samples.
        trainer.inspect_training_checkpoint(resume_path)
        trainer.load_training_checkpoint(resume_path)
        if accelerator.is_main_process:
            print(
                f"Resumed complete training state from {resume_path} "
                f"at optimizer step {trainer.global_step}; each rank resumes "
                f"exactly at local sample {resume_sample_index}"
            )

    # --- Training ---
    # データイテレータは無限 (エポック境界で再シャッフルして継続) なので、train.max_steps で打ち切る
    epochs = int(args.train.get("epochs", 1)) if max_steps == 0 else 10**9
    start_epoch = max(1, trainer.epoch)
    for epoch in range(start_epoch, epochs + 1):
        if trainer.stop_requested:
            break
        trainer.train_epoch(data_loader, epoch)
        if trainer.stop_requested:
            break

    # --- Save resumable state, then a separate model-only inference export ---
    accelerator.wait_for_everyone()
    if trainer.global_step > 0:
        trainer.save_training_checkpoint()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.save_checkpoint(
            str(checkpoint_dir / f"final_step_{trainer.global_step}.safetensors")
        )
        _write_run_record(checkpoint_dir, run_metadata, status="complete")
        if writer is not None:
            writer.finish()
    accelerator.wait_for_everyone()
    # All ranks must explicitly release Accelerate's default process group.
    # Without this, PyTorch 2.4+ reports that NCCL was still initialized when
    # the interpreter exits, even after the final synchronization completed.
    accelerator.end_training()


if __name__ == "__main__":
    main()
