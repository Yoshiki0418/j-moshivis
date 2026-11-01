"""Build and verify immutable manifests for v3 training teacher assets.

The v3 JSONL is only an index.  A row is trainable only when its audio and
the two co-located teacher files still match the bytes that were reviewed:

* ``row["path"]`` (WAV)
* ``<wav parent>/image.jpg``
* ``<wav parent>/assistant_dialogue.refined.json``

The manifest is newline-delimited JSON so both creation and startup
verification stay bounded in memory for the full dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "jmoshivis.asset-manifest.v1"
ASSET_FILENAMES = {
    "image": "image.jpg",
    "alignment": "assistant_dialogue.refined.json",
}
VERIFY_MODES = frozenset({"fast", "full"})
DURATION_TOLERANCE_SECONDS = 1e-3
PROGRESS_INTERVAL_SAMPLES = 1_000


class AssetManifestError(RuntimeError):
    """Raised when an index, asset, or manifest is not safe to train from."""


def _report_progress(operation: str, completed: int, total: int) -> None:
    """Emit sparse, immediately visible progress without affecting artifacts."""

    if (
        completed not in {0, total}
        and completed % PROGRESS_INTERVAL_SAMPLES != 0
    ):
        return
    percentage = 100.0 if total == 0 else completed / total * 100.0
    print(
        f"[asset-manifest] {operation}: {completed:,}/{total:,} samples "
        f"({percentage:.1f}%)",
        flush=True,
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    records = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if line.strip():
                records += 1
            elif line:
                raise AssetManifestError(
                    f"blank JSONL record in {path} at line {line_number}"
                )
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "records": records,
    }


def _read_json_row(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetManifestError(
            f"invalid JSON in {path} at line {line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AssetManifestError(
            f"JSONL record must be an object in {path} at line {line_number}"
        )
    return value


def _row_identity_and_assets(
    row: Mapping[str, Any], *, jsonl_path: Path, line_number: int
) -> tuple[str, list[tuple[str, Path]], str | None]:
    uid = row.get("uid")
    if not isinstance(uid, str) or not uid.strip():
        raise AssetManifestError(
            f"missing stable string uid in {jsonl_path} at line {line_number}"
        )
    auxiliary_task = row.get("auxiliary_task")
    is_unverified_visual = (
        row.get("verification_status") == "unverified_vlm_visual"
        or (
            isinstance(auxiliary_task, Mapping)
            and auxiliary_task.get("grounding_source") == "vlm_visual"
        )
    )
    if is_unverified_visual and row.get("review_required") is not True:
        raise AssetManifestError(
            "unverified VLM-visual sample is missing review_required=true for "
            f"sample {uid!r} in {jsonl_path} at line {line_number}"
        )
    expected_image_asset_sha256: str | None = None
    if row.get("review_required") is True:
        review_id = row.get("review_id")
        evidence = row.get("human_review")
        if (
            not isinstance(review_id, str)
            or len(review_id) != 64
            or any(character not in "0123456789abcdef" for character in review_id)
        ):
            raise AssetManifestError(
                f"review-gated sample {uid!r} has no valid review_id"
            )
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("status") != "approved"
            or evidence.get("review_id") != review_id
        ):
            raise AssetManifestError(
                "review-gated sample lacks exact approved human_review evidence "
                f"for sample {uid!r} (review_id={review_id!r})"
            )
        expected_image_asset_sha256 = row.get("image_asset_sha256")
        if (
            not isinstance(expected_image_asset_sha256, str)
            or len(expected_image_asset_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_image_asset_sha256
            )
        ):
            raise AssetManifestError(
                "review-gated sample has no valid image_asset_sha256 for "
                f"sample {uid!r}"
            )
    audio_value = row.get("path")
    if not isinstance(audio_value, str) or not audio_value.strip():
        raise AssetManifestError(
            f"missing WAV path in {jsonl_path} at line {line_number}"
        )
    audio = Path(audio_value).expanduser().resolve(strict=False)
    if audio.suffix.lower() != ".wav":
        raise AssetManifestError(
            f"teacher audio is not a WAV for sample {uid!r}: {audio}"
        )
    image = audio.parent / ASSET_FILENAMES["image"]
    alignment = audio.parent / ASSET_FILENAMES["alignment"]
    declared_image = row.get("image")
    if declared_image is not None:
        if not isinstance(declared_image, str) or not declared_image.strip():
            raise AssetManifestError(
                f"invalid declared image path for sample {uid!r}"
            )
        if Path(declared_image).expanduser().resolve(strict=False) != image:
            raise AssetManifestError(
                "declared image conflicts with the co-located v3 teacher asset "
                f"for sample {uid!r}: declared={declared_image}, expected={image}"
            )
    return (
        uid,
        [("audio", audio), ("image", image), ("alignment", alignment)],
        expected_image_asset_sha256,
    )


def _row_duration_seconds(
    row: Mapping[str, Any], *, jsonl_path: Path, line_number: int, sample_id: str
) -> float:
    """Return a finite, positive duration declared by the source index."""

    value = row.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetManifestError(
            f"missing/invalid duration for sample {sample_id!r} in "
            f"{jsonl_path} at line {line_number}"
        )
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise AssetManifestError(
            f"duration must be finite and positive for sample {sample_id!r} in "
            f"{jsonl_path} at line {line_number}: {value!r}"
        )
    return duration


def _asset_stat(path: Path, *, role: str, sample_id: str) -> os.stat_result:
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise AssetManifestError(
            f"missing {role} asset for sample {sample_id!r}: {path}"
        ) from error
    if not path.is_file():
        raise AssetManifestError(
            f"{role} asset is not a regular file for sample {sample_id!r}: {path}"
        )
    if stat.st_size <= 0:
        raise AssetManifestError(
            f"empty {role} asset for sample {sample_id!r}: {path}"
        )
    return stat


def _inspect_alignment(
    path: Path,
    sample_id: str,
    *,
    declared_duration_seconds: float,
) -> tuple[list[str], int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetManifestError(
            f"invalid alignment JSON for sample {sample_id!r}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AssetManifestError(
            f"alignment must be a JSON object for sample {sample_id!r}: {path}"
        )
    pretokenized = "token_alignments" in value
    alignments = (
        value.get("token_alignments") if pretokenized else value.get("alignments")
    )
    if not isinstance(alignments, list) or not alignments:
        raise AssetManifestError(
            f"alignment has no token_alignments/alignments for sample {sample_id!r}: {path}"
        )
    roles: set[str] = set()
    for index, item in enumerate(alignments):
        if not isinstance(item, list) or len(item) != 3:
            raise AssetManifestError(
                f"malformed alignment {index} for sample {sample_id!r}: {path}"
            )
        payload, interval, role = item
        if not isinstance(role, str) or not role:
            raise AssetManifestError(
                f"invalid alignment role at index {index} for sample "
                f"{sample_id!r}: {path}"
            )
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in interval
            )
        ):
            raise AssetManifestError(
                f"invalid alignment interval at index {index} for sample "
                f"{sample_id!r}: {path}"
            )
        start, end = (float(interval[0]), float(interval[1]))
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise AssetManifestError(
                f"non-finite/reversed alignment interval at index {index} for "
                f"sample {sample_id!r}: [{start}, {end}] in {path}"
            )
        if end > declared_duration_seconds + DURATION_TOLERANCE_SECONDS:
            raise AssetManifestError(
                f"alignment exceeds declared duration at index {index} for sample "
                f"{sample_id!r}: end={end}, duration={declared_duration_seconds}, "
                f"path={path}"
            )
        if pretokenized:
            if (
                not isinstance(payload, list)
                or not payload
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in payload
                )
            ):
                raise AssetManifestError(
                    f"invalid token IDs at alignment index {index} for sample "
                    f"{sample_id!r}: {path}"
                )
        elif not isinstance(payload, str) or not payload:
            raise AssetManifestError(
                f"invalid text at alignment index {index} for sample "
                f"{sample_id!r}: {path}"
            )
        roles.add(role)
    missing_roles = {"ASSISTANT", "USER"} - roles
    if missing_roles:
        raise AssetManifestError(
            f"alignment is missing required roles {sorted(missing_roles)} for "
            f"sample {sample_id!r}: {path}"
        )
    return sorted(roles), len(alignments)


def _inspect_audio(path: Path, sample_id: str) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            metadata = {
                "channels": wav.getnchannels(),
                "sample_rate": wav.getframerate(),
                "frames": wav.getnframes(),
            }
    except (OSError, EOFError, wave.Error) as error:
        raise AssetManifestError(
            f"invalid WAV for sample {sample_id!r}: {path}: {error}"
        ) from error
    if min(metadata.values()) <= 0:
        raise AssetManifestError(
            f"WAV has invalid stream metadata for sample {sample_id!r}: {path}"
        )
    if metadata["channels"] != 2:
        raise AssetManifestError(
            "training audio must be stereo for sample "
            f"{sample_id!r}: {path} has {metadata['channels']} channel(s)"
        )
    return metadata


def _validate_audio_duration(
    metadata: Mapping[str, int], declared_duration: float, sample_id: str, path: Path
) -> float:
    """Check the JSONL duration against the authoritative WAV header."""

    actual_duration = metadata["frames"] / metadata["sample_rate"]
    tolerance = max(DURATION_TOLERANCE_SECONDS, 1.0 / metadata["sample_rate"])
    if abs(declared_duration - actual_duration) > tolerance:
        raise AssetManifestError(
            "JSONL/WAV duration mismatch for sample "
            f"{sample_id!r}: declared={declared_duration:.9f}s, "
            f"header={actual_duration:.9f}s, tolerance={tolerance:.9f}s, path={path}"
        )
    return actual_duration


def _inspect_image(path: Path, sample_id: str) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except Exception as error:
        raise AssetManifestError(
            f"invalid image for sample {sample_id!r}: {path}: {error}"
        ) from error
    if width <= 0 or height <= 0:
        raise AssetManifestError(
            f"image has invalid dimensions for sample {sample_id!r}: {path}"
        )
    return {"width": width, "height": height, "format": image_format}


def _build_sample_record(
    split: str,
    row_index: int,
    sample_id: str,
    row_sha256: str,
    assets: Sequence[tuple[str, Path]],
    declared_duration_seconds: float,
    expected_image_asset_sha256: str | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    roles: list[str] | None = None
    alignment_count: int | None = None
    for role, path in assets:
        stat = _asset_stat(path, role=role, sample_id=sample_id)
        before_identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        record: dict[str, Any] = {
            "role": role,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        if (
            role == "image"
            and expected_image_asset_sha256 is not None
            and record["sha256"] != expected_image_asset_sha256
        ):
            raise AssetManifestError(
                "review-gated copied image hash differs from image_asset_sha256 "
                f"for sample {sample_id!r}: expected={expected_image_asset_sha256}, "
                f"actual={record['sha256']}, path={path}"
            )
        if role == "audio":
            audio_metadata = _inspect_audio(path, sample_id)
            record.update(audio_metadata)
            record["duration_seconds"] = _validate_audio_duration(
                audio_metadata, declared_duration_seconds, sample_id, path
            )
        elif role == "image":
            record.update(_inspect_image(path, sample_id))
        elif role == "alignment":
            roles, alignment_count = _inspect_alignment(
                path,
                sample_id,
                declared_duration_seconds=declared_duration_seconds,
            )
            record["entries"] = alignment_count
        final_stat = _asset_stat(path, role=role, sample_id=sample_id)
        after_identity = (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        )
        if after_identity != before_identity:
            raise AssetManifestError(
                f"{role} asset changed while hashing/inspecting sample "
                f"{sample_id!r}: {path}"
            )
        records.append(record)
    assert roles is not None and alignment_count is not None
    return {
        "type": "sample",
        "split": split,
        "row_index": row_index,
        "sample_id": sample_id,
        "row_sha256": row_sha256,
        "dialogue_roles": roles,
        "assets": records,
    }


def _ordered_sources(sources: Mapping[str, str | os.PathLike[str]]) -> list[tuple[str, Path]]:
    if not sources:
        raise AssetManifestError("at least one source JSONL is required")
    order = {"train": 0, "eval": 1}
    result = [
        (str(split), Path(path).expanduser().resolve(strict=True))
        for split, path in sources.items()
    ]
    result.sort(key=lambda item: (order.get(item[0], 2), item[0]))
    if len({split for split, _ in result}) != len(result):
        raise AssetManifestError("duplicate split names")
    return result


def _iter_completed_bounded(
    jobs: Iterable[
        tuple[str, int, str, str, Sequence[tuple[str, Path]], float, str | None]
    ],
    *,
    workers: int,
) -> Iterator[dict[str, Any]]:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asset-hash") as pool:
        pending: deque[Future[dict[str, Any]]] = deque()

        def submit_one() -> bool:
            try:
                job = next(iterator)
            except StopIteration:
                return False
            pending.append(pool.submit(_build_sample_record, *job))
            return True

        for _ in range(workers * 2):
            if not submit_one():
                break
        while pending:
            # Preserve JSONL row order even if later hash jobs finish first.
            yield pending.popleft().result()
            submit_one()


def _iter_jobs(
    ordered_sources: Sequence[tuple[str, Path]],
) -> Iterator[
    tuple[str, int, str, str, Sequence[tuple[str, Path]], float, str | None]
]:
    seen_ids: dict[str, tuple[str, int]] = {}
    seen_paths: dict[Path, tuple[str, str, int]] = {}
    for split, jsonl_path in ordered_sources:
        with jsonl_path.open("rb") as handle:
            for row_index, raw_line in enumerate(handle):
                line_number = row_index + 1
                if not raw_line.strip():
                    raise AssetManifestError(
                        f"blank JSONL record in {jsonl_path} at line {line_number}"
                    )
                row = _read_json_row(raw_line, jsonl_path, line_number)
                sample_id, assets, expected_image_asset_sha256 = _row_identity_and_assets(
                    row, jsonl_path=jsonl_path, line_number=line_number
                )
                declared_duration = _row_duration_seconds(
                    row,
                    jsonl_path=jsonl_path,
                    line_number=line_number,
                    sample_id=sample_id,
                )
                if sample_id in seen_ids:
                    previous = seen_ids[sample_id]
                    raise AssetManifestError(
                        f"duplicate sample uid {sample_id!r}: {previous} and "
                        f"({split!r}, {line_number})"
                    )
                seen_ids[sample_id] = (split, line_number)
                for role, path in assets:
                    if path in seen_paths:
                        previous = seen_paths[path]
                        raise AssetManifestError(
                            f"duplicate/conflicting asset path {path}: {previous} and "
                            f"({sample_id!r}, {role!r}, {line_number})"
                        )
                    seen_paths[path] = (sample_id, role, line_number)
                yield (
                    split,
                    row_index,
                    sample_id,
                    hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(),
                    assets,
                    declared_duration,
                    expected_image_asset_sha256,
                )


def build_asset_manifest(
    sources: Mapping[str, str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Hash all teacher assets and atomically write a deterministic manifest."""

    ordered = _ordered_sources(sources)
    source_records = [
        {"split": split, "jsonl": _jsonl_fingerprint(path)}
        for split, path in ordered
    ]
    sample_count = sum(source["jsonl"]["records"] for source in source_records)
    header = {
        "type": "header",
        "schema_version": SCHEMA_VERSION,
        "sources": source_records,
        "sample_count": sample_count,
        "asset_count": sample_count * 3,
    }
    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    _report_progress("generate", 0, sample_count)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".incomplete",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(_canonical_json(header) + b"\n")
            written = 0
            for record in _iter_completed_bounded(
                _iter_jobs(ordered), workers=workers
            ):
                handle.write(_canonical_json(record) + b"\n")
                written += 1
                _report_progress("generate", written, sample_count)
            if written != sample_count:
                raise AssetManifestError(
                    f"source changed while building manifest: expected {sample_count} "
                    f"records, observed {written}"
                )
            # The first pass fingerprints the indexes before asset hashing.
            # Re-read them before publish so an in-place edit with the same
            # line count cannot produce a self-inconsistent manifest.
            final_source_records = [
                {"split": split, "jsonl": _jsonl_fingerprint(path)}
                for split, path in ordered
            ]
            if final_source_records != source_records:
                raise AssetManifestError(
                    "source JSONL changed while building the asset manifest"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    stat = output_path.stat()
    return {
        "path": str(output_path),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(output_path),
        "sample_count": sample_count,
        "asset_count": sample_count * 3,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssetManifestError(
            f"manifest mismatch for {label}: manifest={actual!r}, current={expected!r}"
        )


def verify_asset_manifest(
    manifest: str | os.PathLike[str],
    sources: Mapping[str, str | os.PathLike[str]],
    *,
    mode: str = "fast",
) -> dict[str, Any]:
    """Verify source indexes and every referenced asset against a manifest.

    ``fast`` verifies path, size, and nanosecond mtime. ``full`` additionally
    re-hashes every asset. Both modes hash the relatively small source JSONLs
    and the manifest itself.
    """

    if mode not in VERIFY_MODES:
        raise ValueError(f"verification mode must be one of {sorted(VERIFY_MODES)}")
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    ordered = _ordered_sources(sources)
    current_sources = [
        {"split": split, "jsonl": _jsonl_fingerprint(path)}
        for split, path in ordered
    ]
    digest = hashlib.sha256()
    observed_samples = 0
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    try:
        handle = manifest_path.open("rb")
    except OSError as error:
        raise AssetManifestError(f"cannot open asset manifest {manifest_path}: {error}") from error
    with handle:
        header_line = handle.readline()
        digest.update(header_line)
        header = _read_json_row(header_line, manifest_path, 1)
        _require_equal("schema_version", header.get("schema_version"), SCHEMA_VERSION)
        _require_equal("record type", header.get("type"), "header")
        _require_equal("sources", header.get("sources"), current_sources)
        expected_samples = sum(item["jsonl"]["records"] for item in current_sources)
        _require_equal("sample_count", header.get("sample_count"), expected_samples)
        _require_equal("asset_count", header.get("asset_count"), expected_samples * 3)
        _report_progress(f"verify ({mode})", 0, expected_samples)

        manifest_line = 1
        for split, jsonl_path in ordered:
            with jsonl_path.open("rb") as source_handle:
                for row_index, raw_line in enumerate(source_handle):
                    line_number = row_index + 1
                    row = _read_json_row(raw_line, jsonl_path, line_number)
                    (
                        sample_id,
                        expected_assets,
                        expected_image_asset_sha256,
                    ) = _row_identity_and_assets(
                        row, jsonl_path=jsonl_path, line_number=line_number
                    )
                    declared_duration = _row_duration_seconds(
                        row,
                        jsonl_path=jsonl_path,
                        line_number=line_number,
                        sample_id=sample_id,
                    )
                    record_line = handle.readline()
                    manifest_line += 1
                    if not record_line:
                        raise AssetManifestError(
                            f"asset manifest ended before sample {sample_id!r}"
                        )
                    digest.update(record_line)
                    record = _read_json_row(record_line, manifest_path, manifest_line)
                    _require_equal("sample type", record.get("type"), "sample")
                    _require_equal("sample split", record.get("split"), split)
                    _require_equal("sample row_index", record.get("row_index"), row_index)
                    _require_equal("sample_id", record.get("sample_id"), sample_id)
                    _require_equal(
                        f"row hash for {sample_id}",
                        record.get("row_sha256"),
                        hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(),
                    )
                    dialogue_roles = record.get("dialogue_roles")
                    if (
                        not isinstance(dialogue_roles, list)
                        or not dialogue_roles
                        or any(
                            not isinstance(role, str) or not role
                            for role in dialogue_roles
                        )
                        or dialogue_roles != sorted(set(dialogue_roles))
                    ):
                        raise AssetManifestError(
                            f"invalid dialogue_roles for sample {sample_id!r}"
                        )
                    if sample_id in seen_ids:
                        raise AssetManifestError(f"duplicate sample uid {sample_id!r}")
                    seen_ids.add(sample_id)
                    assets = record.get("assets")
                    if not isinstance(assets, list) or len(assets) != 3:
                        raise AssetManifestError(
                            f"sample {sample_id!r} must contain exactly three assets"
                        )
                    for asset, (expected_role, expected_path) in zip(
                        assets, expected_assets, strict=True
                    ):
                        if not isinstance(asset, dict):
                            raise AssetManifestError(
                                f"malformed asset record for sample {sample_id!r}"
                            )
                        _require_equal("asset role", asset.get("role"), expected_role)
                        _require_equal("asset path", asset.get("path"), str(expected_path))
                        recorded_sha = asset.get("sha256")
                        if (
                            not isinstance(recorded_sha, str)
                            or len(recorded_sha) != 64
                            or any(character not in "0123456789abcdef" for character in recorded_sha)
                        ):
                            raise AssetManifestError(
                                f"invalid {expected_role} sha256 for sample {sample_id!r}"
                            )
                        if (
                            expected_role == "image"
                            and expected_image_asset_sha256 is not None
                        ):
                            _require_equal(
                                f"review image_asset_sha256 for {sample_id}",
                                recorded_sha,
                                expected_image_asset_sha256,
                            )
                        path_key = str(expected_path)
                        if path_key in seen_paths:
                            raise AssetManifestError(
                                f"duplicate/conflicting asset path: {expected_path}"
                            )
                        seen_paths.add(path_key)
                        stat = _asset_stat(
                            expected_path, role=expected_role, sample_id=sample_id
                        )
                        _require_equal(
                            f"{expected_role} size for {sample_id}",
                            asset.get("size_bytes"),
                            stat.st_size,
                        )
                        _require_equal(
                            f"{expected_role} mtime for {sample_id}",
                            asset.get("mtime_ns"),
                            stat.st_mtime_ns,
                        )
                        if expected_role == "audio":
                            audio_metadata = _inspect_audio(expected_path, sample_id)
                            actual_duration = _validate_audio_duration(
                                audio_metadata,
                                declared_duration,
                                sample_id,
                                expected_path,
                            )
                            for field, value in audio_metadata.items():
                                _require_equal(
                                    f"audio {field} for {sample_id}",
                                    asset.get(field),
                                    value,
                                )
                            _require_equal(
                                f"audio duration_seconds for {sample_id}",
                                asset.get("duration_seconds"),
                                actual_duration,
                            )
                        if (
                            expected_role == "image"
                            and expected_image_asset_sha256 is not None
                        ):
                            # Reviewed samples are expected to be sparse. Re-hash
                            # their copied image even in fast mode so a same-stat
                            # replacement cannot sever the human-review binding.
                            _require_equal(
                                f"review copied image sha256 for {sample_id}",
                                expected_image_asset_sha256,
                                _sha256_file(expected_path),
                            )
                        elif mode == "full":
                            _require_equal(
                                f"{expected_role} sha256 for {sample_id}",
                                recorded_sha,
                                _sha256_file(expected_path),
                            )
                    observed_samples += 1
                    _report_progress(
                        f"verify ({mode})", observed_samples, expected_samples
                    )
        trailing = handle.readline()
        if trailing:
            digest.update(trailing)
            raise AssetManifestError(
                f"asset manifest contains extra records after {observed_samples} samples"
            )

    _require_equal("observed sample count", observed_samples, expected_samples)
    stat = manifest_path.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "verification_mode": mode,
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": stat.st_size,
            "sha256": digest.hexdigest(),
        },
        "sources": current_sources,
        "sample_count": observed_samples,
        "asset_count": observed_samples * 3,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="hash and write a manifest")
    generate.add_argument("--train-jsonl", required=True)
    generate.add_argument("--eval-jsonl", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--workers", type=int, default=8)
    verify = subparsers.add_parser("verify", help="verify an existing manifest")
    verify.add_argument("--train-jsonl", required=True)
    verify.add_argument("--eval-jsonl", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--mode", choices=sorted(VERIFY_MODES), default="fast")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = {"train": args.train_jsonl, "eval": args.eval_jsonl}
    if args.command == "generate":
        result = build_asset_manifest(
            sources, args.output, workers=args.workers
        )
    else:
        result = verify_asset_manifest(args.manifest, sources, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
