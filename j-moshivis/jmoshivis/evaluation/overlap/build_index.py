"""Build a reusable fingerprint index of all v3 train/eval images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from ..schema import read_jsonl, sha256_file
from .fingerprints import fingerprint_provenance, image_fingerprints


IMAGE_INDEX_SCHEMA_VERSION = "jmoshivis.image-index.v1"
INDEX_STATE_BUILDING = "building"
INDEX_STATE_COMPLETE = "complete"
CHECKPOINT_ROWS = 100


def _one(item: tuple[str, str]) -> dict:
    uid, path = item
    try:
        return {"uid": uid, "path": path, **image_fingerprints(path), "error": None}
    except Exception as exc:
        return {"uid": uid, "path": path, "error": f"{type(exc).__name__}: {exc}"}


def image_index_metadata_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def collect_train_index_inputs(
    manifests: list[Path],
) -> tuple[dict[str, str], list[dict[str, str | int]]]:
    """Return the indexed UID/path inventory and its manifest provenance."""

    images: dict[str, str] = {}
    input_manifests: list[dict[str, str | int]] = []
    for manifest in manifests:
        record_count = 0
        for line_number, row in read_jsonl(manifest):
            record_count += 1
            path = row.get("image") or row.get("image_path")
            uid = str(row.get("uid") or f"{manifest.name}:{line_number}")
            if path:
                image_path = Path(str(path))
                if not image_path.is_absolute():
                    image_path = manifest.resolve().parent / image_path
                images.setdefault(uid, str(image_path.resolve()))
        input_manifests.append(
            {
                "path": str(manifest.resolve()),
                "sha256": sha256_file(manifest),
                "record_count": record_count,
            }
        )
    return images, input_manifests


def _write_image_index_metadata(index_path: Path, metadata: dict[str, Any]) -> None:
    """Atomically persist a sidecar checkpoint."""

    metadata_path = image_index_metadata_path(index_path)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(metadata_path)
    directory_fd = os.open(metadata_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_image_index_metadata(index_path: Path) -> dict[str, Any]:
    metadata_path = image_index_metadata_path(index_path)
    if not metadata_path.is_file():
        raise RuntimeError(
            f"image fingerprint index metadata sidecar is missing: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read image fingerprint index metadata sidecar {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"image fingerprint index metadata sidecar must contain a JSON object: {metadata_path}"
        )
    return metadata


def _verify_metadata_provenance(
    metadata: dict[str, Any],
    index_path: Path,
    input_manifests: list[dict[str, str | int]],
    num_expected_uids: int,
) -> None:
    if metadata.get("schema_version") != IMAGE_INDEX_SCHEMA_VERSION:
        raise RuntimeError(
            "image fingerprint index schema mismatch: "
            f"found {metadata.get('schema_version')!r}, expected {IMAGE_INDEX_SCHEMA_VERSION!r}"
        )
    expected_index_path = str(index_path.resolve())
    if metadata.get("index") != expected_index_path:
        raise RuntimeError(
            "image fingerprint index path differs from its metadata sidecar: "
            f"found {metadata.get('index')!r}, expected {expected_index_path!r}"
        )
    if metadata.get("fingerprint") != fingerprint_provenance():
        raise RuntimeError(
            "image fingerprint index provenance differs from the current fingerprint "
            "schema, settings, or dependency versions; rebuild the index"
        )

    stored_manifests = metadata.get("input_manifests")
    manifests_match = isinstance(stored_manifests, list) and len(stored_manifests) == len(
        input_manifests
    )
    if manifests_match:
        for stored, expected in zip(stored_manifests, input_manifests):
            if not isinstance(stored, dict) or any(
                stored.get(key) != expected[key] for key in ("path", "sha256")
            ):
                manifests_match = False
                break
            # v1 sidecars created before record_count was added remain valid;
            # newly generated sidecars bind this additional provenance too.
            if "record_count" in stored and stored["record_count"] != expected["record_count"]:
                manifests_match = False
                break
    if not manifests_match:
        raise RuntimeError(
            "training manifest path, SHA-256, or record count differs from the image "
            "fingerprint index metadata"
        )
    stored_expected_uids = metadata.get("num_expected_uids")
    if type(stored_expected_uids) is not int or stored_expected_uids != num_expected_uids:
        raise RuntimeError(
            "training manifest unique image UID count differs from the image fingerprint "
            f"index metadata: found {num_expected_uids}, expected {stored_expected_uids!r}"
        )


def _read_index_prefix(index_path: Path, size: int) -> bytes:
    if size == 0 and not index_path.exists():
        return b""
    if not index_path.is_file():
        raise RuntimeError(f"image fingerprint index is not a file: {index_path}")
    with index_path.open("rb") as handle:
        data = handle.read(size)
    if len(data) != size:
        raise RuntimeError(
            "image fingerprint index is shorter than the committed sidecar checkpoint"
        )
    return data


def _index_uid_status(data: bytes, index_path: Path) -> tuple[set[str], set[str]]:
    successful_uids: set[str] = set()
    error_uids: set[str] = set()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"image fingerprint index is not valid UTF-8: {index_path}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{index_path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"{index_path}:{line_number}: expected a JSON object")
        uid = str(row.get("uid"))
        if row.get("error"):
            error_uids.add(uid)
        else:
            successful_uids.add(uid)
    error_uids -= successful_uids
    return successful_uids, error_uids


def _verify_uid_counts(
    metadata: dict[str, Any],
    successful_uids: set[str],
    error_uids: set[str],
) -> None:
    stored_successful_uids = metadata.get("num_successful_uids")
    stored_error_uids = metadata.get("num_error_uids")
    if (
        type(stored_successful_uids) is not int
        or type(stored_error_uids) is not int
        or stored_successful_uids != len(successful_uids)
        or stored_error_uids != len(error_uids)
    ):
        raise RuntimeError(
            "image fingerprint index successful/error UID counts differ from its metadata "
            f"sidecar: found {len(successful_uids)}/{len(error_uids)}, expected "
            f"{stored_successful_uids!r}/{stored_error_uids!r}"
        )


def verify_image_index_metadata(
    index_path: Path,
    input_manifests: list[dict[str, str | int]],
    num_expected_uids: int,
) -> dict[str, Any]:
    """Verify a completed index sidecar against the index, inputs, and code."""

    metadata = _read_image_index_metadata(index_path)
    _verify_metadata_provenance(metadata, index_path, input_manifests, num_expected_uids)
    state = metadata.get("state", INDEX_STATE_COMPLETE)
    if state != INDEX_STATE_COMPLETE:
        if state == INDEX_STATE_BUILDING:
            raise RuntimeError(
                "image fingerprint index build is unfinished; resume build_index before auditing"
            )
        raise RuntimeError(f"unknown image fingerprint index state: {state!r}")
    transient_fields = {"committed_index_size", "pending_append", "reset_index"}
    if transient_fields.intersection(metadata):
        raise RuntimeError(
            "completed image fingerprint index metadata contains unfinished build state"
        )
    if not index_path.is_file():
        raise RuntimeError(f"image fingerprint index is not a file: {index_path}")
    actual_index_sha256 = sha256_file(index_path)
    if metadata.get("index_sha256") != actual_index_sha256:
        raise RuntimeError(
            "image fingerprint index SHA-256 differs from its metadata sidecar: "
            f"found {actual_index_sha256}, expected {metadata.get('index_sha256')!r}"
        )
    data = index_path.read_bytes()
    successful_uids, error_uids = _index_uid_status(data, index_path)
    _verify_uid_counts(metadata, successful_uids, error_uids)
    return metadata


def _resume_image_index(
    index_path: Path,
    input_manifests: list[dict[str, str | int]],
    num_expected_uids: int,
    *,
    repair: bool,
) -> tuple[dict[str, Any], int]:
    """Validate and optionally recover a complete or transactionally running index."""

    metadata = _read_image_index_metadata(index_path)
    _verify_metadata_provenance(metadata, index_path, input_manifests, num_expected_uids)
    state = metadata.get("state", INDEX_STATE_COMPLETE)
    if state == INDEX_STATE_COMPLETE:
        verify_image_index_metadata(index_path, input_manifests, num_expected_uids)
        return metadata, index_path.stat().st_size
    if state != INDEX_STATE_BUILDING:
        raise RuntimeError(f"unknown image fingerprint index state: {state!r}")

    committed_size = metadata.get("committed_index_size")
    if type(committed_size) is not int or committed_size < 0:
        raise RuntimeError("running image fingerprint index has an invalid committed byte count")
    committed_sha256 = metadata.get("index_sha256")
    if not isinstance(committed_sha256, str):
        raise RuntimeError("running image fingerprint index has no committed SHA-256")

    if metadata.get("reset_index") is True:
        if (
            committed_size != 0
            or committed_sha256 != hashlib.sha256(b"").hexdigest()
            or metadata.get("pending_append") is not None
            or metadata.get("num_successful_uids") != 0
            or metadata.get("num_error_uids") != 0
        ):
            raise RuntimeError("running image fingerprint index has an invalid reset checkpoint")
        normalized = dict(metadata)
        normalized["reset_index"] = False
        normalized["updated_unix"] = time.time()
        if repair:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            _write_image_index_metadata(index_path, normalized)
        return normalized, 0

    if metadata.get("reset_index") not in (None, False):
        raise RuntimeError("running image fingerprint index has an invalid reset flag")
    if not index_path.is_file():
        raise RuntimeError(f"running image fingerprint index is missing: {index_path}")
    actual_size = index_path.stat().st_size
    if actual_size < committed_size:
        raise RuntimeError(
            "image fingerprint index is shorter than the committed sidecar checkpoint"
        )
    committed_data = _read_index_prefix(index_path, committed_size)
    if hashlib.sha256(committed_data).hexdigest() != committed_sha256:
        raise RuntimeError(
            "image fingerprint index committed prefix SHA-256 differs from its metadata sidecar"
        )
    successful_uids, error_uids = _index_uid_status(committed_data, index_path)
    _verify_uid_counts(metadata, successful_uids, error_uids)

    pending = metadata.get("pending_append")
    logical_data = committed_data
    if pending is None:
        if actual_size != committed_size:
            raise RuntimeError(
                "image fingerprint index has bytes beyond its committed sidecar checkpoint"
            )
    else:
        if not isinstance(pending, dict) or not isinstance(pending.get("row"), str):
            raise RuntimeError("running image fingerprint index has invalid pending append data")
        pending_bytes = pending["row"].encode("utf-8")
        expected_end = committed_size + len(pending_bytes)
        expected_sha256 = hashlib.sha256(committed_data + pending_bytes).hexdigest()
        if (
            pending.get("start_offset") != committed_size
            or pending.get("end_offset") != expected_end
            or pending.get("index_sha256") != expected_sha256
            or actual_size > expected_end
        ):
            raise RuntimeError("running image fingerprint index pending append metadata mismatch")
        with index_path.open("rb") as handle:
            handle.seek(committed_size)
            actual_tail = handle.read()
        if actual_tail != pending_bytes[: len(actual_tail)]:
            raise RuntimeError(
                "image fingerprint index pending bytes differ from its metadata sidecar"
            )
        if len(actual_tail) == len(pending_bytes):
            logical_data += pending_bytes
            committed_size = expected_end
            committed_sha256 = expected_sha256
            successful_uids, error_uids = _index_uid_status(logical_data, index_path)

    normalized = dict(metadata)
    normalized.update(
        committed_index_size=committed_size,
        index_sha256=committed_sha256,
        num_successful_uids=len(successful_uids),
        num_error_uids=len(error_uids),
        pending_append=None,
        reset_index=False,
        updated_unix=time.time(),
    )
    if repair:
        # A full pending tail may have reached the page cache just before the
        # writer crashed.  Make it durable before publishing its committed
        # checkpoint; partial tails are truncated to the prior checkpoint.
        with index_path.open("r+b") as handle:
            if actual_size != committed_size:
                handle.truncate(committed_size)
            handle.flush()
            os.fsync(handle.fileno())
        _write_image_index_metadata(index_path, normalized)
    return normalized, committed_size


def _new_running_metadata(
    index_path: Path,
    input_manifests: list[dict[str, str | int]],
    num_expected_uids: int,
) -> dict[str, Any]:
    now = time.time()
    return {
        "schema_version": IMAGE_INDEX_SCHEMA_VERSION,
        "state": INDEX_STATE_BUILDING,
        "created_unix": now,
        "updated_unix": now,
        "input_manifests": input_manifests,
        "num_expected_uids": num_expected_uids,
        "num_successful_uids": 0,
        "num_error_uids": 0,
        "index": str(index_path.resolve()),
        "index_sha256": hashlib.sha256(b"").hexdigest(),
        "fingerprint": fingerprint_provenance(),
        "committed_index_size": 0,
        "pending_append": None,
        "reset_index": True,
    }


def _initialize_image_index(
    index_path: Path,
    input_manifests: list[dict[str, str | int]],
    num_expected_uids: int,
) -> dict[str, Any]:
    """Start a new index, recording reset intent before replacing old bytes."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _new_running_metadata(index_path, input_manifests, num_expected_uids)
    _write_image_index_metadata(index_path, metadata)
    with index_path.open("wb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    metadata["reset_index"] = False
    metadata["updated_unix"] = time.time()
    _write_image_index_metadata(index_path, metadata)
    return metadata


def _begin_index_append(
    index_path: Path,
    metadata: dict[str, Any],
    committed_size: int,
) -> dict[str, Any]:
    running = dict(metadata)
    running.update(
        state=INDEX_STATE_BUILDING,
        committed_index_size=committed_size,
        pending_append=None,
        reset_index=False,
        updated_unix=time.time(),
    )
    _write_image_index_metadata(index_path, running)
    return running


def _finalize_image_index(index_path: Path, metadata: dict[str, Any]) -> None:
    complete = dict(metadata)
    complete["state"] = INDEX_STATE_COMPLETE
    complete["updated_unix"] = time.time()
    for key in ("committed_index_size", "pending_append", "reset_index"):
        complete.pop(key, None)
    _write_image_index_metadata(index_path, complete)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images, input_manifests = collect_train_index_inputs(args.train_manifest)
    metadata_path = image_index_metadata_path(args.output)
    resume_metadata: dict[str, Any] | None = None
    committed_size = 0
    if args.resume:
        output_exists = args.output.exists()
        metadata_exists = metadata_path.exists()
        if output_exists and not metadata_exists:
            raise RuntimeError(
                "cannot resume image fingerprint indexing: required file is missing "
                f"(metadata sidecar): {metadata_path}"
            )
        if metadata_exists:
            resume_metadata, committed_size = _resume_image_index(
                args.output,
                input_manifests,
                len(images),
                repair=not args.dry_run,
            )

    committed_data = (
        _read_index_prefix(args.output, committed_size) if resume_metadata is not None else b""
    )
    successful_uids, error_uids = _index_uid_status(committed_data, args.output)
    # Failed rows are intentionally retried.  A later successful row for the
    # same UID supersedes the earlier error during audit loading.
    existing = set(successful_uids)
    items = [(uid, path) for uid, path in sorted(images.items()) if uid not in existing]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"unique images={len(images)}, indexed={len(existing)}, pending={len(items)}")
    if args.dry_run:
        return 0

    if resume_metadata is None:
        metadata = _initialize_image_index(args.output, input_manifests, len(images))
        committed_size = 0
        successful_uids.clear()
        error_uids.clear()
    elif resume_metadata.get("state", INDEX_STATE_COMPLETE) == INDEX_STATE_COMPLETE:
        metadata = _begin_index_append(args.output, resume_metadata, committed_size)
    else:
        metadata = resume_metadata

    index_hasher = hashlib.sha256()
    with args.output.open("rb") as existing_handle:
        while chunk := existing_handle.read(1024 * 1024):
            index_hasher.update(chunk)
    if (
        args.output.stat().st_size != committed_size
        or index_hasher.hexdigest() != metadata.get("index_sha256")
    ):
        raise RuntimeError("image fingerprint index changed after its resume checkpoint")

    errors = 0
    completed = 0
    with args.output.open("ab") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = iter(pool.map(_one, items))
            while batch := list(islice(results, CHECKPOINT_ROWS)):
                encoded = b"".join(
                    (json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                    for result in batch
                )
                pending_hasher = index_hasher.copy()
                pending_hasher.update(encoded)
                pending_end = committed_size + len(encoded)
                metadata["pending_append"] = {
                    "start_offset": committed_size,
                    "end_offset": pending_end,
                    "index_sha256": pending_hasher.hexdigest(),
                    "row": encoded.decode("utf-8"),
                }
                metadata["updated_unix"] = time.time()
                _write_image_index_metadata(args.output, metadata)

                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

                for result in batch:
                    errors += bool(result.get("error"))
                    uid = str(result.get("uid"))
                    if result.get("error"):
                        if uid not in successful_uids:
                            error_uids.add(uid)
                    else:
                        successful_uids.add(uid)
                        error_uids.discard(uid)
                committed_size = pending_end
                index_hasher = pending_hasher
                metadata.update(
                    committed_index_size=committed_size,
                    index_sha256=index_hasher.hexdigest(),
                    num_successful_uids=len(successful_uids),
                    num_error_uids=len(error_uids),
                    pending_append=None,
                    updated_unix=time.time(),
                )
                _write_image_index_metadata(args.output, metadata)
                completed += len(batch)
                print(f"{completed}/{len(items)} errors={errors}", flush=True)
    _finalize_image_index(args.output, metadata)
    print(f"wrote {args.output}; errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
