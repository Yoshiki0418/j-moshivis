"""Merge benchmark manifests while rebasing all image/audio paths safely."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import replace
from pathlib import Path

from .schema import EvalExample, read_manifest, sha256_file, write_manifest


def _rebase(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _rebase_metadata_assets(
    metadata: dict[str, object], source_manifest: Path, output_dir: Path
) -> None:
    """Rebase file paths stored below metadata in place.

    Babillage COCO/OCR rows keep the target/reference speech under
    ``metadata.reference_audio.audio_path`` rather than the example's input
    ``audio_path``. Leaving that nested path relative to the source manifest
    silently breaks it as soon as several benchmark manifests are merged into
    another directory.
    """

    reference_audio = metadata.get("reference_audio")
    if not isinstance(reference_audio, dict):
        return
    raw_path = reference_audio.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    asset_path = Path(raw_path)
    if not asset_path.is_absolute():
        asset_path = source_manifest.parent / asset_path
    reference_audio["audio_path"] = _rebase(asset_path, output_dir)


def merge(paths: list[Path], output: Path) -> list[EvalExample]:
    merged: list[EvalExample] = []
    for path in paths:
        digest = sha256_file(path)
        for example in read_manifest(path, check_files=True):
            metadata = copy.deepcopy(example.metadata)
            _rebase_metadata_assets(metadata, path, output.parent)
            metadata["source_manifest"] = {
                # Keep provenance portable across workspaces/servers.  The
                # source digest supplies identity; an absolute checkout path
                # would make byte-identical reconstruction impossible.
                "path": os.path.relpath(path.resolve(), output.parent.resolve()),
                "sha256": digest,
            }
            merged.append(
                replace(
                    example,
                    image_path=_rebase(example.resolve_image(path), output.parent) or "",
                    audio_path=_rebase(example.resolve_audio(path), output.parent),
                    metadata=metadata,
                )
            )
    merged.sort(key=lambda item: (item.domain, item.benchmark, item.sample_id))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    examples = merge(args.manifest, args.output)
    counts: dict[str, int] = {}
    for example in examples:
        key = f"{example.domain}/{example.benchmark}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"num_samples": len(examples), "groups": counts}, ensure_ascii=False, indent=2))
    if not args.dry_run:
        count, digest = write_manifest(args.output, examples)
        print(f"wrote {count} samples: {args.output} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
