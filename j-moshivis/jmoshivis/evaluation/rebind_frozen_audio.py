"""Rebind verified frozen question WAVs to a rebuilt clean manifest.

This is intentionally narrower than TTS generation.  It is useful when image
provenance or overlap annotations are corrected while sample IDs and spoken
prompts stay unchanged.  WAV bytes are reused only after prompt and checksum
validation; no synthesis claim is added or changed.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import replace
from pathlib import Path

from .schema import (
    EvalExample,
    ManifestError,
    read_manifest,
    sha256_file,
    validate_question_audio_provenance,
    write_manifest,
)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def rebind_frozen_audio(
    clean_manifest: Path,
    frozen_audio_manifest: Path,
    output_manifest: Path,
) -> list[EvalExample]:
    clean = read_manifest(clean_manifest, check_files=True)
    frozen = read_manifest(frozen_audio_manifest, check_files=True, require_audio=True)
    frozen_by_id = {example.sample_id: example for example in frozen}
    clean_ids = {example.sample_id for example in clean}
    if clean_ids != set(frozen_by_id):
        missing = sorted(clean_ids - set(frozen_by_id))
        extra = sorted(set(frozen_by_id) - clean_ids)
        raise ManifestError(
            "clean and frozen-audio manifests have different sample IDs "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    output: list[EvalExample] = []
    base = output_manifest.resolve().parent
    for current in clean:
        old = frozen_by_id[current.sample_id]
        if current.rendered_prompt != old.rendered_prompt:
            raise ManifestError(f"{current.sample_id}: rendered prompt changed")
        validate_question_audio_provenance(old)
        audio_path = old.resolve_audio(frozen_audio_manifest)
        assert audio_path is not None
        recorded_hash = old.metadata["question_audio"].get("wav_sha256")
        if not recorded_hash or recorded_hash != sha256_file(audio_path):
            raise ManifestError(f"{current.sample_id}: frozen WAV checksum mismatch")
        metadata = copy.deepcopy(current.metadata)
        metadata["question_audio"] = copy.deepcopy(old.metadata["question_audio"])
        rebound = replace(
            current,
            image_path=_relative(current.resolve_image(clean_manifest), base),
            audio_path=_relative(audio_path, base),
            metadata=metadata,
        )
        validate_question_audio_provenance(rebound)
        output.append(rebound)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--frozen-audio-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rebound = rebind_frozen_audio(
        args.clean_manifest,
        args.frozen_audio_manifest,
        args.output_manifest,
    )
    count, digest = write_manifest(args.output_manifest, rebound)
    print(f"rebound {count} frozen WAVs: {args.output_manifest} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
