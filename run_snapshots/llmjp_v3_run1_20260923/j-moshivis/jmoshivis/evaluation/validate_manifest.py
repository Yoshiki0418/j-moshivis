"""Validate an image-evaluation manifest before spending GPU time."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .schema import (
    ManifestError,
    read_manifest,
    sha256_file,
    validate_question_audio_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--require-overlap-clean", action="store_true")
    parser.add_argument(
        "--strict-checksums", action="store_true",
        help="Require and verify declared SHA-256 values for every image and WAV.",
    )
    parser.add_argument("--json", type=Path, help="Optional machine-readable validation report.")
    args = parser.parse_args()

    examples = read_manifest(
        args.manifest,
        check_files=True,
        require_audio=args.require_audio,
    )
    audio_durations: list[float] = []
    verified_images: dict[Path, str] = {}
    verified_audio: dict[Path, str] = {}
    if args.strict_checksums:
        for example in examples:
            image = example.resolve_image(args.manifest).resolve()
            if image not in verified_images:
                verified_images[image] = sha256_file(image)
            actual_image = verified_images[image]
            expected_image = example.source.get("image_sha256")
            if not expected_image:
                raise ManifestError(f"{example.sample_id}: source.image_sha256 is missing")
            if str(expected_image) != actual_image:
                raise ManifestError(f"{example.sample_id}: image checksum mismatch: {image}")
            audio = example.resolve_audio(args.manifest)
            if audio is not None:
                validate_question_audio_provenance(example)
                audio = audio.resolve()
                if not audio.is_file():
                    raise ManifestError(f"{example.sample_id}: audio not found: {audio}")
                if audio not in verified_audio:
                    verified_audio[audio] = sha256_file(audio)
                actual_audio = verified_audio[audio]
                expected_audio = example.metadata.get("question_audio", {}).get("wav_sha256")
                if not expected_audio:
                    raise ManifestError(
                        f"{example.sample_id}: metadata.question_audio.wav_sha256 is missing"
                    )
                if str(expected_audio) != actual_audio:
                    raise ManifestError(f"{example.sample_id}: audio checksum mismatch: {audio}")
    if args.require_audio:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for --require-audio") from exc
        for example in examples:
            audio = example.resolve_audio(args.manifest)
            assert audio is not None
            info = sf.info(audio)
            if info.samplerate != 24_000:
                raise ManifestError(f"{example.sample_id}: expected 24000 Hz, got {info.samplerate}")
            if info.channels != 1:
                raise ManifestError(f"{example.sample_id}: expected mono audio, got {info.channels} channels")
            audio_durations.append(info.duration)

    overlap = Counter()
    for example in examples:
        status = example.metadata.get("overlap_audit", {}).get(
            "status", example.metadata.get("overlap_status", "not_audited")
        )
        overlap[str(status)] += 1
    if args.require_overlap_clean:
        unsafe = sum(count for status, count in overlap.items() if status != "clean")
        if unsafe:
            raise ManifestError(
                f"{unsafe}/{len(examples)} questions are not marked overlap-clean: {dict(overlap)}"
            )

    report = {
        "schema_version": "jmoshivis.image-eval.validation.v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "num_questions": len(examples),
        "num_unique_images": len({str(example.resolve_image(args.manifest).resolve()) for example in examples}),
        "benchmarks": dict(Counter(example.benchmark for example in examples)),
        "domains": dict(Counter(example.domain for example in examples)),
        "tasks": dict(Counter(example.task for example in examples)),
        "question_types": dict(Counter(example.question_type or "unknown" for example in examples)),
        "overlap_status": dict(overlap),
        "audio": {
            "required": args.require_audio,
            "num_files": len(audio_durations),
            "total_hours": sum(audio_durations) / 3600,
            "min_duration_sec": min(audio_durations) if audio_durations else None,
            "max_duration_sec": max(audio_durations) if audio_durations else None,
        },
        "valid": True,
        "strict_checksums": {
            "enabled": args.strict_checksums,
            "verified_unique_images": len(verified_images),
            "verified_unique_audio": len(verified_audio),
        },
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
