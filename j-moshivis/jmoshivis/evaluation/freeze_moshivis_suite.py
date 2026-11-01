"""Freeze a manifest and MoshiVis paper-inference protocol into suite-lock v2."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .audio_protocol import (
    ASSISTANT_PRELUDE_PROTOCOL,
    AUDIO_DIALOGUE_PROTOCOLS,
    BARE_QUESTION_PROTOCOL,
)
from .schema import ManifestError, read_manifest, sha256_file
from .suite_moshivis import (
    PAPER_INPUT_MODALITIES,
    SUITE_LOCK_SCHEMA_V2,
    benchmark_composition,
    verified_asset_inventories,
)


_SUITE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def build_suite_lock(
    manifest_path: Path,
    output_path: Path,
    *,
    suite_id: str,
    paper_input_modalities: Sequence[str] = ("audio", "text"),
    response_tail_seconds: float = 10.0,
    response_overlap_seconds: float = 0.32,
    text_max_new_tokens: int = 96,
    audio_dialogue_protocol: str = BARE_QUESTION_PROTOCOL,
    assistant_prelude_wav: Path | None = None,
    frozen_on: str | None = None,
) -> dict[str, Any]:
    """Build, but do not write, a fully validated suite-lock v2 object."""

    if _SUITE_ID_RE.fullmatch(suite_id) is None:
        raise ValueError("suite_id must contain only letters, numbers, '.', '_' or '-'")
    modalities = [str(value) for value in paper_input_modalities]
    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("paper_input_modalities must be non-empty and unique")
    unsupported = sorted(set(modalities) - PAPER_INPUT_MODALITIES)
    if unsupported:
        raise ValueError("unsupported input modalities: " + ", ".join(unsupported))
    if response_tail_seconds <= 0:
        raise ValueError("response_tail_seconds must be positive")
    if not 0 <= response_overlap_seconds < response_tail_seconds:
        raise ValueError("response_overlap_seconds must be non-negative and below tail")
    if text_max_new_tokens < 1:
        raise ValueError("text_max_new_tokens must be positive")
    if audio_dialogue_protocol not in AUDIO_DIALOGUE_PROTOCOLS:
        raise ValueError(f"unsupported audio dialogue protocol: {audio_dialogue_protocol}")
    if audio_dialogue_protocol == ASSISTANT_PRELUDE_PROTOCOL:
        raise ValueError(
            f"{ASSISTANT_PRELUDE_PROTOCOL} cannot be frozen as a paper suite: "
            "it injects assistant audio without teacher-forcing the synchronized "
            "assistant text stream, so it is not paper-equivalent. Freeze "
            f"{BARE_QUESTION_PROTOCOL}, then use run_inference --diagnostic with "
            "--assistant-prelude-wav for exploratory runs."
        )
    elif assistant_prelude_wav is not None:
        raise ValueError(
            f"assistant_prelude_wav is invalid for {BARE_QUESTION_PROTOCOL}"
        )

    examples = read_manifest(
        manifest_path,
        check_files=True,
        require_audio="audio" in modalities,
    )
    if not examples:
        raise ManifestError("cannot freeze an empty evaluation manifest")
    asset_inventories = verified_asset_inventories(
        examples,
        manifest_path,
        modalities=modalities,
    )
    # Import lazily: score_moshivis itself imports the suite verifier.  At
    # runtime both modules are initialized, while a module-level import here
    # would create a cycle.
    from .score_moshivis import metric_for_example, scoring_provenance

    scoring_protocol = scoring_provenance()
    metrics = {metric_for_example(example) for example in examples}
    if metrics & {"cider", "cider_ja"}:
        missing: list[str] = []
        dependency_versions = scoring_protocol["dependency_versions"]
        dependency_artifacts = scoring_protocol["dependency_artifact_sha256"]
        if not dependency_versions.get("pycocoevalcap"):
            missing.append("pycocoevalcap")
        if "cider" in metrics:
            if not dependency_versions.get("java"):
                missing.append("Java runtime")
            if not dependency_artifacts.get("stanford-corenlp-3.4.1.jar"):
                missing.append("Stanford PTB tokenizer JAR")
        if "cider_ja" in metrics:
            for distribution in ("fugashi", "unidic-lite"):
                if not dependency_versions.get(distribution):
                    missing.append(distribution)

        if missing:
            raise RuntimeError(
                "cannot freeze a CIDEr paper suite without official scoring "
                "dependencies: " + ", ".join(missing)
            )
    try:
        manifest_label = os.path.relpath(
            manifest_path.resolve(), start=output_path.parent.resolve()
        )
    except ValueError:
        manifest_label = str(manifest_path.resolve())
    scope: dict[str, Any] = {
        "prediction": "frame-synchronous model text tokens; no output ASR",
        "paper_generation": "greedy",
        "paper_input_modalities": modalities,
        "response_tail_seconds": float(response_tail_seconds),
        "response_overlap_seconds": float(response_overlap_seconds),
        "text_max_new_tokens": int(text_max_new_tokens),
        "audio_dialogue_protocol": audio_dialogue_protocol,
    }
    return {
        "schema_version": SUITE_LOCK_SCHEMA_V2,
        "suite_id": suite_id,
        "frozen_on": frozen_on or dt.date.today().isoformat(),
        "manifest": {
            "path": manifest_label,
            "sha256": sha256_file(manifest_path),
            "questions": len(examples),
            "benchmark_composition": benchmark_composition(examples),
            "frozen_asset_inventories": asset_inventories,
        },
        "scope": scope,
        "scoring_protocol": scoring_protocol,
    }


def write_new_suite_lock(output_path: Path, lock: dict[str, Any]) -> None:
    """Atomically publish ``lock`` without ever replacing an existing path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite suite lock: {output_path}")
    payload = (
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite suite lock: {output_path}"
            ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument(
        "--paper-input-modalities",
        nargs="+",
        choices=sorted(PAPER_INPUT_MODALITIES),
        default=["audio", "text"],
    )
    parser.add_argument("--response-tail-seconds", type=float, default=10.0)
    parser.add_argument("--response-overlap-seconds", type=float, default=0.32)
    parser.add_argument("--text-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--audio-dialogue-protocol",
        choices=AUDIO_DIALOGUE_PROTOCOLS,
        default=BARE_QUESTION_PROTOCOL,
    )
    parser.add_argument("--assistant-prelude-wav", type=Path)
    parser.add_argument("--frozen-on")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = build_suite_lock(
        args.manifest,
        args.output,
        suite_id=args.suite_id,
        paper_input_modalities=args.paper_input_modalities,
        response_tail_seconds=args.response_tail_seconds,
        response_overlap_seconds=args.response_overlap_seconds,
        text_max_new_tokens=args.text_max_new_tokens,
        audio_dialogue_protocol=args.audio_dialogue_protocol,
        assistant_prelude_wav=args.assistant_prelude_wav,
        frozen_on=args.frozen_on,
    )
    write_new_suite_lock(args.output, lock)
    print(f"suite lock: {args.output}")
    print(f"manifest sha256: {lock['manifest']['sha256']}")
    print(f"questions: {lock['manifest']['questions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
