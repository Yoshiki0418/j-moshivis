"""Create deterministic 24 kHz question audio and an audio-ready manifest.

Run this command with the existing Style-Bert-VITS2 environment.  Audio is
cached by sample ID and the selected voice is a stable hash of that ID, so all
checkpoints receive exactly the same waveform.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import replace
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import EvalExample, read_manifest, sha256_file, write_manifest
from .rng import seed_assigned_device


TARGET_SAMPLE_RATE = 24_000
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_POOL = WORKSPACE / "ssvd/tts/voice_pool.json"
TTS_PARAMS = {
    "sdp_ratio": 0.2,
    "noise": 0.6,
    "noise_w": 0.8,
    "length": 1.0,
    "line_split": False,
    "style_weight": 1.0,
    "pitch_scale": 1.0,
    "intonation_scale": 1.0,
}
TTS_DEPENDENCIES = (
    "torch", "transformers", "style-bert-vits2", "numpy", "scipy", "soundfile"
)


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in TTS_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def choose_voice(sample_id: str, voice_names: list[str]) -> str:
    if not voice_names:
        raise ValueError("at least one evaluation voice is required")
    value = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16], 16)
    return voice_names[value % len(voice_names)]


def render_prompt(question: str, style: str) -> str:
    if style == "verbatim":
        return question.strip()
    if style == "short_answer":
        base = question.strip().rstrip("。.!！?？")
        return f"{base}？ 一言で簡潔に答えてください。"
    raise ValueError(f"unknown prompt style: {style}")


def render_example_prompt(example: EvalExample, style: str) -> str:
    """Render the adapter-defined prompt without discarding its instructions."""

    return render_prompt(example.rendered_prompt, style)


def trim_silence(audio: np.ndarray, threshold: float = 0.01, keep_ms: int = 60) -> np.ndarray:
    indices = np.flatnonzero(np.abs(audio) > threshold)
    if len(indices) == 0:
        return audio
    keep = round(TARGET_SAMPLE_RATE * keep_ms / 1000)
    return audio[max(0, int(indices[0]) - keep) : min(len(audio), int(indices[-1]) + keep)]


def prepare_waveform(
    audio: np.ndarray,
    sample_rate: int,
    *,
    leading_silence: float,
    trailing_silence: float,
) -> tuple[np.ndarray, float, float]:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != TARGET_SAMPLE_RATE:
        from scipy.signal import resample_poly

        factor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE // factor,
            sample_rate // factor,
        ).astype(np.float32)
    waveform = trim_silence(waveform)
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak > 0:
        waveform = waveform / peak * 0.8
    lead = np.zeros(round(leading_silence * TARGET_SAMPLE_RATE), dtype=np.float32)
    trail = np.zeros(round(trailing_silence * TARGET_SAMPLE_RATE), dtype=np.float32)
    speech_start = len(lead) / TARGET_SAMPLE_RATE
    speech_end = (len(lead) + len(waveform)) / TARGET_SAMPLE_RATE
    return np.concatenate([lead, waveform, trail]), speech_start, speech_end


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _voice_asset_fingerprint(spec: dict) -> tuple[str, list[dict[str, str]]]:
    """Fingerprint the exact local voice assets behind a pool entry."""

    root = (WORKSPACE / str(spec["path"])).resolve()
    if root.is_file():
        files = [root]
        display_root = root.parent
    elif root.is_dir():
        # These are the files consumed by tts_engine.prepare_sbv2_dir.  Hashing
        # them is deliberately done once before synthesis, not once/sample.
        files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and (path.suffix in {".safetensors", ".npy"} or path.name == "config.json")
        )
        display_root = root
    else:
        raise FileNotFoundError(f"voice asset path not found: {root}")
    if not files:
        raise FileNotFoundError(f"no hashable model assets found for voice: {root}")
    records = [
        {"path": str(path.relative_to(display_root)), "sha256": sha256_file(path)}
        for path in files
    ]
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), records


def _cache_config_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--voices",
        default="jvnv-M1,jvnv-F1",
        help="Comma-separated fixed evaluation voice pool (default: one male + one female).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--leading-silence", type=float, default=0.35)
    parser.add_argument("--trailing-silence", type=float, default=0.35)
    parser.add_argument(
        "--prompt-style",
        choices=("verbatim", "short_answer"),
        default="verbatim",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_manifest.exists() and not args.dry_run:
        raise FileExistsError(
            "refusing to replace an existing evaluation manifest; choose a new "
            f"versioned output path: {args.output_manifest}"
        )
    voices_requested = [name.strip() for name in args.voices.split(",") if name.strip()]
    examples = read_manifest(args.manifest, check_files=True)
    if args.limit is not None:
        examples = examples[: args.limit]
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    user_specs = {spec["name"]: spec for spec in pool["user"]}
    unknown = sorted(set(voices_requested) - set(user_specs))
    if unknown:
        raise ValueError(f"voices not found in {args.pool}: {', '.join(unknown)}")
    assignments = {example.sample_id: choose_voice(example.sample_id, voices_requested) for example in examples}
    counts = {voice: sum(value == voice for value in assignments.values()) for voice in voices_requested}
    print(f"examples={len(examples)} voices={counts} output={args.output_manifest}")
    if args.dry_run:
        for example in examples[:5]:
            print(
                f"  {example.sample_id} -> {assignments[example.sample_id]}: "
                f"{render_example_prompt(example, args.prompt_style)}"
            )
        return 0

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    # Import the existing loader only after dry-run; it initialises Japanese BERT.
    tts_dir = WORKSPACE / "ssvd/tts"
    sys.path.insert(0, str(tts_dir))
    from tts_engine import BERT_MODEL_ID, BERT_REVISION, Voice, load_bert  # type: ignore
    engine_path = tts_dir / "tts_engine.py"
    engine_sha256 = sha256_file(engine_path)
    prepare_tts_sha256 = sha256_file(Path(__file__))
    rng_seed_sha256 = sha256_file(Path(__file__).with_name("rng.py"))
    pool_sha256 = sha256_file(args.pool)
    dependency_versions = _dependency_versions()
    voice_assets = {
        name: _voice_asset_fingerprint(user_specs[name]) for name in voices_requested
    }
    load_bert()
    built: dict[str, dict] = {}
    for voice_name in voices_requested:
        assigned = [example for example in examples if assignments[example.sample_id] == voice_name]
        if not assigned:
            continue
        print(f"loading voice {voice_name} ({len(assigned)} questions)", flush=True)
        voice = Voice(voice_name, user_specs[voice_name], args.device)
        for index, example in enumerate(assigned, 1):
            # Adapters may provide a benchmark-defined prompt (e.g. numbered
            # choices or an official concise-answer instruction).  ``verbatim``
            # must preserve that exact rendered prompt for both text and audio.
            prompt = render_example_prompt(example, args.prompt_style)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            voice_assets_sha256, voice_asset_records = voice_assets[voice_name]
            cache_config = {
                "schema_version": "jmoshivis.image-eval.tts-cache.v3",
                "sample_id": example.sample_id,
                "prompt_sha256": prompt_hash,
                "prompt_style": args.prompt_style,
                "voice": voice_name,
                "voice_spec": user_specs[voice_name],
                "voice_assets_sha256": voice_assets_sha256,
                "tts_engine_sha256": engine_sha256,
                "prepare_tts_sha256": prepare_tts_sha256,
                "rng_seed_implementation_sha256": rng_seed_sha256,
                "pool_sha256": pool_sha256,
                "bert_model_id": BERT_MODEL_ID,
                "bert_revision": BERT_REVISION,
                "dependency_versions": dependency_versions,
                "python_version": sys.version,
                "tts_params": TTS_PARAMS,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "leading_silence_sec": args.leading_silence,
                "trailing_silence_sec": args.trailing_silence,
                "trim_threshold": 0.01,
                "trim_keep_ms": 60,
            }
            cache_config_sha256 = _cache_config_hash(cache_config)
            destination = args.audio_dir / (
                f"{example.sample_id}_{prompt_hash[:12]}_{cache_config_sha256[:12]}.wav"
            )
            info_path = destination.with_suffix(".json")
            use_cached = False
            if destination.exists() and info_path.is_file() and args.resume:
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    use_cached = bool(
                        info.get("question_sha256") == prompt_hash
                        and info.get("voice") == voice_name
                        and info.get("cache_config_sha256") == cache_config_sha256
                        and info.get("wav_sha256") == sha256_file(destination)
                    )
                except (json.JSONDecodeError, OSError):
                    use_cached = False
                if not use_cached:
                    print(
                        f"  rebuilding incomplete or invalid cache: {example.sample_id}",
                        flush=True,
                    )
            if not use_cached:
                synthesis_seed = int(
                    hashlib.sha256(f"tts:{example.sample_id}".encode("utf-8")).hexdigest()[:8], 16
                )
                seed_assigned_device(synthesis_seed, args.device)
                sample_rate, raw = voice.synth(
                    prompt,
                    style=voice.default_style,
                    **TTS_PARAMS,
                )
                waveform, speech_start, speech_end = prepare_waveform(
                    raw,
                    sample_rate,
                    leading_silence=args.leading_silence,
                    trailing_silence=args.trailing_silence,
                )
                temporary_wav = destination.with_name(destination.name + ".tmp.wav")
                sf.write(temporary_wav, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")
                wav_sha256 = sha256_file(temporary_wav)
                info = {
                    "sample_id": example.sample_id,
                    "prompt_text": prompt,
                    "prompt_style": args.prompt_style,
                    "question_sha256": prompt_hash,
                    "voice": voice_name,
                    "voice_spec": user_specs[voice_name],
                    "voice_assets_sha256": voice_assets_sha256,
                    "voice_assets": voice_asset_records,
                    "engine": "Style-Bert-VITS2/AIVM",
                    "tts_engine_sha256": engine_sha256,
                    "prepare_tts_sha256": prepare_tts_sha256,
                    "rng_seed_implementation_sha256": rng_seed_sha256,
                    "pool_sha256": pool_sha256,
                    "bert_model_id": BERT_MODEL_ID,
                    "bert_revision": BERT_REVISION,
                    "dependency_versions": dependency_versions,
                    "python_version": sys.version,
                    "cache_config_sha256": cache_config_sha256,
                    "style": voice.default_style,
                    "synthesis_seed": synthesis_seed,
                    "tts_params": TTS_PARAMS,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "duration_sec": len(waveform) / TARGET_SAMPLE_RATE,
                    "speech_start_sec": speech_start,
                    "speech_end_sec": speech_end,
                    "leading_silence_sec": args.leading_silence,
                    "trailing_silence_sec": args.trailing_silence,
                    "wav_sha256": wav_sha256,
                }
                temporary_info = info_path.with_suffix(info_path.suffix + ".tmp")
                temporary_info.write_text(
                    json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                # Either both final artifacts validate on resume, or the pair
                # is rebuilt.  Replacing the WAV first prevents metadata from
                # ever blessing bytes that have not reached the final path.
                temporary_wav.replace(destination)
                temporary_info.replace(info_path)
            built[example.sample_id] = {"path": destination, "info": info}
            if index % 25 == 0 or index == len(assigned):
                print(f"  {voice_name}: {index}/{len(assigned)}", flush=True)
        del voice

    output_examples: list[EvalExample] = []
    base = args.output_manifest.resolve().parent
    for example in examples:
        entry = built[example.sample_id]
        metadata = copy.deepcopy(example.metadata)
        metadata["question_audio"] = entry["info"]
        output_examples.append(
            replace(
                example,
                image_path=_relative(example.resolve_image(args.manifest), base),
                prompt_text=entry["info"]["prompt_text"],
                audio_path=_relative(entry["path"], base),
                metadata=metadata,
            )
        )
    count, digest = write_manifest(args.output_manifest, output_examples)
    print(f"wrote {count} samples: {args.output_manifest} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
