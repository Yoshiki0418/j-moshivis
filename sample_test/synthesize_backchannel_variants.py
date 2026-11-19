#!/usr/bin/env python3
"""Generate three audition-only Honoka variants using cached CPU models.

Run with the sbv2 Python environment. Existing outputs are never overwritten.
The names identify synthesis settings, not a verified perceptual judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "eval_results/backchannel_voice_warmth_20261117/assets"
BERT_REVISION = "547b0e8b044fba3f9b84d0ab9f990440bd130c8b"
BERT_DIR = Path.home() / ".cache/huggingface/hub/models--ku-nlp--deberta-v2-large-japanese-char-wwm/snapshots" / BERT_REVISION
WHISPER_PATH = Path.home() / ".cache/whisper/large-v3-turbo.pt"
SEED = 42
RATE = 24000
VARIANTS = [
    {"id": "un_expressive", "text": "うん。", "style": "uresii_uresii", "style_weight": 1.0, "length": 1.15, "intonation_scale": 1.35},
    {"id": "unun_neutral", "text": "うんうん。", "style": "ノーマル", "style_weight": 1.0, "length": 1.10, "intonation_scale": 1.20},
    {"id": "unun_expressive", "text": "うんうん。", "style": "uresii_uresii", "style_weight": 0.8, "length": 1.10, "intonation_scale": 1.35},
]
COMMON_PARAMS = {"speaker_id": 0, "sdp_ratio": 0.2, "noise": 0.6, "noise_w": 0.8, "line_split": False, "pitch_scale": 1.0, "use_assist_text": False}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def offline_cpu(threads):
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "false", "NUMBA_CACHE_DIR": "/tmp/backchannel_warmth_numba"})

    def deny_connection(*args, **kwargs):
        raise RuntimeError("Network disabled for this local-only audition")

    socket.socket.connect = deny_connection
    socket.create_connection = deny_connection
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def versions(names):
    return {name: importlib.metadata.version(name) for name in names}


def prepare_audio(audio, source_rate):
    import numpy as np
    from scipy.signal import resample_poly

    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono TTS output, got {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    divisor = math.gcd(source_rate, RATE)
    audio = resample_poly(audio, RATE // divisor, source_rate // divisor)
    source_length = len(audio)
    frame = round(0.010 * RATE)
    padded = np.pad(audio, (0, (-len(audio)) % frame))
    frame_rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    threshold = max(1e-5, float(frame_rms.max()) * 10 ** (-35 / 20))
    active_frames = frame_rms > threshold
    active_indices = np.flatnonzero(active_frames)
    if not len(active_indices):
        raise ValueError("TTS output contains no detectable active audio")
    active_mask = np.repeat(active_frames, frame)[:len(audio)]
    active_rms_before = float(np.sqrt(np.mean(audio[active_mask] ** 2)))
    margin = round(0.030 * RATE)
    active_start = int(active_indices[0] * frame)
    active_end = min(len(audio), int((active_indices[-1] + 1) * frame))
    requested_start, requested_end = active_start - margin, active_end + margin
    start, end = max(0, requested_start), min(len(audio), requested_end)
    audio = np.pad(audio[start:end], (max(0, -requested_start), max(0, requested_end - len(audio))))
    gain = min(0.12 / active_rms_before, 0.75 / float(np.max(np.abs(audio))))
    audio *= gain
    # A 3 ms endpoint fade is confined to the retained silence margins.
    fade = round(0.003 * RATE)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return audio.astype(np.float32), {
        "source_rate": source_rate, "resampled_source_duration": source_length / RATE,
        "activity_frame_ms": 10, "activity_threshold_db_below_max_frame_rms": -35,
        "activity_threshold_absolute": threshold, "margin_ms": 30, "endpoint_fade_ms": 3,
        "trim_start_seconds": start / RATE, "trim_end_seconds": (source_length - end) / RATE,
        "added_start_padding_seconds": max(0, -requested_start) / RATE,
        "added_end_padding_seconds": max(0, requested_end - source_length) / RATE,
        "target_active_rms": 0.12, "peak_cap": 0.75, "gain": gain,
        "active_rms_before": active_rms_before, "active_rms_after": active_rms_before * gain,
        "peak_after": float(np.max(np.abs(audio))), "duration_seconds": len(audio) / RATE,
    }


def synthesize(out, threads):
    import numpy as np
    import soundfile as sf
    import torch
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    expected = [out / f"{item['id']}.flac" for item in VARIANTS] + [out / "audit.json"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite existing synthesis outputs; choose a new --out directory")
    model_dir = ROOT / "models/tts/aivis/honoka"
    started = time.monotonic()
    print("Loading cached BERT and Honoka on CPU", flush=True)
    bert_models.load_model(Languages.JP, str(BERT_DIR)).float()
    bert_models.load_tokenizer(Languages.JP, str(BERT_DIR))
    model = TTSModel(model_path=str(model_dir / "honoka.safetensors"), config_path=str(model_dir / "config.json"), style_vec_path=str(model_dir / "style_vectors.npy"), device="cpu")
    model.load()
    load_seconds = time.monotonic() - started
    print(f"Loaded TTS in {load_seconds:.1f}s", flush=True)
    results = {}
    for variant in VARIANTS:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        params = {**COMMON_PARAMS, **{key: value for key, value in variant.items() if key != "id"}}
        tick = time.monotonic()
        sample_rate, raw = model.infer(language=Languages.JP, **params)
        audio, preprocessing = prepare_audio(raw, sample_rate)
        path = out / f"{variant['id']}.flac"
        sf.write(path, audio, RATE, format="FLAC", subtype="PCM_16")
        reopened, reopened_rate = sf.read(path)
        assert reopened_rate == RATE and reopened.ndim == 1
        assert np.isfinite(reopened).all() and np.max(np.abs(reopened)) < 0.751
        elapsed = time.monotonic() - tick
        results[variant["id"]] = {"audio_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "text": variant["text"], "params": params, "language": "JP", "seed": SEED, "preprocessing": preprocessing, "sha256": sha256(path), "synthesis_seconds": elapsed, "format": {"sample_rate": RATE, "channels": 1, "subtype": sf.info(path).subtype}}
        print(f"{variant['id']}: {len(audio) / RATE:.3f}s audio, {elapsed:.1f}s compute, active RMS {preprocessing['active_rms_after']:.4f}", flush=True)
    audit = {"purpose": "Audition-only voice variants; names describe settings and do not certify perceived warmth", "human_listened": False, "device": "cpu", "network_disabled": True, "cpu_threads": threads, "seed_reset_per_variant": SEED, "script_sha256": sha256(__file__), "versions": versions(["torch", "numpy", "scipy", "soundfile", "style-bert-vits2", "transformers"]), "bert": {"path": str(BERT_DIR), "revision": BERT_REVISION}, "model_files": {name: {"path": str(model_dir / name), "sha256": sha256(model_dir / name)} for name in ["honoka.safetensors", "config.json", "style_vectors.npy"]}, "tts_load_seconds": load_seconds, "variants": results}
    write_new_json(out / "audit.json", audit)
    print(f"Wrote {out / 'audit.json'}", flush=True)


def transcribe(out, threads):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import whisper as base_whisper
    import whisper_timestamped as whisper

    timings_path = out / "timings.json"
    if timings_path.exists():
        raise FileExistsError(f"Refusing to overwrite {timings_path}")
    if not WHISPER_PATH.is_file():
        raise FileNotFoundError(f"Cached Whisper checkpoint required: {WHISPER_PATH}")
    print("Loading cached Whisper turbo on CPU", flush=True)
    tick = time.monotonic()
    checkpoint_hash = sha256(WHISPER_PATH)
    model = whisper.load_model(str(WHISPER_PATH), device="cpu")
    model.set_alignment_heads(base_whisper._ALIGNMENT_HEADS["large-v3-turbo"])
    print(f"Loaded ASR in {time.monotonic() - tick:.1f}s", flush=True)
    params = {"language": "ja", "task": "transcribe", "temperature": 0.0, "beam_size": 1, "fp16": False, "vad": False, "condition_on_previous_text": False, "initial_prompt": None, "seed": SEED, "verbose": False}
    records = {}
    for variant in VARIANTS:
        path = out / f"{variant['id']}.flac"
        audio, rate = sf.read(path, dtype="float32")
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        tick = time.monotonic()
        result = whisper.transcribe(model, audio, **params)
        elapsed = time.monotonic() - tick
        records[variant["id"]] = {"audio_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "transcript": result.get("text", ""), "words": [{"text": word["text"], "start": word["start"], "end": word["end"], **({"confidence": word["confidence"]} if "confidence" in word else {})} for segment in result.get("segments", []) for word in segment.get("words", [])], "segments": result.get("segments", []), "asr_seconds": elapsed, "method": "whisper_timestamped", "model": "large-v3-turbo", "model_sha256": checkpoint_hash, "device": "cpu", "cpu_threads": threads, "parameters": params, "audio_sha256": sha256(path), "versions": versions(["openai-whisper", "whisper-timestamped"]), "note": "Actual model output; auxiliary recognition and word times do not measure perceived warmth."}
        print(f"ASR {variant['id']}: {result.get('text', '')!r}, {elapsed:.1f}s", flush=True)
    write_new_json(timings_path, records)
    print(f"Wrote {timings_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase", choices=["synth", "asr"], required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    offline_cpu(args.cpu_threads)
    (synthesize if args.phase == "synth" else transcribe)(args.out, args.cpu_threads)


if __name__ == "__main__":
    main()
