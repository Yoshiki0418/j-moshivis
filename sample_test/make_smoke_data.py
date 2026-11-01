"""Create a small, self-contained v3-format dataset for training smoke tests.

The samples are synthetic and have no scientific value.  They only exercise
stereo audio decoding, alignment/token interleaving, mandatory image loading,
mixed speech/speechless batching, checkpoint save, and resume validation.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


SAMPLE_RATE = 24_000


def _write_stereo_wav(path: Path, *, duration_seconds: float, seed: int) -> None:
    frames = round(duration_seconds * SAMPLE_RATE)
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    # Deterministic, low-amplitude signals in alternating input/output channels.
    user = 0.04 * np.sin(2 * math.pi * (180 + seed * 7) * time)
    assistant = 0.04 * np.sin(2 * math.pi * (240 + seed * 11) * time)
    stereo = np.stack([assistant, user], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _write_image(path: Path, *, seed: int) -> None:
    base = (40 + seed * 31) % 256
    image = Image.new("RGB", (96, 96), (base, 90, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 84, 78), outline=(240, 230, 30), width=5)
    draw.ellipse((32, 32, 64, 64), fill=(30, 210, 120))
    image.save(path, format="JPEG", quality=95)


def _sample(root: Path, *, split: str, index: int, duration_seconds: float) -> dict:
    uid = f"smoke-{split}-{index:03d}"
    directory = root / "data_stereo" / uid
    directory.mkdir(parents=True, exist_ok=False)
    audio_path = directory / "stereo_dialogue.wav"
    image_path = directory / "image.jpg"
    _write_stereo_wav(audio_path, duration_seconds=duration_seconds, seed=index)
    _write_image(image_path, seed=index)

    # Keep both roles and sorted, positive intervals: the production manifest
    # validates exactly the invariants required by the crop/interleaver path.
    alignments = [
        ["この画像には何がありますか。", [0.10, 0.65], "USER"],
        ["緑の丸と黄色い枠があります。", [0.85, 1.55], "ASSISTANT"],
    ]
    (directory / "assistant_dialogue.refined.json").write_text(
        json.dumps({"alignments": alignments}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "uid": uid,
        "path": str(audio_path.resolve()),
        "image": str(image_path.resolve()),
        "duration": duration_seconds,
        "text": "緑の丸と黄色い枠があります。",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(
    output_dir: Path,
    *,
    train_samples: int,
    eval_samples: int,
    duration_seconds: float,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty smoke directory: {output_dir}"
        )
    if train_samples < 1 or eval_samples < 1:
        raise ValueError("train_samples and eval_samples must both be positive")
    if duration_seconds < 1.6:
        raise ValueError("duration_seconds must be at least 1.6")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        _sample(
            output_dir,
            split="train",
            index=index,
            duration_seconds=duration_seconds,
        )
        for index in range(train_samples)
    ]
    eval_rows = [
        _sample(
            output_dir,
            split="eval",
            index=train_samples + index,
            duration_seconds=duration_seconds,
        )
        for index in range(eval_samples)
    ]
    _write_jsonl(output_dir / "train_smoke.jsonl", train_rows)
    _write_jsonl(output_dir / "eval_smoke.jsonl", eval_rows)
    print(f"smoke dataset: {output_dir}")
    print(f"train={len(train_rows)}, eval={len(eval_rows)}, duration={duration_seconds}s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_dataset(
        args.output_dir.expanduser().resolve(),
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        duration_seconds=args.duration_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
