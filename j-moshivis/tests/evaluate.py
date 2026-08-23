"""J-MoshiVis evaluation script.

For each evaluation sample (image + user-side audio + ground-truth dialogue),
this script:

    1. Runs streaming inference (mimi -> moshi_vis.step) to obtain the
       assistant's text + audio tokens.
    2. Decodes the text tokens with the SentencePiece tokenizer.
    3. Optionally decodes the assistant audio tokens via mimi for inspection.
    4. Computes a battery of metrics and writes a per-sample JSONL log plus
       an aggregate summary JSON.

Metrics
-------
    Relevance (image grounding)
      - clip_score_gen     : cosine(sim) between Japanese-Stable-CLIP image
                              embedding and the generated text embedding.
      - clip_score_ref     : same, but using the ground-truth assistant text
                              (acts as an upper-bound baseline per sample).
      - clip_text_sim      : cosine(sim) between generated text and ground
                              truth text (semantic similarity).

    Fluency (language quality)
      - text_token_entropy : Shannon entropy of generated text token IDs.
      - unique_token_ratio : |unique tokens| / |total tokens|.
      - bigram_repeat_rate : 1 - |unique bigrams| / |total bigrams|.

    Grounding heuristic
      - generic_phrase_hit : True if any well-known generic phrase is in the
                              generation (rule-based, see GENERIC_PHRASES).

    Cross-attention activity
      - gate_mean / std / max : statistics over the per-step CA gate value.

    Audio sanity (only computed when --decode-audio is set)
      - audio_rms          : RMS of the decoded waveform.
      - audio_active_ratio : fraction of frames with energy above a small
                              threshold (a proxy for "did the model speak?").

Usage
-----
    python tests/evaluate.py \\
        --checkpoint /workspace/j-moshivis/checkpoints/step_23000.safetensors \\
        --num-samples 30 \\
        --output /workspace/j-moshivis/eval_results/step_23000.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import get_moshi_vis
from moshi.models.loaders import get_mimi


TARGET_SR = 24000

# Phrases that strongly signal a non-grounded, generic Japanese reply. Used
# only as a coarse heuristic alongside CLIP. Keep terse — generic enough to
# trip on shortcut answers but not on legitimate descriptions.
GENERIC_PHRASES = [
    "何ですか",
    "教えてください",
    "わかりません",
    "知りたいです",
    "どう思いますか",
]


@dataclass
class SampleResult:
    sample_id: str
    image_path: str
    audio_path: str
    reference_text: str
    generated_text: str
    num_text_tokens: int
    duration_processed_sec: float
    text_token_entropy: float
    unique_token_ratio: float
    bigram_repeat_rate: float
    generic_phrase_hit: bool
    gate_mean: float
    gate_std: float
    gate_max: float
    gate_min: float
    clip_score_gen: float | None = None
    clip_score_ref: float | None = None
    clip_text_sim: float | None = None
    audio_rms: float | None = None
    audio_active_ratio: float | None = None
    audio_out_path: str | None = None
    error: str | None = None


@dataclass
class Aggregate:
    num_samples: int
    num_failed: int
    avg_clip_score_gen: float | None
    avg_clip_score_ref: float | None
    avg_clip_text_sim: float | None
    avg_text_token_entropy: float
    avg_unique_token_ratio: float
    avg_bigram_repeat_rate: float
    generic_phrase_rate: float
    avg_gate_mean: float
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IO + sample discovery
# ---------------------------------------------------------------------------
def _is_valid_sample(d: str) -> bool:
    return (
        os.path.exists(os.path.join(d, "image.jpg"))
        and os.path.exists(os.path.join(d, "user_dialogue.wav"))
        and os.path.exists(os.path.join(d, "dialogue.json"))
    )


def discover_samples(root: str, n: int, seed: int) -> list[str]:
    all_dirs = sorted(glob.glob(os.path.join(root, "*")))
    valid = [d for d in all_dirs if _is_valid_sample(d)]
    if not valid:
        raise RuntimeError(f"No valid samples found under {root}")
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:n]


def read_reference_text(sample_dir: str) -> str:
    """Concatenate all ASSISTANT turns from dialogue.json."""
    with open(os.path.join(sample_dir, "dialogue.json"), encoding="utf-8") as f:
        data = json.load(f)
    turns = data.get("dialogue", [])
    asst = [t["text"] for t in turns if t.get("speaker") in ("アシスタント", "assistant", "ASSISTANT")]
    return " ".join(asst).strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_jmoshivis(args, device: str, dtype: torch.dtype):
    print(f"[load] Mimi + MoshiVis on {device} ({dtype})")
    config = KyuteyeConfig.from_yml(args.config)

    mimi_weight = hf_hub_download(
        repo_id="kyutai/moshika-vis-pytorch-bf16",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    mimi = get_mimi(mimi_weight, device)
    mimi.eval()

    moshi_vis, image_embedder = get_moshi_vis(
        config,
        moshi_weight=Path(args.checkpoint),
        device=device,
        dtype=dtype,
    )
    moshi_vis.eval()
    image_embedder.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    return mimi, moshi_vis, image_embedder, sp, config


def load_clip(name: str, device: str):
    """Load a CLIP-style model that exposes get_image_features / get_text_features.

    `stabilityai/japanese-stable-clip-vit-l-16` is the highest quality option for
    Japanese but is gated on HF. If access is denied, point the user to
    workarounds rather than silently using a worse model.
    """
    print(f"[load] CLIP: {name}")
    try:
        model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        proc = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
        return model, tok, proc
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg or "restricted" in msg.lower():
            print(
                "\n[ERROR] CLIP model is gated and HF access is not granted.\n"
                f"        Model: {name}\n"
                "        Fix one of:\n"
                f"          (a) Request access at https://huggingface.co/{name}\n"
                "              and ensure HF_TOKEN points to that approved account.\n"
                "          (b) Pass --clip-model with an open alternative, e.g.\n"
                "              --clip-model line-corporation/clip-japanese-base\n"
                "              --clip-model openai/clip-vit-base-patch32   (English; sanity only)\n"
                "          (c) Pass --no-clip to skip CLIP metrics entirely.\n",
                flush=True,
            )
        raise


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def encode_image(image_embedder, image_processor: ImageProcessor, img_path: str,
                 device: str, dtype: torch.dtype):
    image_tensor = image_processor(img_path).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        out = image_embedder(image_tensor)
    return out["cross_attention_src"]


def encode_audio(mimi, wav_path: str, max_duration: float, device: str):
    """Load + mono-mix + length-cap a wav, then encode with mimi.

    Uses soundfile (not torchaudio) so we don't depend on the torchaudio C++
    extension, which in this env is built against an incompatible CUDA.
    The dataset is already at TARGET_SR (24 kHz); we assert rather than
    resample, since adding a resampler would re-introduce the dependency.
    """
    data, sr = sf.read(wav_path, always_2d=True)  # shape: (T, C)
    if sr != TARGET_SR:
        raise RuntimeError(
            f"Expected {TARGET_SR} Hz audio, got {sr} Hz at {wav_path}. "
            "Pre-resample the dataset or wire in a resampler."
        )
    mono = data.mean(axis=1).astype(np.float32)  # (T,)
    target_len = int(max_duration * TARGET_SR)
    if mono.shape[0] > target_len:
        mono = mono[:target_len]
    waveform = torch.from_numpy(mono).view(1, 1, -1).to(device)  # (B=1, C=1, T)
    with torch.no_grad():
        codes = mimi.encode(waveform)
    return codes, waveform


def run_streaming(moshi_vis, codes: torch.Tensor, ca_src,
                  max_frames: int) -> tuple[list[int], list[torch.Tensor], list[float]]:
    """Run streaming generation, returning text tokens, audio tokens, gate values."""
    text_token_ids: list[int] = []
    audio_token_chunks: list[torch.Tensor] = []
    gate_values: list[float] = []

    steps = min(codes.shape[-1], max_frames)
    with torch.no_grad(), moshi_vis.streaming():
        for t in range(steps):
            chunk = codes[:, :, t : t + 1]
            tokens, gate = moshi_vis.step(chunk, ca_src=ca_src)
            gate_values.append(float(gate))
            if tokens is not None:
                text_token_ids.append(int(tokens[0, 0, 0].item()))
                audio_token_chunks.append(tokens[:, 1:, :].cpu())
    return text_token_ids, audio_token_chunks, gate_values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def shannon_entropy(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = len(token_ids)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def bigram_repeat_rate(token_ids: list[int]) -> float:
    if len(token_ids) < 2:
        return 0.0
    bigrams = list(zip(token_ids[:-1], token_ids[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def has_generic_phrase(text: str) -> bool:
    return any(p in text for p in GENERIC_PHRASES)


def clean_decoded_text(text: str) -> str:
    return text.replace("⁇", "").strip()


@torch.no_grad()
def clip_image_embed(model, processor, img_path: str, device: str) -> torch.Tensor:
    image = Image.open(img_path).convert("RGB")
    inputs = {k: v.to(device) for k, v in processor(images=image, return_tensors="pt").items()}
    feats = model.get_image_features(**inputs)
    return feats / feats.norm(p=2, dim=-1, keepdim=True)


@torch.no_grad()
def clip_text_embed(model, tokenizer, text: str, device: str) -> torch.Tensor | None:
    if not text or len(text) < 2:
        return None
    inputs = tokenizer(text=[text], padding=True, truncation=True,
                       max_length=77, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)
    return feats / feats.norm(p=2, dim=-1, keepdim=True)


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a @ b.T).item())


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------
def evaluate_sample(args, sample_dir: str, mimi, moshi_vis, image_embedder,
                    sp, image_processor, clip_bundle, device, dtype) -> SampleResult:
    sample_id = os.path.basename(sample_dir.rstrip("/"))
    img_path = os.path.join(sample_dir, "image.jpg")
    wav_path = os.path.join(sample_dir, "user_dialogue.wav")
    ref_text = read_reference_text(sample_dir)

    try:
        ca_src = encode_image(image_embedder, image_processor, img_path, device, dtype)
        codes, _ = encode_audio(mimi, wav_path, args.max_duration, device)
        max_frames = int(args.max_duration * mimi.frame_rate)
        text_ids, audio_chunks, gates = run_streaming(moshi_vis, codes, ca_src, max_frames)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            sample_id=sample_id, image_path=img_path, audio_path=wav_path,
            reference_text=ref_text, generated_text="",
            num_text_tokens=0, duration_processed_sec=0.0,
            text_token_entropy=0.0, unique_token_ratio=0.0, bigram_repeat_rate=0.0,
            generic_phrase_hit=False,
            gate_mean=0.0, gate_std=0.0, gate_max=0.0, gate_min=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    gen_text = clean_decoded_text(sp.decode(text_ids))
    duration = codes.shape[-1] / float(mimi.frame_rate)

    gate_arr = np.asarray(gates, dtype=np.float64) if gates else np.zeros(1)

    result = SampleResult(
        sample_id=sample_id, image_path=img_path, audio_path=wav_path,
        reference_text=ref_text, generated_text=gen_text,
        num_text_tokens=len(text_ids), duration_processed_sec=duration,
        text_token_entropy=shannon_entropy(text_ids),
        unique_token_ratio=(len(set(text_ids)) / len(text_ids)) if text_ids else 0.0,
        bigram_repeat_rate=bigram_repeat_rate(text_ids),
        generic_phrase_hit=has_generic_phrase(gen_text),
        gate_mean=float(gate_arr.mean()), gate_std=float(gate_arr.std()),
        gate_max=float(gate_arr.max()), gate_min=float(gate_arr.min()),
    )

    # CLIP-based metrics
    if clip_bundle is not None:
        clip_model, clip_tok, clip_proc = clip_bundle
        try:
            img_emb = clip_image_embed(clip_model, clip_proc, img_path, device)
            gen_emb = clip_text_embed(clip_model, clip_tok, gen_text, device)
            ref_emb = clip_text_embed(clip_model, clip_tok, ref_text, device)
            result.clip_score_gen = cosine(img_emb, gen_emb)
            result.clip_score_ref = cosine(img_emb, ref_emb)
            result.clip_text_sim = cosine(gen_emb, ref_emb)
        except Exception as exc:  # noqa: BLE001
            result.error = f"clip: {type(exc).__name__}: {exc}"

    # Audio metrics + optional save
    if args.decode_audio and audio_chunks:
        try:
            audio_tokens = torch.cat(audio_chunks, dim=-1).to(device)
            with torch.no_grad():
                wav = mimi.decode(audio_tokens).squeeze().detach().cpu().numpy()
            result.audio_rms = float(np.sqrt(np.mean(wav ** 2)))
            result.audio_active_ratio = float(np.mean(np.abs(wav) > 1e-3))
            if args.save_audio_dir:
                os.makedirs(args.save_audio_dir, exist_ok=True)
                out_path = os.path.join(args.save_audio_dir, f"{sample_id}.wav")
                sf.write(out_path, wav, TARGET_SR)
                result.audio_out_path = out_path
        except Exception as exc:  # noqa: BLE001
            result.error = (result.error or "") + f" | audio: {type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def _mean_or_none(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def aggregate_results(results: list[SampleResult], cfg: dict[str, Any]) -> Aggregate:
    ok = [r for r in results if r.error is None and r.num_text_tokens > 0]
    failed = len(results) - len(ok)
    if not ok:
        return Aggregate(
            num_samples=len(results), num_failed=failed,
            avg_clip_score_gen=None, avg_clip_score_ref=None, avg_clip_text_sim=None,
            avg_text_token_entropy=0.0, avg_unique_token_ratio=0.0,
            avg_bigram_repeat_rate=0.0, generic_phrase_rate=0.0, avg_gate_mean=0.0,
            config=cfg,
        )
    return Aggregate(
        num_samples=len(results), num_failed=failed,
        avg_clip_score_gen=_mean_or_none([r.clip_score_gen for r in ok]),
        avg_clip_score_ref=_mean_or_none([r.clip_score_ref for r in ok]),
        avg_clip_text_sim=_mean_or_none([r.clip_text_sim for r in ok]),
        avg_text_token_entropy=float(np.mean([r.text_token_entropy for r in ok])),
        avg_unique_token_ratio=float(np.mean([r.unique_token_ratio for r in ok])),
        avg_bigram_repeat_rate=float(np.mean([r.bigram_repeat_rate for r in ok])),
        generic_phrase_rate=float(np.mean([r.generic_phrase_hit for r in ok])),
        avg_gate_mean=float(np.mean([r.gate_mean for r in ok])),
        config=cfg,
    )


def print_per_sample(results: list[SampleResult]) -> None:
    print("\n" + "=" * 100)
    print("PER-SAMPLE")
    print("=" * 100)
    for r in results:
        if r.error:
            print(f" [FAIL] {r.sample_id}: {r.error}")
            continue
        clip_g = f"{r.clip_score_gen:.3f}" if r.clip_score_gen is not None else "  -  "
        clip_r = f"{r.clip_score_ref:.3f}" if r.clip_score_ref is not None else "  -  "
        sim = f"{r.clip_text_sim:.3f}" if r.clip_text_sim is not None else "  -  "
        marker = "[GENERIC]" if r.generic_phrase_hit else "[ok]    "
        snippet = r.generated_text.replace("\n", " ")[:60]
        print(f" {marker} clip(gen)={clip_g} clip(ref)={clip_r} sim={sim} "
              f"gate={r.gate_mean:.3f} ent={r.text_token_entropy:.2f} | {snippet}")


def print_summary(agg: Aggregate) -> None:
    print("\n" + "=" * 60)
    print(f"AGGREGATE  ({agg.num_samples - agg.num_failed}/{agg.num_samples} ok)")
    print("=" * 60)

    def _fmt(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "n/a"

    print("Image grounding (cosine, higher = more relevant)")
    print(f"  CLIP(image, generated) : {_fmt(agg.avg_clip_score_gen)}")
    print(f"  CLIP(image, reference) : {_fmt(agg.avg_clip_score_ref)}    "
          "<- per-sample upper bound")
    print(f"  CLIP(gen,   reference) : {_fmt(agg.avg_clip_text_sim)}")
    print("\nFluency")
    print(f"  Token entropy          : {agg.avg_text_token_entropy:.4f}")
    print(f"  Unique-token ratio     : {agg.avg_unique_token_ratio:.4f}")
    print(f"  Bigram repeat rate     : {agg.avg_bigram_repeat_rate:.4f}")
    print("\nGrounding heuristic")
    print(f"  Generic-phrase rate    : {agg.generic_phrase_rate:.2%}")
    print("\nCross-attention activity")
    print(f"  Mean gate value        : {agg.avg_gate_mean:.4f}")

    # Quick verbal verdict
    if agg.avg_clip_score_gen is None:
        verdict = "n/a (CLIP disabled)"
    elif agg.avg_text_token_entropy < 1.5:
        verdict = "Model collapse (low diversity)"
    elif agg.generic_phrase_rate > 0.6:
        verdict = "Likely shortcut learning (mostly generic)"
    elif agg.avg_clip_score_gen < 0.20:
        verdict = "Weak image grounding"
    elif (agg.avg_clip_score_ref is not None
          and agg.avg_clip_score_gen >= 0.8 * agg.avg_clip_score_ref):
        verdict = "Approaching reference-level grounding"
    else:
        verdict = "Generating, but below reference grounding"
    print(f"\nVerdict: {verdict}")


def save_results(results: list[SampleResult], agg: Aggregate, output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    payload = {
        "aggregate": asdict(agg),
        "samples": [asdict(r) for r in results],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[save] wrote {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="/workspace/j-moshivis/checkpoints/step_23000.safetensors")
    p.add_argument("--config", default="/workspace/j-moshivis/configs/j-moshi-vis.yaml")
    p.add_argument("--tokenizer", default="/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")
    p.add_argument("--data-root", default="/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo")
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--max-duration", type=float, default=30.0,
                   help="Maximum input audio duration per sample, in seconds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Override device (e.g. cuda:0, cpu). Auto-detect if unset.")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip CLIP-based metrics (faster sanity runs).")
    p.add_argument("--clip-model", default="stabilityai/japanese-stable-clip-vit-l-16",
                   help="HF model id used for CLIP metrics. Default is gated; "
                        "see load_clip() for open alternatives.")
    p.add_argument("--decode-audio", action="store_true",
                   help="Decode generated audio tokens via mimi (slower).")
    p.add_argument("--save-audio-dir", default=None,
                   help="If set, save decoded waveforms here. Implies --decode-audio.")
    p.add_argument("--output", default=None,
                   help="Output JSON. Defaults to eval_results/<checkpoint-stem>.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.save_audio_dir:
        args.decode_audio = True
    if args.output is None:
        stem = Path(args.checkpoint).stem
        args.output = f"/workspace/j-moshivis/eval_results/{stem}.json"

    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[env] device={device}, dtype={dtype}")

    samples = discover_samples(args.data_root, args.num_samples, args.seed)
    print(f"[data] {len(samples)} samples (root={args.data_root})")

    mimi, moshi_vis, image_embedder, sp, _config = load_jmoshivis(args, device, dtype)
    image_processor = ImageProcessor()
    clip_bundle = None if args.no_clip else load_clip(args.clip_model, device)

    cfg_dump = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tokenizer": args.tokenizer,
        "data_root": args.data_root,
        "num_samples": args.num_samples,
        "max_duration": args.max_duration,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "no_clip": args.no_clip,
        "clip_model": None if args.no_clip else args.clip_model,
        "decode_audio": args.decode_audio,
        "frame_rate": float(mimi.frame_rate),
    }

    t0 = time.time()
    results: list[SampleResult] = []
    for sample_dir in tqdm(samples, desc="evaluate"):
        results.append(evaluate_sample(
            args, sample_dir, mimi, moshi_vis, image_embedder,
            sp, image_processor, clip_bundle, device, dtype,
        ))
    cfg_dump["wall_time_sec"] = round(time.time() - t0, 2)

    agg = aggregate_results(results, cfg_dump)
    print_per_sample(results)
    print_summary(agg)
    save_results(results, agg, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())