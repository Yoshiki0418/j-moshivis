"""Text-prompt runtime for paired MoshiVis image-understanding evaluation.

This runtime deliberately bypasses the audio/Depformer path.  It applies the
same per-codebook delay prefix as ``forward_speech`` to raw speechless frames:
audio is the model's zero token, except where a delayed stream still carries
its initial token.  Evaluating the same image/question in this mode tells us
whether a failure comes from visual learning itself or from text-to-speech
transfer.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from moshi.utils.sampling import sample_token

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import get_moshi_vis

from .response import SPECIAL_TEXT_IDS
from .rng import seed_assigned_device
from .runtime import SAMPLING_CONFIG
from .text_diagnostics import (
    diagnose_text_logits,
    require_finite_scalar,
    require_finite_tensor,
    summarize_text_decisions,
)


TEXT_RUNTIME_VERSION = "speechless-text-autoregressive.v3"


@dataclass(slots=True)
class TextRuntimeConfig:
    checkpoint: Path
    model_config: Path
    tokenizer: Path
    device: str = "cuda:0"
    max_new_tokens: int = 96
    use_sampling: bool = False
    seed: int = 42


class SpeechlessDelayState:
    """Stream raw speechless frames through Moshi's codebook delays."""

    def __init__(self, lm: Any, *, batch_size: int = 1) -> None:
        self.lm = lm
        self.delays = tuple(int(delay) for delay in lm.delays)
        if len(self.delays) != int(lm.num_codebooks):
            raise ValueError(
                f"delay/codebook mismatch: {len(self.delays)} != {lm.num_codebooks}"
            )
        if any(delay < 0 for delay in self.delays):
            raise ValueError("speechless delays must be non-negative")
        self.initial = lm.get_initial_token().expand(batch_size, -1, -1)
        self.history: list[torch.Tensor] = []

    def step(self, text_token_id: int) -> torch.Tensor:
        """Return one delayed model-input frame and retain its raw source."""

        raw = torch.full(
            (self.initial.shape[0], self.lm.num_codebooks, 1),
            self.lm.zero_token_id,
            dtype=torch.long,
            device=self.initial.device,
        )
        raw[:, 0, 0] = int(text_token_id)
        position = len(self.history)
        streams: list[torch.Tensor] = []
        for codebook, delay in enumerate(self.delays):
            if delay == 0:
                source = raw
            elif position < delay:
                source = self.initial
            else:
                source = self.history[position - delay]
            streams.append(source[:, codebook : codebook + 1, :])
        delayed = torch.cat(streams, dim=1)
        self.history.append(raw)
        return delayed


class JmoshiVisTextRuntime:
    """Autoregressive text-only inference matching the speechless train path."""

    def __init__(self, config: TextRuntimeConfig) -> None:
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.config = config
        self.device = config.device
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        kyuteye = KyuteyeConfig.from_yml(config.model_config)
        self.model, self.image_embedder = get_moshi_vis(
            kyuteye,
            moshi_weight=config.checkpoint,
            device=self.device,
            dtype=self.dtype,
            gen_kwargs={
                "use_sampling": config.use_sampling,
                "temp": SAMPLING_CONFIG["audio_temperature"],
                "top_k": SAMPLING_CONFIG["audio_top_k"],
                "temp_text": SAMPLING_CONFIG["text_temperature"],
                "top_k_text": SAMPLING_CONFIG["text_top_k"],
            },
        )
        self.model.eval()
        self.image_embedder.eval()
        self.image_processor = ImageProcessor.from_config(kyuteye, device=self.device)
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(config.tokenizer))
        self.cache_cross_attention = bool(kyuteye.xa_shared)

    def seed_for_sample(self, sample_id: str) -> int:
        payload = f"{self.config.seed}\0text\0{sample_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF

    def _set_sample_seed(self, sample_seed: int) -> None:
        seed_assigned_device(sample_seed, self.device)

    @torch.inference_mode()
    def _encode_image(self, image_path: Path) -> tuple[torch.Tensor, Any]:
        image = self.image_processor(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to decode image: {image_path}")
        source = self.image_embedder(image.unsqueeze(0).to(self.dtype))["cross_attention_src"]
        source = source.to(self.dtype)
        conditioning = self.model.precompute_ca_kv(source) if self.cache_cross_attention else source
        return source, conditioning

    @torch.inference_mode()
    def infer(self, image_path: Path, prompt_text: str, *, sample_id: str) -> dict[str, Any]:
        prompt_ids = list(self.tokenizer.encode_as_ids(prompt_text.strip()))
        if not prompt_ids:
            raise ValueError("text prompt tokenized to an empty sequence")
        sample_seed = self.seed_for_sample(sample_id)
        self._set_sample_seed(sample_seed)

        started = time.perf_counter()
        raw_ca, conditioning = self._encode_image(image_path)
        image_seconds = time.perf_counter() - started
        lm = self.model.lm_model
        generated_ids: list[int] = []
        events: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        gates: list[float] = []
        generation_started = time.perf_counter()

        # The initial frame is the same causal prefix used by forward_speech.
        # Its prediction of prompt token 0 is ignored; subsequent prompt tokens
        # are teacher-forced, and logits after the final prompt token predict the
        # first answer token.
        with lm.streaming():
            delay_state = SpeechlessDelayState(lm, batch_size=1)
            _, logits, gate = lm.forward_text(
                delay_state.initial,
                cross_attention_src=conditioning,
            )
            require_finite_tensor(logits, name="initial text logits")
            gates.append(require_finite_scalar(gate, name="initial gate"))
            for token_id in prompt_ids:
                _, logits, gate = lm.forward_text(
                    delay_state.step(token_id),
                    cross_attention_src=conditioning,
                )
                require_finite_tensor(logits, name="prompt text logits")
                gates.append(require_finite_scalar(gate, name="prompt gate"))

            for index in range(self.config.max_new_tokens):
                decisions.append(diagnose_text_logits(logits, frame=index))
                token = sample_token(
                    logits.float(),
                    self.config.use_sampling,
                    SAMPLING_CONFIG["text_temperature"],
                    SAMPLING_CONFIG["text_top_k"],
                )
                token_id = int(token[0, 0, 0].item())
                generated_ids.append(token_id)
                events.append(
                    {
                        "token_index": index,
                        "token_id": token_id,
                        "piece": self.tokenizer.id_to_piece(token_id),
                    }
                )
                if token_id == self.tokenizer.eos_id():
                    break
                _, logits, gate = lm.forward_text(
                    delay_state.step(token_id),
                    cross_attention_src=conditioning,
                )
                require_finite_tensor(logits, name="generated-token text logits")
                gates.append(require_finite_scalar(gate, name="generated-token gate"))

        generation_seconds = time.perf_counter() - generation_started
        content_ids = [token_id for token_id in generated_ids if token_id not in SPECIAL_TEXT_IDS]
        diagnostics = summarize_text_decisions(
            decisions,
            question_end_frame=0,
            frame_rate=1.0,
        )
        # In text mode the existing field is measured in generated tokens, not
        # seconds.  Keep the unambiguous token value and remove the misleading
        # audio-specific alias.
        diagnostics["first_lexical_argmax_delay_tokens"] = diagnostics.pop(
            "first_lexical_argmax_delay_sec"
        )
        return {
            "answer_text": self.tokenizer.decode(content_ids).replace("⁇", "").strip(),
            "generated_text_all": self.tokenizer.decode(content_ids).replace("⁇", "").strip(),
            "answer_token_ids": generated_ids,
            "raw_text_events": events,
            "raw_text_decisions": decisions,
            "text_decision_diagnostics": diagnostics,
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(generated_ids),
            "sample_seed": sample_seed,
            "gate": {
                "mean": sum(gates) / len(gates) if gates else None,
                "min": min(gates) if gates else None,
                "max": max(gates) if gates else None,
            },
            "timing": {
                "image_encode_sec": image_seconds,
                "generation_sec": generation_seconds,
                "total_sec": time.perf_counter() - started,
            },
            "debug": {
                "raw_ca_shape": list(raw_ca.shape),
                "cross_attention_conditioning": (
                    "cached_shared_kv" if isinstance(conditioning, tuple) else "raw_per_layer"
                ),
            },
        }
