"""J-MoshiVis audio-prompt runtime used by the benchmark CLI.

Heavy ML dependencies live in this module so manifest validation and scoring
remain CPU-only and importable from lightweight environments.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import get_moshi_vis
from moshi.models.loaders import get_mimi

from .audio_protocol import ASSISTANT_PRELUDE_PROTOCOL, BARE_QUESTION_PROTOCOL
from .rng import seed_assigned_device
from .response import SPECIAL_TEXT_IDS, extract_answer_events
from .text_diagnostics import (
    diagnose_text_logits,
    require_finite_scalar,
    require_finite_tensor,
    summarize_text_decisions,
)


TARGET_SAMPLE_RATE = 24_000
RUNTIME_VERSION = "audio-stream-dialogue-protocol-text-diagnostics.v5"
SAMPLING_CONFIG = {
    "audio_temperature": 0.5,
    "audio_top_k": 250,
    "text_temperature": 0.5,
    "text_top_k": 10,
}


@dataclass(slots=True)
class RuntimeConfig:
    checkpoint: Path
    model_config: Path
    tokenizer: Path
    mimi_weight: str
    audio_dialogue_protocol: str = BARE_QUESTION_PROTOCOL
    assistant_prelude_wav: Path | None = None
    device: str = "cuda:0"
    response_seconds: float = 10.0
    response_overlap_seconds: float = 0.32
    use_sampling: bool = False
    seed: int = 42


@dataclass(frozen=True, slots=True)
class AudioDialogueFrame:
    """One frame with explicitly separated user and assistant streams."""

    frame: int
    phase: str
    user_tokens: torch.Tensor
    forced_output_audio_tokens: torch.Tensor | None


def iter_audio_dialogue_frames(
    question_codes: torch.Tensor,
    *,
    silent_user_prelude_codes: torch.Tensor | None = None,
    assistant_prelude_codes: torch.Tensor | None = None,
) -> Iterator[AudioDialogueFrame]:
    """Yield a dialogue timeline without ever mixing output into user audio."""

    if question_codes.dim() != 3:
        raise ValueError("question_codes must have shape [B, K, T]")
    if (silent_user_prelude_codes is None) != (assistant_prelude_codes is None):
        raise ValueError(
            "silent user and assistant prelude codes must be supplied together"
        )
    offset = 0
    if silent_user_prelude_codes is not None:
        assert assistant_prelude_codes is not None
        if silent_user_prelude_codes.dim() != 3 or assistant_prelude_codes.dim() != 3:
            raise ValueError("prelude codes must have shape [B, K, T]")
        if silent_user_prelude_codes.shape[0] != question_codes.shape[0]:
            raise ValueError("prelude and question batch sizes differ")
        if silent_user_prelude_codes.shape[-1] != assistant_prelude_codes.shape[-1]:
            raise ValueError("silent-user and assistant prelude frame counts differ")
        for frame in range(silent_user_prelude_codes.shape[-1]):
            yield AudioDialogueFrame(
                frame=frame,
                phase="assistant_prelude",
                user_tokens=silent_user_prelude_codes[:, :, frame : frame + 1],
                forced_output_audio_tokens=assistant_prelude_codes[
                    :, :, frame : frame + 1
                ],
            )
        offset = int(silent_user_prelude_codes.shape[-1])
    for local_frame in range(question_codes.shape[-1]):
        yield AudioDialogueFrame(
            frame=offset + local_frame,
            phase="question_and_response",
            user_tokens=question_codes[:, :, local_frame : local_frame + 1],
            forced_output_audio_tokens=None,
        )


def stream_codebook_counts(lm_model: Any) -> tuple[int, int]:
    """Return user-input and assistant-output codec counts used by ``step``."""

    assistant_output = int(lm_model.num_audio_codebooks_out)
    user_input = int(
        lm_model.num_codebooks - assistant_output - lm_model.audio_offset
    )
    if user_input < 1 or assistant_output < 1:
        raise ValueError(
            "invalid streaming codebook layout: "
            f"total={lm_model.num_codebooks}, output={assistant_output}, "
            f"audio_offset={lm_model.audio_offset}"
        )
    return user_input, assistant_output


class JmoshiVisAudioRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.device = config.device
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        kyuteye = KyuteyeConfig.from_yml(config.model_config)
        mimi_path = config.mimi_weight
        if not Path(mimi_path).is_file():
            raise FileNotFoundError(
                "Mimi checkpoint must be a pinned local file so its bytes can be "
                f"hashed before inference: {mimi_path}"
            )
        self.mimi = get_mimi(mimi_path, device=self.device)
        self.mimi.eval()
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
        self.image_processor = ImageProcessor.from_config(
            kyuteye, device=self.device
        )
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(config.tokenizer))
        self.frame_rate = float(self.mimi.frame_rate)
        self.cache_cross_attention = bool(kyuteye.xa_shared)
        self.assistant_prelude_codes: torch.Tensor | None = None
        self.silent_user_prelude_codes: torch.Tensor | None = None
        self.assistant_prelude_num_samples = 0
        if config.audio_dialogue_protocol == BARE_QUESTION_PROTOCOL:
            if config.assistant_prelude_wav is not None:
                raise ValueError(
                    "bare-question protocol cannot accept an assistant prelude WAV"
                )
        elif config.audio_dialogue_protocol == ASSISTANT_PRELUDE_PROTOCOL:
            if config.assistant_prelude_wav is None:
                raise ValueError(
                    "assistant-prelude protocol requires assistant_prelude_wav"
                )
            self._prepare_assistant_prelude(config.assistant_prelude_wav)
        else:
            raise ValueError(
                f"unknown audio dialogue protocol: {config.audio_dialogue_protocol}"
            )

    @torch.inference_mode()
    def _prepare_assistant_prelude(self, audio_path: Path) -> None:
        """Encode a fixed greeting on output and matching silence on input."""

        waveform, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        if sample_rate != TARGET_SAMPLE_RATE:
            raise RuntimeError(
                f"expected {TARGET_SAMPLE_RATE} Hz prelude audio, got "
                f"{sample_rate} Hz: {audio_path}"
            )
        mono = waveform.mean(axis=1)
        if len(mono) == 0:
            raise ValueError(f"assistant prelude WAV is empty: {audio_path}")
        if not np.isfinite(mono).all():
            raise ValueError(f"assistant prelude WAV contains non-finite samples: {audio_path}")
        assistant = torch.from_numpy(mono).view(1, 1, -1).to(self.device)
        silent_user = torch.zeros_like(assistant)
        assistant_codes = self.mimi.encode(assistant)
        silent_codes = self.mimi.encode(silent_user)
        if assistant_codes.shape[-1] != silent_codes.shape[-1]:
            raise RuntimeError(
                "Mimi produced different frame counts for equal-length assistant "
                "prelude and user silence"
            )
        expected_input, expected_output = stream_codebook_counts(self.model.lm_model)
        if assistant_codes.shape[1] != expected_output:
            raise RuntimeError(
                "assistant prelude codec count does not match model output: "
                f"{assistant_codes.shape[1]} != {expected_output}"
            )
        if silent_codes.shape[1] != expected_input:
            raise RuntimeError(
                "silent user prelude codec count does not match model input: "
                f"{silent_codes.shape[1]} != {expected_input}"
            )
        self.assistant_prelude_codes = assistant_codes
        self.silent_user_prelude_codes = silent_codes
        self.assistant_prelude_num_samples = int(len(mono))

    def seed_for_sample(self, sample_id: str) -> int:
        """Return a stable per-sample seed, independent of run/resume order."""

        payload = f"{self.config.seed}\0{sample_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF

    def _set_sample_seed(self, sample_seed: int) -> None:
        seed_assigned_device(sample_seed, self.device)

    @torch.inference_mode()
    def _encode_image(self, image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_processor(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to decode image: {image_path}")
        source = self.image_embedder(image.unsqueeze(0).to(self.dtype))["cross_attention_src"]
        source = source.to(self.dtype)
        # Shared cross-attention projections can be cached once.  For a valid
        # non-shared configuration each layer must project the raw image source
        # with its own weights, so passing a layer-0 K/V pair would be wrong.
        conditioning = self.model.precompute_ca_kv(source) if self.cache_cross_attention else source
        return source, conditioning

    @torch.inference_mode()
    def infer(
        self,
        image_path: Path,
        audio_path: Path,
        *,
        speech_end_sec: float | None = None,
        sample_id: str,
    ) -> dict[str, Any]:
        sample_seed = self.seed_for_sample(sample_id)
        self._set_sample_seed(sample_seed)
        waveform, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        if sample_rate != TARGET_SAMPLE_RATE:
            raise RuntimeError(
                f"expected {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate} Hz: {audio_path}"
            )
        mono = waveform.mean(axis=1)
        question_samples = len(mono)
        tail = np.zeros(round(self.config.response_seconds * TARGET_SAMPLE_RATE), dtype=np.float32)
        mono_with_tail = np.concatenate([mono, tail])
        tensor = torch.from_numpy(mono_with_tail).view(1, 1, -1).to(self.device)

        started = time.perf_counter()
        raw_ca, cached_ca = self._encode_image(image_path)
        image_seconds = time.perf_counter() - started
        codes = self.mimi.encode(tensor)
        encode_seconds = time.perf_counter() - started - image_seconds

        audio_duration_sec = question_samples / TARGET_SAMPLE_RATE
        response_anchor_sec = audio_duration_sec if speech_end_sec is None else float(speech_end_sec)
        if not 0 < response_anchor_sec <= audio_duration_sec:
            raise ValueError(
                f"speech_end_sec must be within audio duration: {response_anchor_sec} vs {audio_duration_sec}"
            )
        prelude_frames = (
            0
            if self.assistant_prelude_codes is None
            else int(self.assistant_prelude_codes.shape[-1])
        )
        question_start_frame = prelude_frames
        question_end_frame = question_start_frame + int(
            round(response_anchor_sec * self.frame_rate)
        )
        overlap_frames = int(round(self.config.response_overlap_seconds * self.frame_rate))
        response_start_frame = max(0, question_end_frame - overlap_frames)
        events: list[dict[str, Any]] = []
        text_decisions: list[dict[str, Any]] = []
        gates: list[float] = []
        generation_started = time.perf_counter()
        with self.model.streaming():
            timeline = iter_audio_dialogue_frames(
                codes,
                silent_user_prelude_codes=self.silent_user_prelude_codes,
                assistant_prelude_codes=self.assistant_prelude_codes,
            )
            for dialogue_frame in timeline:
                frame = dialogue_frame.frame
                tokens, gate = self.model.step(
                    dialogue_frame.user_tokens,
                    ca_src=cached_ca,
                    forced_output_audio_tokens=(
                        dialogue_frame.forced_output_audio_tokens
                    ),
                )
                text_logits = self.model.get_streaming_attribute("text_logits")
                if text_logits is None:
                    raise RuntimeError("model did not expose text logits for diagnostics")
                if frame >= response_start_frame:
                    text_decisions.append(diagnose_text_logits(text_logits, frame=frame))
                else:
                    require_finite_tensor(
                        text_logits, name=f"text logits at frame {frame}"
                    )
                gate_value = require_finite_scalar(gate, name=f"gate at frame {frame}")
                gates.append(gate_value)
                if tokens is None:
                    continue
                token_id = int(tokens[0, 0, 0].item())
                events.append(
                    {
                        "frame": frame,
                        "time_sec": frame / self.frame_rate,
                        "token_id": token_id,
                        "piece": self.tokenizer.id_to_piece(token_id),
                        "gate": gate_value,
                        "phase": dialogue_frame.phase,
                    }
                )
        generation_seconds = time.perf_counter() - generation_started

        answer_events = extract_answer_events(
            events,
            question_end_frame=question_end_frame,
            overlap_frames=overlap_frames,
        )
        answer_ids = [int(event["token_id"]) for event in answer_events]
        all_content_ids = [
            int(event["token_id"]) for event in events
            if int(event["token_id"]) not in SPECIAL_TEXT_IDS
            and event["phase"] != "assistant_prelude"
        ]
        gate_array = np.asarray(gates, dtype=np.float64)
        return {
            "answer_text": self.tokenizer.decode(answer_ids).replace("⁇", "").strip(),
            "generated_text_all": self.tokenizer.decode(all_content_ids).replace("⁇", "").strip(),
            "answer_token_ids": answer_ids,
            "raw_text_events": events,
            "raw_text_decisions": text_decisions,
            "text_decision_diagnostics": summarize_text_decisions(
                text_decisions,
                question_end_frame=question_end_frame,
                frame_rate=self.frame_rate,
            ),
            "question_audio_duration_sec": audio_duration_sec,
            "question_speech_end_sec": response_anchor_sec,
            "question_start_frame": question_start_frame,
            "question_end_frame": question_end_frame,
            "response_seconds": self.config.response_seconds,
            "audio_dialogue_protocol": self.config.audio_dialogue_protocol,
            "assistant_prelude": {
                "enabled": self.assistant_prelude_codes is not None,
                "duration_sec": (
                    self.assistant_prelude_num_samples / TARGET_SAMPLE_RATE
                    if self.assistant_prelude_num_samples
                    else 0.0
                ),
                "num_samples": self.assistant_prelude_num_samples,
                "num_frames": prelude_frames,
                "assistant_audio_stream": (
                    "teacher_forced_mimi_codes"
                    if self.assistant_prelude_codes is not None
                    else "none"
                ),
                "user_audio_stream": (
                    "encoded_silence"
                    if self.silent_user_prelude_codes is not None
                    else "question_from_frame_zero"
                ),
                "text_stream": (
                    "model_generated_not_synchronized_and_excluded_from_answer"
                    if self.assistant_prelude_codes is not None
                    else "model_generated"
                ),
            },
            "sample_seed": sample_seed,
            "processed_frames": prelude_frames + int(codes.shape[-1]),
            "frame_rate": self.frame_rate,
            "gate": {
                "mean": float(gate_array.mean()) if gates else None,
                "std": float(gate_array.std()) if gates else None,
                "min": float(gate_array.min()) if gates else None,
                "max": float(gate_array.max()) if gates else None,
            },
            "timing": {
                "image_encode_sec": image_seconds,
                "audio_encode_sec": encode_seconds,
                "generation_sec": generation_seconds,
                "total_sec": time.perf_counter() - started,
            },
            "debug": {
                "raw_ca_shape": list(raw_ca.shape),
                "cross_attention_conditioning": (
                    "cached_shared_kv" if isinstance(cached_ca, tuple) else "raw_per_layer"
                ),
                "cached_k_shape": list(cached_ca[0].shape) if isinstance(cached_ca, tuple) else None,
                "audio_codes_shape": list(codes.shape),
                "assistant_prelude_codes_shape": (
                    list(self.assistant_prelude_codes.shape)
                    if self.assistant_prelude_codes is not None
                    else None
                ),
                "silent_user_prelude_codes_shape": (
                    list(self.silent_user_prelude_codes.shape)
                    if self.silent_user_prelude_codes is not None
                    else None
                ),
            },
        }
