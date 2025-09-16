"""Streaming helpers for J‑MoshiVis
==================================

This file provides *one‑shot* and *token‑by‑token* streaming utilities on top
of :class:`jmvis.models.jm_vis_model.JMoshiVisModel`.

The original Moshi repository ships an advanced `StreamingModule` that manages
CUDA graphs, KV‑cache pinning, and block‑wise audio decoding. For early
experiments – and for unit‑tests / CPU environments – we offer here a **minimal
pure‑PyTorch fallback** that:

* keeps the LM's *past_key_values* alive across calls;
* fuses a *fixed* (already pre‑processed) image once at session start;
* exposes a simple `step()` method that consumes a *single* token and returns
  the **next token logits** in \*realtime.

You can later swap this out for the high‑performance StreamingContainer from
Kyutai's codebase by subclassing :class:`BaseStreamer`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import Tensor

from .jm_vis_model import JMoshiVisModel

logger = logging.getLogger(__name__)

__all__ = [
    "StreamingSession",
    "batch_stream",
]


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@dataclass
class SessionState:
    """Holds KV‑cache between decoding steps."""

    past_key_values: Optional[Tuple[Tuple[Tensor, ...], ...]] = None  # type: ignore
    generated: List[int] | None = None


class StreamingSession:
    """Lightweight token‑by‑token decoder for J‑MoshiVis.

    Example
    -------
    >>> session = StreamingSession(model, pixel_values, prompt="これは何の画像？")
    >>> while True:
    ...     next_id, logits = session.step()
    ...     print(model.tokenizer.decode([next_id]), end="", flush=True)
    ...     if next_id == model.tokenizer.eos_token_id:
    ...         break
    """

    def __init__(
        self,
        model: JMoshiVisModel,
        pixel_values: Tensor,
        prompt: str = "",
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> None:
        self.model = model.eval()  # ensure eval‑mode
        self.device = _device_of(model)
        self.pixel_values = pixel_values.to(self.device)
        self.temperature = float(temperature)
        self.max_length = int(max_length)

        # Tokenize prompt (may be empty)
        toks = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.input_ids = toks.input_ids  # (1, T)
        self.attn_mask = toks.attention_mask  # (1, T)

        self.state = SessionState(past_key_values=None, generated=[])

    # ------------------------------------------------------------------
    # Step‑wise decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> Tuple[int, Tensor]:
        """Generate *one* next token and append it to the context.

        Returns
        -------
        next_token_id : int
        next_token_logits : Tensor shape ``(V,)``
        """
        # Forward pass (with or without past_kv)
        outputs = self.model.backbone(
            input_ids=self.input_ids[:, -1:],  # last token only
            attention_mask=self.attn_mask[:, -1:],
            pixel_values=self.pixel_values,
            past_key_values=self.state.past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits: Tensor = outputs.logits[:, -1, :]  # (1,V)
        self.state.past_key_values = outputs.past_key_values  # update cache

        # Sample next token
        next_token = self._sample(logits[0])  # int
        # Append
        next_token_tensor = torch.tensor([[next_token]], device=self.device)
        self.input_ids = torch.cat([self.input_ids, next_token_tensor], dim=-1)
        self.attn_mask = torch.cat(
            [self.attn_mask, torch.ones_like(next_token_tensor)], dim=-1
        )
        self.state.generated.append(next_token)

        # Truncate context if exceeds max_length (simple sliding window)
        if self.input_ids.size(1) > self.max_length:
            excess = self.input_ids.size(1) - self.max_length
            self.input_ids = self.input_ids[:, excess:]
            self.attn_mask = self.attn_mask[:, excess:]
        return next_token, logits[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _sample(self, logits: Tensor) -> int:
        if self.temperature == 0.0:
            return int(logits.argmax(dim=-1))
        probs = (logits / self.temperature).softmax(dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return int(idx)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def generate_until_eos(self, eos_id: Optional[int] = None) -> str:
        eos_id = eos_id or self.model.tokenizer.eos_token_id
        while True:
            token, _ = self.step()
            if token == eos_id:
                break
        return self.model.tokenizer.decode(self.state.generated, skip_special_tokens=True)


# =====================================================================
# Batch convenience (non‑streaming) – for quick sanity tests
# =====================================================================
@torch.no_grad()
def batch_stream(
    model: JMoshiVisModel,
    texts: List[str],
    images: Tensor,
    max_new_tokens: int = 32,
) -> List[str]:
    """One‑shot batch generation – wraps `JMoshiVisModel.generate`."""
    return model.generate(texts, images, max_new_tokens=max_new_tokens)
