"""Mimi audio codec wrapper for J‑MoshiVis
=======================================

*Purpose*
---------
Convert raw 16‑kHz **mono** waveform tensors ⟶ scalar *token IDs* that Moshi
models expect, and (optionally) do the inverse.  At training time we only need
*encoding* (waveform → tokens). During inference you may want to decode tokens
back to waveform for TTS playback; we include a basic stub.

The original Kyutai implementation sits in `moshi/audio/codec.py` and depends
on their proprietary Mimi codec binaries.  This wrapper tries to import it; if
unavailable, we gracefully fall back to a **µ‑law 8‑bit encoder** that yields
values in `[0,255]`.  That keeps unit tests and data pipelines functional on
any machine, though **audio quality is obviously far worse**.

Public API
~~~~~~~~~~
```
encode(waveform: Tensor, sr: int = 16000) -> List[int]
decode(tokens: List[int], sr: int = 16000) -> Tensor  # (1, N)
```
"""
from __future__ import annotations

import logging
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["encode", "decode", "mimi_available"]

try:
    # Kyutai's official Mimi codec
    from moshi.audio.codec import (
        encode as mimi_encode,  # type: ignore
        decode as mimi_decode,  # type: ignore
    )

    def encode(waveform: Tensor, sr: int = 16000) -> List[int]:  # noqa: D401
        """Encode with official Mimi codec (16‑kHz required)."""
        if sr != 16000:
            raise ValueError("Mimi codec expects 16‑kHz audio")
        return mimi_encode(waveform.numpy().flatten().tolist())  # type: ignore

    def decode(tokens: List[int], sr: int = 16000) -> Tensor:  # noqa: D401
        """Decode back to waveform (float Tensor in [‑1,1])."""
        if sr != 16000:
            raise ValueError("Mimi codec expects 16‑kHz audio")
        wav = torch.tensor(mimi_decode(tokens), dtype=torch.float32)
        return wav.unsqueeze(0)

    mimi_available = True
    logger.info("Using official Mimi codec bindings")

except ModuleNotFoundError:  # pragma: no cover – fallback path

    import numpy as np

    logger.warning(
        "Official Mimi codec not found – using µ‑law fallback (quality ↓)"
    )

    MU_LAW_QUANT = 256

    # Helper: µ‑law encode/decode (8‑bit) – from Torchaudio implementation
    def _mu_law_encode(x: Tensor, mu: int = MU_LAW_QUANT):
        mu = float(mu - 1)
        x_clamped = torch.clamp(x, -1.0, 1.0)
        fx = torch.sign(x_clamped) * torch.log1p(mu * torch.abs(x_clamped)) / torch.log1p(torch.tensor(mu))
        return ((fx + 1) / 2 * mu + 0.5).to(torch.long)

    def _mu_law_decode(x: Tensor, mu: int = MU_LAW_QUANT):
        mu = float(mu - 1)
        x = x.float()
        fx = 2.0 * (x / mu) - 1.0
        return torch.sign(fx) * (1.0 / mu) * (torch.pow(1 + mu, torch.abs(fx)) - 1.0)

    def encode(waveform: Tensor, sr: int = 16000) -> List[int]:  # noqa: D401
        """µ‑law 8‑bit encode (fallback)."""
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)  # convert to mono
        tokens = _mu_law_encode(waveform)
        return tokens.cpu().tolist()

    def decode(tokens: List[int], sr: int = 16000) -> Tensor:  # noqa: D401
        """µ‑law decode (fallback)."""
        tensor = torch.tensor(tokens, dtype=torch.long)
        wav = _mu_law_decode(tensor)
        return wav.unsqueeze(0)  # (1, N)

    mimi_available = False
