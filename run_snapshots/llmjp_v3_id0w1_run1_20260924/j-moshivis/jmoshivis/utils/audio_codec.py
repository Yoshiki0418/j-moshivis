"""Mimi audio codec wrapper for J-MoshiVis
=======================================

*Purpose*
---------
Convert raw 16-kHz mono waveform tensors ⟶ scalar *token IDs* that Moshi
models expect, and (optionally) do the inverse.
"""

import logging
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["encode", "decode", "mimi_available", "load_mimi"]

_mimi_model = None  # global MimiModel instance
mimi_available = False


def load_mimi(repo_id: str = "kyutai/moshika-vis-pytorch-bf16",
              filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
              device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load MimiModel from Hugging Face Hub safetensors checkpoint."""
    global _mimi_model, mimi_available
    try:
        from moshi.models import MimiModel
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = load_file(weights_path)

        _mimi_model = MimiModel.build_from_pretrained(state_dict).to(device).eval()
        mimi_available = True
        logger.info(f"Loaded MimiModel from {repo_id}/{filename} on {device}")
    except Exception as e:
        logger.warning(f"Failed to load MimiModel: {e}")
        mimi_available = False


def encode(waveform: Tensor, sr: int = 16000) -> List[int]:
    """Encode waveform to Mimi tokens."""
    if mimi_available and _mimi_model is not None:
        if sr != _mimi_model.sample_rate:
            raise ValueError(f"MimiModel expects { _mimi_model.sample_rate } Hz audio, got {sr}")
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [B=1,C=1,T]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # assume [C,T]
        codes = _mimi_model.encode(waveform.to(next(_mimi_model.parameters()).device))
        return codes.squeeze(0).cpu().tolist()
    else:
        # fallback µ-law
        return _mu_law_encode_fallback(waveform)


def decode(tokens: List[int], sr: int = 16000) -> Tensor:
    """Decode tokens back to waveform."""
    if mimi_available and _mimi_model is not None:
        codes = torch.tensor(tokens, dtype=torch.long, device=next(_mimi_model.parameters()).device)
        codes = codes.view(1, _mimi_model.num_codebooks, -1)  # [B,K,T]
        wav = _mimi_model.decode(codes)
        return wav.cpu()
    else:
        return _mu_law_decode_fallback(tokens)


# =========================
# µ-law fallback
# =========================
MU_LAW_QUANT = 256

def _mu_law_encode_fallback(x: Tensor, mu: int = MU_LAW_QUANT) -> List[int]:
    mu = float(mu - 1)
    x_clamped = torch.clamp(x, -1.0, 1.0)
    fx = torch.sign(x_clamped) * torch.log1p(mu * torch.abs(x_clamped)) / torch.log1p(torch.tensor(mu))
    return ((fx + 1) / 2 * mu + 0.5).to(torch.long).cpu().tolist()

def _mu_law_decode_fallback(tokens: List[int], mu: int = MU_LAW_QUANT) -> Tensor:
    x = torch.tensor(tokens, dtype=torch.long)
    mu = float(mu - 1)
    fx = 2.0 * (x.float() / mu) - 1.0
    wav = torch.sign(fx) * (1.0 / mu) * (torch.pow(1 + mu, torch.abs(fx)) - 1.0)
    return wav.unsqueeze(0)  # (1, N)
