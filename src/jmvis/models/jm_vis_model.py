"""J-MoshiVis composite model
=================================

This module knits together three frozen components
  1. **J-Moshi (Helium) language backbone** – causal LM that generates speech-text tokens.
  2. **Vision encoder** (e.g. *google/paligemma‐3b‐448* or *vit-g/14*) – produces visual token embeddings.
  3. **Cross-Attention-Gated adapter** – light module that lets the language
     backbone attend to the visual tokens. Only this adapter (and an optional
     projection layer) is trainable.

The implementation purposefully keeps the first production version minimal yet
functional. Power-users can later patch the adapter hooks into every
transformer layer; here we fuse after the *final* hidden state for rapid
prototyping / CI tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
try:
    # Moshi専用クラスが利用可能なら優先
    from transformers import MoshiForCausalLM  # Transformers 4.5x 以降
    LMClass = MoshiForCausalLM
except Exception:
    LMClass = AutoModelForCausalLM

from .adapter import CrossAttentionGatedAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "JMoshiVisModel",
]


class JMoshiVisModel(nn.Module):
    """Minimal end-to-end *J-MoshiVis* model.

    Parameters
    ----------
    moshi_name : str
        HF Hub repo for Moshi weights (e.g. "kyutai-labs/moshi").
    jm_checkpoint : str | Path | None
        Path to *J-Moshi* fine-tuned checkpoint (.safetensors or .bin). If
        ``None`` the base Moshi weights are used.
    vision_name : str
        HF Hub repo for image encoder (defaults to "google/paligemma-3b-448")
    hidden_dim : int | None
        Hidden size of language model. If ``None`` it will be inferred after LM
        is loaded.
    freeze_backbone / freeze_vision : bool
        If ``True`` (default) the underlying backbone is set to ``eval`` and
        ``requires_grad=False``.
    adapter_kwargs : dict
        Arguments forwarded to :class:`CrossAttentionGatedAdapter`.
    """

    def __init__(
        self,
        moshi_name: str = "kyutai/moshiko-pytorch-bf16",
        jm_checkpoint: Union[str, Path, None] = None,
        vision_name: str = "google/paligemma-3b-448",
        hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
        freeze_vision: bool = True,
        adapter_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        adapter_kwargs = adapter_kwargs or {}

        # ────────────────────────────────────────────────────────────────────
        # 1) Language backbone  (Helium / Moshi)
        # ────────────────────────────────────────────────────────────────────
        logger.info("Loading Moshi backbone: %s", moshi_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(moshi_name)
        if jm_checkpoint is not None:
            logger.info("Loading J-Moshi checkpoint: %s", jm_checkpoint)
            self.backbone.load_state_dict(torch.load(jm_checkpoint, map_location="cpu"), strict=False)

        if freeze_backbone:
            _freeze_module(self.backbone)

        # Infer hidden size if not provided
        if hidden_dim is None:
            hidden_dim = self.backbone.config.hidden_size

        # ────────────────────────────────────────────────────────────────────
        # 2) Vision encoder
        # ────────────────────────────────────────────────────────────────────
        logger.info("Loading vision encoder: %s", vision_name)
        self.vision_encoder = AutoModel.from_pretrained(vision_name)
        if freeze_vision:
            _freeze_module(self.vision_encoder)

        # Map vision hidden_dim → language hidden_dim if they differ
        vision_hidden = getattr(self.vision_encoder.config, "hidden_size", hidden_dim)
        if vision_hidden != hidden_dim:
            logger.info("Adding projection layer %d → %d", vision_hidden, hidden_dim)
            self.vision_proj = nn.Linear(vision_hidden, hidden_dim, bias=False)
        else:
            self.vision_proj = nn.Identity()

        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)
        self.vision_encoder.eval()

        # ────────────────────────────────────────────────────────────────────
        # 3) Cross-attention gated adapter
        # ────────────────────────────────────────────────────────────────────
        try:
            vision_width = self.vision_encoder.config.hidden_size
        except AttributeError:
            # SiglipModel や他のラッパー構造の場合のフォールバック
            vision_width = getattr(getattr(self.vision_encoder, "vision_model", None), "config", None).hidden_size
        llm_width = self.backbone.config.hidden_size
        num_heads = self.backbone.config.num_attention_heads
        self.adapter = CrossAttentionGatedAdapter(
            hidden_size=llm_width,
            vision_width=vision_width,
            num_heads=num_heads,
            dropout=0.0,
        )

        # ────────────────────────────────────────────────────────────────────
        # Tokenizer (needed for convenience APIs)
        # ────────────────────────────────────────────────────────────────────
        self.tokenizer = _load_tokenizer_robust(moshi_name)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ======================================================================
    # Forward & helpers
    # ======================================================================
    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Return *projected* visual tokens (B, S, H)."""
        enc = getattr(self.vision_encoder, "vision_model", self.vision_encoder)
        outputs = enc(pixel_values=pixel_values,
                    output_hidden_states=True, return_dict=True)
        feats = outputs.last_hidden_state  # InferenceTensor
        feats = feats.clone()              # ★ここが重要（detachではダメ）
        feats = self.vision_proj(feats)    # 以降は通常の autograd OK
        return feats

    def _fuse(self, hidden_states: Tensor, visual_tokens: Tensor) -> Tensor:
        """Fuse visual context via adapter (B, T, H)."""
        return self.adapter(hidden_states, visual_tokens)

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Standard LM forward with visual conditioning.

        Returns logits of shape (B, T, V) where V = vocab size.
        """
        # 1) Text → hidden states (no softmax)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states: Tensor = backbone_outputs.hidden_states[-1]  # (B,T,H)

        # 2) Image → visual tokens
        visual_tokens = self.encode_image(pixel_values)  # (B,S,H)

        # 3) Cross-attention fuse
        fused = self._fuse(hidden_states, visual_tokens)

        # 4) LM head (weight sharing inside backbone)
        logits = self.backbone.lm_head(fused)
        return logits

    # ----------------------------------------------------------------------
    # Convenience greedy decoder (blocking, real-time not implemented here)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        text_prompts: Union[str, List[str]],
        images: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> List[str]:
        """Greedy/sample decode conditioned on ``images``.

        *images* – pre-processed pixel values of shape (B, C, H, W).
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        device = next(self.parameters()).device
        pixel_values = images.to(device)

        # Tokenize text
        toks = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_ids = toks.input_ids
        attn_mask = toks.attention_mask

        # Prepare kv_cache by running backbone once (latent cache opt omitted)
        with torch.no_grad():
            logits = self.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attn_mask,
            )
        # Now iterative decoding (naïve greedy/sample) – small helper
        generated: List[List[int]] = input_ids.tolist()
        for _ in range(max_new_tokens):
            next_token = _sample_next_token(logits[:, -1, :], temperature)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones_like(next_token)], dim=-1
            )
            logits = self.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attn_mask,
            )
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return texts


# ==========================================================================
# Helper utilities
# ==========================================================================

def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _sample_next_token(logit: Tensor, temperature: float) -> Tensor:
    if temperature == 0.0:
        return logit.argmax(dim=-1, keepdim=True)
    probs = (logit / temperature).softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


import os
import logging
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def _load_tokenizer_robust(model_id: str):
    """
    1) まず AutoTokenizer（slow優先）を試す
    2) 失敗したら repo を落として SPMファイル(tokenizer_spm_32k_3.model など)を直接読む
    """
    # 1) AutoTokenizer (fastを避ける)
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    except Exception as e:
        logger.warning("AutoTokenizer failed → fallback to SentencePiece: %s", e)

    # 2) 直接ダウンロードして SPM を探す
    local_dir = snapshot_download(model_id)
    candidates = [
        "tokenizer_spm_32k_3.model",
        "tokenizer.model",
        "spiece.model",
    ]
    spm_path = None
    for name in candidates:
        p = os.path.join(local_dir, name)
        if os.path.exists(p):
            spm_path = p
            break
    if spm_path is None:
        raise RuntimeError(
            f"No SentencePiece model found in {local_dir}. Looked for: {candidates}"
        )

    # SPMを扱える汎用トークナイザで読み込む（Llama→ダメならT5）
    try:
        from transformers import LlamaTokenizer
        tok = LlamaTokenizer(vocab_file=spm_path)
    except Exception:
        from transformers import T5Tokenizer
        tok = T5Tokenizer(vocab_file=spm_path)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

