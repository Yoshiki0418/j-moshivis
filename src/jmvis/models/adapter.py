"""Cross‑Attention + Gate adapter for J‑MoshiVis

This module injects a lightweight cross‑attention block that lets the (frozen)
Helium/J‑Moshi language backbone attend to visual tokens produced by a frozen
Vision Encoder (e.g. PaliGemma‑3B‑448 or ViT‑G/14).
Only the parameters inside this file are updated during fine‑tuning.

Notation
========
B – batch size • T – #audio/text tokens • S – #visual tokens • H – hidden dim
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "CrossAttentionGatedAdapter",
]


# class CrossAttentionGatedAdapter(nn.Module):
#     """Multi‑head cross‑attention with a learnable residual gate.

#     The adapter is meant to be *shared* across all transformer layers to keep
#     the parameter count low (≈ 200 M when H=4096, n_heads=32).
#     """

#     def __init__(
#         self,
#         hidden_dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         if hidden_dim % num_heads != 0:
#             raise ValueError("hidden_dim must be divisible by num_heads")

#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
#         self.scale = 1.0 / math.sqrt(self.head_dim)

#         # Projections
#         self.q_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.k_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.v_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)

#         self.dropout = nn.Dropout(dropout)

#         # A single scalar gate; sigmoid(gate) in (0,1)
#         self.gate = nn.Parameter(torch.tensor(0.0))

#     def _shape(self, x: Tensor, B: int, L: int) -> Tensor:
#         """(B, L, H) ⇒ (B, n_heads, L, head_dim)"""
#         return (
#             x.view(B, L, self.num_heads, self.head_dim)
#             .transpose(1, 2)  # (B, h, L, d)
#             .contiguous()
#         )

#     def forward(
#         self,
#         hidden_states: Tensor,  # (B, T, H)
#         visual_states: Tensor,  # (B, S, H)
#         attention_mask: Tensor | None = None,  # broadcastable to (B, 1, T, S)
#     ) -> Tensor:
#         B, T, _ = hidden_states.size()
#         S = visual_states.size(1)

#         # Project
#         q = self._shape(self.q_proj(hidden_states), B, T)
#         k = self._shape(self.k_proj(visual_states), B, S)
#         v = self._shape(self.v_proj(visual_states), B, S)

#         # Scaled dot‑product attention
#         attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,h,T,S)
#         if attention_mask is not None:
#             attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
#         attn_probs = self.dropout(attn_scores.softmax(dim=-1))
#         context = torch.matmul(attn_probs, v)  # (B,h,T,d)

#         # Merge heads & project back
#         context = (
#             context.transpose(1, 2)  # (B,T,h,d)
#             .contiguous()
#             .view(B, T, self.hidden_dim)
#         )
#         fused = self.out_proj(context)

#         # Gated residual: x + σ(g)·Δ
#         gate = torch.sigmoid(self.gate)
#         return hidden_states + gate * fused

class CrossAttentionGatedAdapter(nn.Module):
    def __init__(self, hidden_size: int, vision_width: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size      # LLM幅 (例: 4096)
        self.vision_width = vision_width    # 視覚幅 (例: 1152)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q はテキストから（in=hidden_size）
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # K/V は視覚から（in=vision_width）→ LLM幅へ合わせる
        self.k_proj = nn.Linear(vision_width, hidden_size, bias=False)
        self.v_proj = nn.Linear(vision_width, hidden_size, bias=False)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int, S: int):
        # x: (B*S, hidden_size) → (B, num_heads, S, head_dim)
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states: torch.Tensor, visual_states: torch.Tensor):
        # hidden_states: (B, S_t, hidden_size)
        # visual_states: (B, S_v, vision_width)
        B, S_t, H = hidden_states.shape
        Bv, S_v, C_v = visual_states.shape
        assert B == Bv, "batch mismatch"
        assert C_v == self.vision_width, f"vision width mismatch: {C_v} != {self.vision_width}"
        assert H == self.hidden_size, f"LLM width mismatch: {H} != {self.hidden_size}"

        # dtype/device をテキスト側に合わせる（混在を防止）
        visual_states = visual_states.to(hidden_states.dtype)

        q = self.q_proj(hidden_states.reshape(B * S_t, H))
        k = self.k_proj(visual_states.reshape(B * S_v, C_v))
        v = self.v_proj(visual_states.reshape(B * S_v, C_v))

        q = self._shape(q, B, S_t)   # (B, h, S_t, d)
        k = self._shape(k, B, S_v)   # (B, h, S_v, d)
        v = self._shape(v, B, S_v)   # (B, h, S_v, d)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)                 # (B, h, S_t, d)
        context = context.permute(0, 2, 1, 3).contiguous()      # (B, S_t, h, d)
        context = context.view(B, S_t, self.hidden_size)        # (B, S_t, H)

        out = self.out_proj(context)                            # (B, S_t, H)
        # ここで residual を足すかは設計次第。足すなら:
        # out = out + hidden_states
        return out


if __name__ == "__main__":
    # Minimal sanity check
    B, T, S, H = 2, 16, 32, 512
    x = torch.randn(B, T, H)
    v = torch.randn(B, S, H)
    adapter = CrossAttentionGatedAdapter(hidden_dim=H, num_heads=8)
    y = adapter(x, v)
    assert y.shape == x.shape
    print("Adapter forward pass OK →", y[0, 0, :5])
