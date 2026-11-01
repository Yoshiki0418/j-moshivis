"""Deterministic evaluation seeding without touching unassigned CUDA devices."""

from __future__ import annotations

import random

import numpy as np


def seed_assigned_device(seed: int, device: str) -> None:
    """Seed host RNGs and only the CUDA generator named by ``device``.

    ``torch.manual_seed`` is intentionally not used: current PyTorch versions
    implement it by calling ``torch.cuda.manual_seed_all``.  On a shared
    multi-GPU host that can initialize or mutate generators for peer devices
    which the evaluation process does not own.
    """

    import torch

    seed = int(seed)
    random.seed(seed)
    # NumPy's legacy global RandomState accepts only uint32 seeds.
    np.random.seed(seed & 0xFFFF_FFFF)
    # Seed the CPU generator directly; torch.manual_seed would also seed CUDA.
    torch.default_generator.manual_seed(seed)

    target = torch.device(device)
    if target.type != "cuda":
        return
    index = target.index
    if index is None:
        index = torch.cuda.current_device()
    try:
        generator = torch.cuda.default_generators[index]
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError(f"CUDA RNG device is unavailable: {target}") from exc
    generator.manual_seed(seed)
