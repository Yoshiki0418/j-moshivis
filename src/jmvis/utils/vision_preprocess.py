"""Vision preprocessing utilities for J-MoshiVis
================================================
The frozen vision encoder (e.g. PaliGemma‐3B‐448, ViT-G/14) expects **448×448
RGB** images, normalized to *(-1,1)*.  For convenience this module exports a
`build_transform()` helper compatible with **torchvision/fiftyone style**
transforms as well as a high-level `preprocess()` shortcut usable in CLI/REST
endpoints.

Typical usage
-------------
```python
from jmvis.utils.vision_preprocess import preprocess

pixels = preprocess("cat.jpg", device="cuda")  # (1,3,448,448) float32
logits = model.encode_image(pixels)            # (1,S,H)
```
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Union

import torch

try:
    from PIL import Image
    from torchvision import transforms as T
except ImportError as e:  # pragma: no cover – minimal CI
    raise RuntimeError(
        "torchvision>=0.15 and Pillow are required for vision preprocessing"
    ) from e

logger = logging.getLogger(__name__)

__all__ = ["build_transform", "preprocess"]


# ---------------------------------------------------------------------------
# Transform builder
# ---------------------------------------------------------------------------

def build_transform(
    img_size: int = 448,
    crop_mode: str = "center",
    normalize: bool = True,
) -> Callable[[Image.Image], torch.Tensor]:
    """Create a torchvision transform pipeline.

    Parameters
    ----------
    img_size : int
        Final square size (448 for PaliGemma; 224 for ViT-B/16, etc.)
    crop_mode : {"center", "random"}
        Which crop strategy to use; *center* is deterministic and preferred for
        inference, *random* for data augmentation.
    normalize : bool
        If ``True`` scale to (-1,1) via `(x-0.5)/0.5`.
    """
    resize = T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC)
    if crop_mode == "center":
        crop = T.CenterCrop(img_size)
    elif crop_mode == "random":
        crop = T.RandomCrop(img_size)
    else:
        raise ValueError("crop_mode must be 'center' or 'random'")

    trans: list = [resize, crop, T.ToTensor()]
    if normalize:
        trans.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return T.Compose(trans)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def preprocess(
    img: Union[str, Path, Image.Image],
    img_size: int = 448,
    device: str | torch.device = "cpu",
    crop: str = "center",
) -> torch.Tensor:
    """Load image from *path* or PIL and return (1,3,H,W) tensor ready for encoder."""
    if not isinstance(img, Image.Image):
        img = Image.open(Path(img)).convert("RGB")
    transform = build_transform(img_size=img_size, crop_mode=crop)
    tensor = transform(img).unsqueeze(0)  # add batch dim
    return tensor.to(device)
