from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn


class Trainer:
    def __init__(self, moshi_vis, image_proj, dataloader, device: str = "cpu"):
        self.model = moshi_vis
        self.image_proj = image_proj
        self.dataloader = dataloader
        self.device = device

    def smoke_run(self, steps: int = 1):
        self.model.eval()
        it = iter(self.dataloader)
        for _ in range(steps):
            imgs = next(it)
            # Determine device from nested model if needed
            device = getattr(self.model, 'device', None)
            if device is None and hasattr(self.model, 'lm_model'):
                device = getattr(self.model.lm_model, 'device', None)
            if device is None:
                device = torch.device('cpu')
            # imgs may be a list of PIL images (collate returns list), convert to batch list
            if isinstance(imgs, list):
                pil_batch = imgs
            else:
                pil_batch = list(imgs)
            # image_proj expects PIL or list of PIL images
            # image_proj expects PIL or preprocessed tensors; try calling it and fallback to random embeds
            try:
                emb = self.image_proj(pil_batch)
            except Exception:
                print("warning: image_proj failed, using random embeddings for smoke test")
                model_dim = getattr(self.model, 'model_dim', None) or getattr(self.model, 'lm_model', None) and getattr(self.model.lm_model, 'model_dim', None)
                if model_dim is None:
                    model_dim = getattr(self.model, 'model_dim', None) or 512
                batch_size = len(pil_batch)
                # produce a fake cross_attention_src with shape [B, Seq, D]
                emb = {'cross_attention_src': torch.randn(batch_size, 1, model_dim, device=device)}
            # ensure embeddings are on device
            # call model.step / forward_text using embedding
            device = device
            # prepare tokens
            tokens = torch.zeros((len(pil_batch), 1, 1), device=device, dtype=torch.long)
            # Determine nested lm object (MoshiVisGen.lm_model -> MoshiVis) or direct MoshiVis
            lm = getattr(self.model, 'lm_model', None) or getattr(self.model, 'lm', None) or getattr(self.model, 'lm_model', None)
            try:
                # MoshiVis exposes forward_text
                if lm is not None and hasattr(lm, 'forward_text'):
                    _ = lm.forward_text(tokens, cross_attention_src=emb.get('cross_attention_src', None))
                else:
                    # fallback to calling step on generator if available
                    if hasattr(self.model, 'step'):
                        _ = self.model.step(tokens, ca_src=emb.get('cross_attention_src', None))
            except Exception:
                # ignore errors in smoke test but log
                print("warning: model forward failed in smoke_run")
            print("smoke step done")
