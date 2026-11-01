import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    logits = logits.reshape(-1, logits.size(-1)).float()
    target = target.reshape(-1)
    weights = weights.reshape(-1)

    denom = torch.sum(weights)
    if denom == 0:
        # Keep a zero-valued graph connection so DDP sees trainable Depformer
        # parameters as used on all-speechless microbatches.  Delayed padding
        # logits may be NaN, hence nan_to_num before forming the graph-zero.
        return torch.nan_to_num(logits, nan=0.0).sum() * 0.0

    # Delayed/undelayed sequences deliberately fill invalid rows with NaN.
    # Exclude them before cross_entropy: masking a NaN loss afterwards does not
    # guarantee finite gradients through the softmax backward.
    valid = weights > 0.0
    valid_logits = logits[valid]
    valid_target = target[valid]
    valid_weights = weights[valid]
    mb_loss = F.cross_entropy(valid_logits, valid_target, reduction="none")
    mb_loss = torch.sum(mb_loss * valid_weights) / torch.sum(valid_weights)

    return mb_loss
