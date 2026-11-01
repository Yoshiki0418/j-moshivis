"""Diagnostics for frame-level text-token decisions during streaming inference."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch

from .response import SPECIAL_TEXT_IDS


TEXT_DECISION_DIAGNOSTIC_VERSION = "pad-vs-lexical-logits.v1"
ALIGNMENT_PAD_ID = 3
END_PAD_ID = 0


def require_finite_scalar(value: Any, *, name: str) -> float:
    """Convert one diagnostic scalar and reject silent numerical failures."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value, got shape {tuple(value.shape)}")
        result = float(value.detach().float().item())
    else:
        result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is non-finite: {result}")
    return result


def require_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    """Raise when a model output tensor contains NaN or infinity."""

    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def diagnose_text_logits(logits: torch.Tensor, *, frame: int) -> dict[str, Any]:
    """Reduce one frame's vocabulary logits to auditable PAD/content signals.

    The model currently has one batch item, one text stream and one time step.
    Keeping only the relevant probabilities and margins avoids copying the full
    vocabulary distribution into every prediction row.
    """

    if logits.ndim != 4 or logits.shape[:3] != (1, 1, 1):
        raise ValueError(f"expected text logits [1, 1, 1, V], got {tuple(logits.shape)}")
    vector = logits[0, 0, 0].float()
    first_lexical_id = max(SPECIAL_TEXT_IDS) + 1
    if SPECIAL_TEXT_IDS != set(range(first_lexical_id)):
        raise RuntimeError("text diagnostic assumes contiguous leading special-token IDs")
    if vector.numel() <= first_lexical_id:
        raise ValueError(f"vocabulary is too small for lexical tokens: {vector.numel()}")

    argmax_id_tensor = vector.argmax()
    lexical_logits = vector[first_lexical_id:]
    lexical_offset = lexical_logits.argmax()
    best_lexical_id_tensor = lexical_offset + first_lexical_id
    best_lexical_logit = lexical_logits[lexical_offset]
    alignment_pad_logit = vector[ALIGNMENT_PAD_ID]
    end_pad_logit = vector[END_PAD_ID]
    best_pad_logit = torch.maximum(alignment_pad_logit, end_pad_logit)
    log_normalizer = torch.logsumexp(vector, dim=0)
    special_log_mass = torch.logsumexp(vector[:first_lexical_id], dim=0)

    # One device-to-host synchronization per frame keeps the diagnostic cheap
    # enough to leave enabled for both greedy and sampling runs.
    values = torch.stack(
        (
            argmax_id_tensor,
            best_lexical_id_tensor,
            torch.exp(alignment_pad_logit - log_normalizer),
            torch.exp(end_pad_logit - log_normalizer),
            torch.exp(special_log_mass - log_normalizer),
            torch.exp(best_lexical_logit - log_normalizer),
            alignment_pad_logit - best_lexical_logit,
            end_pad_logit - best_lexical_logit,
            best_pad_logit - best_lexical_logit,
            torch.isfinite(vector).all(),
        )
    ).detach().cpu().tolist()
    if not bool(values[9]):
        raise RuntimeError(f"text logits contain non-finite values at frame {frame}")
    argmax_id = int(values[0])
    if argmax_id == ALIGNMENT_PAD_ID:
        argmax_class = "alignment_pad"
    elif argmax_id == END_PAD_ID:
        argmax_class = "end_pad"
    elif argmax_id in SPECIAL_TEXT_IDS:
        argmax_class = "other_special"
    else:
        argmax_class = "lexical"
    return {
        "decision_frame": int(frame),
        "argmax_id": argmax_id,
        "argmax_class": argmax_class,
        "best_lexical_id": int(values[1]),
        "probability": {
            "alignment_pad_id_3": float(values[2]),
            "end_pad_id_0": float(values[3]),
            "all_special_ids_0_to_6": float(values[4]),
            "best_lexical": float(values[5]),
        },
        "logit_margin_vs_best_lexical": {
            "alignment_pad_id_3": float(values[6]),
            "end_pad_id_0": float(values[7]),
            "best_of_ids_0_and_3": float(values[8]),
        },
    }


def summarize_text_decisions(
    decisions: Sequence[dict[str, Any]],
    *,
    question_end_frame: int,
    frame_rate: float,
) -> dict[str, Any]:
    """Aggregate response-window logit decisions without hiding raw frames."""

    class_names = ("alignment_pad", "end_pad", "other_special", "lexical")
    counts = {
        name: sum(item["argmax_class"] == name for item in decisions)
        for name in class_names
    }
    total = len(decisions)

    probability_names = (
        "alignment_pad_id_3",
        "end_pad_id_0",
        "all_special_ids_0_to_6",
        "best_lexical",
    )
    margin_names = (
        "alignment_pad_id_3",
        "end_pad_id_0",
        "best_of_ids_0_and_3",
    )

    def means(group: str, names: Sequence[str]) -> dict[str, float | None]:
        return {
            name: (
                sum(float(item[group][name]) for item in decisions) / total
                if total
                else None
            )
            for name in names
        }

    first_lexical = next(
        (int(item["decision_frame"]) for item in decisions if item["argmax_class"] == "lexical"),
        None,
    )
    return {
        "version": TEXT_DECISION_DIAGNOSTIC_VERSION,
        "num_response_window_frames": total,
        "argmax_count": counts,
        "argmax_rate": {
            name: (counts[name] / total if total else None) for name in class_names
        },
        "mean_probability": means("probability", probability_names),
        "mean_logit_margin_vs_best_lexical": means(
            "logit_margin_vs_best_lexical", margin_names
        ),
        "first_lexical_argmax_frame": first_lexical,
        "first_lexical_argmax_delay_sec": (
            (first_lexical - question_end_frame) / frame_rate
            if first_lexical is not None
            else None
        ),
    }
