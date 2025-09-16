"""Evaluation metrics utilities for J-MoshiVis
===========================================
This module bundles lightweight wrappers for common vision–language benchmarks
used in MoshiVis:

* **BLEU‑4**, **CIDEr**          – Image captioning (STAIR / COCO‑ja)
* **VQA accuracy**              – ja‑VQA (translation of VQAv2)
* **Latency stats (mean, p95)** – Real‑time streaming evaluation

Dependencies are *optional*; each metric will gracefully skip with a warning if
the required package is missing so that CI can still pass on barebones images.
"""
from __future__ import annotations

import logging
import time
from typing import List, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "bleu4",
    "cider",
    "vqa_accuracy",
    "latency_stats",
]

# ---------------------------------------------------------------------------
# BLEU‑4 (sacrebleu or NLTK fallback)
# ---------------------------------------------------------------------------

def bleu4(preds: List[str], refs: List[Sequence[str]]) -> float:  # noqa: D401
    """Compute corpus BLEU‑4.

    *preds* – list of predicted strings (len N)
    *refs*  – list of reference lists (len N, each K variants)
    """
    if len(preds) != len(refs):
        raise ValueError("Pred/Ref length mismatch")

    try:
        import sacrebleu  # type: ignore

        bleu = sacrebleu.corpus_bleu(preds, list(zip(*refs)))  # sacrebleu wants list[list]
        return bleu.score  # already 0‑100 scale
    except ModuleNotFoundError:
        logger.warning("sacrebleu not installed – falling back to nltk")
        try:
            import nltk  # type: ignore
            from nltk.translate.bleu_score import corpus_bleu  # type: ignore

            nltk.download("punkt", quiet=True)
            # NLTK returns 0‑1, we scale ×100
            return corpus_bleu(refs, preds) * 100.0  # type: ignore[arg-type]
        except ModuleNotFoundError:
            logger.error("Neither sacrebleu nor nltk present – BLEU unavailable")
            return float("nan")


# ---------------------------------------------------------------------------
# CIDEr (pycocoevalcap required)
# ---------------------------------------------------------------------------

def cider(preds: List[str], refs: List[Sequence[str]]) -> float:  # noqa: D401
    """Compute CIDEr (returns 0‑100)."""
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore
    except ModuleNotFoundError:
        logger.error("pycocoevalcap not installed – CIDEr unavailable")
        return float("nan")

    gts, res = {}, {}
    for i, (p, r) in enumerate(zip(preds, refs)):
        gts[i] = [str(s) for s in r]
        res[i] = [str(p)]
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score


# ---------------------------------------------------------------------------
# VQA accuracy (case‑insensitive exact match, numeric tolerance)
# ---------------------------------------------------------------------------

def _normalize(ans: str) -> str:
    return ans.lower().strip()


def vqa_accuracy(preds: List[str], gts: List[str]) -> float:  # noqa: D401
    if len(preds) != len(gts):
        raise ValueError("Pred/GT length mismatch")
    correct = 0
    for p, t in zip(preds, gts):
        p_n, t_n = _normalize(p), _normalize(t)
        if p_n == t_n:
            correct += 1
            continue
        # numeric leniency (e.g. "2" vs "two")
        try:
            if float(p_n) == float(t_n):
                correct += 1
        except ValueError:
            pass
    return correct * 100.0 / len(preds)


# ---------------------------------------------------------------------------
# Latency stats (mean / p95 in ms)
# ---------------------------------------------------------------------------

def latency_stats(latencies_ms: List[float]):  # noqa: D401
    """Return mean and 95‑percentile latency in *ms* (float, float)."""
    if not latencies_ms:
        raise ValueError("latencies_ms is empty")
    arr = sorted(latencies_ms)
    mean = sum(arr) / len(arr)
    p95 = arr[int(0.95 * len(arr)) - 1]
    return mean, p95
