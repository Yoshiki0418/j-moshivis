"""Japanese CIDEr with one reproducible MeCab tokenizer on both sides."""

from __future__ import annotations

import unicodedata
from typing import Sequence

from .metrics_moshivis import CiderScore


TOKENIZATION = "NFKC + fugashi/UniDic words; Unicode punctuation removed"


def japanese_cider_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> CiderScore:
    """Score Japanese captions using the canonical COCO CIDEr formula.

    The English PTB tokenizer leaves most Japanese sentences as one token.
    MeCab segments both generated text and references identically before
    handing them to pycocoevalcap's unmodified CIDEr scorer.
    """

    if len(predictions) != len(references) or not predictions:
        raise ValueError("Japanese CIDEr requires equally sized nonempty inputs")
    if any(len(item) < 2 for item in references):
        raise ValueError("Japanese CIDEr requires multiple references per image")
    try:
        import fugashi
        from pycocoevalcap.cider.cider import Cider
    except ImportError as error:
        raise RuntimeError(
            "Japanese CIDEr requires fugashi, unidic-lite, and pycocoevalcap"
        ) from error

    tagger = fugashi.Tagger()

    def tokenize(value: str) -> str:
        words = []
        for word in tagger(unicodedata.normalize("NFKC", str(value))):
            surface = word.surface
            if surface and not all(unicodedata.category(char).startswith("P") for char in surface):
                words.append(surface)
        return " ".join(words)

    ground_truth = {
        index: [tokenize(reference) for reference in item]
        for index, item in enumerate(references)
    }
    results = {
        index: [tokenize(prediction)]
        for index, prediction in enumerate(predictions)
    }
    corpus_score, sample_scores = Cider().compute_score(ground_truth, results)
    return CiderScore(
        score=float(corpus_score),
        per_example=tuple(float(score) for score in sample_scores),
        tokenization=TOKENIZATION,
    )
