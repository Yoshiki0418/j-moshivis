"""SakanaAI-compatible Japanese ROUGE used by JA-VG-VQA-500.

This mirrors the public ``evolutionary-model-merge`` evaluator: neologdn
normalisation, emoji removal, fugashi ``-Owakati`` tokenisation, and
``rouge-score==0.1.2``'s BootstrapAggregator.  Dependencies are optional so the
default lightweight scorer remains usable in the training environment.  The
published evaluator additionally emits a ``*_ja`` variant after filtering
non-Japanese generations with fastText ``lid.176.ftz``; that filter is only
enabled when an explicit, hashable model path is supplied.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np


BOOTSTRAP_SEED = 42


class MecabTokenizer:
    def __init__(self) -> None:
        try:
            import emoji
            import neologdn
            from fugashi import Tagger
        except ImportError as exc:
            raise RuntimeError(
                "Official Japanese ROUGE dependencies are missing. Install "
                "j-moshivis/evaluation/requirements-eval.txt."
            ) from exc
        self._emoji = emoji
        self._neologdn = neologdn
        self._tagger = Tagger("-Owakati")

    def normalize_answer(self, text: str) -> str:
        text = "".join("" if self._emoji.is_emoji(char) else char for char in text)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        text = self._neologdn.normalize(text)
        return " ".join(text.split())

    def tokenize(self, text: str) -> list[str]:
        return self._tagger.parse(self.normalize_answer(text)).split()


def official_rouge_ja(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    try:
        from rouge_score import rouge_scorer, scoring
    except ImportError as exc:
        raise RuntimeError(
            "rouge-score is missing. Install j-moshivis/evaluation/requirements-eval.txt."
        ) from exc
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=MecabTokenizer(),
    )
    aggregator = scoring.BootstrapAggregator()
    for reference, prediction in zip(references, predictions):
        aggregator.add_scores(scorer.score(reference, prediction))
    # rouge-score's BootstrapAggregator draws from NumPy's process-global RNG.
    # Freeze that draw while restoring the caller's state so repeated scoring
    # produces byte-identical summaries without perturbing inference code.
    random_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        result = aggregator.aggregate()
    finally:
        np.random.set_state(random_state)
    return {name: value.mid.fmeasure * 100 for name, value in result.items()}


def sakana_rouge_scores(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    language_id_model: str | Path | None = None,
) -> dict[str, float]:
    """Return SakanaAI's MeCab ROUGE, optionally including strict-JA scores."""

    result = official_rouge_ja(references, predictions)
    if language_id_model is None:
        return result
    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError(
            "fasttext is required for language-filtered SakanaAI ROUGE. "
            "Install j-moshivis/evaluation/requirements-eval.txt."
        ) from exc
    model = fasttext.load_model(str(language_id_model))

    def japanese_probability(text: str) -> float:
        labels, probabilities = model.predict(text.replace("\n", " ").strip() or " ", k=-1)
        return dict(zip(labels, probabilities)).get("__label__ja", 0.0)

    filtered: list[str] = []
    for reference, prediction in zip(references, predictions):
        if japanese_probability(reference) >= 0.5 and japanese_probability(prediction) < 0.5:
            filtered.append("")
        else:
            filtered.append(prediction)
    strict = official_rouge_ja(references, filtered)
    result.update({f"{name}_ja": value for name, value in strict.items()})
    return result
