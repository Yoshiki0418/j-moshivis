"""Metrics used by the three visual benchmarks in MoshiVis.

MoshiVis evaluates OCR-VQA with accuracy, VQAv2 with the official VQA
accuracy, and COCO Captions with CIDEr.  This module keeps those metrics
separate from the Japanese conversational diagnostics in :mod:`metrics` so a
paper result cannot silently change when a diagnostic normalizer changes.

The implementation is CPU-only and has no mandatory third-party dependency.
For CIDEr, the numerical scorer follows the COCO ``CiderScorer`` algorithm.
COCO's official pipeline applies the Stanford PTB tokenizer before the
scorer; callers that require byte-for-byte leaderboard comparability should
PTB-tokenize the captions first and pass ``tokenized=True``.  The default
tokenizer is deterministic and dependency-free, and is intended for local
development and ordinary English COCO captions.

References
----------
* PaliGemma OCR-VQA evaluator (lower-cased exact equality):
  https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/vqa.py
* Official VQAv2 evaluator:
  https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
* Official COCO CIDEr scorer:
  https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/cider/cider_scorer.py
"""

from __future__ import annotations

import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


# PaliGemma sets ``to_lower=True`` for OCR-VQA and otherwise compares the
# complete decoded answer.  In particular, punctuation and internal spaces
# remain significant; MoshiVis Appendix B.2 explicitly discusses that
# formatting sensitivity in the audio condition.
def normalize_ocr_vqa_answer(text: str) -> str:
    """Return the normalization used for PaliGemma-style OCR-VQA accuracy.

    This intentionally performs *only* Unicode-aware lower-casing.  Model
    decoding is responsible for removing BOS/EOS and surrounding generation
    artifacts before this function is called.
    """

    return str(text).lower()


def ocr_vqa_exact_match(prediction: str, references: Iterable[str]) -> float:
    """Score lower-cased, full-string OCR-VQA equality against any reference."""

    normalized_prediction = normalize_ocr_vqa_answer(prediction)
    return float(
        any(
            normalized_prediction == normalize_ocr_vqa_answer(reference)
            for reference in references
        )
    )


def ocr_vqa_accuracy(
    predictions: Sequence[str], references: Sequence[Sequence[str] | str]
) -> float:
    """Return mean OCR-VQA exact-match accuracy in the range ``[0, 1]``."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("OCR-VQA accuracy requires at least one example")
    scores = []
    for prediction, item_references in zip(predictions, references):
        refs = [item_references] if isinstance(item_references, str) else item_references
        if not refs:
            raise ValueError("each OCR-VQA example requires at least one reference")
        scores.append(ocr_vqa_exact_match(prediction, refs))
    return sum(scores) / len(scores)


# These tables and punctuation rules reproduce the official VQAv2 evaluator.
_VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'owsat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_VQA_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_VQA_ARTICLES = frozenset(("a", "an", "the"))
_VQA_PERIOD_RE = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VQA_COMMA_NUMBER_RE = re.compile(r"(\d)(,)(\d)")
_VQA_PUNCTUATION = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
)


def _vqa_clean_whitespace(text: str) -> str:
    return str(text).replace("\n", " ").replace("\t", " ").strip()


def _vqa_process_punctuation(text: str) -> str:
    output = text
    comma_in_number = _VQA_COMMA_NUMBER_RE.search(text) is not None
    for punctuation in _VQA_PUNCTUATION:
        if punctuation + " " in text or " " + punctuation in text or comma_in_number:
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    return _VQA_PERIOD_RE.sub("", output)


def _vqa_process_digits_articles(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _VQA_NUMBER_MAP.get(word, word)
        if word not in _VQA_ARTICLES:
            words.append(word)
    return " ".join(_VQA_CONTRACTIONS.get(word, word) for word in words)


def normalize_vqav2_answer(text: str) -> str:
    """Apply the official VQAv2 punctuation/digit/article normalization."""

    cleaned = _vqa_clean_whitespace(text)
    return _vqa_process_digits_articles(_vqa_process_punctuation(cleaned))


def vqav2_accuracy(
    prediction: str,
    human_answers: Sequence[str],
    *,
    require_ten: bool = True,
) -> float:
    """Return official VQAv2 leave-one-out accuracy for one question.

    The canonical annotation contains ten human answers.  The official score
    averages ten leave-one-annotator-out comparisons, hence one, two, and
    three matching annotators yield 0.3, 0.6, and 0.9 rather than the common
    shortcut 1/3, 2/3, and 1.0.  Set ``require_ten=False`` only for fixtures or
    non-standard VQA-style datasets.
    """

    answers = [_vqa_clean_whitespace(answer) for answer in human_answers]
    if require_ten and len(answers) != 10:
        raise ValueError(f"VQAv2 requires exactly 10 human answers, got {len(answers)}")
    if not answers:
        raise ValueError("VQAv2 accuracy requires at least one human answer")

    result = _vqa_clean_whitespace(prediction)
    # This conditional is part of the official implementation.  When all ten
    # annotators provided exactly the same spelling, no further normalization
    # is applied.
    if len(set(answers)) > 1:
        answers = [normalize_vqav2_answer(answer) for answer in answers]
        result = normalize_vqav2_answer(result)

    leave_one_out_scores = []
    for held_out_index in range(len(answers)):
        matches = sum(
            answer == result
            for index, answer in enumerate(answers)
            if index != held_out_index
        )
        leave_one_out_scores.append(min(1.0, matches / 3.0))
    return sum(leave_one_out_scores) / len(leave_one_out_scores)


def vqav2_corpus_accuracy(
    predictions: Sequence[str],
    human_answers: Sequence[Sequence[str]],
    *,
    require_ten: bool = True,
) -> float:
    """Return mean official VQAv2 accuracy in the range ``[0, 1]``."""

    if len(predictions) != len(human_answers):
        raise ValueError("predictions and human_answers must have the same length")
    if not predictions:
        raise ValueError("VQAv2 corpus accuracy requires at least one example")
    return sum(
        vqav2_accuracy(prediction, answers, require_ten=require_ten)
        for prediction, answers in zip(predictions, human_answers)
    ) / len(predictions)


_COCO_REMOVED_PUNCTUATION = frozenset(
    ("''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
     ".", "?", "!", ",", ":", "-", "--", "...", ";")
)
_COCO_TOKEN_RE = re.compile(r"\.\.\.|--|``|''|[\w]+(?:'[\w]+)?|[^\w\s]", re.UNICODE)
_PTB_SUFFIX_RE = re.compile(r"(?i)^(.+)(n't|'ll|'re|'ve|'s|'m|'d)$")


def tokenize_coco_caption(text: str) -> str:
    """Return a deterministic, dependency-free COCO-style token string.

    It lower-cases, splits common Penn Treebank contraction suffixes, and
    removes the same punctuation token list as COCO's ``PTBTokenizer``.  For
    exact leaderboard reproduction on unusual punctuation, use the official
    Stanford tokenizer externally and pass ``tokenized=True`` to
    :func:`cider_score`.
    """

    raw_tokens = _COCO_TOKEN_RE.findall(str(text).replace("\n", " ").lower())
    tokens: list[str] = []
    for token in raw_tokens:
        suffix_match = _PTB_SUFFIX_RE.match(token)
        if suffix_match:
            tokens.extend((suffix_match.group(1), suffix_match.group(2)))
        else:
            tokens.append(token)
    return " ".join(token for token in tokens if token not in _COCO_REMOVED_PUNCTUATION)


def _ngram_counts(sentence: str, max_order: int) -> Counter[tuple[str, ...]]:
    words = sentence.split()
    counts: Counter[tuple[str, ...]] = Counter()
    for order in range(1, max_order + 1):
        counts.update(
            tuple(words[start:start + order])
            for start in range(len(words) - order + 1)
        )
    return counts


@dataclass(frozen=True, slots=True)
class CiderScore:
    """Corpus and per-example COCO CIDEr values.

    ``score`` and ``per_example`` use the native COCO scorer scale.  MoshiVis
    reports the conventional paper scale, exposed as ``score_x100`` (for
    example a native 1.25 is reported as 125).
    """

    score: float
    per_example: tuple[float, ...]
    tokenization: str

    @property
    def score_x100(self) -> float:
        return self.score * 100.0

    @property
    def per_example_x100(self) -> tuple[float, ...]:
        return tuple(value * 100.0 for value in self.per_example)


def cider_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    max_order: int = 4,
    sigma: float = 6.0,
    tokenized: bool = False,
) -> CiderScore:
    """Compute COCO CIDEr with the official scorer formula.

    Parameters
    ----------
    predictions:
        One generated caption per image.
    references:
        One or more reference captions per image.  Document frequencies are
        computed over this complete corpus, so results must never be averaged
        from independently scored shards.
    max_order, sigma:
        Official defaults: 1--4 grams and Gaussian length sigma 6.
    tokenized:
        If true, every caption must already be a space-separated output of
        the official COCO PTB tokenizer.  If false, the deterministic local
        tokenizer above is applied.
    """

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("CIDEr requires at least one example")
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if any(not item_references for item_references in references):
        raise ValueError("each CIDEr example requires at least one reference")

    tokenizer = (lambda value: str(value)) if tokenized else tokenize_coco_caption
    cooked_predictions = [
        _ngram_counts(tokenizer(prediction), max_order) for prediction in predictions
    ]
    cooked_references = [
        [_ngram_counts(tokenizer(reference), max_order) for reference in item]
        for item in references
    ]

    document_frequency: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for item_references in cooked_references:
        unique_ngrams = {
            ngram for reference in item_references for ngram in reference
        }
        for ngram in unique_ngrams:
            document_frequency[ngram] += 1.0
    reference_log_count = math.log(float(len(cooked_references)))

    def counts_to_vector(
        counts: Counter[tuple[str, ...]],
    ) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
        vectors: list[dict[tuple[str, ...], float]] = [
            {} for _ in range(max_order)
        ]
        squared_norms = [0.0 for _ in range(max_order)]
        # The reference implementation uses bigram count as its length term
        # (``n == 1`` where n is zero-indexed); preserve that behavior.
        length = 0
        for ngram, term_frequency in counts.items():
            order_index = len(ngram) - 1
            log_document_frequency = math.log(
                max(1.0, document_frequency[ngram])
            )
            weight = float(term_frequency) * (
                reference_log_count - log_document_frequency
            )
            vectors[order_index][ngram] = weight
            squared_norms[order_index] += weight * weight
            if order_index == 1:
                length += term_frequency
        norms = [math.sqrt(value) for value in squared_norms]
        return vectors, norms, length

    def similarity(
        prediction_vector: list[dict[tuple[str, ...], float]],
        reference_vector: list[dict[tuple[str, ...], float]],
        prediction_norm: list[float],
        reference_norm: list[float],
        prediction_length: int,
        reference_length: int,
    ) -> list[float]:
        penalty = math.exp(
            -((float(prediction_length - reference_length) ** 2)
              / (2.0 * sigma * sigma))
        )
        values = [0.0 for _ in range(max_order)]
        for order_index in range(max_order):
            for ngram, prediction_weight in prediction_vector[order_index].items():
                reference_weight = reference_vector[order_index].get(ngram, 0.0)
                values[order_index] += (
                    min(prediction_weight, reference_weight) * reference_weight
                )
            denominator = (
                prediction_norm[order_index] * reference_norm[order_index]
            )
            if denominator:
                values[order_index] /= denominator
            values[order_index] *= penalty
        return values

    per_example: list[float] = []
    for prediction_counts, item_references in zip(
        cooked_predictions, cooked_references
    ):
        prediction_vector, prediction_norm, prediction_length = counts_to_vector(
            prediction_counts
        )
        accumulated = [0.0 for _ in range(max_order)]
        for reference_counts in item_references:
            reference_vector, reference_norm, reference_length = counts_to_vector(
                reference_counts
            )
            values = similarity(
                prediction_vector,
                reference_vector,
                prediction_norm,
                reference_norm,
                prediction_length,
                reference_length,
            )
            accumulated = [
                total + value for total, value in zip(accumulated, values)
            ]
        score = (
            sum(accumulated)
            / max_order
            / len(item_references)
            * 10.0
        )
        per_example.append(score)

    return CiderScore(
        score=sum(per_example) / len(per_example),
        per_example=tuple(per_example),
        tokenization="pretokenized_ptb" if tokenized else "dependency_free_coco",
    )


def official_coco_cider_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> CiderScore:
    """Run the unmodified COCO PTB-tokenizer + CIDEr package when installed.

    This is the strict path for publishable comparison with MoshiVis/COCO.
    Keeping it optional lets the other two metrics and pre-tokenized CIDEr run
    in a minimal CPU environment. Install ``pycocoevalcap`` (and a Java
    runtime required by its bundled Stanford tokenizer) to enable this path.
    """

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("CIDEr requires at least one example")
    if any(not item_references for item_references in references):
        raise ValueError("each CIDEr example requires at least one reference")
    if shutil.which("java") is None:
        raise RuntimeError(
            "official COCO CIDEr requires a Java runtime for the Stanford PTB "
            "tokenizer; install Java and ensure 'java' is on PATH"
        )
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as error:
        raise RuntimeError(
            "official COCO CIDEr requires pycocoevalcap; install the frozen "
            "evaluation dependency before producing paper numbers"
        ) from error

    ground_truth = {
        index: [{"caption": str(reference)} for reference in item_references]
        for index, item_references in enumerate(references)
    }
    results = {
        index: [{"caption": str(prediction)}]
        for index, prediction in enumerate(predictions)
    }
    tokenizer = PTBTokenizer()
    tokenized_ground_truth = tokenizer.tokenize(ground_truth)
    tokenized_results = tokenizer.tokenize(results)
    corpus_score, sample_scores = Cider().compute_score(
        tokenized_ground_truth, tokenized_results
    )
    return CiderScore(
        score=float(corpus_score),
        per_example=tuple(float(score) for score in sample_scores),
        tokenization="official_coco_ptb",
    )
