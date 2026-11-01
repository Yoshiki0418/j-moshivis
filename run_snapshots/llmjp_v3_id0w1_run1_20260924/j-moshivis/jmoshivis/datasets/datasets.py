import itertools
import json
import logging
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import torch

import numpy as np
import sphn
import torch.distributed as dist
import sentencepiece as spm

from jmoshivis.distributed import get_rank

from .interleaver import InterleavedTokenizer, Sample
from jmoshivis.conditioners.base import ConditionAttributes, TensorCondition
from jmoshivis.models.image_projection import ImageProjection, ImageProcessor

logger = logging.getLogger("dataset")


AudioChunkPath = tuple[str, float]
_LOADED_DATASETS: dict[Path, list[AudioChunkPath]] = {}
_JSONL_OFFSETS: dict[Path, tuple[int, ...]] = {}
SPEECH_CHUNK_POLICY = "one_random_crop_per_row"


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, rank: int, world_size: int, instruct_tokenizer: InterleavedTokenizer
) -> list[AudioChunkPath]:
    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    duration = instruct_tokenizer.duration_sec
    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    chunks: list[AudioChunkPath] = []
    for line in lines:
        data = json.loads(line)
        start_sec = 0
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration

    main_logger_info(f"{path} loaded and chunked.")
    _LOADED_DATASETS[path] = chunks

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        # Filesystem traversal order is not guaranteed.  Keeping this stable is
        # necessary for seeded sampling to be reproducible across machines.
        jsonl_files = sorted(self.path.rglob("*jsonl"))
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []

    sample_sources = pretrain_data

    for source in sample_sources.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source "
                "as <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        assert weight > 0, (
            f"Make sure to define strictly positive data sampling weights, not {weight}"
        )

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or directory "
                "that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights


def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    device: torch.device,
    shuffle_pretrain: bool = True,
    image_embedder: ImageProjection | None = None,
    image_root: str | None = None,
    speech_chunk_policy: str = SPEECH_CHUNK_POLICY,
    speech_decode_workers: int = 4,
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)

    shuffle = not is_eval and shuffle_pretrain

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
            image_root=image_root,
            image_embedder=image_embedder,
            device=device,
            speech_chunk_policy=speech_chunk_policy,
            speech_decode_workers=speech_decode_workers,
        )
        for source in sources
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def build_speechless_dataset(
    pretrain_data: str,
    text_tokenizer,  # ← HF tokenizer など (encode を持つもの)
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    device: torch.device,
    shuffle_pretrain: bool = True,
    image_embedder: ImageProjection | None = None,
    image_root: str | None = None,
    num_audio_streams: int = 16,   # = dep_q
    audio_pad_id: int = 0,        # audio padding 用 ID
    target_len: int | None = None,
    text_loss_padding_id: int = -1,
) -> Iterator[Sample]:
    """
    テキストのみ（speechless）データセット用の Iterator を構築する。

    - pretrain_data: "path1:weight1,path2:weight2,..." 形式（parse_data_sources と同じ）
    - text_tokenizer: encode(text, add_special_tokens=False) を持つテキスト用 tokenizer
    - seed / rank / world_size: 分散学習用
    - is_eval: True の場合は 1 周だけ回して終了（chain）、False の場合は interleave で無限ループ
    - shuffle_pretrain: 学習時のみ各エポックでシャッフルするかどうか
    - image_embedder / image_root: 画像条件付け（ConditionAttributes）用
    - num_audio_streams: MoshiVis の dep_q（Mimi の n_q と揃える）
    - audio_pad_id: audio コードの padding ID（トレーナ側でマスクする前提）
    """
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)
    shuffle = not is_eval and shuffle_pretrain

    dataset_iterators = [
        get_speechless_dataset_iterator(
            source=source,
            text_tokenizer=text_tokenizer,
            image_embedder=image_embedder,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
            image_root=image_root,
            device=device,
            num_audio_streams=num_audio_streams,
            audio_pad_id=audio_pad_id,
            target_len=target_len,
            text_loss_padding_id=text_loss_padding_id,
        )
        for source in sources
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def _seeded_rng(seed: int | None, *stream_ids: int) -> np.random.RandomState:
    """Return a stable RNG for one logical data stream.

    ``RandomState`` accepts an array seed.  Splitting streams this way avoids
    coupling row order, modality order, and source selection to the number of
    random calls made by another stream.
    """

    seed_words = [int(0 if seed is None else seed) & 0xFFFFFFFF]
    seed_words.extend(int(value) & 0xFFFFFFFF for value in stream_ids)
    return np.random.RandomState(seed=np.asarray(seed_words, dtype=np.uint32))


def _jsonl_offsets(path: Path) -> tuple[int, ...]:
    """Index JSONL byte offsets without retaining all row contents in memory."""

    path = path.resolve()
    cached = _JSONL_OFFSETS.get(path)
    if cached is not None:
        return cached

    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            offsets.append(offset)

    result = tuple(offsets)
    _JSONL_OFFSETS[path] = result
    return result


def iter_sharded_jsonl_lines(
    path: Path,
    *,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int | None,
    epoch: int,
    drop_remainder: bool = False,
    start_position: int = 0,
) -> Iterator[str]:
    """Yield a deterministic, disjoint JSONL shard for one rank.

    All ranks construct the same global row order and then take
    ``row_order[rank::world_size]``.  Training sets ``drop_remainder`` so the
    final ``N % world_size`` rows are excluded *before* sharding.  Every rank
    then consumes the same number of rows and reaches the next epoch together.
    Evaluation leaves ``drop_remainder`` disabled and visits every row, allowing
    rank lengths to differ by one.  Byte-offset indexing keeps memory
    proportional to the number of rows rather than the dialogue text.
    ``start_position`` is a rank-local cursor; slicing offsets lets checkpoint
    resume avoid reading or preprocessing already-consumed rows.
    """

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if start_position < 0:
        raise ValueError(
            f"start_position must be non-negative, got {start_position}"
        )

    offsets = _jsonl_offsets(path)
    row_order = np.arange(len(offsets), dtype=np.int64)
    if shuffle:
        path_stream_id = zlib.crc32(str(path.resolve()).encode("utf-8"))
        _seeded_rng(seed, epoch, path_stream_id, 0x4A534F4E).shuffle(row_order)

    dropped = 0
    if drop_remainder:
        dropped = len(row_order) % world_size
        if dropped:
            row_order = row_order[: len(row_order) - dropped]
    rank_row_order = row_order[rank::world_size]
    if start_position > len(rank_row_order):
        raise ValueError(
            f"start_position={start_position} exceeds rank-local row count "
            f"{len(rank_row_order)} for {path}"
        )
    if rank == 0:
        logger.info(
            "JSONL rank-sharding policy path=%s epoch=%d rows=%d world_size=%d "
            "drop_remainder=%s dropped=%d start_position=%d",
            path,
            epoch,
            len(offsets),
            world_size,
            drop_remainder,
            dropped,
            start_position,
        )

    with path.open("rb") as handle:
        for row_index in rank_row_order[start_position:]:
            handle.seek(offsets[int(row_index)])
            yield handle.readline().decode("utf-8")


def _sharded_jsonl_row_count(
    path: Path, *, rank: int, world_size: int, drop_remainder: bool
) -> int:
    """Return a rank's row count without reading or shuffling row contents."""

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    row_count = len(_jsonl_offsets(path))
    if drop_remainder:
        row_count -= row_count % world_size
    if rank >= row_count:
        return 0
    return (row_count - 1 - rank) // world_size + 1


def speech_crop_start_sec(
    *,
    sample_key: str,
    audio_duration_sec: float,
    crop_duration_sec: float,
    seed: int | None,
    epoch: int,
    is_eval: bool,
    policy: str = SPEECH_CHUNK_POLICY,
) -> float:
    """Choose one deterministic crop for a JSONL row.

    Training crops long rows uniformly using only ``seed + epoch + stable UID``;
    evaluation always starts at zero.  Thus each row contributes exactly one
    optimizer sample per iterator epoch instead of a full chunk plus an
    equally weighted, possibly millisecond-long tail.
    """

    if policy != SPEECH_CHUNK_POLICY:
        raise ValueError(
            f"unsupported speech_chunk_policy={policy!r}; expected "
            f"{SPEECH_CHUNK_POLICY!r}"
        )
    if audio_duration_sec <= 0 or crop_duration_sec <= 0:
        raise ValueError(
            "audio and crop durations must be positive, got "
            f"{audio_duration_sec} and {crop_duration_sec}"
        )
    max_start = max(0.0, audio_duration_sec - crop_duration_sec)
    if is_eval or max_start == 0.0:
        return 0.0
    if not sample_key:
        raise ValueError("sample_key must be non-empty")
    sample_stream_id = zlib.crc32(sample_key.encode("utf-8"))
    rng = _seeded_rng(seed, epoch, sample_stream_id, 0x43524F50)
    return float(rng.uniform(0.0, max_start))


def stable_speech_sample_key(sample: dict[str, Any]) -> str:
    """Return a crop key stable across absolute dataset-root relocation."""

    uid = sample.get("uid")
    if uid:
        return str(uid)
    audio_path = str(sample.get("path", ""))
    parent_name = Path(audio_path).parent.name
    if not parent_name:
        raise RuntimeError(
            f"speech sample has neither uid nor a usable audio parent: {sample!r}"
        )
    return parent_name


def resolve_v3_image_path(
    sample: dict[str, Any], *, image_root: str | None
) -> Path:
    """Resolve the authoritative v3 co-located image path.

    ``image_root`` is retained only for API compatibility.  v3 has no stable
    mapping from its content-addressed audio UID to that legacy directory, so
    silently falling back to it could condition a dialogue on the wrong image.
    The required rule is always ``<wav parent>/image.jpg``; an explicit JSONL
    ``image`` field, when present, must agree with it.
    """

    del image_root
    audio_value = sample.get("path")
    if not audio_value:
        raise RuntimeError(f"speech sample has no audio path: {sample!r}")
    expected = Path(str(audio_value)).parent / "image.jpg"
    declared = sample.get("image")
    if declared is not None and Path(str(declared)).resolve() != expected.resolve():
        raise RuntimeError(
            "v3 image path disagrees with the co-located image rule "
            f"(sample={sample.get('uid', expected.parent.name)!r}, "
            f"audio_path={audio_value}, declared_image={declared}, "
            f"expected_image={expected})"
        )
    return expected


def require_image_condition(
    sample: dict[str, Any],
    *,
    image_processor: Any,
    device: torch.device,
    image_root: str | None,
) -> ConditionAttributes:
    """Build mandatory visual conditioning or fail with sample provenance."""

    audio_path = str(sample.get("path", "<missing>"))
    sample_id = str(sample.get("uid", Path(audio_path).parent.name))
    image_path = resolve_v3_image_path(sample, image_root=image_root)
    if not image_path.is_file():
        raise RuntimeError(
            "required image is missing "
            f"(sample={sample_id}, audio_path={audio_path}, image_path={image_path})"
        )
    try:
        image = image_processor(image_path)
        if image is None:
            raise ValueError("ImageProcessor returned None")
        image_tensor = image.unsqueeze(0).to(device)
    except Exception as error:
        raise RuntimeError(
            "failed to decode/process required image "
            f"(sample={sample_id}, audio_path={audio_path}, image_path={image_path}): "
            f"{type(error).__name__}: {error}"
        ) from error

    condition = ConditionAttributes()
    condition.tensor["image"] = TensorCondition(
        tensor=image_tensor,
        mask=torch.ones(1, 1, dtype=torch.bool, device=device),
    )
    return condition


@dataclass(frozen=True)
class DecodedSpeechRow:
    sample: dict[str, Any]
    waveform: np.ndarray
    start_sec: float


def _decode_speech_row(
    line: str,
    *,
    crop_duration_sec: float,
    sample_rate: int,
    seed: int | None,
    epoch: int,
    is_eval: bool,
    policy: str,
) -> DecodedSpeechRow:
    sample = json.loads(line)
    audio_path = str(sample.get("path", ""))
    sample_id = str(sample.get("uid", Path(audio_path).parent.name))
    try:
        duration = float(sample["duration"])
        start_sec = speech_crop_start_sec(
            sample_key=stable_speech_sample_key(sample),
            audio_duration_sec=duration,
            crop_duration_sec=crop_duration_sec,
            seed=seed,
            epoch=epoch,
            is_eval=is_eval,
            policy=policy,
        )
        waveform, decoded_rate = sphn.read(
            audio_path,
            start_sec=start_sec,
            duration_sec=crop_duration_sec,
            sample_rate=sample_rate,
        )
        if decoded_rate != sample_rate:
            raise ValueError(
                f"decoder returned sample_rate={decoded_rate}, expected {sample_rate}"
            )
        if waveform.ndim != 2 or waveform.shape[0] != 2 or waveform.shape[-1] == 0:
            raise ValueError(
                "expected non-empty stereo waveform [2, samples], got "
                f"{waveform.shape}"
            )
    except Exception as error:
        raise RuntimeError(
            "failed to decode speech sample "
            f"(sample={sample_id}, audio_path={audio_path}, "
            f"start_sec={locals().get('start_sec', None)}): "
            f"{type(error).__name__}: {error}"
        ) from error
    return DecodedSpeechRow(sample=sample, waveform=waveform, start_sec=start_sec)


def iter_decoded_speech_rows(
    lines: Iterator[str],
    *,
    crop_duration_sec: float,
    sample_rate: int,
    seed: int | None,
    epoch: int,
    is_eval: bool,
    policy: str,
    num_workers: int,
) -> Iterator[DecodedSpeechRow]:
    """Decode a bounded number of rows concurrently while preserving order."""

    if num_workers < 1:
        raise ValueError(f"speech_decode_workers must be >= 1, got {num_workers}")
    line_iterator = iter(lines)
    with ThreadPoolExecutor(
        max_workers=num_workers, thread_name_prefix="speech-decode"
    ) as executor:
        pending: deque[Future[DecodedSpeechRow]] = deque()

        def submit_one() -> bool:
            try:
                line = next(line_iterator)
            except StopIteration:
                return False
            pending.append(
                executor.submit(
                    _decode_speech_row,
                    line,
                    crop_duration_sec=crop_duration_sec,
                    sample_rate=sample_rate,
                    seed=seed,
                    epoch=epoch,
                    is_eval=is_eval,
                    policy=policy,
                )
            )
            return True

        for _ in range(num_workers):
            if not submit_one():
                break
        while pending:
            yield pending.popleft().result()
            submit_one()


def pad_or_truncate_text_ids(
    ids: list[int], target_len: int | None, *, padding_id: int = -1
) -> list[int]:
    """Make a speechless sequence fixed-length without teaching artificial PAD.

    ``padding_id`` defaults to MoshiVis' ``zero_token_id`` (-1), which is
    accepted by ``ClampedEmbedding`` and excluded by ``text_logits_mask``.
    Importantly, existing token ID 0 values in ``ids`` are left untouched.
    """

    if target_len is None:
        return list(ids)
    if target_len < 0:
        raise ValueError(f"target_len must be non-negative, got {target_len}")
    if len(ids) >= target_len:
        return list(ids[:target_len])
    return list(ids) + [padding_id] * (target_len - len(ids))


def _speech_fraction(speech_ratio: float) -> Fraction:
    if not 0.0 <= speech_ratio <= 1.0:
        raise ValueError(f"speech_ratio must be in [0, 1], got {speech_ratio}")
    fraction = Fraction(str(speech_ratio)).limit_denominator(1000)
    if abs(float(fraction) - speech_ratio) > 1e-9:
        raise ValueError(
            f"speech_ratio={speech_ratio} cannot be represented exactly with "
            "a denominator <= 1000"
        )
    return fraction


def _local_modality_window_spec(
    *,
    speech_ratio: float,
    world_size: int,
    local_batch_size: int,
    effective_window_size: int | None,
) -> tuple[int, int]:
    """Return ``(local_window_size, local_speech_samples)`` or fail exactly."""

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if local_batch_size < 1:
        raise ValueError(
            f"local_batch_size must be >= 1, got {local_batch_size}"
        )
    ratio = _speech_fraction(speech_ratio)
    if effective_window_size is not None:
        if effective_window_size < 1:
            raise ValueError(
                "effective_window_size must be >= 1, "
                f"got {effective_window_size}"
            )
        if effective_window_size % world_size != 0:
            raise ValueError(
                f"effective_window_size={effective_window_size} must be divisible "
                f"by world_size={world_size}"
            )
        local_window_size = effective_window_size // world_size
    else:
        # Backward-compatible standalone use: choose a minimal local window
        # that represents the ratio and contains whole local microbatches.
        local_window_size = ratio.denominator * local_batch_size
    if local_window_size % local_batch_size != 0:
        raise ValueError(
            f"local rank window size {local_window_size} must be divisible by "
            f"local_batch_size={local_batch_size}"
        )
    speech_numerator = ratio.numerator * local_window_size
    if speech_numerator % ratio.denominator != 0:
        raise ValueError(
            f"local rank window size {local_window_size} cannot contain "
            f"speech_ratio={speech_ratio} exactly"
        )
    return local_window_size, speech_numerator // ratio.denominator


def iter_stratified_modalities(
    *,
    speech_ratio: float,
    seed: int | None,
    rank: int,
    world_size: int,
    local_batch_size: int = 1,
    effective_window_size: int | None = None,
) -> Iterator[bool]:
    """Yield a reproducible speech/speechless schedule for one DDP rank.

    Every rank gets the exact requested ratio within each optimizer window,
    not merely across the union of ranks.  A separately seeded permutation is
    generated for every ``(rank, optimizer_window)`` pair.  This keeps speech
    and speechless iterator cursors synchronized across ranks while still
    varying their within-window ordering.
    """

    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    local_window_size, local_speech_samples = _local_modality_window_spec(
        speech_ratio=speech_ratio,
        world_size=world_size,
        local_batch_size=local_batch_size,
        effective_window_size=effective_window_size,
    )

    window_index = 0
    while True:
        pattern = np.zeros(local_window_size, dtype=np.bool_)
        pattern[:local_speech_samples] = True
        _seeded_rng(seed, rank, window_index, 0x4D4F4441).shuffle(pattern)
        yield from (bool(value) for value in pattern)
        window_index += 1


def iter_stratified_source_modalities(
    *,
    source_probabilities: list[float],
    speech_ratio: float,
    seed: int | None,
    rank: int,
    world_size: int,
    local_batch_size: int = 1,
    effective_window_size: int | None = None,
    start_sample_index: int = 0,
) -> Iterator[tuple[int, bool]]:
    """Yield deterministic ``(source_id, is_speech)`` choices for one rank.

    ``start_sample_index`` replays only the cheap CPU schedule and suppresses
    its prefix.  The corresponding per-source iterator cursors are advanced
    separately using rank-local JSONL offsets, so resume never re-encodes the
    skipped audio or images.
    """

    if not source_probabilities:
        raise ValueError("at least one data-source probability is required")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if start_sample_index < 0:
        raise ValueError(
            f"start_sample_index must be non-negative, got {start_sample_index}"
        )
    local_window_size, local_speech_samples = _local_modality_window_spec(
        speech_ratio=speech_ratio,
        world_size=world_size,
        local_batch_size=local_batch_size,
        effective_window_size=effective_window_size,
    )

    # This stream intentionally excludes rank: local sample N selects the same
    # source on all ranks, avoiding rank-dependent source cursor drift.
    source_rng = _seeded_rng(seed, 0x534F5552)
    sample_index = 0
    window_index = 0
    while True:
        source_ids = source_rng.choice(
            len(source_probabilities),
            size=local_window_size,
            p=source_probabilities,
        )
        common_pattern = np.zeros(local_window_size, dtype=np.bool_)
        common_pattern[:local_speech_samples] = True
        _seeded_rng(seed, window_index, 0x4D4F4442).shuffle(common_pattern)

        rank_pattern = common_pattern.copy()
        for source_id in range(len(source_probabilities)):
            positions = np.flatnonzero(source_ids == source_id)
            assignments = rank_pattern[positions].copy()
            _seeded_rng(
                seed, rank, window_index, source_id, 0x4D4F4443
            ).shuffle(assignments)
            rank_pattern[positions] = assignments

        for source_id_raw, use_speech_raw in zip(source_ids, rank_pattern):
            if sample_index >= start_sample_index:
                yield int(source_id_raw), bool(use_speech_raw)
            sample_index += 1
        window_index += 1


def stratified_prefix_cursor_counts(
    *,
    source_probabilities: list[float],
    speech_ratio: float,
    seed: int | None,
    rank: int,
    world_size: int,
    local_batch_size: int = 1,
    effective_window_size: int | None = None,
    sample_count: int,
) -> tuple[list[int], list[int]]:
    """Count consumed speech/text samples per source before a resume cursor."""

    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}")
    speech_counts = [0] * len(source_probabilities)
    speechless_counts = [0] * len(source_probabilities)
    choices = iter_stratified_source_modalities(
        source_probabilities=source_probabilities,
        speech_ratio=speech_ratio,
        seed=seed,
        rank=rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
        effective_window_size=effective_window_size,
    )
    for source_id, use_speech in itertools.islice(choices, sample_count):
        counts = speech_counts if use_speech else speechless_counts
        counts[source_id] += 1
    return speech_counts, speechless_counts


def stratified_interleave_iterators(
    speech_iterators: list[Iterator],
    speechless_iterators: list[Iterator],
    *,
    source_probabilities: list[float],
    speech_ratio: float,
    seed: int | None,
    rank: int,
    world_size: int,
    local_batch_size: int = 1,
    effective_window_size: int | None = None,
    start_sample_index: int = 0,
) -> Iterator:
    """Interleave modalities exactly while sampling data sources by weight.

    Source choice at a given local position is identical on every rank.  The
    rank-specific modality permutation is shuffled only within positions for
    the same source, preserving identical per-source speech and speechless
    cursor counts at every optimizer-window boundary.
    """

    if len(speech_iterators) != len(speechless_iterators):
        raise ValueError("speech and speechless iterator counts must match")
    if len(speech_iterators) != len(source_probabilities):
        raise ValueError("iterator and source probability counts must match")
    if not speech_iterators:
        raise ValueError("at least one data-source iterator is required")
    choices = iter_stratified_source_modalities(
        source_probabilities=source_probabilities,
        speech_ratio=speech_ratio,
        seed=seed,
        rank=rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
        effective_window_size=effective_window_size,
        start_sample_index=start_sample_index,
    )
    for source_id, use_speech in choices:
        iterator = (
            speech_iterators[source_id]
            if use_speech
            else speechless_iterators[source_id]
        )
        yield next(iterator)


def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    image_embedder: ImageProjection,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    image_root: str | None,
    device: torch.device,
    speech_chunk_policy: str = SPEECH_CHUNK_POLICY,
    speech_decode_workers: int = 4,
    initial_skip_samples: int = 0,
) -> Iterator[Sample]:
    image_processor = ImageProcessor.from_config(
        image_embedder.kyuteye_config, device=device
    )
    if speech_chunk_policy != SPEECH_CHUNK_POLICY:
        raise ValueError(
            f"unsupported speech_chunk_policy={speech_chunk_policy!r}"
        )
    if speech_decode_workers < 1:
        raise ValueError(
            f"speech_decode_workers must be >= 1, got {speech_decode_workers}"
        )
    if initial_skip_samples < 0:
        raise ValueError(
            f"initial_skip_samples must be non-negative, got {initial_skip_samples}"
        )
    remaining_skip = initial_skip_samples
    epoch = 0
    while True:
        jsonl_files = list(source.jsonl_files)
        if shuffle_at_epoch:
            _seeded_rng(seed, epoch, 0x46494C45).shuffle(jsonl_files)
        for jsonl_file in jsonl_files:
            file_row_count = _sharded_jsonl_row_count(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                drop_remainder=not is_finite,
            )
            file_start_position = min(remaining_skip, file_row_count)
            remaining_skip -= file_start_position
            if file_start_position == file_row_count:
                continue
            lines = iter_sharded_jsonl_lines(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                shuffle=shuffle_at_epoch,
                seed=seed,
                epoch=epoch,
                drop_remainder=not is_finite,
                start_position=file_start_position,
            )
            decoded_rows = iter_decoded_speech_rows(
                lines,
                crop_duration_sec=instruct_tokenizer.duration_sec,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                seed=seed,
                epoch=epoch,
                is_eval=is_finite,
                policy=speech_chunk_policy,
                num_workers=speech_decode_workers,
            )
            for decoded in decoded_rows:
                # Visual conditioning is mandatory.  Validate it before the
                # comparatively expensive GPU audio tokenization so corrupt
                # rows cannot silently become image-free training examples.
                condition = require_image_condition(
                    decoded.sample,
                    image_processor=image_processor,
                    device=device,
                    image_root=image_root,
                )
                sample_out = instruct_tokenizer(
                    decoded.waveform,
                    decoded.start_sec,
                    str(decoded.sample["path"]),
                )
                sample_out.condition_attributes = condition
                yield sample_out

        if is_finite:
            if remaining_skip:
                raise ValueError(
                    f"initial_skip_samples={initial_skip_samples} exceeds finite "
                    f"rank-local dataset size by {remaining_skip}"
                )
            break
        print(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


def get_speechless_dataset_iterator(
    source: DataDir | DataFile,
    text_tokenizer,              # LLM用 tokenizer
    image_embedder: ImageProjection,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    image_root: str | None,
    device: torch.device,
    num_audio_streams: int = 8,   # = dep_q (Mimi の n_q と揃える)
    audio_pad_id: int = 0,        # audio padding 用 ID（trainer 側でマスクする前提）
    target_len: int | None = None,
    text_loss_padding_id: int = -1,
    initial_skip_samples: int = 0,
) -> Iterator[Sample]:
    image_processor = ImageProcessor.from_config(
        image_embedder.kyuteye_config, device=device
    )

    def _encode_text(text_tokenizer, text: str):
        # SentencePiece の場合
        if isinstance(text_tokenizer, spm.SentencePieceProcessor):
            return text_tokenizer.encode_as_ids(text)
        # HF tokenizer の場合を想定
        return text_tokenizer.encode(text, add_special_tokens=False)

    if initial_skip_samples < 0:
        raise ValueError(
            f"initial_skip_samples must be non-negative, got {initial_skip_samples}"
        )
    remaining_skip = initial_skip_samples
    epoch = 0

    while True:
        jsonl_files = list(source.jsonl_files)
        if shuffle_at_epoch:
            # All ranks use the same file permutation. Row-level sharding is
            # applied after a shared global row permutation below.
            _seeded_rng(seed, epoch, 0x46494C45).shuffle(jsonl_files)

        for jsonl_file in jsonl_files:
            file_row_count = _sharded_jsonl_row_count(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                drop_remainder=not is_finite,
            )
            file_start_position = min(remaining_skip, file_row_count)
            remaining_skip -= file_start_position
            if file_start_position == file_row_count:
                continue
            lines = iter_sharded_jsonl_lines(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                shuffle=shuffle_at_epoch,
                seed=seed,
                epoch=epoch,
                drop_remainder=not is_finite,
                start_position=file_start_position,
            )
            for line in lines:
                data = json.loads(line)
                text = data["text"]

                # Image conditioning is not optional for this visual dataset.
                # Resolve and decode it before constructing the training sample
                # so no rank can silently continue with an image-free example.
                condition = require_image_condition(
                    data,
                    image_processor=image_processor,
                    device=device,
                    image_root=image_root,
                )

                # 1. テキストをトークン化。固定長の人工末尾には
                # zero_token_id (-1) を使い、text loss から除外する。
                # tokenizer が実際に出力した ID 0 はそのまま保持する。
                ids = pad_or_truncate_text_ids(
                    _encode_text(text_tokenizer, text),
                    target_len,
                    padding_id=text_loss_padding_id,
                )
                text_len = len(ids)

                # text_codes: [1, 1, T]
                text_codes = torch.tensor(
                    ids, dtype=torch.long, device=device
                ).unsqueeze(0).unsqueeze(0)

                # audio_codes: [1, dep_q, T] を padding ID で埋める
                audio_codes = torch.full(
                    (1, num_audio_streams, text_len),
                    fill_value=audio_pad_id,
                    dtype=torch.long,
                    device=device,
                )

                # codes: [1, 1 + dep_q, T]
                codes = torch.cat([text_codes, audio_codes], dim=1)

                sample_out = Sample(codes=codes)
                sample_out.condition_attributes = condition

                yield sample_out

        if is_finite:
            if remaining_skip:
                raise ValueError(
                    f"initial_skip_samples={initial_skip_samples} exceeds finite "
                    f"rank-local dataset size by {remaining_skip}"
                )
            break
        print(f"[speechless] Rank {rank} finished epoch {epoch + 1}")
        epoch += 1


def build_mixed_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    text_tokenizer,                # SentencePiece など
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    device: torch.device,
    shuffle_pretrain: bool = True,
    image_embedder: ImageProjection | None = None,
    image_root: str | None = None,
    speech_ratio: float = 0.5,     # speech : speechless の比率
    target_len: int | None = None,
    local_batch_size: int = 1,
    effective_window_size: int | None = None,
    text_loss_padding_id: int = -1,
    speech_chunk_policy: str = SPEECH_CHUNK_POLICY,
    speech_decode_workers: int = 4,
    start_sample_index: int = 0,
) -> Iterator[Sample]:
    """
    音声あり(speech) + テキストのみ(speechless) を
    同じ pretrain_data から混合して返す Iterator を構築する。
    - pretrain_data: "path1:weight1,path2:weight2,..." 形式
    - speech_ratio: 各データソースに対して speech : speechless の比率
    """
    sources, base_probs = parse_data_sources(pretrain_data=pretrain_data)
    shuffle = not is_eval and shuffle_pretrain
    if start_sample_index < 0:
        raise ValueError(
            f"start_sample_index must be non-negative, got {start_sample_index}"
        )
    if is_eval and start_sample_index:
        raise ValueError("evaluation does not support a non-zero resume cursor")

    speech_skip_counts, speechless_skip_counts = stratified_prefix_cursor_counts(
        source_probabilities=base_probs,
        speech_ratio=speech_ratio,
        seed=seed,
        rank=rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
        effective_window_size=effective_window_size,
        sample_count=start_sample_index,
    )

    speech_iters: list[Iterator[Sample]] = []
    speechless_iters: list[Iterator[Sample]] = []

    # dep_q / padding ID は Mimi 側に揃える
    num_audio_streams = 16
    audio_pad_id = -1

    for source_id, src in enumerate(sources):
        # --- speech 用 iterator ---
        speech_it = get_dataset_iterator(
            source=src,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
            image_root=image_root,
            image_embedder=image_embedder,
            device=device,
            speech_chunk_policy=speech_chunk_policy,
            speech_decode_workers=speech_decode_workers,
            initial_skip_samples=speech_skip_counts[source_id],
        )
        speech_iters.append(speech_it)

        # --- speechless 用 iterator ---
        speechless_it = get_speechless_dataset_iterator(
            source=src,
            text_tokenizer=text_tokenizer,
            image_embedder=image_embedder,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
            image_root=image_root,
            device=device,
            num_audio_streams=num_audio_streams,
            audio_pad_id=audio_pad_id,
            target_len=target_len,
            text_loss_padding_id=text_loss_padding_id,
            initial_skip_samples=speechless_skip_counts[source_id],
        )
        speechless_iters.append(speechless_it)

    if is_eval:
        # Preserve the existing evaluation behavior: enumerate both views.
        paired_iters = itertools.chain.from_iterable(
            zip(speech_iters, speechless_iters)
        )
        combined_iterator = itertools.chain.from_iterable(paired_iters)
    else:
        combined_iterator = stratified_interleave_iterators(
            speech_iters,
            speechless_iters,
            source_probabilities=base_probs,
            speech_ratio=speech_ratio,
            seed=seed,
            rank=rank,
            world_size=world_size,
            local_batch_size=local_batch_size,
            effective_window_size=effective_window_size,
            start_sample_index=start_sample_index,
        )

    return combined_iterator


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
