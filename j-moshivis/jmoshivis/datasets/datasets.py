import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
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
        jsonl_files = list(self.path.rglob("*jsonl"))
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


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng


image_processor = ImageProcessor()


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
) -> Iterator[Sample]:
    epoch = 1
    while True:
        for jsonl_file in source.jsonl_files:
            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                num_threads=4,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                pad_last_segment=True,
            )
            if shuffle_at_epoch:
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed += 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)
            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]

                # === 1. 音声トークン化 ===
                sample_out = instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])

                # === 2. 画像条件付け ===
                wav_path = Path(sample["path"])
                uid_dir = wav_path.parent
                image_path = uid_dir / "image.jpg"
                cond_attr = None
                if (
                    image_embedder is not None
                    and image_path is not None
                    and Path(image_path).exists()
                ):
                    try:
                        image_tensor = image_processor(image_path).unsqueeze(0)
                        with torch.no_grad():
                            outputs = image_embedder(image_tensor)          # ← dict が返る
                            image_embed = outputs["cross_attention_src"]  # [1, N_tokens, D]
                        mask = torch.ones(
                            1, image_embed.shape[1], dtype=torch.bool, device=device
                        )

                        cond_attr = ConditionAttributes()
                        cond_attr.tensor["image"] = TensorCondition(
                            tensor=image_embed, mask=mask
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to embed image: {image_path}, {e}")
                        cond_attr = None

                # === 3. Sampleとして返す ===
                if cond_attr is not None:
                    sample_out.condition_attributes = cond_attr

                yield sample_out

        if is_finite:
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
) -> Iterator[Sample]:
    def _encode_text(text_tokenizer, text: str):
        # SentencePiece の場合
        if isinstance(text_tokenizer, spm.SentencePieceProcessor):
            return text_tokenizer.encode_as_ids(text)
        # HF tokenizer の場合を想定
        return text_tokenizer.encode(text, add_special_tokens=False)

    epoch = 1
    rng = get_rng(seed, rank) if seed is not None else None

    while True:
        jsonl_files = source.jsonl_files
        if shuffle_at_epoch and rng is not None:
            rng.shuffle(jsonl_files)

        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for i, line in enumerate(f):
                    if i % world_size != rank:
                        continue

                    data = json.loads(line)
                    text = data["text"]
                    wav_path = Path(data["path"])
                    uid_dir = wav_path.parent
                    image_path = uid_dir / "image.jpg"

                    # 1. テキストをトークン化
                    ids = _encode_text(text_tokenizer, text)

                    # --- 🔧 ここで長さを揃える ---
                    if target_len is not None:
                        if len(ids) > target_len:
                            ids = ids[:target_len]
                        elif len(ids) < target_len:
                            # SentencePiece の pad_id を使う（なければ 0 ）
                            pad_id = (
                                text_tokenizer.pad_id()
                                if hasattr(text_tokenizer, "pad_id")
                                and text_tokenizer.pad_id() >= 0
                                else 0
                            )
                            ids = ids + [pad_id] * (target_len - len(ids))

                    text_len = len(ids) if target_len is None else target_len

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

                    # 2. 画像条件付け
                    cond_attr = None
                    if image_embedder is not None and image_path.exists():
                        image_tensor = image_processor(image_path).unsqueeze(0)
                        with torch.no_grad():
                            outputs = image_embedder(image_tensor)
                            image_embed = outputs["cross_attention_src"]
                        mask = torch.ones(
                            1, image_embed.shape[1], dtype=torch.bool, device=device
                        )
                        cond_attr = ConditionAttributes()
                        cond_attr.tensor["image"] = TensorCondition(
                            tensor=image_embed, mask=mask
                        )

                    if cond_attr is not None:
                        sample_out.condition_attributes = cond_attr

                    yield sample_out

        if is_finite:
            break
        print(f"[speechless] Rank {rank} finished epoch {epoch}")
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
) -> Iterator[Sample]:
    """
    音声あり(speech) + テキストのみ(speechless) を
    同じ pretrain_data から混合して返す Iterator を構築する。
    - pretrain_data: "path1:weight1,path2:weight2,..." 形式
    - speech_ratio: 各データソースに対して speech : speechless の比率
    """
    sources, base_probs = parse_data_sources(pretrain_data=pretrain_data)
    shuffle = not is_eval and shuffle_pretrain

    iters: list[Iterator[Sample]] = []
    probs: list[float] = []

    # dep_q / padding ID は Mimi 側に揃える
    num_audio_streams = 16
    audio_pad_id = -1

    for src, w in zip(sources, base_probs):
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
        )
        iters.append(speech_it)
        probs.append(w * speech_ratio)

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
        )
        iters.append(speechless_it)
        probs.append(w * (1.0 - speech_ratio))

    # weight を正規化
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()

    if is_eval:
        # 評価時は単純連結でもいいし、同じく interleave でもよい
        combined_iterator = itertools.chain.from_iterable(iters)
    else:
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            iters, probabilities=probs, rng=rng
        )

    return combined_iterator


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
