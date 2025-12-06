from typing import Any, Iterator

import torch
from .args import DataArgs
from .datasets import build_dataset, build_speechless_dataset, build_mixed_dataset
from .interleaver import Batch
from jmoshivis.models.image_projection import ImageProjection


def build_data_loader(
    instruct_tokenizer: Any,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    image_root: str | None = None,
    image_embedder: ImageProjection | None = None,
    device: torch.device = torch.device("cpu"),
    mode: str = "speech",          # "speech" | "speechless" | "mixed"
    text_tokenizer: Any | None = None,
    target_len: int = 125,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."

    if mode == "speech":
        pretrain_data = args.train_data if not is_eval else args.eval_data
        dataset = build_dataset(
            pretrain_data=pretrain_data,
            instruct_tokenizer=instruct_tokenizer,
            seed=seed,
            rank=rank,
            world_size=world_size,
            is_eval=is_eval,
            shuffle_pretrain=args.shuffle,
            image_root=image_root,
            image_embedder=image_embedder,
            device=device,
        )

    elif mode == "speechless":
        assert text_tokenizer is not None
        pretrain_data = args.train_data   # ← config に追加想定
        dataset = build_speechless_dataset(
            pretrain_data=pretrain_data,
            text_tokenizer=text_tokenizer,
            seed=seed,
            rank=rank,
            world_size=world_size,
            is_eval=is_eval,
            shuffle_pretrain=args.shuffle,
            image_root=image_root,
            image_embedder=image_embedder,
            device=device,
            num_audio_streams=16,
            audio_pad_id=-1,
            target_len=target_len,
        )

    elif mode == "mixed":
        assert text_tokenizer is not None
        pretrain_data = args.train_data
        dataset = build_mixed_dataset(
            pretrain_data=pretrain_data,          # 例: config 側で別フィールドに切る
            instruct_tokenizer=instruct_tokenizer,
            text_tokenizer=text_tokenizer,
            seed=seed,
            rank=rank,
            world_size=world_size,
            is_eval=is_eval,
            device=device,
            shuffle_pretrain=args.shuffle,
            image_root=image_root,
            image_embedder=image_embedder,
            speech_ratio=args.speech_ratio,             # 例: 0.5 とか
            target_len=target_len,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Batch 化は今のままでOK ---
    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []
