import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with  "path" and "duration" fields for each audio .wav file.
    """

    train_data: str = ""
    shuffle: bool = False
    eval_data: str = ""
    asset_manifest: str | None = None
    asset_manifest_verification: str = ""
    image_root: str = ""
    speech_ratio: float = 0.25
    speech_chunk_policy: str = "one_random_crop_per_row"
    speech_decode_workers: int = 4


@dataclass()
class TrainerArgs(Serializable):
    text_padding_weight: float = 0.1
    end_of_text_padding_weight: float | None = None
    first_codebook_weight_multiplier: float = 3.0
