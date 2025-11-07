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


@dataclass()
class TrainerArgs(Serializable):
    text_padding_weight: float = 0.1
    first_codebook_weight_multiplier: float = 3.0
