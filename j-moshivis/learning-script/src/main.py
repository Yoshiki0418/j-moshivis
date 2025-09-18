import sys
from pathlib import Path

# Ensure repo root is on sys.path so `jmoshivis` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Ensure project root is on sys.path so `src` package imports work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch
import yaml

from src.config.config import load_config
from src.modules.dataset import CocoLikeDataset, get_dataloader
from src.modules.trainer import Trainer
from jmoshivis.models.loaders import get_moshi_vis
from jmoshivis.config.kyuteye_config import KyuteyeConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/learning_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)

    # Load YAML as dict so we can sanitize values before creating KyuteyeConfig
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Ensure delays length is not too long (or missing)
    moshi_cfg = cfg_dict.get('moshi', {})
    n_q = moshi_cfg.get('n_q', 8)
    delays = moshi_cfg.get('delays', None)
    if delays is None or len(delays) > (n_q + 1):
        moshi_cfg['delays'] = tuple([0] * (n_q + 1))
        cfg_dict['moshi'] = moshi_cfg

    # Construct KyuteyeConfig from dict
    cfg = KyuteyeConfig(**{k: v for sub in cfg_dict.values() for k, v in (sub.items() if isinstance(sub, dict) else ())})

    # Get model and image projection
    try:
        moshi_vis, image_proj = get_moshi_vis(cfg, moshi_weight=None, device=args.device)
    except AssertionError:
        # Fallback: loaders.get_moshi_vis passes an empty dict as moshi_weight which
        # triggers MoshiVisGen.from_config to try loading an empty state dict.
        # Create MoshiVisGen and ImageProjection directly without loading weights.
        from jmoshivis.models.moshivis import MoshiVisGen
        from jmoshivis.models.image_projection import ImageProjection

        moshi_vis = MoshiVisGen.from_config(cfg, moshi_weight=None, device=args.device)
        image_proj = ImageProjection.from_config(cfg, moshi_vis.model_dim, None, args.device)

    # Minimal dataset/dataloader for smoke test
    # Dataset path: prefer repository-level data/COCO, fallback to provided relative path
    repo_root = Path(__file__).resolve().parents[3]
    coco_path = repo_root / 'data' / 'COCO' / 'val2014'
    if not coco_path.exists():
        # fallback to workspace-level data
        coco_path = Path('/workspace') / 'data' / 'COCO' / 'val2014'

    dataset = CocoLikeDataset(root_dir=coco_path, image_size=cfg.image.image_size)
    dataloader = get_dataloader(dataset, batch_size=2)

    trainer = Trainer(moshi_vis, image_proj, dataloader, device=args.device)
    trainer.smoke_run(steps=2)


if __name__ == "__main__":
    main()
