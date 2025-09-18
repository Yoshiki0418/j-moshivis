import yaml
from pathlib import Path


def load_config(path: Path | str) -> dict:
    p = Path(path)
    with open(p, "r") as f:
        return yaml.safe_load(f)
