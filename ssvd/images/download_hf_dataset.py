#!/usr/bin/env python3
"""Download a public Hugging Face dataset snapshot with pinned provenance."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--allow-pattern", action="append", default=[])
    args = parser.parse_args()

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    revision = info.sha
    args.local_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern or None,
        max_workers=args.max_workers,
    )
    atomic_json(
        args.local_dir / "source_info.json",
        {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "requested_revision": args.revision,
            "resolved_revision": revision,
            "downloaded_unix": time.time(),
            "local_dir": str(Path(resolved).resolve()),
        },
    )
    print(f"downloaded {args.repo_id}@{revision} -> {resolved}")


if __name__ == "__main__":
    main()
