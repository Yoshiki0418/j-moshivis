#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Karpathy-style COCO JSON (with Japanese captions in `raw`) into JSONL.

Input schema (example):
{
  "images": [
    {
      "filepath": "val2014",
      "filename": "COCO_val2014_000000391895.jpg",
      "imgid": 0,
      "split": "test",  # train / val / test / restval
      "sentences": [
        {"raw": "赤いヘルメットをかぶった人がバイクに跨っている", "imgid": 0, "sentid": 770337},
        ...
      ],
      "cocoid": 391895
    },
    ...
  ]
}

Output (one record per caption, JSONL per split):
{"id":"0_770337","image_path":"/abs/or/rel/path/val2014/COCO_val2014_000000391895.jpg","text":"赤いヘルメットをかぶった人がバイクに跨っている","imgid":0,"sentid":770337,"cocoid":391895}

Usage:
  python scripts/coco_ja_to_jsonl.py \
      --input dataset_coco_ja.json \
      --images-root /path/to/coco \
      --out-dir data/jmvis/coco-ja-jsonl \
      --max-captions-per-image 5 \
      --dedup-text --require-exists --restval-to-train

Then load with:
  from datasets import load_dataset
  ds_train = load_dataset("json", data_files="data/jmvis/coco-ja-jsonl/train.jsonl")["train"]
"""

import argparse
import json
import os
import random
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to dataset_coco_ja.json")
    p.add_argument("--images-root", required=True,
                   help="Root directory that contains {train2014,val2014,...} or the dirs named in `filepath`")
    p.add_argument("--out-dir", required=True, help="Directory to write {train,val,test}.jsonl")
    p.add_argument("--max-captions-per-image", type=int, default=None,
                   help="If set: keep at most K captions per image. If None: keep all.")
    p.add_argument("--sample-mode", choices=["first", "random"], default="first",
                   help="When limiting captions per image: pick the 'first' or 'random'")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--dedup-text", action="store_true", help="Deduplicate identical text per image")
    p.add_argument("--dedup-global", action="store_true", help="Deduplicate identical text globally (across images)")
    p.add_argument("--require-exists", action="store_true", help="Skip records whose image file does not exist")
    p.add_argument("--normalize", action="store_true",
                   help="Apply Unicode NFKC normalization and whitespace cleanup")
    p.add_argument("--prefix", type=str, default=None,
                   help="Optional string to prepend to each caption (e.g. '<image> 説明: ')")
    p.add_argument("--template", type=str, default=None,
                   help="Optional template with {caption}; e.g. '<image> この画像を説明: {caption}'")
    p.add_argument("--restval-to-train", action="store_true", default=True,
                   help="Map 'restval' split into 'train' (enabled by default)")
    p.add_argument("--write-combined", action="store_true",
                   help="Also write combined.jsonl that concatenates all splits")
    return p.parse_args()

def clean_text(s: str, do_norm: bool) -> str:
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    # collapse spaces (including zenkaku spaces)
    s = s.replace("\u3000", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    if do_norm:
        s = unicodedata.normalize("NFKC", s)
    return s

def format_text(caption: str, prefix: Optional[str], template: Optional[str]) -> str:
    if template:
        return template.replace("{caption}", caption)
    if prefix:
        return prefix + caption
    return caption

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    args = parse_args()
    random.seed(args.seed)

    ensure_dir(args.out_dir)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    images: List[Dict] = data.get("images", [])
    if not images:
        print("No 'images' in input JSON.", file=sys.stderr)
        sys.exit(1)

    # Collect per-split buckets
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    stats = {
        "images_total": 0,
        "records_written": 0,
        "skipped_missing_file": 0,
        "skipped_empty_text": 0,
        "skipped_dedup_local": 0,
        "skipped_dedup_global": 0,
    }
    seen_global = set()  # for --dedup-global

    for im in images:
        stats["images_total"] += 1
        split = (im.get("split") or "train").lower()
        if split == "restval" and args.restval_to_train:
            split = "train"

        filepath = im.get("filepath", "")
        filename = im.get("filename", "")
        rel_path = os.path.join(filepath, filename) if filepath else filename
        image_path = os.path.join(args.images_root, rel_path)

        # Optionally skip if file doesn't exist
        if args.require_exists and not os.path.exists(image_path):
            # We still want to count per caption for clarity, but we cannot read captions until we check sentences.
            # Count each sentence as skipped to be conservative.
            stats["skipped_missing_file"] += len(im.get("sentences", []))
            continue

        # Gather captions
        sentences = im.get("sentences", [])
        # Local dedup set per image
        seen_local = set()

        # Build list of candidate captions (cleaned, filtered)
        caps: List[Dict] = []
        for s in sentences:
            raw = s.get("raw")  # Japanese text per your dataset
            text = clean_text(raw, do_norm=args.normalize)
            if not text:
                stats["skipped_empty_text"] += 1
                continue
            if args.dedup_text:
                if text in seen_local:
                    stats["skipped_dedup_local"] += 1
                    continue
                seen_local.add(text)
            if args.dedup_global:
                if text in seen_global:
                    stats["skipped_dedup_global"] += 1
                    continue
                seen_global.add(text)

            caps.append({
                "text": text,
                "sentid": s.get("sentid"),
                "imgid": s.get("imgid", im.get("imgid")),
            })

        # Apply per-image cap sampling/limiting
        if args.max_captions_per_image is not None and len(caps) > args.max_captions_per_image:
            if args.sample_mode == "random":
                caps = random.sample(caps, args.max_captions_per_image)
            else:  # 'first'
                caps = caps[:args.max_captions_per_image]

        # Add to bucket records
        for c in caps:
            caption = format_text(c["text"], args.prefix, args.template)
            rec = {
                "id": f"{im.get('imgid')}_{c.get('sentid')}",
                "image_path": image_path,
                "text": caption,
                "imgid": im.get("imgid"),
                "sentid": c.get("sentid"),
                "cocoid": im.get("cocoid"),
                "split": split,
                # Keep relative too (optional, could help if you move roots later)
                "rel_image_path": rel_path,
            }
            buckets[split].append(rec)
            stats["records_written"] += 1

    # Write per-split JSONL
    written_files = []
    for split, recs in buckets.items():
        if not recs:
            continue
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as w:
            for r in recs:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        written_files.append(out_path)
        print(f"[OK] Wrote {len(recs):,} records -> {out_path}")

    if args.write_combined and written_files:
        combined = os.path.join(args.out_dir, "combined.jsonl")
        with open(combined, "w", encoding="utf-8") as w:
            for split in ("train", "val", "test"):
                for rec in buckets.get(split, []):
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote combined -> {combined}")

    # Print stats
    print("\n==== Summary ====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("splits:", {k: len(v) for k, v in buckets.items()})

if __name__ == "__main__":
    main()
