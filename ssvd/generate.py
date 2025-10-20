# pylint: disable=C0413,C0411
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "fire",
#     "numpy<2",
#     "rich",
#     "torch==2.2.0",
#     "tqdm",
#     "transformers==4.47.0",
#     "triton",
# ]
# ///
"""Generate dialogues and store them in a database"""


import json
import logging
import os
import random
import sqlite3
from collections import defaultdict
from hashlib import sha256
from math import ceil
from typing import Dict, Literal, Optional

import datasets
import fire
import rich
import torch
from multiturn_instruct import MTCInstruct
from multiturn_prompting import run_multiturn_pipeline
from transformers import pipeline, Pipeline
from utils import postprocess_synth_annot, preprocess_pixelprose_captions

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_pipeline(
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    device: Optional[str | torch.device] = "cuda",
) -> Pipeline:
    """Initialize the Qwen2 pipeline (multilingual, Japanese-capable)"""
    print(f"Loading pipeline: {model}", flush=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        device=device,
        torch_dtype=torch.bfloat16,  # Qwen2はBF16対応
    )
    print(f"Done Loading {model}.", flush=True)
    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"
    return pipe


def get_captions(
    dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
    split: str = "train",
) -> datasets.Dataset:
    """Return captions and image ids for the given dataset

    The output returns an iterable over dicts containing the dfiels:
      * `uid`
      * `caption`
    """
    if dataset == "docci":
        return (
            datasets.load_dataset("google/docci", split=split)
            .select_columns(["example_id", "description"])
            .rename_column("example_id", "uid")
            .rename_column("description", "caption")
        )
    if dataset == "pixmo":
        ds = datasets.load_dataset("allenai/pixmo-cap", split=split).select_columns(
            ["image_url", "caption"]
        )
        return ds.add_column(
            "uid", [sha256(x.encode()).hexdigest() for x in ds["image_url"]]
        ).remove_columns("image_url")
    if dataset == "pixelprose":
        return (
            datasets.load_dataset("tomg-group-umd/pixelprose", split=split)
            .select_columns(["uid", "vlm_caption"])
            .rename_column("vlm_caption", "caption")
            .map(preprocess_pixelprose_captions, input_columns="caption")
        )
    raise NotImplementedError("Unsupported dataset", dataset)


class Launcher:
    """fire entry point"""

    @staticmethod
    def __get_db_file__(
        out_dir: str = "./synthetic_visual_dialogues",
        dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
    ) -> str:
        return os.path.join(out_dir, f"{dataset}_ssvd.db")

    @staticmethod
    def __get_table_name__(
        task: str,
        split: str = "train",
    ) -> str:
        return f"{split}_{task}"

    @staticmethod
    def __get_annot_file__(
        task: str,
        out_dir: str = "./synthetic_visual_dialogues",
        dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
        split: str = "train",
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> str:
        return os.path.join(
            out_dir,
            f"{task}_{dataset}_{split}_{start_idx:05d}_{end_idx:05d}_ssvd_temp.jsonl",
        )

    def watch(
        self,
        task: str,
        dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
        split: str = "train",
        out_dir: str = "./synthetic_visual_dialogues",
        idx: int = 0,
    ) -> None:
        """Visualize all dialogue for the given image sample"""
        db_path = Launcher.__get_db_file__(out_dir=out_dir, dataset=dataset)
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = db.cursor()
        table_name = Launcher.__get_table_name__(task, split)
        try:
            itr = cursor.execute(f"SELECT DISTINCT uid FROM {table_name}")
            for _ in range(idx + 1):
                uid = itr.fetchone()[0]

            lines = cursor.execute(
                f'SELECT idx, text FROM {table_name} WHERE uid = "{uid}" ORDER BY idx, turn'
            ).fetchall()
            past_idx = -1
            line_idx = 0
            for dialogue_idx, line in lines:
                if dialogue_idx != past_idx:
                    past_idx = dialogue_idx
                    rich.print(f"\n[magenta]Dialogue {past_idx + 1}[/magenta]")
                    line_idx = 0
                color = "cyan" if line_idx % 2 == 0 else "yellow"
                speaker = "USER" if line_idx % 2 == 0 else "MOSHIVIS"
                rich.print(f"[bold]{speaker}:[/bold] [{color}]{line}[/{color}]")
                line_idx += 1
        finally:
            cursor.close()
            db.close()

    def run(
        self,
        task: str,
        dataset: Literal["docci", "pixelprose", "pixmo"] = "docci",
        split: str = "train",
        start_idx: int = 0,
        end_idx: int = -1,
        out_dir: str = "./synthetic_visual_dialogues",
        batch_size: int = 8,
        temperature: float = 0.3,
        convo_length: int = 16,
        num_retries: int = 5,
        overwrite: Literal["yes", "no", "resume"] = "no",
        verbose: bool = False,
        output_format: Literal["jsonl", "db", "both"] = "jsonl",
    ) -> None:
        """Generate synthetic visual dialogues for the given dataset

        :param output_format: 出力先を選択 ("jsonl", "db", "both")
        """
        task = task.lower()
        try:
            MTCInstruct(task)
        except ValueError as e:
            raise NotImplementedError(f"Unknown MTC Instruct pipeline {task}") from e

        descriptions = get_captions(dataset=dataset, split=split)
        if end_idx < 0:
            end_idx = len(descriptions)
        rich.print(f"Found {len(descriptions)} samples in {dataset}-{split}")
        descriptions = descriptions.select(range(start_idx, end_idx))
        rich.print(f"{len(descriptions)} samples after shard selection")

        out_dir = os.path.abspath(out_dir)
        out_file = Launcher.__get_annot_file__(
            task, out_dir, dataset, split, start_idx, end_idx
        )
        db_file = Launcher.__get_db_file__(out_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)

        # --- JSONL の場合は既存ファイルの uid を読み込み ---
        existing_uids = set()
        if output_format in {"jsonl", "both"}:
            jsonl_path = os.path.join(out_dir, f"{task}_{dataset}_{split}.jsonl")
            if os.path.exists(jsonl_path):
                with open(jsonl_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            obj = json.loads(line)
                            existing_uids.add(obj["uid"])
                        except Exception:
                            continue
                print(f"Found {len(existing_uids)} existing JSONL dialogues")

        # DBが必要な場合のみ初期化
        if output_format in {"db", "both"}:
            annotations_db = sqlite3.connect(db_file, timeout=60, isolation_level=None)
            annotdb_cursor = annotations_db.cursor()
            table_name = Launcher.__get_table_name__(task, split)
            trial = 0
            while (trial := trial + 1) < 5:
                try:
                    annotdb_cursor.execute(
                        f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        uid INTEGER,
                        idx INTEGER,
                        turn INTEGER,
                        speaker INTEGER,
                        text TEXT,
                        PRIMARY KEY(uid, idx, turn)
                        )
                    """
                    )
                    trial = 5
                except sqlite3.OperationalError:
                    pass
        else:
            annotations_db, annotdb_cursor, table_name = None, None, None

        track_idx_per_uid: Dict[str, int] = defaultdict(lambda: 0)

        og_length = len(descriptions)
        descriptions = [x for x in descriptions if x["uid"] not in existing_uids]
        print(
            f"Found {len(descriptions)} / {og_length} captions without an associated dialogue"
        )

        # pipeline
        rich.print(f"Annotations will be generated in [yellow]{out_file}[/yellow]")
        hf_pipeline = get_pipeline()
        try:
            num_rows_written = 0
            for retry_idx in range(num_retries):
                print(
                    f"Run {retry_idx + 1} / {num_retries} (max): {len(descriptions)} samples left to process"
                )
                if len(descriptions) == 0:
                    break
                failed_uids = set()
                num_total_batches = int(ceil(len(descriptions) / batch_size))
                for batch_idx in range(num_total_batches):
                    indices = list(
                        range(
                            batch_size * batch_idx,
                            min(len(descriptions), batch_size * (batch_idx + 1)),
                        )
                    )
                    captions = [descriptions[i]["caption"].strip() for i in indices]
                    uids = [descriptions[i]["uid"] for i in indices]

                    run_multiturn_pipeline(
                        hf_pipeline,
                        captions=[x.strip() for x in captions],
                        img_ids=[str(x) for x in uids],
                        out_file=out_file,
                        batch_size=min(len(captions), batch_size),
                        convo_length=random.randint(4, convo_length // 2) * 2,
                        setting=task,
                        temperature=temperature,
                    )

                    # Post-process
                    with open(out_file, "r") as f:
                        for result in f.readlines():
                            data = json.loads(result)
                            rows = postprocess_synth_annot(
                                uid=data["uid"],
                                res=data["res"],
                                idx=track_idx_per_uid,
                                trim_first_question=task == "comb",
                            )
                            if len(rows) == 0:
                                failed_uids.add(data["uid"])
                                continue

                            # JSONL 出力
                            if output_format in {"jsonl", "both"}:
                                with open(os.path.join(out_dir, f"{task}_{dataset}_{split}.jsonl"),
                                        "a", encoding="utf-8") as fout:
                                    dialogue = [
                                        {"speaker": "ユーザー" if row[3] == 1 else "アシスタント", "text": row[4]}
                                        for row in rows
                                    ]
                                    json.dump({"uid": data["uid"], "dialogue": dialogue}, fout, ensure_ascii=False)
                                    fout.write("\n")

                            # DB 出力
                            if output_format in {"db", "both"}:
                                for line in rows:
                                    try:
                                        annotdb_cursor.execute(
                                            f"INSERT OR REPLACE INTO {table_name} VALUES(?, ?, ?, ?, ?)",
                                            line,
                                        )
                                        num_rows_written += 1
                                    except (
                                        sqlite3.OperationalError,
                                        sqlite3.IntegrityError,
                                    ):
                                        continue

                    rich.print(
                        f"  [magenta]Batch {(batch_idx + 1):05d}/{num_total_batches:05d}[/magenta]: "
                        f"[cyan]wrote {num_rows_written} rows so far[/cyan]"
                    )
                descriptions = [s for s in descriptions if s["uid"] in failed_uids]
        finally:
            if output_format in {"db", "both"}:
                annotations_db.commit()
                annotdb_cursor.close()
                annotations_db.close()


if __name__ == "__main__":
    # Example:
    # ```bash
    #    python scripts/preprocessing/synthetic_annots/annotate_docci.py --task tns
    # ```
    # """
    fire.Fire(Launcher)
