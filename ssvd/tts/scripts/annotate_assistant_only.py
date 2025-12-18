#!/usr/bin/env python3
# python3 annotate_assistant_only.py --local --verbose /workspace/data/speech/data.jsonl

import argparse
import gc
import gzip
import importlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import sphn
import submitit
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper

transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@contextmanager
def write_and_rename(path: Path, mode: str = "w", suffix: str = ".tmp", pid: bool = False):
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode, encoding="utf-8") as f:
        yield f
    os.rename(tmp_path, path)


def load_audio_paths(egs_path: Path) -> list[Path]:
    open_fn = gzip.open if str(egs_path).lower().endswith(".gz") else open
    with open_fn(egs_path, "rt", encoding="utf-8") as fp:
        lines = fp.readlines()
    paths: list[Path] = []
    for line in lines:
        d = json.loads(line)
        paths.append(Path(d["path"]))
    return paths


def init_logging(verbose: bool = False):
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


def process_one_assistant_only(
    in_file: Path,
    out_file: Path,
    language: str,
    w_model,
    params: "Params",
):
    """
    ステレオ音声から「アシスタント側チャンネル」だけを参照し、
    すべて speaker="ASSISTANT" として alignments を作る。
    """
    logger.debug("Loading stereo audio %s", in_file)
    gc.collect()
    torch.cuda.empty_cache()

    # sphn.read は (チャンネル数, サンプル数) の ndarray を返す
    x, sr = sphn.read(in_file)
    if x.ndim == 1:
        logger.warning("Mono file detected, skipping: %s", in_file)
        return
    if x.shape[0] <= params.assistant_channel:
        logger.warning(
            "File has only %d channels (assistant_channel=%d), skipping: %s",
            x.shape[0],
            params.assistant_channel,
            in_file,
        )
        return

    x = torch.from_numpy(x).cuda()

    # 指定されたチャンネルのみ利用
    ch = params.assistant_channel
    speaker_label = "ASSISTANT"
    logger.debug("Processing assistant channel %d (%s)", ch, speaker_label)

    vocals = x[ch][None]
    vocals = F.resample(vocals, sr, SAMPLE_RATE)
    vocals_np = vocals.cpu().numpy()[0]

    this_duration = vocals_np.shape[-1] / SAMPLE_RATE
    if this_duration < 0.1:  # 短すぎる音声はスキップ
        logger.info("Audio too short (%.3f sec), skipping: %s", this_duration, in_file)
        return

    logger.debug(
        "Transcribing assistant channel %d in %s, duration %.1f",
        ch,
        language,
        this_duration,
    )

    pipe_output = whisper.transcribe(
        w_model,
        vocals_np,
        language=language,
        vad=True,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    all_chunks = []
    for segment in pipe_output.get("segments", []):
        if "words" not in segment:
            continue
        for word in segment["words"]:
            try:
                all_chunks.append(
                    {
                        "text": word["text"],
                        "timestamp": (word["start"], word["end"]),
                        # ← 常に ASSISTANT としてラベル付け
                        "speaker": speaker_label,
                    }
                )
            except KeyError:
                logger.error(
                    "Missing key in %s (channel %d): %r", in_file, ch, word
                )

    # 1語もなければファイルは作らない
    if not all_chunks:
        logger.info("No words detected for %s, skipping output.", in_file)
        return

    # 開始時刻でソート
    all_chunks.sort(key=lambda item: item["timestamp"][0])

    # Moshi-finetune 形式に整形
    outputs = {
        "alignments": [
            [chunk["text"], chunk["timestamp"], chunk["speaker"]]
            for chunk in all_chunks
        ]
    }

    logger.debug("Whisper applied to assistant channel only.")
    with write_and_rename(out_file, "w", pid=True) as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    logger.info("✅ Wrote file %s", out_file)


def run(params: "Params", shard: int = 0):
    init_logging(params.verbose)
    local_rank = 0
    logger.info("Hello, world, this is shard %d / %d.", shard, params.shards)
    params.shard = shard
    torch.cuda.set_device(local_rank)

    logger.info("Loading Whisper model: %s", params.whisper_model)
    device = torch.device(f"cuda:%d" % local_rank)
    w_model = whisper.load_model(params.whisper_model, device=device)

    logger.info("Loading egs %s.", params.egs)
    paths = load_audio_paths(params.egs)
    kept_paths = paths[shard :: params.shards]
    logger.info("Processing %d files out of %d.", len(kept_paths), len(paths))
    del paths

    for idx, path in enumerate(tqdm(kept_paths, desc="Annotating assistant channel")):
        # 出力ファイル名は分かりやすく変更しても良い（例: assistant_dialogue.json）
        out_file = path.parent / "assistant_dialogue.json"
        err_file = path.parent / "assistant_dialogue.json.err"

        if out_file.exists():
            logger.debug("Output file %s already exists, skipping.", out_file)
            continue
        if err_file.exists() and not params.rerun_errors:
            logger.debug("Error file %s exists, skipping.", err_file)
            continue

        try:
            process_one_assistant_only(
                path,
                out_file,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
        except Exception as err:
            if "cuda" in repr(err).lower():
                logger.error(
                    "CUDA error processing %s. Aborting.", path, exc_info=True
                )
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
            continue


@dataclass
class Params:
    egs: Path
    verbose: bool
    lang: str
    whisper_model: str
    keep_silence_in_segments: float
    rerun_errors: bool
    shards: int
    assistant_channel: int = 0
    shard: int = 0


def main():
    parser = argparse.ArgumentParser(
        description="Annotate stereo audio dialogues using only the assistant channel."
    )
    parser.add_argument(
        "egs", type=Path, help="Path to the .jsonl file listing audio files."
    )
    parser.add_argument(
        "--log_folder",
        type=Path,
        default=Path.home() / "tmp" / "mass_annotate_assistant",
    )
    parser.add_argument(
        "-S",
        "--shards",
        type=int,
        default=1,
        help="Number of shards to schedule for parallel processing.",
    )
    parser.add_argument(
        "--lang", default="ja", help="Language of the audio files."
    )
    parser.add_argument(
        "--partition",
        default="",
        help="SLURM partition to use (if not running locally).",
    )
    parser.add_argument(
        "--whisper_model",
        default="medium",
        help="Whisper model size (e.g., tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--rerun_errors",
        action="store_true",
        help="Ignore previous errors and rerun failed files.",
    )
    parser.add_argument(
        "--keep_silence_in_segments",
        type=float,
        default=0.1,
        help="Seconds of silence to keep at segment boundaries.",
    )
    parser.add_argument(
        "--assistant_channel",
        type=int,
        default=0,
        help="Index of the assistant channel in stereo audio (0-based).",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="Run locally on a single shard (for debugging).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()
    init_logging(args.verbose)

    # Params に渡す項目を整形
    kwargs = {
        k: v
        for k, v in args.__dict__.items()
        if k not in ["local", "partition", "log_folder"]
    }
    params = Params(**kwargs)

    if args.local:
        params.shards = 1
        run(params)
    else:
        # SLURM での分散実行
        executor = submitit.SlurmExecutor(folder=args.log_folder)
        executor.update_parameters(
            cpus_per_task=6,
            ntasks_per_node=1,
            gpus_per_node=1,
            time=60 * 24 * 10,
            partition=args.partition,
            job_name="annotate_assistant_only",
        )
        jobs = executor.map_array(run, [params] * args.shards, range(args.shards))
        print(f"Scheduled {len(jobs)} jobs. First job id: {jobs[0].job_id}")


if __name__ == "__main__":
    main()
