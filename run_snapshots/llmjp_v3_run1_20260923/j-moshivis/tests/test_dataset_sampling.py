from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from jmoshivis.datasets.datasets import (
    DataFile,
    SPEECH_CHUNK_POLICY,
    _decode_speech_row,
    build_mixed_dataset,
    get_dataset_iterator,
    get_speechless_dataset_iterator,
    iter_sharded_jsonl_lines,
    iter_stratified_modalities,
    iter_stratified_source_modalities,
    pad_or_truncate_text_ids,
    require_image_condition,
    speech_crop_start_sec,
    stable_speech_sample_key,
    stratified_interleave_iterators,
    stratified_prefix_cursor_counts,
)
from jmoshivis.datasets.interleaver import (
    InterleavedTokenizer,
    Sample,
    slice_overlapping_alignments,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [0, 7]


def _constant_iterator(value: str):
    while True:
        yield value


class SpeechlessPaddingTests(unittest.TestCase):
    def test_padding_uses_loss_mask_id_without_replacing_real_zero(self) -> None:
        self.assertEqual(
            pad_or_truncate_text_ids([0, 11, 0], 6),
            [0, 11, 0, -1, -1, -1],
        )
        self.assertEqual(pad_or_truncate_text_ids([0, 11, 12], 2), [0, 11])

    def test_speechless_sample_contains_masked_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "data.jsonl"
            item_dir = Path(tmpdir) / "missing"
            item_dir.mkdir()
            Image.new("RGB", (4, 4), "white").save(item_dir / "image.jpg")
            jsonl_path.write_text(
                json.dumps({"text": "dummy", "path": str(item_dir / "item.wav")})
                + "\n",
                encoding="utf-8",
            )
            embedder = SimpleNamespace(kyuteye_config=object())

            with patch(
                "jmoshivis.datasets.datasets.ImageProcessor.from_config",
                return_value=lambda _: torch.zeros(3, 4, 4),
            ):
                iterator = get_speechless_dataset_iterator(
                    source=DataFile(jsonl_path),
                    text_tokenizer=_Tokenizer(),
                    image_embedder=embedder,
                    rank=0,
                    world_size=1,
                    is_finite=True,
                    seed=42,
                    shuffle_at_epoch=False,
                    image_root=None,
                    device=torch.device("cpu"),
                    num_audio_streams=2,
                    audio_pad_id=-1,
                    target_len=5,
                )
                sample = next(iterator)

        self.assertEqual(sample.codes.shape, (1, 3, 5))
        self.assertEqual(sample.codes[0, 0].tolist(), [0, 7, -1, -1, -1])
        self.assertTrue((sample.codes[0, 1:] == -1).all())
        self.assertIsNotNone(sample.condition_attributes)


class SpeechCropTests(unittest.TestCase):
    def test_crop_is_reproducible_varies_by_epoch_and_eval_starts_at_zero(self) -> None:
        kwargs = dict(
            sample_key="content-addressed-uid",
            audio_duration_sec=75.0,
            crop_duration_sec=60.0,
            seed=42,
            is_eval=False,
            policy=SPEECH_CHUNK_POLICY,
        )
        epoch0 = speech_crop_start_sec(epoch=0, **kwargs)
        repeated = speech_crop_start_sec(epoch=0, **kwargs)
        epoch1 = speech_crop_start_sec(epoch=1, **kwargs)

        self.assertEqual(epoch0, repeated)
        self.assertNotEqual(epoch0, epoch1)
        self.assertGreaterEqual(epoch0, 0.0)
        self.assertLessEqual(epoch0, 15.0)
        self.assertEqual(
            speech_crop_start_sec(epoch=99, **{**kwargs, "is_eval": True}),
            0.0,
        )
        self.assertEqual(
            speech_crop_start_sec(
                sample_key=kwargs["sample_key"],
                audio_duration_sec=45.0,
                crop_duration_sec=60.0,
                seed=42,
                epoch=0,
                is_eval=False,
            ),
            0.0,
        )

    def test_crop_key_is_independent_of_absolute_dataset_root(self) -> None:
        left = {
            "uid": "same-uid",
            "path": "/old/server/same-uid/stereo_dialogue.wav",
        }
        right = {
            "uid": "same-uid",
            "path": "/new/server/same-uid/stereo_dialogue.wav",
        }
        self.assertEqual(stable_speech_sample_key(left), stable_speech_sample_key(right))
        starts = [
            speech_crop_start_sec(
                sample_key=stable_speech_sample_key(sample),
                audio_duration_sec=75.0,
                crop_duration_sec=60.0,
                seed=42,
                epoch=3,
                is_eval=False,
            )
            for sample in (left, right)
        ]
        self.assertEqual(starts[0], starts[1])
        self.assertEqual(
            stable_speech_sample_key(
                {"path": "/another/root/fallback-uid/stereo_dialogue.wav"}
            ),
            "fallback-uid",
        )

    def test_one_long_row_yields_one_sample_and_rank_shards_are_disjoint(self) -> None:
        class FakeTokenizer:
            duration_sec = 60.0
            mimi = SimpleNamespace(sample_rate=100)

            def __init__(self) -> None:
                self.calls: list[tuple[str, float]] = []

            def __call__(self, wav, start_sec, path):
                self.calls.append((Path(path).parent.name, start_sec))
                return Sample(torch.zeros((1, 3, 2), dtype=torch.long))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "data.jsonl"
            rows = []
            for index in range(5):
                item = root / f"uid-{index}"
                item.mkdir()
                Image.new("RGB", (4, 4), "white").save(item / "image.jpg")
                rows.append(
                    {
                        "uid": f"uid-{index}",
                        "path": str(item / "stereo_dialogue.wav"),
                        "duration": 75.0,
                    }
                )
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            calls_by_rank = []
            with patch(
                "jmoshivis.datasets.datasets.sphn.read",
                return_value=(np.zeros((2, 6000), dtype=np.float32), 100),
            ), patch(
                "jmoshivis.datasets.datasets.ImageProcessor.from_config",
                return_value=lambda _: torch.zeros(3, 4, 4),
            ):
                for rank in range(2):
                    tokenizer = FakeTokenizer()
                    samples = list(
                        get_dataset_iterator(
                            source=DataFile(manifest),
                            instruct_tokenizer=tokenizer,
                            image_embedder=SimpleNamespace(kyuteye_config=object()),
                            rank=rank,
                            world_size=2,
                            is_finite=True,
                            seed=11,
                            shuffle_at_epoch=True,
                            image_root="/unused/legacy/root",
                            device=torch.device("cpu"),
                            speech_decode_workers=2,
                        )
                    )
                    self.assertEqual(len(samples), len(tokenizer.calls))
                    calls_by_rank.append({uid for uid, _ in tokenizer.calls})

        self.assertTrue(calls_by_rank[0].isdisjoint(calls_by_rank[1]))
        self.assertEqual(calls_by_rank[0] | calls_by_rank[1], {f"uid-{i}" for i in range(5)})
        self.assertEqual(sum(map(len, calls_by_rank)), 5)

    def test_audio_decode_error_has_sample_and_path(self) -> None:
        line = json.dumps(
            {
                "uid": "broken-uid",
                "path": "/dataset/broken/stereo_dialogue.wav",
                "duration": 70.0,
            }
        )
        with patch(
            "jmoshivis.datasets.datasets.sphn.read",
            side_effect=OSError("decode failed"),
        ), self.assertRaisesRegex(
            RuntimeError, "broken-uid.*stereo_dialogue.wav"
        ):
            _decode_speech_row(
                line,
                crop_duration_sec=60.0,
                sample_rate=24_000,
                seed=1,
                epoch=0,
                is_eval=False,
                policy=SPEECH_CHUNK_POLICY,
            )


class ImageConditioningIntegrityTests(unittest.TestCase):
    def test_missing_image_fails_with_sample_audio_and_image_paths(self) -> None:
        sample = {
            "uid": "missing-image",
            "path": "/dataset/missing-image/stereo_dialogue.wav",
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "missing-image.*stereo_dialogue.wav.*image.jpg",
        ):
            require_image_condition(
                sample,
                image_processor=lambda _: torch.zeros(3, 2, 2),
                device=torch.device("cpu"),
                image_root="/legacy/root/is/not/a/fallback",
            )

    def test_image_processor_none_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            item = Path(tmpdir) / "uid"
            item.mkdir()
            Image.new("RGB", (4, 4), "white").save(item / "image.jpg")
            sample = {
                "uid": "decode-none",
                "path": str(item / "stereo_dialogue.wav"),
            }
            with self.assertRaisesRegex(
                RuntimeError, "decode-none.*ImageProcessor returned None"
            ):
                require_image_condition(
                    sample,
                    image_processor=lambda _: None,
                    device=torch.device("cpu"),
                    image_root=None,
                )


class AlignmentCropBoundaryTests(unittest.TestCase):
    def test_alignment_crossing_crop_start_is_kept_at_frame_zero(self) -> None:
        alignments = [
            ([10], (9.8, 10.2), "ASSISTANT"),
            ([11], (10.4, 10.6), "ASSISTANT"),
            ([12], (12.1, 12.2), "ASSISTANT"),
        ]

        cropped = slice_overlapping_alignments(
            alignments, start_sec=10.0, duration_sec=2.0
        )

        self.assertEqual([item[0] for item in cropped], [[10], [11]])
        self.assertAlmostEqual(cropped[0][1][0], -0.2)
        self.assertAlmostEqual(cropped[0][1][1], 0.2)

    def test_full_duplex_input_is_sorted_before_crop_filter(self) -> None:
        # Dialogue order can place a USER interruption after an ASSISTANT token
        # whose start is later.  The crop result must still be chronological.
        alignments = [
            ([10], (9.8, 10.2), "ASSISTANT"),
            ([12], (10.5, 10.8), "ASSISTANT"),
            ([11], (10.1, 10.7), "USER"),
        ]

        cropped = slice_overlapping_alignments(
            alignments, start_sec=10.0, duration_sec=1.0
        )

        self.assertEqual([item[0] for item in cropped], [[10], [11], [12]])
        self.assertEqual([item[2] for item in cropped], ["ASSISTANT", "USER", "ASSISTANT"])

    def test_long_overlap_is_not_hidden_by_expired_intervals(self) -> None:
        alignments = [
            ([20], (1.0, 12.0), "USER"),
            ([21], (2.0, 3.0), "ASSISTANT"),
            ([22], (4.0, 5.0), "ASSISTANT"),
            ([23], (10.5, 10.8), "ASSISTANT"),
        ]

        cropped = slice_overlapping_alignments(
            alignments, start_sec=10.0, duration_sec=1.0
        )

        self.assertEqual([item[0] for item in cropped], [[20], [23]])
        self.assertAlmostEqual(cropped[0][1][0], -9.0)


class InterleavedTokenizerDeviceTests(unittest.TestCase):
    def test_audio_tensor_uses_mimi_parameter_device_not_implicit_cuda(self) -> None:
        class FakeMimi(torch.nn.Module):
            frame_rate = 2.0

            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.observed_device = None

            def encode(self, waveform: torch.Tensor) -> torch.Tensor:
                self.observed_device = waveform.device
                return torch.zeros((2, 2, 4), dtype=torch.long, device=waveform.device)

        class FakeInterleaver:
            zero_padding = -1

            def prepare_item(self, alignments, duration, pretokenized):
                del alignments, duration, pretokenized
                return torch.zeros((1, 1, 4), dtype=torch.long)

        with tempfile.TemporaryDirectory() as directory:
            item = Path(directory)
            audio_path = item / "stereo_dialogue.wav"
            (item / "assistant_dialogue.refined.json").write_text(
                json.dumps(
                    {
                        "token_alignments": [
                            [[1], [0.1, 0.2], "USER"],
                            [[2], [0.3, 0.4], "ASSISTANT"],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mimi = FakeMimi()
            tokenizer = InterleavedTokenizer(
                mimi=mimi,
                interleaver=FakeInterleaver(),
                duration_sec=2.0,
            )
            sample = tokenizer(
                np.zeros((2, 8), dtype=np.float32),
                start_sec=0.0,
                path=str(audio_path),
            )

        self.assertEqual(mimi.observed_device, mimi.anchor.device)
        self.assertEqual(tuple(sample.codes.shape), (1, 5, 4))


class JsonlShuffleTests(unittest.TestCase):
    def test_rank_local_start_position_matches_uninterrupted_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(17)),
                encoding="utf-8",
            )
            kwargs = dict(
                path=path,
                rank=1,
                world_size=3,
                shuffle=True,
                seed=321,
                epoch=4,
                drop_remainder=True,
            )
            full = list(iter_sharded_jsonl_lines(**kwargs))
            resumed = list(
                iter_sharded_jsonl_lines(**kwargs, start_position=3)
            )

        self.assertEqual(resumed, full[3:])

    def test_training_epoch_shuffle_is_equal_disjoint_and_drops_only_remainder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(31)),
                encoding="utf-8",
            )

            def ids(rank: int, epoch: int) -> list[int]:
                return [
                    json.loads(line)["id"]
                    for line in iter_sharded_jsonl_lines(
                        path,
                        rank=rank,
                        world_size=3,
                        shuffle=True,
                        seed=123,
                        epoch=epoch,
                        drop_remainder=True,
                    )
                ]

            epoch0 = [ids(rank, 0) for rank in range(3)]
            repeated = [ids(rank, 0) for rank in range(3)]
            epoch1 = [ids(rank, 1) for rank in range(3)]

        self.assertEqual(epoch0, repeated)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual([len(shard) for shard in epoch0], [10, 10, 10])
        flattened = list(itertools.chain.from_iterable(epoch0))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(flattened), 30)
        self.assertEqual(len(set(range(31)) - set(flattened)), 1)

    def test_eval_sharding_keeps_every_row_and_allows_one_row_imbalance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(31)),
                encoding="utf-8",
            )
            shards = [
                [
                    json.loads(line)["id"]
                    for line in iter_sharded_jsonl_lines(
                        path,
                        rank=rank,
                        world_size=3,
                        shuffle=False,
                        seed=123,
                        epoch=0,
                        drop_remainder=False,
                    )
                ]
                for rank in range(3)
            ]

        self.assertEqual([len(shard) for shard in shards], [11, 10, 10])
        flattened = list(itertools.chain.from_iterable(shards))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(range(31)))


class ExactMixedDatasetResumeTests(unittest.TestCase):
    class _SpeechTokenizer:
        duration_sec = 60.0
        mimi = SimpleNamespace(sample_rate=100)

        def __init__(self) -> None:
            self.calls: list[tuple[str, float]] = []

        def __call__(self, wav, start_sec, path):
            del wav
            uid = Path(path).parent.name
            source_id, row_id = (
                int(part[1:]) for part in uid.split("-")
            )
            self.calls.append((uid, start_sec))
            code = (
                1_000_000
                + source_id * 100_000
                + row_id * 10_000
                + int(round(start_sec * 100))
            )
            return Sample(torch.full((1, 17, 1), code, dtype=torch.long))

    class _TextTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [int(text)]

    def test_multisource_resume_matches_rows_modalities_and_crop_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifests = []
            for source_id in range(2):
                manifest = root / f"source-{source_id}.jsonl"
                rows = []
                for row_id in range(5):
                    item = root / f"s{source_id}-r{row_id}"
                    item.mkdir()
                    Image.new("RGB", (4, 4), "white").save(item / "image.jpg")
                    rows.append(
                        {
                            "uid": item.name,
                            "path": str(item / "stereo_dialogue.wav"),
                            "duration": 75.0,
                            "text": str(2_000_000 + source_id * 100 + row_id),
                        }
                    )
                manifest.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                manifests.append(manifest)

            pretrain_data = ",".join(f"{path}:0.5" for path in manifests)
            schedule_kwargs = dict(
                source_probabilities=[0.5, 0.5],
                speech_ratio=0.5,
                seed=876,
                world_size=2,
                local_batch_size=1,
                effective_window_size=8,
            )
            resume_at = 64
            tail_size = 16

            for rank in range(2):
                with self.subTest(rank=rank):
                    image_calls: list[str] = []

                    def process_image(path):
                        image_calls.append(str(path))
                        return torch.zeros(3, 4, 4)

                    baseline_tokenizer = self._SpeechTokenizer()
                    with patch(
                        "jmoshivis.datasets.datasets.sphn.read",
                        return_value=(np.zeros((2, 6000), dtype=np.float32), 100),
                    ) as baseline_decode, patch(
                        "jmoshivis.datasets.datasets.ImageProcessor.from_config",
                        return_value=process_image,
                    ):
                        baseline_iterator = build_mixed_dataset(
                            pretrain_data=pretrain_data,
                            instruct_tokenizer=baseline_tokenizer,
                            text_tokenizer=self._TextTokenizer(),
                            seed=876,
                            rank=rank,
                            world_size=2,
                            is_eval=False,
                            device=torch.device("cpu"),
                            image_embedder=SimpleNamespace(kyuteye_config=object()),
                            speech_ratio=0.5,
                            target_len=1,
                            local_batch_size=1,
                            effective_window_size=8,
                            speech_decode_workers=1,
                        )
                        baseline = [
                            sample.codes.clone()
                            for sample in itertools.islice(
                                baseline_iterator, resume_at + tail_size
                            )
                        ]
                        baseline_iterator.close()
                        baseline_decode_count = baseline_decode.call_count
                    self.assertEqual(len(image_calls), resume_at + tail_size)

                    image_calls.clear()
                    resumed_tokenizer = self._SpeechTokenizer()
                    with patch(
                        "jmoshivis.datasets.datasets.sphn.read",
                        return_value=(np.zeros((2, 6000), dtype=np.float32), 100),
                    ) as resumed_decode, patch(
                        "jmoshivis.datasets.datasets.ImageProcessor.from_config",
                        return_value=process_image,
                    ):
                        resumed_iterator = build_mixed_dataset(
                            pretrain_data=pretrain_data,
                            instruct_tokenizer=resumed_tokenizer,
                            text_tokenizer=self._TextTokenizer(),
                            seed=876,
                            rank=rank,
                            world_size=2,
                            is_eval=False,
                            device=torch.device("cpu"),
                            image_embedder=SimpleNamespace(kyuteye_config=object()),
                            speech_ratio=0.5,
                            target_len=1,
                            local_batch_size=1,
                            effective_window_size=8,
                            speech_decode_workers=1,
                            start_sample_index=resume_at,
                        )
                        resumed = [
                            sample.codes.clone()
                            for sample in itertools.islice(
                                resumed_iterator, tail_size
                            )
                        ]
                        resumed_iterator.close()

                    for actual, expected in zip(resumed, baseline[resume_at:]):
                        torch.testing.assert_close(
                            actual, expected, rtol=0.0, atol=0.0
                        )
                    self.assertEqual(len(image_calls), tail_size)

                    choices = list(
                        itertools.islice(
                            iter_stratified_source_modalities(
                                **schedule_kwargs, rank=rank
                            ),
                            resume_at + tail_size,
                        )
                    )
                    prefix_speech = sum(
                        use_speech for _, use_speech in choices[:resume_at]
                    )
                    suffix_speech = sum(
                        use_speech for _, use_speech in choices[resume_at:]
                    )
                    self.assertEqual(
                        resumed_tokenizer.calls,
                        baseline_tokenizer.calls[prefix_speech:],
                    )
                    self.assertEqual(resumed_decode.call_count, suffix_speech)
                    self.assertEqual(
                        baseline_decode_count, prefix_speech + suffix_speech
                    )


class StratifiedModalityTests(unittest.TestCase):
    @staticmethod
    def _global_schedule(
        ratio: float,
        *,
        world_size: int = 2,
        local_batch_size: int = 1,
        global_samples: int = 32,
    ) -> list[bool]:
        samples_per_micro_step = world_size * local_batch_size
        micro_steps = global_samples // samples_per_micro_step
        local_samples = micro_steps * local_batch_size
        by_rank = [
            list(
                itertools.islice(
                    iter_stratified_modalities(
                        speech_ratio=ratio,
                        seed=42,
                        rank=rank,
                        world_size=world_size,
                        local_batch_size=local_batch_size,
                        effective_window_size=16,
                    ),
                    local_samples,
                )
            )
            for rank in range(world_size)
        ]
        result: list[bool] = []
        for micro_step in range(micro_steps):
            for rank in range(world_size):
                start = micro_step * local_batch_size
                result.extend(by_rank[rank][start : start + local_batch_size])
        return result

    def test_supported_ratios_are_exact_in_every_effective_window(self) -> None:
        for ratio, expected in ((0.0, 0), (0.25, 4), (0.5, 8), (1.0, 16)):
            with self.subTest(ratio=ratio):
                schedule = self._global_schedule(ratio)
                self.assertEqual(sum(schedule[:16]), expected)
                self.assertEqual(sum(schedule[16:32]), expected)

    def test_schedule_handles_physical_batch_larger_than_one(self) -> None:
        schedule = self._global_schedule(0.25, local_batch_size=2)
        self.assertEqual(sum(schedule[:16]), 4)
        self.assertEqual(sum(schedule[16:32]), 4)

    def test_each_rank_has_exact_ratio_in_each_optimizer_window(self) -> None:
        schedules = []
        for rank in range(2):
            schedule = list(
                itertools.islice(
                    iter_stratified_modalities(
                        speech_ratio=0.25,
                        seed=42,
                        rank=rank,
                        world_size=2,
                        local_batch_size=1,
                        effective_window_size=16,
                    ),
                    32,
                )
            )
            schedules.append(schedule)
            for offset in range(0, 32, 8):
                self.assertEqual(sum(schedule[offset : offset + 8]), 2)

        # Rank is part of the deterministic shuffle stream, while repeated
        # construction for one rank remains byte-for-byte reproducible.
        self.assertNotEqual(schedules[0], schedules[1])
        repeated = list(
            itertools.islice(
                iter_stratified_modalities(
                    speech_ratio=0.25,
                    seed=42,
                    rank=0,
                    world_size=2,
                    local_batch_size=1,
                    effective_window_size=16,
                ),
                32,
            )
        )
        self.assertEqual(schedules[0], repeated)

    def test_long_running_modality_cursor_counts_match_across_ranks(self) -> None:
        counts = []
        for rank in range(4):
            schedule = itertools.islice(
                iter_stratified_modalities(
                    speech_ratio=0.25,
                    seed=123,
                    rank=rank,
                    world_size=4,
                    local_batch_size=1,
                    effective_window_size=32,
                ),
                8_000,
            )
            speech_count = sum(schedule)
            counts.append((speech_count, 8_000 - speech_count))
        self.assertEqual(counts, [(2_000, 6_000)] * 4)

    def test_reproducible_and_validates_impossible_window(self) -> None:
        kwargs = dict(
            speech_ratio=0.25,
            seed=99,
            rank=0,
            world_size=2,
            effective_window_size=16,
        )
        first = list(itertools.islice(iter_stratified_modalities(**kwargs), 32))
        second = list(itertools.islice(iter_stratified_modalities(**kwargs), 32))
        self.assertEqual(first, second)

        with self.assertRaisesRegex(ValueError, "local rank window size 10"):
            next(
                iter_stratified_modalities(
                    speech_ratio=0.25,
                    seed=1,
                    rank=0,
                    world_size=1,
                    effective_window_size=10,
                )
            )

    def test_interleaver_never_draws_disabled_modality(self) -> None:
        speech = [_constant_iterator("speech")]
        speechless = [_constant_iterator("speechless")]
        only_text = stratified_interleave_iterators(
            speech,
            speechless,
            source_probabilities=[1.0],
            speech_ratio=0.0,
            seed=7,
            rank=0,
            world_size=1,
            effective_window_size=16,
        )
        self.assertEqual(
            list(itertools.islice(only_text, 20)), ["speechless"] * 20
        )

    def test_multisource_cursors_stay_synchronized_across_ranks(self) -> None:
        by_rank = []
        for rank in range(4):
            iterator = stratified_interleave_iterators(
                [_constant_iterator("speech-0"), _constant_iterator("speech-1")],
                [
                    _constant_iterator("speechless-0"),
                    _constant_iterator("speechless-1"),
                ],
                source_probabilities=[0.3, 0.7],
                speech_ratio=0.25,
                seed=77,
                rank=rank,
                world_size=4,
                local_batch_size=1,
                effective_window_size=32,
            )
            by_rank.append(list(itertools.islice(iterator, 8_000)))

        expected_counts = Counter(by_rank[0])
        expected_sources = [value.rsplit("-", 1)[1] for value in by_rank[0]]
        for values in by_rank[1:]:
            self.assertEqual(Counter(values), expected_counts)
            self.assertEqual(
                [value.rsplit("-", 1)[1] for value in values],
                expected_sources,
            )

    def test_resume_schedule_and_per_source_cursors_match_prefix_exactly(self) -> None:
        kwargs = dict(
            source_probabilities=[0.2, 0.3, 0.5],
            speech_ratio=0.25,
            seed=2026,
            rank=1,
            world_size=2,
            local_batch_size=2,
            effective_window_size=16,
        )
        resume_at = 74
        tail_size = 37
        uninterrupted = list(
            itertools.islice(
                iter_stratified_source_modalities(**kwargs),
                resume_at + tail_size,
            )
        )
        resumed = list(
            itertools.islice(
                iter_stratified_source_modalities(
                    **kwargs, start_sample_index=resume_at
                ),
                tail_size,
            )
        )
        self.assertEqual(resumed, uninterrupted[resume_at:])

        speech_counts, speechless_counts = stratified_prefix_cursor_counts(
            **kwargs, sample_count=resume_at
        )
        expected_speech = [0, 0, 0]
        expected_speechless = [0, 0, 0]
        for source_id, use_speech in uninterrupted[:resume_at]:
            counts = expected_speech if use_speech else expected_speechless
            counts[source_id] += 1
        self.assertEqual(speech_counts, expected_speech)
        self.assertEqual(speechless_counts, expected_speechless)


if __name__ == "__main__":
    unittest.main()
