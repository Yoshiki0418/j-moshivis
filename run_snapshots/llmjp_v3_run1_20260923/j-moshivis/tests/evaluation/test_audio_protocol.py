from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from jmoshivis.evaluation.audio_protocol import (
    ASSISTANT_PRELUDE_PROTOCOL,
    BARE_QUESTION_PROTOCOL,
    audio_protocol_contract_from_lock,
    suite_freezes_audio_protocol,
)
from jmoshivis.evaluation.run_inference import RUN_INVARIANT_KEYS
from jmoshivis.evaluation.runtime import (
    iter_audio_dialogue_frames,
    stream_codebook_counts,
)
from jmoshivis.models.moshivis import _select_output_audio_tokens
from jmoshivis.evaluation.schema import sha256_file


class AudioDialogueTimelineTests(unittest.TestCase):
    def test_user_stream_uses_eight_not_the_full_sixteen_audio_codebooks(self) -> None:
        lm_model = SimpleNamespace(
            num_audio_codebooks_in=16,
            num_audio_codebooks_out=8,
            num_codebooks=17,
            audio_offset=1,
        )

        user_input, assistant_output = stream_codebook_counts(lm_model)

        self.assertEqual(user_input, 8)
        self.assertEqual(assistant_output, 8)
        self.assertNotEqual(user_input, lm_model.num_audio_codebooks_in)

    def test_bare_question_keeps_the_historical_user_only_timeline(self) -> None:
        question = torch.tensor([[[10, 11], [20, 21]]])

        frames = list(iter_audio_dialogue_frames(question))

        self.assertEqual([frame.frame for frame in frames], [0, 1])
        self.assertEqual(
            [frame.phase for frame in frames],
            ["question_and_response", "question_and_response"],
        )
        self.assertTrue(torch.equal(frames[0].user_tokens, question[:, :, 0:1]))
        self.assertTrue(torch.equal(frames[1].user_tokens, question[:, :, 1:2]))
        self.assertIsNone(frames[0].forced_output_audio_tokens)
        self.assertIsNone(frames[1].forced_output_audio_tokens)

    def test_prelude_keeps_assistant_audio_out_of_the_user_stream(self) -> None:
        question = torch.tensor([[[10, 11], [20, 21]]])
        silent_user = torch.zeros((1, 2, 2), dtype=torch.long)
        assistant = torch.tensor([[[100, 101], [200, 201], [300, 301]]])

        frames = list(
            iter_audio_dialogue_frames(
                question,
                silent_user_prelude_codes=silent_user,
                assistant_prelude_codes=assistant,
            )
        )

        self.assertEqual([frame.frame for frame in frames], [0, 1, 2, 3])
        self.assertEqual(
            [frame.phase for frame in frames],
            [
                "assistant_prelude",
                "assistant_prelude",
                "question_and_response",
                "question_and_response",
            ],
        )
        for index in range(2):
            self.assertTrue(
                torch.equal(frames[index].user_tokens, silent_user[:, :, index : index + 1])
            )
            self.assertTrue(
                torch.equal(
                    frames[index].forced_output_audio_tokens,
                    assistant[:, :, index : index + 1],
                )
            )
        self.assertTrue(torch.equal(frames[2].user_tokens, question[:, :, 0:1]))
        self.assertIsNone(frames[2].forced_output_audio_tokens)

    def test_prelude_requires_a_matched_stream_pair(self) -> None:
        question = torch.zeros((1, 2, 2), dtype=torch.long)
        assistant = torch.zeros((1, 2, 2), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            list(
                iter_audio_dialogue_frames(
                    question,
                    assistant_prelude_codes=assistant,
                )
            )


class AssistantAudioForcingTests(unittest.TestCase):
    def test_forcing_bypasses_depformer_generation(self) -> None:
        forced = torch.tensor([[[1], [2]]], dtype=torch.long)
        called = False

        def generate() -> torch.Tensor:
            nonlocal called
            called = True
            return torch.full((1, 2), 99, dtype=torch.long)

        selected = _select_output_audio_tokens(
            forced,
            batch_size=1,
            num_codebooks=2,
            audio_card=1024,
            device=torch.device("cpu"),
            generate=generate,
        )

        self.assertFalse(called)
        self.assertTrue(torch.equal(selected, torch.tensor([[1, 2]])))

    def test_unforced_path_uses_depformer_generation(self) -> None:
        generated = torch.tensor([[7, 8]], dtype=torch.long)
        selected = _select_output_audio_tokens(
            None,
            batch_size=1,
            num_codebooks=2,
            audio_card=1024,
            device=torch.device("cpu"),
            generate=lambda: generated,
        )
        self.assertIs(selected, generated)

    def test_forced_codec_ids_are_strictly_validated(self) -> None:
        kwargs = {
            "batch_size": 1,
            "num_codebooks": 2,
            "audio_card": 4,
            "device": torch.device("cpu"),
            "generate": lambda: torch.zeros((1, 2), dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "shape"):
            _select_output_audio_tokens(torch.zeros((1, 2), dtype=torch.long), **kwargs)
        with self.assertRaisesRegex(TypeError, "integer"):
            _select_output_audio_tokens(torch.zeros((1, 2, 1)), **kwargs)
        with self.assertRaisesRegex(ValueError, "outside"):
            _select_output_audio_tokens(
                torch.tensor([[[0], [4]]], dtype=torch.long), **kwargs
            )


class AudioProtocolProvenanceTests(unittest.TestCase):
    def test_vision_v1_defaults_to_bare_and_keeps_frozen_suite_hash(self) -> None:
        project = Path(__file__).resolve().parents[2]
        lock_path = project / "evaluation/suite_v1.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))

        contract = audio_protocol_contract_from_lock(lock_path)

        self.assertEqual(
            contract["frozen_audio_dialogue_protocol"], BARE_QUESTION_PROTOCOL
        )
        self.assertIsNone(contract["frozen_assistant_prelude_wav_sha256"])
        frozen_suite_hash = lock["judge_protocol"]["implementation_file_sha256"][
            "j-moshivis/jmoshivis/evaluation/suite.py"
        ]
        self.assertEqual(
            sha256_file(project / "jmoshivis/evaluation/suite.py"),
            frozen_suite_hash,
        )

    def test_prelude_protocol_needs_matching_suite_digest(self) -> None:
        contract = {
            "frozen_audio_dialogue_protocol": ASSISTANT_PRELUDE_PROTOCOL,
            "frozen_assistant_prelude_wav_sha256": "abc",
        }
        self.assertTrue(
            suite_freezes_audio_protocol(
                contract,
                protocol=ASSISTANT_PRELUDE_PROTOCOL,
                assistant_prelude_sha256="abc",
            )
        )
        self.assertFalse(
            suite_freezes_audio_protocol(
                contract,
                protocol=ASSISTANT_PRELUDE_PROTOCOL,
                assistant_prelude_sha256="different",
            )
        )
        self.assertTrue(
            suite_freezes_audio_protocol(
                {"frozen_audio_dialogue_protocol": BARE_QUESTION_PROTOCOL},
                protocol=BARE_QUESTION_PROTOCOL,
                assistant_prelude_sha256=None,
            )
        )

    def test_protocol_asset_is_a_resume_invariant(self) -> None:
        self.assertIn("audio_dialogue_protocol", RUN_INVARIANT_KEYS)
        self.assertIn("assistant_prelude_wav_sha256", RUN_INVARIANT_KEYS)
        self.assertIn("assistant_prelude_wav_size", RUN_INVARIANT_KEYS)


if __name__ == "__main__":
    unittest.main()
