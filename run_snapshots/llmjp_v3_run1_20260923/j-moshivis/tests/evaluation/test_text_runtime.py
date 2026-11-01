from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from jmoshivis.evaluation.text_runtime import (
    JmoshiVisTextRuntime,
    SpeechlessDelayState,
)
from jmoshivis.models.lm_utils import _delay_sequence


class _Tokenizer:
    def encode_as_ids(self, text: str) -> list[int]:
        return [10, 11] if text else []

    def eos_id(self) -> int:
        return 2

    def id_to_piece(self, token_id: int) -> str:
        return {2: "</s>", 7: "答"}.get(token_id, str(token_id))

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_piece(value) for value in token_ids)


class _LanguageModel:
    zero_token_id = -1
    num_codebooks = 3
    delays = (0, 0, 1)

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    @contextlib.contextmanager
    def streaming(self):
        yield

    def get_initial_token(self) -> torch.Tensor:
        return torch.full((1, self.num_codebooks, 1), 99, dtype=torch.long)

    def forward_text(self, frame: torch.Tensor, *, cross_attention_src):
        del cross_attention_src
        self.inputs.append(frame.clone())
        logits = torch.full((1, 1, 1, 12), -10.0)
        # initial, prompt-0, prompt-1, answer-0.  The final prompt predicts a
        # lexical answer; feeding that answer predicts EOS.
        next_id = 7 if len(self.inputs) == 3 else 2
        logits[..., next_id] = 10.0
        return torch.zeros(1, 1, 2), logits, 0.25


class TextRuntimeTests(unittest.TestCase):
    def test_streaming_delay_state_matches_batch_speechless_preprocessing(self) -> None:
        lm = _LanguageModel()
        # Full target sequence: prompt 10,11; generated answer 7, EOS 2.
        raw = torch.full((1, lm.num_codebooks, 4), lm.zero_token_id, dtype=torch.long)
        raw[:, 0, :] = torch.tensor([10, 11, 7, 2])
        initial = lm.get_initial_token()
        expected = torch.cat(
            [initial, _delay_sequence(list(lm.delays), raw, initial)], dim=2
        )[:, :, :-1]

        state = SpeechlessDelayState(lm)
        streamed = torch.cat(
            [state.initial] + [state.step(token) for token in (10, 11, 7)], dim=2
        )

        self.assertTrue(torch.equal(streamed, expected))

    def test_speechless_generation_uses_training_delay_prefix(self) -> None:
        runtime = object.__new__(JmoshiVisTextRuntime)
        lm = _LanguageModel()
        runtime.config = SimpleNamespace(
            seed=7,
            max_new_tokens=4,
            use_sampling=False,
        )
        runtime.device = "cpu"
        runtime.tokenizer = _Tokenizer()
        runtime.model = SimpleNamespace(lm_model=lm)
        runtime._encode_image = lambda _: (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))

        with tempfile.TemporaryDirectory() as directory:
            result = runtime.infer(Path(directory) / "image.png", "質問", sample_id="sample")

        self.assertEqual(result["answer_text"], "答")
        self.assertEqual(result["answer_token_ids"], [7, 2])
        self.assertEqual(result["prompt_token_count"], 2)
        self.assertEqual(result["generated_token_count"], 2)
        self.assertTrue(torch.equal(lm.inputs[0], torch.full((1, 3, 1), 99)))
        for index, (frame, text_id) in enumerate(zip(lm.inputs[1:], (10, 11, 7))):
            self.assertEqual(int(frame[0, 0, 0]), text_id)
            expected_audio = torch.tensor([-1, 99] if index == 0 else [-1, -1])
            self.assertTrue(torch.equal(frame[0, 1:, 0], expected_audio))


if __name__ == "__main__":
    unittest.main()
