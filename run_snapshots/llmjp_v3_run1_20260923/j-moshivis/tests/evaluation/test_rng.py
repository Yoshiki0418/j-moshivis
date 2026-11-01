from __future__ import annotations

import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from jmoshivis.evaluation.rng import seed_assigned_device


class AssignedDeviceSeedTests(unittest.TestCase):
    def test_cpu_host_rngs_are_reproducible_without_global_cuda_seed(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.default_generator.get_state()
        try:
            with (
                patch.object(torch, "manual_seed", side_effect=AssertionError("global seed")),
                patch.object(
                    torch.cuda,
                    "manual_seed_all",
                    side_effect=AssertionError("all-device seed"),
                ),
            ):
                seed_assigned_device(0x1_0000_0007, "cpu")
                first = (random.random(), float(np.random.random()), torch.rand(3))
                seed_assigned_device(0x1_0000_0007, "cpu")
                second = (random.random(), float(np.random.random()), torch.rand(3))
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            self.assertTrue(torch.equal(first[2], second[2]))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.default_generator.set_state(torch_state)

    def test_cuda_seed_targets_only_the_explicit_device_generator(self) -> None:
        peer = Mock()
        assigned = Mock()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.default_generator.get_state()
        try:
            with (
                patch.object(torch, "manual_seed", side_effect=AssertionError("global seed")),
                patch.object(
                    torch.cuda,
                    "manual_seed_all",
                    side_effect=AssertionError("all-device seed"),
                ),
                patch.object(torch.cuda, "default_generators", (peer, assigned)),
                patch.object(
                    torch.cuda,
                    "current_device",
                    side_effect=AssertionError(
                        "explicit device must not consult current device"
                    ),
                ),
            ):
                seed_assigned_device(1234, "cuda:1")
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.default_generator.set_state(torch_state)

        peer.manual_seed.assert_not_called()
        assigned.manual_seed.assert_called_once_with(1234)


if __name__ == "__main__":
    unittest.main()
