from pathlib import Path
import unittest

import torch
from PIL import Image
from torchvision import transforms

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import (
    _is_allowed_shared_mha_alias_missing,
    _is_model_visual_adapter_key,
    _partition_training_checkpoint,
    _reset_image_projection_adapter,
    _resolve_visual_adapter_init_mode,
)
from jmoshivis.modules.cross_attention import XAGate


REPO_ROOT = Path(__file__).resolve().parents[1]


class GateCompatibilityTest(unittest.TestCase):
    def test_official_sigmoid_shift_at_zero(self) -> None:
        gate = XAGate(
            activation="sigmoid",
            conditional_gating=True,
            dims=8,
        )
        for parameter in gate.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.ones(1, 2, 8)

        output, weight = gate(x)

        expected = torch.sigmoid(torch.tensor(-4.0))
        torch.testing.assert_close(weight, torch.full_like(weight, expected))
        torch.testing.assert_close(output, torch.full_like(output, expected))
        self.assertEqual(gate.sigmoid_shift, 4.0)

    def test_shift_is_configurable(self) -> None:
        gate = XAGate(activation="sigmoid", sigmoid_shift=2.0)
        _, weight = gate(torch.ones(1, 1, 1))
        torch.testing.assert_close(weight, torch.sigmoid(torch.tensor([[[-2.0]]])))


class ImagePreprocessingTest(unittest.TestCase):
    def test_checked_in_config_uses_explicit_bicubic_resize(self) -> None:
        config = KyuteyeConfig.from_yml(REPO_ROOT / "configs" / "j-moshi-vis.yaml")
        processor = ImageProcessor.from_config(config, device="cpu")
        resize = processor.transform.transforms[0]

        self.assertEqual(processor.size, (448, 448))
        self.assertEqual(processor.interpolation, "bicubic")
        self.assertEqual(config.moshi_constructor_kwargs["xa_sigmoid_shift"], 4.0)
        self.assertEqual(resize.interpolation, transforms.InterpolationMode.BICUBIC)
        output = processor.transform(Image.new("RGB", (16, 24), color="white"))
        self.assertEqual(tuple(output.shape), (3, 448, 448))

    def test_invalid_interpolation_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported image interpolation"):
            ImageProcessor(device="cpu", interpolation="lanczos")  # type: ignore[arg-type]


class VisualAdapterInitializationTest(unittest.TestCase):
    def test_only_noncanonical_shared_mha_aliases_may_be_missing(self) -> None:
        self.assertTrue(
            _is_allowed_shared_mha_alias_missing(
                "llm.transformer.layers.1.cross_attention.mha.in_proj_weight"
            )
        )
        self.assertFalse(
            _is_allowed_shared_mha_alias_missing(
                "llm.transformer.layers.0.cross_attention.mha.in_proj_weight"
            )
        )
        self.assertFalse(
            _is_allowed_shared_mha_alias_missing(
                "llm.transformer.layers.1.cross_attention.gate.alpha.0.weight"
            )
        )
        self.assertFalse(
            _is_allowed_shared_mha_alias_missing(
                "llm.transformer.layers.1.norm_cross.alpha"
            )
        )

    def test_explicit_and_legacy_modes(self) -> None:
        self.assertEqual(
            _resolve_visual_adapter_init_mode(
                {"visual_adapter_init": "inherit_visual_adapter"}
            ),
            "inherit_visual_adapter",
        )
        self.assertEqual(
            _resolve_visual_adapter_init_mode({"inherit_from_pretrained": False}),
            "random_visual_adapter",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _resolve_visual_adapter_init_mode(
                {
                    "visual_adapter_init": "inherit_visual_adapter",
                    "inherit_from_pretrained": False,
                }
            )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _resolve_visual_adapter_init_mode({"visual_adapter_init": "partial"})

    def test_model_adapter_group_includes_norm_cross(self) -> None:
        self.assertTrue(
            _is_model_visual_adapter_key(
                "llm.transformer.layers.0.cross_attention.mha.in_proj_weight"
            )
        )
        self.assertTrue(
            _is_model_visual_adapter_key(
                "llm.transformer.layers.0.norm_cross.alpha"
            )
        )
        self.assertFalse(_is_model_visual_adapter_key("llm.text_emb.weight"))

    def test_random_mode_excludes_every_model_adapter_group(self) -> None:
        tensor = torch.ones(1)
        weights = {
            "module.llm.text_emb.weight": tensor,
            "module.llm.transformer.layers.0.cross_attention.gate.alpha.0.weight": tensor,
            "module.llm.transformer.layers.0.norm_cross.alpha": tensor,
            "module.image_prefix.enc.weight": tensor,
            "module.image_prefix.proj_xa.weight": tensor,
        }

        image_state, random_model_state, groups = _partition_training_checkpoint(
            weights, inherit_visual_adapter=False
        )
        _, inherited_model_state, _ = _partition_training_checkpoint(
            weights, inherit_visual_adapter=True
        )

        self.assertEqual(set(random_model_state), {"llm.text_emb.weight"})
        self.assertIn("llm.transformer.layers.0.cross_attention.gate.alpha.0.weight", inherited_model_state)
        self.assertIn("llm.transformer.layers.0.norm_cross.alpha", inherited_model_state)
        self.assertEqual(set(image_state), {"enc.weight", "proj_xa.weight"})
        self.assertEqual(len(groups["image_encoder"]), 1)
        self.assertEqual(len(groups["image_projection_and_norm"]), 1)

    def test_random_image_adapter_reset_preserves_encoder(self) -> None:
        class DummyProjection(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = torch.nn.Linear(2, 2, bias=False)
                self.proj_extra = None
                self.proj_xa = torch.nn.Linear(2, 2)
                self.norm_extra = None
                self.norm_xa = torch.nn.LayerNorm(2)

        projection = DummyProjection()
        for parameter in projection.parameters():
            torch.nn.init.constant_(parameter, 7.0)
        encoder_before = projection.enc.weight.detach().clone()

        _reset_image_projection_adapter(projection)  # type: ignore[arg-type]

        torch.testing.assert_close(projection.enc.weight, encoder_before)
        self.assertFalse(torch.all(projection.proj_xa.weight == 7.0))
        torch.testing.assert_close(
            projection.norm_xa.weight,
            torch.ones_like(projection.norm_xa.weight),
        )


if __name__ == "__main__":
    unittest.main()
