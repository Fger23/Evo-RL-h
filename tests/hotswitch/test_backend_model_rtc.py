"""Offline integration regression for PI05's real clean-prefix sampler."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, _install_tokenwise_adarms


class SamplerCore:
    def __init__(self):
        self.config = SimpleNamespace(num_inference_steps=3, rtc_training_max_delay=2,
                                      chunk_size=4, max_action_dim=3)
        self.rtc_processor = None
        self.times = []
        self.inputs = []
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
            forward=lambda **kwargs: ([None, None], None),
        )

    def _rtc_enabled(self):
        return False

    def embed_prefix(self, *args):
        return torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)

    def _prepare_attention_masks_4d(self, value):
        return value[:, None]

    def denoise_step(self, *, x_t, timestep, **kwargs):
        self.times.append(timestep.clone())
        self.inputs.append(x_t.clone())
        return torch.ones_like(x_t)


class CleanPrefixTests(unittest.TestCase):
    def test_cached_expert_norm_shim_matches_training_tokenwise_semantics(self):
        class LegacyNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(4, 12)
                self.cond_dim = 4

            def _norm(self, x):
                return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)

            def forward(self, x, cond=None):
                normed = self._norm(x)
                if cond is None:
                    return normed, None
                scale, shift, gate = self.dense(cond).unsqueeze(1).chunk(3, dim=-1)
                return normed * (1 + scale) + shift, gate

        norm = LegacyNorm()
        x = torch.randn(2, 5, 4)
        cond = torch.randn(2, 4)
        original_scalar, original_gate = norm(x, cond)
        names = list(norm.state_dict())
        _install_tokenwise_adarms(norm)
        _install_tokenwise_adarms(norm)  # idempotent on repeated initialization
        tokenwise, gate = norm(x, cond[:, None].expand(2, 5, 4))
        torch.testing.assert_close(tokenwise, original_scalar)
        torch.testing.assert_close(gate, original_gate.expand_as(gate))
        self.assertEqual(tokenwise.shape, x.shape)
        self.assertEqual(list(norm.state_dict()), names)

    def call(self, core, **kwargs):
        return PI05Pytorch.sample_actions(
            core, [], [], torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool),
            noise=torch.full((1, 4, 3), 5.0), **kwargs,
        )

    def test_prefix_hard_clamp_and_zero_time_at_every_flow_step(self):
        core = SamplerCore()
        prefix = torch.tensor([[[11.0, 12.0], [13.0, 14.0]]])
        output = self.call(core, action_prefix=prefix, prefix_steps=2)
        self.assertTrue(torch.equal(output[:, :2, :2], prefix))
        for time, noisy in zip(core.times, core.inputs, strict=True):
            self.assertTrue(torch.equal(time[:, :2], torch.zeros(1, 2)))
            self.assertTrue(torch.equal(noisy[:, :2, :2], prefix))
        torch.testing.assert_close(output[:, 2:], torch.full((1, 2, 3), 4.0))

    def test_no_prefix_preserves_ordinary_sampler(self):
        core = SamplerCore()
        output = self.call(core)
        torch.testing.assert_close(output, torch.full((1, 4, 3), 4.0))
        self.assertTrue(all(time.shape == (1,) for time in core.times))

    def test_invalid_range_shape_and_nonfinite_prefix_rejected(self):
        for kwargs in (
            {"prefix_steps": 3, "action_prefix": torch.zeros(1, 3, 2)},
            {"prefix_steps": 1, "action_prefix": None},
            {"prefix_steps": 0, "action_prefix": torch.zeros(1, 1, 2)},
            {"prefix_steps": 1, "action_prefix": torch.zeros(1, 2, 2)},
            {"prefix_steps": 1, "action_prefix": torch.full((1, 1, 2), float("nan"))},
        ):
            with self.assertRaises(ValueError):
                self.call(SamplerCore(), **kwargs)


class TinyCheckpointPolicy(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model = torch.nn.Linear(1, 1, bias=False)

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state


class CheckpointLoadingTests(unittest.TestCase):
    def load(self, state):
        with patch("transformers.utils.cached_file", return_value="offline_fixture"), patch(
            "safetensors.torch.load_file", return_value=state
        ):
            return PI05Policy.from_pretrained.__func__(
                TinyCheckpointPolicy, "/offline/base", config=SimpleNamespace(device="cpu"), strict=True
            )

    def test_strict_checkpoint_mismatch_raises_instead_of_random_policy(self):
        with self.assertRaisesRegex(RuntimeError, "restoration failed"):
            self.load({"incorrect_weight": torch.ones(1, 1)})

    def test_load_failure_raises_instead_of_random_policy(self):
        with (
            patch("transformers.utils.cached_file", side_effect=FileNotFoundError("offline missing")),
            self.assertRaisesRegex(RuntimeError, "restoration failed"),
        ):
            PI05Policy.from_pretrained.__func__(
                TinyCheckpointPolicy, "/missing/base", config=SimpleNamespace(device="cpu"), strict=True
            )

    def test_compatible_strict_checkpoint_loads_real_weights(self):
        policy = self.load({"weight": torch.full((1, 1), 7.0)})
        self.assertEqual(float(policy.model.weight.detach()[0, 0]), 7.0)


if __name__ == "__main__":
    unittest.main()
