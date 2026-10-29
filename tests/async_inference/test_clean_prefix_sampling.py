"""Exercise the real PI05 sampler with a fake denoiser, without importing Transformers.

This is a CPU control-flow test, not a PI05 model forward validation. Extracting the
sampler lets protocol tests run even when the custom Transformers branch is absent.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F  # noqa: N812


def real_sampler():
    source = Path(__file__).parents[2] / "src/lerobot/policies/pi05/modeling_pi05.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "sample_actions",
            "pad_vector",
            "make_att_2d_masks",
        }:
            node.decorator_list = []
            node.returns = None
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
            selected.append(node)
    namespace = {"torch": torch, "F": F, "ACTION": "action"}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"),
        namespace,
    )
    return namespace["sample_actions"]


class CleanPrefixTests(unittest.TestCase):
    def setUp(self):
        self.times = []
        self.fake = SimpleNamespace(
            config=SimpleNamespace(
                num_inference_steps=3,
                chunk_size=4,
                max_action_dim=2,
                rtc_training_max_delay=2,
                output_features={"action": SimpleNamespace(shape=(1,))},
            ),
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
                forward=lambda **kwargs: (None, {}),
            ),
            rtc_processor=None,
            embed_prefix=lambda *args: (
                torch.zeros(1, 1, 2),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, dtype=torch.bool),
            ),
            _prepare_attention_masks_4d=lambda x: x,
            _rtc_enabled=lambda: False,
        )

        def denoise(**kwargs):
            self.times.append(kwargs["timestep"].clone())
            return torch.full_like(kwargs["x_t"], 100)

        self.fake.denoise_step = denoise
        self.sampler = real_sampler()

    def run_sampler(self, **kwargs):
        return self.sampler(
            self.fake,
            [],
            [],
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            noise=torch.ones(1, 4, 2),
            **kwargs,
        )

    def test_hard_clamp_each_step_and_time_condition(self):
        prefix = torch.tensor([[[0.25], [0.75]]])
        result = self.run_sampler(prefix_steps=2, action_prefix=prefix)
        self.assertTrue(torch.equal(result[:, :2, :1], prefix))
        self.assertTrue(torch.equal(result[:, :2, 1:], torch.zeros(1, 2, 1)))
        for time in self.times:
            self.assertEqual(tuple(time.shape), (1, 4))
            self.assertTrue(torch.equal(time[:, :2], torch.zeros(1, 2)))
            self.assertTrue((time[:, 2:] > 0).all())

    def test_bootstrap_and_invalid_prefix(self):
        result = self.run_sampler()
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaisesRegex(ValueError, "training_max_delay"):
            self.run_sampler(prefix_steps=3, action_prefix=torch.zeros(1, 3, 1))
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.run_sampler(prefix_steps=0, action_prefix=torch.zeros(1, 0, 1))
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.run_sampler(prefix_steps=1, action_prefix=torch.full((1, 1, 1), float("nan")))


if __name__ == "__main__":
    unittest.main()
