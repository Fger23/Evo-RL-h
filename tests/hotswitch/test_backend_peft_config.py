"""Parse new LoRA controls and audit both branches on a tiny offline module."""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import draccus
import torch
from torch import nn

from lerobot.configs.default import PeftConfig
from lerobot.optim.factory import _peft_trainable_params, make_optimizer_and_scheduler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.scripts.hotswitch_training import collate_smoke_sample, joint_targets, predict_with_target_hooks


def tiny_policy():
    def layer():
        attention = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attention, name, nn.Linear(2, 2))
        mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, name, nn.Linear(2, 2))
        return nn.ModuleDict({"self_attn": attention, "mlp": mlp})

    body = nn.Module()
    body.paligemma_with_expert = nn.Module()
    body.paligemma_with_expert.paligemma = nn.Module()
    body.paligemma_with_expert.paligemma.language_model = nn.Module()
    body.paligemma_with_expert.paligemma.language_model.layers = nn.ModuleList([layer()])
    body.paligemma_with_expert.gemma_expert = nn.Module()
    body.paligemma_with_expert.gemma_expert.model = nn.Module()
    body.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList([layer()])
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        setattr(body, name, nn.Linear(2, 2))
    policy = nn.Module()
    policy.model = body
    return policy


class PeftConfigTests(unittest.TestCase):
    def test_checkpoint_tokenizer_copy_precedes_external_config_source(self):
        config = PI05Config(device="cpu", tokenizer_name="/external/tokenizer")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "tokenizer").mkdir()
            with patch("lerobot.policies.factory.PolicyProcessorPipeline", SimpleNamespace(from_pretrained=lambda **kwargs: kwargs)):
                preprocessor, _ = make_pre_post_processors(config, pretrained_path=checkpoint)
            self.assertEqual(preprocessor["overrides"]["tokenizer_processor"]["tokenizer_name"],
                             str((checkpoint / "tokenizer").resolve()))

    def test_explicit_tokenizer_override_precedes_checkpoint_copy(self):
        config = PI05Config(device="cpu", tokenizer_name="/external/tokenizer")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "tokenizer").mkdir()
            with patch("lerobot.policies.factory.PolicyProcessorPipeline", SimpleNamespace(from_pretrained=lambda **kwargs: kwargs)):
                preprocessor, _ = make_pre_post_processors(
                    config, pretrained_path=checkpoint,
                    preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": "/explicit/tokenizer"}},
                )
            self.assertEqual(preprocessor["overrides"]["tokenizer_processor"]["tokenizer_name"], "/explicit/tokenizer")

    def test_inference_hooks_prove_all_targets_and_are_removed(self):
        model = nn.ModuleDict({"vlm": nn.Linear(2, 2), "expert": nn.Linear(2, 2)})
        policy = SimpleNamespace(get_base_model=lambda: model,
                                 predict_action_chunk=lambda batch: model["expert"](model["vlm"](batch)))
        _, counts = predict_with_target_hooks(policy, torch.zeros(1, 2), {"target_modules": ["vlm", "expert"]})
        self.assertEqual(counts, {"vlm": 1, "expert": 1})
        self.assertFalse(model["vlm"]._forward_hooks)
        self.assertFalse(model["expert"]._forward_hooks)

    def test_inference_target_bypass_fails_and_still_removes_hooks(self):
        model = nn.ModuleDict({"vlm": nn.Linear(2, 2), "expert": nn.Linear(2, 2)})
        policy = SimpleNamespace(get_base_model=lambda: model,
                                 predict_action_chunk=lambda batch: model["expert"](batch))
        with self.assertRaisesRegex(AssertionError, "bypassed"):
            predict_with_target_hooks(policy, torch.zeros(1, 2), {"target_modules": ["vlm", "expert"]})
        self.assertFalse(model["vlm"]._forward_hooks)
        self.assertFalse(model["expert"]._forward_hooks)

    def test_smoke_collates_action_horizon_as_real_training_batch(self):
        batch = collate_smoke_sample({"action": torch.zeros(50, 12), "observation.state": torch.zeros(12), "task": "straighten cash"})
        self.assertEqual(tuple(batch["action"].shape), (1, 50, 12))
        self.assertEqual(batch["task"], ["straighten cash"])

    def test_peft_optimizer_keeps_only_adapters_and_group_options(self):
        model = nn.Linear(2, 2)
        model.weight.requires_grad_(False)
        model.get_optim_params = lambda: [
            {"params": [model.weight], "weight_decay": 1.0},
            {"params": [model.bias], "weight_decay": 0.0},
        ]
        cfg = SimpleNamespace(peft=PeftConfig(), use_policy_training_preset=True, steps=1,
                              optimizer=SimpleNamespace(build=lambda params: torch.optim.AdamW(params, lr=0.01)),
                              scheduler=None)
        optimizer, _ = make_optimizer_and_scheduler(cfg, model)
        self.assertEqual([id(p) for group in optimizer.param_groups for p in group["params"]], [id(model.bias)])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_peft_optimizer_rejects_missing_and_duplicate_adapters(self):
        model = nn.Linear(2, 2)
        for params in ([model.weight], [model.weight, model.bias, model.bias]):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                _peft_trainable_params(model, params)

    def test_alpha_dropout_parse_and_peft_forwarding(self):
        config = draccus.decode(PeftConfig, {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0})
        policy = object.__new__(PI05Policy)
        nn.Module.__init__(policy)
        policy.config = SimpleNamespace(pretrained_path="/explicit/base")
        result = policy._build_peft_config(dataclasses.asdict(config))
        self.assertEqual(result.r, 32)
        self.assertEqual(result.lora_alpha, 64)
        self.assertEqual(result.lora_dropout, 0.0)
        self.assertEqual(result.modules_to_save, [])
        target = result.target_modules
        self.assertIn("paligemma", target)
        self.assertIn("gemma_expert", target)
        self.assertIn("time_mlp_in", target)

    def test_joint_and_extended_audits_cover_real_module_paths(self):
        model = tiny_policy()
        narrow = joint_targets(model)
        broad = joint_targets(model, extended=True)
        self.assertEqual([len(narrow[key]) for key in ("vlm", "expert", "projections")], [2, 2, 4])
        self.assertEqual([len(broad[key]) for key in ("vlm", "expert", "projections")], [7, 7, 4])
        self.assertEqual(narrow["vision"], [])

    def test_missing_expert_fails_audit(self):
        model = tiny_policy()
        model.model.paligemma_with_expert.gemma_expert = nn.Module()
        with self.assertRaisesRegex(ValueError, "audit failed"):
            joint_targets(model)

    def test_custom_transformers_paligemma_model_wrapper(self):
        model = tiny_policy()
        paligemma = model.model.paligemma_with_expert.paligemma
        inner = nn.Module()
        inner.language_model = paligemma.language_model
        del paligemma.language_model
        paligemma.model = inner
        targets = joint_targets(model)
        self.assertEqual(len(targets["vlm"]), 2)
        self.assertTrue(all("paligemma.model.language_model" in name for name in targets["vlm"]))


if __name__ == "__main__":
    unittest.main()
