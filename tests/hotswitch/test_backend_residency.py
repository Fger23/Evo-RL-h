"""Actual tiny PEFT loading/numerics and independent storage, CPU only."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from torch import nn

from lerobot.async_inference.hotswitch.backend import (
    DualModelBackend,
    DualUnmergedBackend,
    SharedAdapterBackend,
    configure_inference_precision,
    inference_compute_precision,
)
from lerobot.configs.policies import PreTrainedConfig


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(1)

    def forward(self, inputs):
        return self.proj(inputs.to(self.proj.weight.dtype))

    def reset(self):
        pass

    def predict_action_chunk(self, batch, action_prefix=None, prefix_steps=0):
        action = self(batch["input"]).float().unsqueeze(1).repeat(1, 4, 1)
        if prefix_steps:
            action[:, :prefix_steps] = action_prefix
        return action


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = SimpleNamespace(dtype="bfloat16", device="original", compile_model=True,
                                      rtc_config="legacy", image_features={}, chunk_size=4)
        self.original_config = copy.deepcopy(vars(self.config))
        self.factory_dtypes = []
        self.processor_dtypes = []
        self.tasks = {}
        self.bundles = {}
        trained = get_peft_model(TinyPolicy(), LoraConfig(r=1, lora_alpha=1, lora_dropout=0, target_modules=["proj"]))
        trained.peft_config["default"].base_model_name_or_path = str(self.base)
        with torch.no_grad():
            trained.base_model.model.proj.lora_A["default"].weight.copy_(torch.tensor([[0.001, 0.0]]))
            trained.base_model.model.proj.lora_B["default"].weight.fill_(1)
        for task_id in ("A", "A_copy"):
            directory = self.root / task_id
            trained.save_pretrained(directory)
            save_file({"input": torch.tensor([[100.0, -100.0]])}, str(directory / "obs.safetensors"))
            self.tasks[task_id] = directory
            self.bundles[directory] = (directory, {
                "task_id": task_id, "task_text": "offline CPU fixture", "acp": {"enabled": False},
                "rtc": {"mode": "async", "max_delay": 0, "prefix_steps": 0},
                "action_schema": {"names": ["joint"], "units": ["fixture"], "control_hz": 30},
                "offline_observation": "obs.safetensors",
            })
            (directory / "manifest.json").write_text(json.dumps(self.bundles[directory][1]), encoding="utf-8")
        prior = inference_compute_precision()
        self.addCleanup(torch.set_float32_matmul_precision, prior["float32_matmul_precision"])
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", prior["cuda_matmul_allow_tf32"])
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", prior["cudnn_allow_tf32"])
        patches = (
            patch("lerobot.async_inference.hotswitch.backend.base_fingerprint", return_value={"same": "base"}),
            patch("lerobot.async_inference.hotswitch.backend.load_task_bundle", side_effect=lambda path, fingerprint: self.bundles[Path(path)]),
            patch("lerobot.async_inference.hotswitch.backend.check_compatible"),
            patch.object(PreTrainedConfig, "from_pretrained", return_value=self.config),
            patch("lerobot.policies.factory.get_policy_class", return_value=SimpleNamespace(from_pretrained=self.make_base)),
            patch("lerobot.policies.factory.make_pre_post_processors", side_effect=self.make_processors),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def make_base(self, path, *, config, local_files_only):
        self.assertEqual(Path(path), self.base)
        self.assertTrue(local_files_only)
        self.factory_dtypes.append(config.dtype)
        return TinyPolicy().to(dtype=getattr(torch, config.dtype))

    def make_processors(self, config, **kwargs):
        self.processor_dtypes.append(config.dtype)
        return lambda value: value, lambda value: value

    def test_independent_peft_never_merges_and_matches_shared_bf16(self):
        with patch("peft.tuners.lora.model.LoraModel.merge_and_unload", side_effect=AssertionError("must not merge")) as merge:
            shared = SharedAdapterBackend(self.base, self.tasks, device="cpu")
            dual = DualUnmergedBackend(self.base, self.tasks, device="cpu")
        merge.assert_not_called()
        self.assertEqual(shared.load_counts, {"base": 1, "adapter": 2, "warmup": 0})
        self.assertEqual(dual.load_counts, {"base": 2, "adapter": 2, "warmup": 0})
        self.assertEqual(dual.storage_report["mode"], "dual_independent_peft")
        self.assertTrue(dual.storage_report["independent_model_storages"])
        storages = [{p.untyped_storage().data_ptr() for p in model.parameters()} for model in dual.models.values()]
        self.assertFalse(storages[0] & storages[1])
        for task_id in dual.task_ids:
            shared.activate(task_id)
            dual.activate(task_id)
            self.assertEqual(dual.policy.active_adapters, [task_id])
            expected, _ = shared.infer(task_id, shared.offline_observation(task_id))
            actual, _ = dual.infer(task_id, dual.offline_observation(task_id))
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(vars(self.config), self.original_config)

    def test_dtype_override_reaches_base_factory_and_processors_without_mutating_config(self):
        backend = DualModelBackend(self.base, self.tasks, device="cpu", dtype="float32")
        self.assertEqual(self.factory_dtypes, ["float32", "float32"])
        self.assertEqual(self.processor_dtypes, ["float32", "float32"])
        self.assertEqual(vars(self.config), self.original_config)
        self.assertEqual(backend.storage_report["mode"], "dual_merged")
        self.assertEqual(backend.compute_precision, {"float32_matmul_precision": "highest",
                                                  "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False})
        for task_id, groups in backend.parameter_dtypes.items():
            self.assertEqual(set(groups["base"]), {"torch.float32"})
            self.assertEqual(groups["adapter"], {})
            self.assertEqual(backend.task_metadata[task_id]["inference_dtype"], "float32")
            self.assertTrue(all(not p.requires_grad for p in backend.models[task_id].parameters()))

    def test_default_preserves_precision_and_bf16_merge_can_fail_original_gate(self):
        prior = inference_compute_precision()
        shared = SharedAdapterBackend(self.base, self.tasks, device="cpu")
        merged = DualModelBackend(self.base, self.tasks, device="cpu")
        self.assertEqual(prior, inference_compute_precision())
        self.assertEqual(self.factory_dtypes, ["bfloat16"] * 3)
        expected, _ = shared.infer("A", shared.offline_observation("A"))
        actual, _ = merged.infer("A", merged.offline_observation("A"))
        self.assertFalse(torch.allclose(expected, actual, atol=0.02, rtol=0.01))
        self.assertEqual(vars(self.config), self.original_config)

    def test_independent_mode_rejects_shared_storage(self):
        backend = DualUnmergedBackend(self.base, self.tasks, device="cpu")
        backend.models["A_copy"] = backend.models["A"]
        with self.assertRaisesRegex(RuntimeError, "share parameter storage"):
            backend.parameter_storage_report()

    def test_independent_mode_checks_actual_active_adapter_before_commit(self):
        backend = DualUnmergedBackend(self.base, self.tasks, device="cpu")
        peer = backend.models["A_copy"]
        peer.base_model.active_adapter = ["A"]
        with patch.object(peer, "set_adapter"), self.assertRaisesRegex(RuntimeError, "did not activate"):
            backend.activate("A_copy")
        self.assertEqual(backend.active_task, "A")

    def test_invalid_dtype_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Inference dtype"):
            configure_inference_precision("float16")


if __name__ == "__main__":
    unittest.main()
