"""Hardware-free package integrity and actual PI05 target-path tests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lerobot.async_inference.task_bundle import (
    TaskBundle,
    audit_targets,
    content_fingerprint,
    export_task,
    validate_task_pair,
)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"fake weights for integrity testing")
        (self.base / "config.json").write_text("{}")
        self.source = self.root / "checkpoint"
        self.source.mkdir()
        (self.source / "adapter_model.safetensors").write_bytes(b"simulated adapter")
        (self.source / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": str(self.base),
                    "peft_type": "LORA",
                    "r": 32,
                    "lora_alpha": 64,
                    "modules_to_save": [],
                }
            )
        )
        (self.source / "config.json").write_text(json.dumps({"rtc_training_max_delay": 3, "chunk_size": 8}))
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (self.source / name).write_text("{}")
        self.meta = {
            "task_id": "A",
            "task_text": "test",
            "action_schema": {"names": ["joint"], "units": ["rad"], "fps": 30},
            "tokenizer_fingerprint": "test-tokenizer",
            "camera_mapping": {},
            "rtc": {"mode": "training_time", "prefix_steps": 2},
            "acp": {"enabled": False},
            "training_source": {"test": True},
            "target_layers": ["language", "expert"],
            "lerobot_features": {},
        }

    def export(self, task_id="A", **overrides):
        metadata = dict(self.meta, task_id=task_id, **overrides)
        return export_task(self.source, self.root / task_id, metadata, self.base)

    def test_fingerprint_content_and_path_independence(self):
        import shutil

        clone = self.root / "renamed"
        shutil.copytree(self.base, clone)
        self.assertEqual(content_fingerprint(clone), content_fingerprint(self.base))
        (clone / "model.safetensors").write_bytes(b"different")
        self.assertNotEqual(content_fingerprint(clone), content_fingerprint(self.base))

    def test_export_load_and_compatible_pair(self):
        a, b = TaskBundle.load(self.export()), TaskBundle.load(self.export("B"))
        validate_task_pair([a, b])
        self.assertEqual(a.manifest["base_fingerprint"], content_fingerprint(self.base))

    def test_tamper_rejected(self):
        path = self.export()
        (path / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            TaskBundle.load(path)

    def test_base_mismatch_rejected(self):
        other = self.root / "other"
        other.mkdir()
        (other / "model.safetensors").write_bytes(b"not same base")
        with self.assertRaisesRegex(ValueError, "different Base"):
            export_task(self.source, self.root / "bad", self.meta, other)

    def test_action_order_incompatibility(self):
        a = TaskBundle.load(self.export())
        b = TaskBundle.load(
            self.export("B", action_schema={"names": ["different"], "units": ["rad"], "fps": 30})
        )
        with self.assertRaisesRegex(ValueError, "action_schema"):
            validate_task_pair([a, b])

    def test_training_prefix_range_and_acp_tag_rejected(self):
        with self.assertRaisesRegex(ValueError, "training range"):
            self.export(rtc={"mode": "training_time", "prefix_steps": 4})
        with self.assertRaisesRegex(ValueError, "untagged"):
            self.export("B", task_text="test\nAdvantage: positive")

    def test_joint_target_actual_paths(self):
        names = [
            "model.paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj",
            "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
            *[f"model.{p}" for p in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")],
            "model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        ]
        policy = SimpleNamespace(named_modules=lambda: [(name, object()) for name in names])
        report = audit_targets(policy)
        self.assertEqual(len(report["targets"]), 6)
        self.assertEqual(report["groups"]["vision"], [])
        self.assertEqual(len(audit_targets(policy, vision=True)["groups"]["vision"]), 1)
        missing = SimpleNamespace(named_modules=lambda: [(n, object()) for n in names[1:]])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit_targets(missing)


if __name__ == "__main__":
    unittest.main()
