"""Offline content-integrity and compatibility tests; no hardware imports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lerobot.async_inference.hotswitch.manifest import (
    base_fingerprint,
    canonical_digest,
    check_compatible,
    export_task,
    load_task_bundle,
)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.config = {"type": "pi05", "chunk_size": 4, "rtc_training_max_delay": 0,
                       "output_features": {"action": {"shape": [2]}}, "tokenizer_max_length": 200}
        (self.base / "config.json").write_text(json.dumps(self.config))
        (self.base / "model.safetensors").write_bytes(b"real-content-not-a-filename")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        (self.checkpoint / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": str(self.base), "r": 32, "lora_alpha": 64,
            "peft_type": "LORA", "target_modules": ["v", "e", "a"], "bias": "none", "modules_to_save": [],
        }))
        (self.checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-content")
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (self.checkpoint / filename).write_text(json.dumps({"steps": [{"registry_name": "normalizer", "state_file": "stats.safetensors"}]}))
        (self.checkpoint / "stats.safetensors").write_bytes(b"normalization-content")
        (self.checkpoint / "tokenizer").mkdir()
        (self.checkpoint / "tokenizer" / "tokenizer.json").write_text('{"test": "tokenizer content"}')
        (self.checkpoint / "tokenizer" / "tokenizer_config.json").write_text('{"tokenizer_class": "FixtureTokenizer"}')
        self.targets = self.root / "targets.json"
        self.targets.write_text(json.dumps({"vlm": ["v"], "expert": ["e"], "projections": ["a"], "target_modules": ["v", "e", "a"]}))
        self.offline = self.root / "obs.safetensors"
        self.offline.write_bytes(b"offline-observation")
        self.metadata = {"task_id": "A", "task_text": "straighten cash", "action_schema": {
            "names": ["left", "right"], "units": ["degrees", "degrees"], "control_hz": 30},
            "rtc": {"mode": "async", "max_delay": 0, "prefix_steps": 0},
            "acp": {"enabled": False}, "camera_map": {}, "tokenizer": {"name": "local", "max_length": 200}}

    def tearDown(self):
        self.tmp.cleanup()

    def export(self):
        destination = self.root / "A"
        export_task(self.checkpoint, self.base, destination, self.metadata, self.targets, self.offline)
        return destination

    def test_bundle_contains_processor_state_and_validates(self):
        destination = self.export()
        _, manifest = load_task_bundle(destination, base_fingerprint(self.base))
        self.assertIn("stats.safetensors", manifest["artifacts"])

    def test_new_export_requires_tokenizer_config_without_partial_bundle(self):
        (self.checkpoint / "tokenizer/tokenizer_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires tokenizer_config"):
            self.export()
        self.assertFalse((self.root / "A").exists())

    def test_legacy_tokenizer_json_only_bundle_still_has_integrity_validation(self):
        destination = self.export()
        (destination / "tokenizer/tokenizer_config.json").unlink()
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["artifacts"]["tokenizer/tokenizer_config.json"]
        manifest["tokenizer"]["sha256"] = canonical_digest({name: digest for name, digest in manifest["artifacts"].items()
                                                             if name.startswith("tokenizer/")})
        path.write_text(json.dumps(manifest))
        load_task_bundle(destination)
        (destination / "tokenizer/tokenizer.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_task_bundle(destination)

    def test_benchmark_provenance_binds_actual_bundle_and_rejects_changed_adapter(self):
        from lerobot.async_inference.hotswitch.manifest import digest_file
        from lerobot.scripts.hotswitch_cli import _benchmark_input_provenance

        destination = self.export()
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"base_path": str(self.base), "tasks": {"A": str(destination)}}))
        source_manifest = self.root / "source_snapshot_manifest.json"
        source_manifest.write_text('{"cpu_fixture": true}')
        args = SimpleNamespace(registry=str(registry), source_manifest=str(source_manifest))
        evidence = _benchmark_input_provenance(args)
        self.assertEqual(evidence["registry_sha256"], digest_file(registry))
        self.assertEqual(evidence["bundle_manifests"]["A"]["sha256"], digest_file(destination / "manifest.json"))
        self.assertEqual(evidence["source_manifest_sha256"], digest_file(source_manifest))
        self.assertIn("adapter_model.safetensors", evidence["bundle_manifests"]["A"]["artifacts"])
        (destination / "adapter_model.safetensors").write_bytes(b"changed adapter after measurement setup")
        with self.assertRaisesRegex(ValueError, "checksum"):
            _benchmark_input_provenance(args)

    def test_base_content_drift_is_rejected(self):
        destination = self.export()
        (self.base / "model.safetensors").write_bytes(b"other-base-same-path")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_task_bundle(destination, base_fingerprint(self.base))

    def test_adapter_and_normalization_corruption_is_rejected(self):
        for name in ("adapter_model.safetensors", "stats.safetensors"):
            destination = self.export()
            (destination / name).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_task_bundle(destination)
            import shutil
            shutil.rmtree(destination)

    def test_different_action_order_is_incompatible(self):
        first = load_task_bundle(self.export())[1]
        second = dict(first, action_schema={**first["action_schema"], "names": ["right", "left"]})
        with self.assertRaisesRegex(ValueError, "action_schema"):
            check_compatible([first, second])

    def camera_bundle(self):
        cameras = ("observation.images.left_fixed", "observation.images.left_wrist", "observation.images.right_wrist")
        self.config["input_features"] = {
            "observation.state": {"type": "STATE", "shape": [2]},
            **{name: {"type": "VISUAL", "shape": [3, 480, 640]} for name in cameras},
        }
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        self.metadata["camera_map"] = {name: name for name in cameras}
        return self.export()

    def test_equal_feature_dicts_with_different_camera_order_are_incompatible(self):
        first = load_task_bundle(self.camera_bundle())[1]
        second = copy.deepcopy(first)
        second["model_structure"]["input_features"] = dict(reversed(list(first["model_structure"]["input_features"].items())))
        # Ordinary dict comparison cannot distinguish this unsafe camera order.
        self.assertEqual(first["model_structure"], second["model_structure"])
        with self.assertRaisesRegex(ValueError, "ordered VISUAL"):
            check_compatible([first, second])

    def test_manifest_camera_order_must_match_its_actual_policy_config(self):
        destination = self.camera_bundle()
        manifest_file = destination / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        features = manifest["model_structure"]["input_features"]
        manifest["model_structure"]["input_features"] = dict(reversed(list(features.items())))
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "camera order"):
            load_task_bundle(destination)

    def test_control_frequency_rejects_nan_infinity_and_nonpositive_values(self):
        destination = self.export()
        manifest_file = destination / "manifest.json"
        original = json.loads(manifest_file.read_text())
        for frequency in (float("nan"), float("inf"), float("-inf"), 0, -1, True):
            with self.subTest(frequency=frequency):
                manifest = copy.deepcopy(original)
                manifest["action_schema"]["control_hz"] = frequency
                manifest_file.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    load_task_bundle(destination)

    def test_pre_tagged_acp_is_rejected(self):
        self.metadata["task_text"] += "\nAdvantage: positive"
        with self.assertRaisesRegex(ValueError, "untagged"):
            self.export()

    def test_acp_requires_explicit_positive_single_branch_beta_one(self):
        self.metadata["acp"] = {"enabled": True, "prompt_strategy": "positive", "cfg_enabled": False, "beta": 1.0}
        destination = self.export()
        manifest_file = destination / "manifest.json"
        original = json.loads(manifest_file.read_text())
        for options in ({"cfg_enabled": True, "beta": 1}, {"cfg_enabled": False, "beta": 2},
                        {"cfg_enabled": False, "beta": float("nan")}, {"cfg_enabled": False},
                        {"beta": 1}, {"cfg_enabled": 0, "beta": 1}, {"cfg_enabled": False, "beta": True}):
            with self.subTest(options=options):
                manifest = copy.deepcopy(original)
                manifest["acp"] = {"enabled": True, "prompt_strategy": "positive", **options}
                manifest_file.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "ACP deployment"):
                    load_task_bundle(destination)

    def test_manifest_cannot_reference_outside_package(self):
        destination = self.export()
        manifest_file = destination / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["artifacts"]["../obs.safetensors"] = "wrong"
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_task_bundle(destination)


if __name__ == "__main__":
    unittest.main()
