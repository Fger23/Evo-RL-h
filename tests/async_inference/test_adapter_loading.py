"""CPU checks of the additional Adapter loader; no robot, CUDA or PI05 model."""

import copy
import json

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file

from lerobot.policies.pi05 import adapter_inference as loader


def saved_config():
    return {
        "type": "pi05",
        "device": "cuda:0",
        "use_peft": True,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [12]},
            **{
                f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 480, 640]}
                for name in loader.CAMERAS
            },
        },
        "output_features": {"action": {"type": "ACTION", "shape": [12]}},
        "rtc_training_max_delay": 37,
        "tokenizer_name": "/original/tokenizer",
        "gradient_checkpointing": True,
        "compile_model": False,
        "rtc_config": None,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
    }


def test_saved_config_decodes_features_without_changing_native_math():
    saved = saved_config()
    original = copy.deepcopy(saved)
    cfg = loader._native_config(saved)
    assert saved == original
    assert cfg.device == "cpu" and cfg.use_peft is False
    assert cfg.input_features["observation.state"].shape == (12,)
    assert cfg.output_features["action"].shape == (12,)
    assert cfg.max_delay == 37 and cfg.action_conditioning is True
    assert cfg.rtc_config is None and cfg.gradient_checkpointing is False
    assert cfg.num_inference_steps == 10 and cfg.dtype == "bfloat16"


def test_config_unknown_field_is_rejected():
    saved = saved_config()
    saved["unrecognized_training_fork_option"] = True
    with pytest.raises(Exception):
        loader._native_config(saved)


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        self.model = torch.nn.Module()
        self.model.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.model.alias = self.model.weight

    def _fix_pytorch_state_dict_keys(self, state, config):
        assert config is self.config
        return state


def test_strict_base_restores_only_genuine_tied_alias():
    base = TinyBase()
    value = torch.ones(2, 2)
    loader._restore_base(base, {"weight": value})
    assert torch.equal(base.model.weight, value)
    assert base.model.weight is base.model.alias


@pytest.mark.parametrize(
    "state", [{}, {"weight": torch.ones(3, 3)}, {"weight": torch.ones(2, 2), "extra": torch.zeros(1)}]
)
def test_base_missing_wrong_shape_and_unexpected_keys_fail(state):
    with pytest.raises(RuntimeError):
        loader._restore_base(TinyBase(), state)


def test_inconsistent_tied_base_weights_fail():
    with pytest.raises(ValueError, match="inconsistent tied"):
        loader._restore_base(TinyBase(), {"weight": torch.ones(2, 2), "alias": torch.zeros(2, 2)})


def test_conflicting_native_remap_alias_fails():
    with pytest.raises(ValueError, match="Conflicting Base aliases"):
        loader._restore_base(TinyBase(), {"weight": torch.ones(2, 2), "model.weight": torch.zeros(2, 2)})


def test_real_cpu_peft_roundtrip_and_missing_corrupt_state(tmp_path):
    base = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    peft = get_peft_model(base, LoraConfig(target_modules=["0", "1"], r=2, lora_alpha=4))
    with torch.no_grad():
        for name, value in peft.named_parameters():
            if "lora_" in name:
                value.fill_(0.125)
    expected = {key: value.detach().clone() for key, value in get_peft_model_state_dict(peft).items()}
    peft.save_pretrained(tmp_path)
    restored = PeftModel.from_pretrained(
        torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)), tmp_path, is_trainable=False
    )
    loader._verify_adapter_state(restored, expected)
    missing = dict(expected)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="tensor keys"):
        loader._verify_adapter_state(restored, missing)
    corrupt = {key: value.clone() for key, value in expected.items()}
    corrupt[next(iter(corrupt))].add_(1)
    with pytest.raises(ValueError, match="not restored exactly"):
        loader._verify_adapter_state(restored, corrupt)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Small schema fixture; substitute only the production checkpoint SHA pin."""
    root = tmp_path / "A"
    base = tmp_path / "base"
    root.mkdir()
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"fake base for file-gate tests only")
    (base / "config.json").write_text("{}")
    (root / "config.json").write_text(json.dumps(saved_config()))
    adapter_config = {
        "base_model_name_or_path": str(base),
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 64,
        "bias": "none",
        "modules_to_save": [],
        "target_modules": sorted(loader._expected_targets()),
    }
    (root / "adapter_config.json").write_text(json.dumps(adapter_config))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (root / name).write_text(json.dumps({"steps": [{"registry_name": "fixture_processor"}]}))
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer.json").write_text("{}")
    tensors = {
        f"base_model.model.{target}.lora_{side}.weight": torch.zeros((32, 1) if side == "A" else (1, 32))
        for target in adapter_config["target_modules"]
        for side in ("A", "B")
    }
    save_file(tensors, root / "adapter_model.safetensors")
    pin = loader._sha256(root / "adapter_model.safetensors")
    monkeypatch.setattr(loader, "FORMAL_ADAPTER_SHA256", pin)
    artifacts = {
        path.relative_to(root).as_posix(): loader._sha256(path) for path in root.rglob("*") if path.is_file()
    }
    manifest = {
        "task_id": "A",
        "artifacts": artifacts,
        "training_source": {"steps": 160000, "formal_training_completed": True, "mechanism_copy": False},
        "base_fingerprint": {
            "files": {name: loader._sha256(base / name) for name in ("model.safetensors", "config.json")}
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, base


def test_complete_bundle_gate_and_same_base_relocation(bundle, tmp_path):
    root, base = bundle
    checked = loader.check_adapter_files(root)
    assert checked["target_count"] == 76 and checked["tensor_count"] == 152
    assert checked["tokenizer_path"] == str(root / "tokenizer")
    assert checked["config"]["tokenizer_name"] == "/original/tokenizer"
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for path in base.iterdir():
        (relocated / path.name).write_bytes(path.read_bytes())
    assert loader.check_adapter_files(root, relocated)["base_path"] == str(relocated)


def test_corrupt_saved_processor_or_base_is_rejected(bundle):
    root, base = bundle
    (root / "policy_preprocessor.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact SHA256"):
        loader.check_adapter_files(root)


def test_wrong_base_bytes_are_rejected(bundle):
    root, base = bundle
    (base / "model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="Base SHA256"):
        loader.check_adapter_files(root)


def test_production_adapter_sha_pin_is_enforced(bundle, monkeypatch):
    root, base = bundle
    monkeypatch.setattr(loader, "FORMAL_ADAPTER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="formal A160000 checkpoint"):
        loader.check_adapter_files(root)
