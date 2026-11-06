"""Load the exported A160000 LoRA with native PI05 inference, without merging.

This module is used only by the optional Adapter server entry.  The upstream
PI05 policy, processor implementations, and the async inference wire protocol
remain unchanged.  All paths refer to local server files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

CAMERAS = ("left_fixed", "left_wrist", "right_wrist")
FORMAL_ADAPTER_SHA256 = "ae2253b81accf9d2f417f3d720137ddd5ca6f5c4809f576cdf779352f6574f22"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Artifact must be a local file within {root}: {name}")
    return path


def _tensor_header(path: Path) -> dict:
    """Inspect the tensor inventory without importing torch or creating a model."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def _expected_targets() -> set[str]:
    targets = {
        f"model.{name}" for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    }
    for branch in ("paligemma.model.language_model", "gemma_expert.model"):
        for index in range(18):
            for projection in ("q_proj", "v_proj"):
                targets.add(f"model.paligemma_with_expert.{branch}.layers.{index}.self_attn.{projection}")
    return targets


def check_adapter_files(path: str | Path, base_override: str | Path | None = None) -> dict:
    """Validate the formal exported bundle and its Base, without GPU/model imports.

    The export contains the exact training processors, tokenizer, and hashes.
    A bare ``adapter_model.safetensors`` is deliberately insufficient.
    ``base_override`` permits relocating the same Base bytes; it does not permit
    substituting a different model.
    """
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Adapter path must be the exported bundles/A directory")
    manifest = _json(_local_file(root, "manifest.json"))
    source = manifest.get("training_source", {})
    if (
        manifest.get("task_id") != "A"
        or source.get("steps") != 160000
        or source.get("formal_training_completed") is not True
        or source.get("mechanism_copy") is not False
    ):
        raise ValueError("Expected the completed formal A160000 export, not a pilot or A_copy")
    artifacts = manifest.get("artifacts")
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("Use the complete exported Adapter bundle with saved processors and hashes")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or _sha256(_local_file(root, name)) != expected:
            raise ValueError(f"Adapter bundle artifact SHA256 mismatch: {name}")
    if artifacts["adapter_model.safetensors"] != FORMAL_ADAPTER_SHA256:
        raise ValueError("Adapter weights differ from the formal A160000 checkpoint")

    config = _json(root / "config.json")
    expected_inputs = {"observation.state": {"type": "STATE", "shape": [12]}}
    expected_inputs.update(
        {f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 480, 640]} for name in CAMERAS}
    )
    if (
        config.get("type") != "pi05"
        or config.get("input_features") != expected_inputs
        or config.get("output_features") != {"action": {"type": "ACTION", "shape": [12]}}
        or config.get("chunk_size") != 50
        or config.get("max_action_dim") != 32
        or config.get("max_state_dim") != 32
        or config.get("dtype") != "bfloat16"
        or config.get("paligemma_variant") != "gemma_2b"
        or config.get("action_expert_variant") != "gemma_300m"
    ):
        raise ValueError("Adapter is not the expected 12D, three-camera PI05 A160000 schema")
    if config.get("rtc_training_max_delay") != 37:
        raise ValueError("Adapter training RTC metadata differs from A160000")

    adapter = _json(root / "adapter_config.json")
    targets = adapter.get("target_modules", [])
    if (
        not isinstance(targets, list)
        or len(targets) != 76
        or set(targets) != _expected_targets()
        or adapter.get("peft_type") != "LORA"
        or adapter.get("r") != 32
        or adapter.get("lora_alpha") != 64
        or adapter.get("bias") != "none"
        or adapter.get("modules_to_save") not in (None, [])
        or adapter.get("use_dora")
        or adapter.get("use_rslora")
        or adapter.get("rank_pattern")
        or adapter.get("alpha_pattern")
    ):
        raise ValueError("Adapter must use its saved 76 joint LoRA targets, rank32 and alpha64")
    header = _tensor_header(root / "adapter_model.safetensors")
    expected_keys = {
        f"base_model.model.{target}.lora_{side}.weight" for target in targets for side in ("A", "B")
    }
    if set(header) != expected_keys:
        raise ValueError("Adapter tensor inventory differs from the saved 76 LoRA targets")
    for key, entry in header.items():
        shape = entry.get("shape", [])
        if (
            len(shape) != 2
            or entry.get("dtype") != "F32"
            or (shape[0] if ".lora_A." in key else shape[1]) != 32
        ):
            raise ValueError(f"Invalid Adapter tensor shape or dtype: {key}")

    base = Path(base_override or adapter["base_model_name_or_path"]).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError("Base must be a local pretrained model directory")
    fingerprint = manifest.get("base_fingerprint", {}).get("files", {})
    if set(fingerprint) != {"model.safetensors", "config.json"}:
        raise ValueError("Bundle has no exact Base weights/config fingerprint")
    for name, expected in fingerprint.items():
        if _sha256(_local_file(base, name)) != expected:
            raise ValueError(f"Base SHA256 mismatch: {name}")
    tokenizer = root / "tokenizer"
    # The additional server entry overrides the saved tokenizer_name with this
    # hash-checked directory.  The historical external tokenizer path is neither
    # changed on disk nor used to load the inference tokenizer.
    if not tokenizer.is_dir() or "tokenizer/tokenizer.json" not in artifacts:
        raise ValueError("Exported Adapter tokenizer is missing")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        steps = _json(root / name).get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Saved processor has no steps: {name}")
        for step in steps:
            state_file = step.get("state_file")
            if state_file and state_file not in artifacts:
                raise ValueError(f"Saved processor state is not hash-bound: {state_file}")
    return {
        "adapter_path": str(root),
        "base_path": str(base),
        "tokenizer_path": str(tokenizer),
        "config": config,
        "manifest": manifest,
        "adapter_config": adapter,
        "target_count": len(targets),
        "tensor_count": len(header),
        "manifest_sha256": _sha256(root / "manifest.json"),
    }


def _native_config(saved: dict):
    import draccus

    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    payload = copy.deepcopy(saved)
    if payload.pop("type") != "pi05":
        raise ValueError("Only PI05 Adapter configs are supported")
    # These are the two fields added by the earlier training fork.  The native
    # sampler has no clean-prefix interface; no RTC math is patched here.
    delay = payload.pop("rtc_training_max_delay")
    payload.pop("tokenizer_name")  # the server overrides this with bundle/tokenizer
    payload.update(
        device="cpu",
        use_peft=False,
        compile_model=False,
        gradient_checkpointing=False,
        rtc_config=None,
        action_conditioning=bool(delay),
        max_delay=delay,
    )
    # Decode nested PolicyFeature/enum values; reject every other unknown field.
    return draccus.decode(PI05Config, payload)


def _restore_base(policy, state: dict) -> None:
    """Use the upstream remapper, then require complete Base restoration."""
    fixed = policy._fix_pytorch_state_dict_keys(state, policy.config)
    remapped = {}
    for key, value in fixed.items():
        new_key = key if key.startswith("model.") else f"model.{key}"
        if new_key in remapped and not remapped[new_key].equal(value):
            raise ValueError(f"Conflicting Base aliases after native key remapping: {new_key}")
        remapped[new_key] = value
    # Safetensors may omit an alias of the tied language embedding/lm_head.
    # Fill only genuinely shared storage aliases, never unrelated missing keys.
    groups = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    for names in groups.values():
        present = [name for name in names if name in remapped]
        if len(names) > 1 and present:
            value = remapped[present[0]]
            if any(not remapped[name].equal(value) for name in present[1:]):
                raise ValueError(f"Base checkpoint has inconsistent tied weights: {present}")
            for name in names:
                remapped.setdefault(name, value)
    policy.load_state_dict(remapped, strict=True)


def _verify_adapter_state(model, expected: dict) -> None:
    import torch
    from peft import get_peft_model_state_dict

    actual = get_peft_model_state_dict(model)
    if set(actual) != set(expected):
        raise ValueError("Loaded PEFT tensor keys differ from saved Adapter; refusing inference")
    for key, value in expected.items():
        loaded = actual[key].detach().cpu()
        if loaded.shape != value.shape or not torch.isfinite(loaded).all():
            raise ValueError(f"Loaded Adapter tensor has an invalid shape/value: {key}")
        if not torch.equal(loaded, value.to(dtype=loaded.dtype)):
            raise ValueError(f"Adapter weight was not restored exactly: {key}")


def load_adapter_policy(path: str | Path, device: str, base_override: str | Path | None = None):
    """Return an eval, frozen, unmerged native PI05+PEFT policy on CPU.

    The native server subsequently calls ``policy.to(device)`` as usual.  Base
    load errors propagate; the original PI05 ``from_pretrained`` fallback to a
    randomly initialized model is intentionally not used by this new entry.
    """
    from peft import PeftModel
    from safetensors.torch import load_file

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checked = check_adapter_files(path, base_override)
    logger.info(
        "Loading native PI05 Base on CPU; inference target=%s, Adapter=%s, targets=%d",
        device,
        checked["adapter_path"],
        checked["target_count"],
    )
    config = _native_config(checked["config"])
    policy = PI05Policy(config)
    base_state = load_file(str(Path(checked["base_path"]) / "model.safetensors"), device="cpu")
    _restore_base(policy, base_state)
    del base_state
    model = PeftModel.from_pretrained(
        policy, checked["adapter_path"], is_trainable=False, local_files_only=True
    )
    expected = load_file(str(Path(checked["adapter_path"]) / "adapter_model.safetensors"), device="cpu")
    _verify_adapter_state(model, expected)
    model.eval()
    model.requires_grad_(False)
    model.config.use_peft = True
    model.config.device = device
    model.reset()
    logger.info(
        "Verified complete Base restoration and all %d LoRA tensors; unmerged eval inference", len(expected)
    )
    return model
