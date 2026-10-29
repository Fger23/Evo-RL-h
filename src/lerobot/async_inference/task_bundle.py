"""Local, content-verified deployment bundles. No model imports or hardware access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
STRUCTURE_KEYS = (
    "paligemma_variant",
    "action_expert_variant",
    "max_state_dim",
    "max_action_dim",
    "chunk_size",
    "image_resolution",
    "input_features",
    "output_features",
    "tokenizer_max_length",
    "dtype",
    "num_inference_steps",
    "empty_cameras",
)


def content_fingerprint(directory: str | Path) -> str:
    """Hash all weight and configuration bytes, independent of directory name."""
    directory = Path(directory).resolve(strict=True)
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".bin", ".json", ".model"}
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"No local model weights in {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def joint_target_pattern(extended: bool = False, vision: bool = False) -> str:
    """Use actual PI05 module paths, including its time_mlp projections."""
    attention = "q|k|v|o" if extended else "q|v"
    transformer = rf".*\.self_attn\.({attention})_proj"
    if extended:
        transformer = rf"({transformer}|.*\.mlp\.(gate|up|down)_proj)"
    branches = [rf"model\.paligemma_with_expert\.(paligemma\.language_model|gemma_expert)\.{transformer}"]
    branches.append(r"model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)")
    if vision:
        branches.append(
            r"model\.paligemma_with_expert\.paligemma\.vision_tower\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)"
        )
    return "(" + "|".join(branches) + ")"


def audit_targets(policy, pattern: str | None = None, *, extended=False, vision=False) -> dict:
    pattern = pattern or joint_target_pattern(extended, vision)

    def matches(name):
        if isinstance(pattern, str):
            return re.fullmatch(pattern, name) is not None
        return any(name == target or name.endswith("." + target) for target in pattern)

    names = [name for name, module in policy.named_modules() if matches(name)]
    groups = {
        "language": [n for n in names if ".paligemma.language_model." in n],
        "expert": [n for n in names if ".gemma_expert." in n],
        "projections": [
            n
            for n in names
            if re.fullmatch(r"model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)", n)
        ],
        "vision": [n for n in names if ".vision_tower." in n],
    }
    if not groups["language"] or not groups["expert"] or len(groups["projections"]) != 4:
        raise ValueError(f"Joint PI05 targets incomplete: { {k: len(v) for k, v in groups.items()} }")
    if vision and not groups["vision"]:
        raise ValueError("Requested vision adapter but no SigLIP layers matched")
    return {
        "pattern": pattern if isinstance(pattern, str) else sorted(pattern),
        "targets": names,
        "groups": groups,
    }


@dataclass(frozen=True)
class TaskBundle:
    root: Path
    manifest: dict

    @property
    def task_id(self):
        return self.manifest["task_id"]

    @property
    def adapter_path(self):
        return self.root / "adapter"

    @classmethod
    def load(cls, path: str | Path) -> TaskBundle:
        root = Path(path).resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "task_id",
            "base_path",
            "base_fingerprint",
            "adapter_fingerprint",
            "task_text",
            "action_schema",
            "tokenizer_fingerprint",
            "camera_mapping",
            "rtc",
            "acp",
            "training_source",
            "target_layers",
            "policy_config",
            "lerobot_features",
        }
        if required - manifest.keys():
            raise ValueError(f"Missing task manifest fields: {sorted(required - manifest.keys())}")
        if not isinstance(manifest["task_id"], str):
            raise ValueError("task_id must be a nonempty registered identifier")
        if not isinstance(manifest["task_text"], str) or not manifest["task_text"].strip():
            raise ValueError("task_text must contain the confirmed task instruction")
        if manifest["schema_version"] != SCHEMA_VERSION or not re.fullmatch(
            r"[A-Za-z0-9_-]+", manifest["task_id"]
        ):
            raise ValueError("Invalid bundle schema/task_id")
        schema = manifest["action_schema"]
        fps = schema.get("fps")
        if (
            not schema.get("names")
            or len(schema["names"]) != len(schema.get("units", []))
            or not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(fps)
            or fps <= 0
            or any(
                not isinstance(value, str) or not value for value in schema["names"] + schema.get("units", [])
            )
        ):
            raise ValueError("Explicit action order, units and positive fps required")
        if len(set(schema["names"])) != len(schema["names"]):
            raise ValueError("Duplicate action names")
        rtc = manifest["rtc"]
        if rtc.get("mode") not in {"async", "training_time"}:
            raise ValueError(
                "Hot-switch backend supports async and clean-prefix training_time; legacy RTC must use its existing path"
            )
        maximum = manifest["policy_config"].get("rtc_training_max_delay", 0)
        delay = rtc.get("prefix_steps", 0)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
            raise ValueError("Checkpoint rtc_training_max_delay must be a nonnegative integer")
        if not isinstance(delay, int) or delay < 0 or (rtc["mode"] == "async" and delay) or delay > maximum:
            raise ValueError("Deployment prefix length exceeds checkpoint training range")
        if rtc["mode"] == "training_time" and (maximum <= 0 or delay <= 0):
            raise ValueError("training_time requires a trained positive maximum and deployment prefix")
        acp = manifest["acp"]
        if acp.get("enabled", False) and acp.get("deployment_prompt") != "positive":
            raise ValueError("Only explicit positive single-branch ACP deployment is supported")
        if acp.get("cfg", False) or acp.get("beta", 1) != 1:
            raise ValueError("CFG is not enabled in the initial hot-switch backend")
        if "Advantage:" in manifest["task_text"]:
            raise ValueError("task_text must be untagged; backend owns ACP conditioning")
        if (
            json.loads((root / "adapter" / "config.json").read_text(encoding="utf-8"))
            != manifest["policy_config"]
        ):
            raise ValueError("Manifest policy_config differs from the saved checkpoint")
        action = manifest["policy_config"].get("output_features", {}).get("action")
        if action and action.get("shape", [None])[0] != len(schema["names"]):
            raise ValueError("Action schema dimension differs from checkpoint")
        if (
            not manifest["target_layers"]
            or not manifest["training_source"]
            or not manifest["tokenizer_fingerprint"]
        ):
            raise ValueError("Target layers, tokenizer fingerprint and training provenance required")
        if content_fingerprint(root / "adapter") != manifest["adapter_fingerprint"]:
            raise ValueError("Task bundle content changed since export")
        cfg = json.loads((root / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg.get("modules_to_save"):
            raise ValueError("modules_to_save unsupported for shared hot-switch v1")
        if cfg.get("peft_type") != "LORA":
            raise ValueError("Joint hot-switch bundles require LoRA")
        if cfg.get("bias", "none") != "none":
            raise ValueError("Shared Base bias training is unsupported; use bias='none'")
        if cfg.get("use_dora", False) or cfg.get("layer_replication") or cfg.get("trainable_token_indices"):
            raise ValueError(
                "DoRA, replicated layers and trainable tokenizer embeddings require separate validation"
            )
        return cls(root, manifest)


def validate_task_pair(bundles: list[TaskBundle]) -> None:
    if not bundles or len({b.task_id for b in bundles}) != len(bundles):
        raise ValueError("Registry must have unique task IDs")
    reference = bundles[0].manifest
    for bundle in bundles:
        current = bundle.manifest
        for key in (
            "base_fingerprint",
            "action_schema",
            "tokenizer_fingerprint",
            "camera_mapping",
            "target_layers",
        ):
            if current[key] != reference[key]:
                raise ValueError(f"Incompatible task {bundle.task_id}: {key}")
        for key in STRUCTURE_KEYS:
            if current["policy_config"].get(key) != reference["policy_config"].get(key):
                raise ValueError(f"Incompatible task {bundle.task_id}: policy_config.{key}")
        for key in (
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "rank_pattern",
            "alpha_pattern",
            "lora_bias",
            "use_rslora",
            "fan_in_fan_out",
        ):
            first = json.loads((bundles[0].adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            config = json.loads((bundle.adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            if config.get(key) != first.get(key):
                raise ValueError(f"Incompatible adapter structure: {key}")


def export_task(
    adapter_path: str | Path, destination: str | Path, metadata: dict, base_path: str | Path
) -> Path:
    """Copy a trainer checkpoint (processors included) into a verified local bundle."""
    source, destination, base = (
        Path(adapter_path).resolve(strict=True),
        Path(destination),
        Path(base_path).resolve(strict=True),
    )
    if destination.exists():
        raise FileExistsError(destination)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    adapter = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    claimed = adapter.get("base_model_name_or_path")
    if not claimed or not Path(claimed).is_dir():
        raise ValueError("Adapter must reference an available local Base, resolved before export")
    fingerprint = content_fingerprint(base)
    if content_fingerprint(claimed) != fingerprint:
        raise ValueError("Adapter was trained against a different Base")
    destination.mkdir(parents=True)
    shutil.copytree(source, destination / "adapter")
    manifest = dict(metadata)
    manifest.update(
        schema_version=SCHEMA_VERSION,
        base_path=str(base),
        base_fingerprint=fingerprint,
        adapter_fingerprint=content_fingerprint(destination / "adapter"),
        policy_config=json.loads((source / "config.json").read_text(encoding="utf-8")),
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    TaskBundle.load(destination)
    return destination
