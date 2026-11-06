"""Content-addressed, local task bundles. No model framework is imported here."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
STRUCTURE_KEYS = (
    "type", "paligemma_variant", "action_expert_variant", "max_state_dim", "max_action_dim",
    "input_features", "output_features", "chunk_size", "image_resolution", "tokenizer_max_length",
    "dtype", "num_inference_steps", "normalization_mapping", "rtc_training_max_delay",
)
TOKENIZER_ASSET_NAMES = frozenset({
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "chat_template.jinja", "chat_template.json",
})
_VERSIONED_TOKENIZER_JSON = re.compile(r"tokenizer\.\d+(?:\.\d+)*(?:[-+][A-Za-z0-9_.-]+)?\.json")


def _tokenizer_asset_files(source):
    """Supported local HF tokenizer data only; custom tokenizer code is unsupported."""
    if Path(source).is_symlink():
        raise ValueError("Tokenizer source directory symlink is unsupported")
    root = Path(source).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Tokenizer source must be a local directory")

    def asset(relative):
        path = root / relative
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError(f"Tokenizer asset symlink is unsupported: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"Tokenizer asset escapes source or is not a file: {relative}")
        if resolved.stat().st_size == 0:
            raise ValueError(f"Tokenizer asset is empty: {relative}")
        return resolved

    configuration = root / "tokenizer_config.json"
    if not configuration.exists():
        raise ValueError("Tokenizer export requires tokenizer_config.json")
    config = read_json(asset("tokenizer_config.json"))
    if not isinstance(config, dict) or config.get("auto_map"):
        raise ValueError("Tokenizer config must be an object without unsupported custom tokenizer code")
    selected = {}
    for entry in sorted(root.iterdir()):
        if entry.name in TOKENIZER_ASSET_NAMES or _VERSIONED_TOKENIZER_JSON.fullmatch(entry.name):
            selected[entry.name] = asset(entry.name)
    templates = root / "chat_templates"
    if templates.is_symlink():
        raise ValueError("Tokenizer chat_templates directory symlink is unsupported")
    if templates.exists():
        if not templates.is_dir():
            raise ValueError("Tokenizer chat_templates must be a directory")
        for entry in sorted(templates.iterdir()):
            if entry.suffix == ".jinja":
                relative = "chat_templates/" + entry.name
                selected[relative] = asset(relative)

    def reference(value):
        if value is None:
            return
        if (not isinstance(value, str) or "\\" in value or ":" in value
                or PurePosixPath(value).is_absolute() or any(part in (".", "..") for part in value.split("/"))
                or value not in selected):
            raise ValueError(f"Unsupported or missing tokenizer asset reference: {value!r}")

    for key in ("tokenizer_file", "vocab_file", "merges_file", "sp_model_file", "sentencepiece_model_file",
                "chat_template_file", "special_tokens_map_file", "added_tokens_file"):
        reference(config.get(key))
    fast_files = config.get("fast_tokenizer_files", [])
    if fast_files is None:
        fast_files = []
    if not isinstance(fast_files, list):
        raise ValueError("fast_tokenizer_files must be a list of local asset names")
    for filename in fast_files:
        if not isinstance(filename, str):
            raise ValueError("fast_tokenizer_files must contain local asset names")
        reference(filename)
    template_files = config.get("chat_template_files", [])
    if template_files is None:
        template_files = []
    if isinstance(template_files, dict):
        template_files = list(template_files.values())
    if not isinstance(template_files, list):
        raise ValueError("chat_template_files must contain local asset references")
    for filename in template_files:
        if not isinstance(filename, str):
            raise ValueError("chat_template_files must contain local asset references")
        reference(filename)
    # chat_template itself contains inline Jinja text, not a filesystem path.
    # Versioned JSONs are retained for HF's version selection, but cannot replace
    # a supported default asset: an incompatible future version may be ignored.
    fast_data = "tokenizer.json" in selected
    slow_data = any(name in selected for name in ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.txt"))
    bpe_data = "vocab.json" in selected and "merges.txt" in selected
    if not (fast_data or slow_data or bpe_data):
        raise ValueError("Tokenizer export has no supported tokenizer data assets")
    return selected


def copy_tokenizer_assets(source, destination):
    """Copy supported tokenizer assets byte-for-byte, excluding all model weights."""
    files = _tokenizer_asset_files(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(Path(source).resolve(strict=True)):
        raise ValueError("Tokenizer destination must not modify its source directory")
    destination.mkdir(parents=True)
    for relative, path in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return list(files)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def base_fingerprint(base_path):
    base_path = Path(base_path).resolve(strict=True)
    files = sorted({*base_path.glob("*.safetensors"), *base_path.glob("*.safetensors.index.json")})
    if not files or not (base_path / "config.json").is_file():
        raise ValueError("Base must contain config.json and local safetensors weights")
    files.append(base_path / "config.json")
    content = {p.name: digest_file(p) for p in files}
    return {"sha256": canonical_digest(content), "files": content}


def _local_file(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Bundle artifact escapes package: {relative}")
    return path


def adapter_structure(config):
    structure = {k: config.get(k) for k in (
        "peft_type", "r", "lora_alpha", "lora_dropout", "target_modules", "rank_pattern", "alpha_pattern",
        "bias", "modules_to_save", "use_dora", "use_rslora",
    )}
    for key in ("target_modules", "modules_to_save"):
        if isinstance(structure[key], list):
            structure[key] = sorted(structure[key])
    return structure


def validate_target_audit(audit, adapter):
    branches = ("vlm", "expert", "projections")
    if not all(audit.get(key) for key in branches):
        raise ValueError("Target audit must cover language backbone, expert and projections")
    expected = set(audit["target_modules"])
    covered = set(sum((audit.get(key, []) for key in (*branches, "vision")), []))
    if expected != covered:
        raise ValueError("Target audit branch coverage differs from target_modules")
    modules = adapter["target_modules"]
    if isinstance(modules, str):
        names = [entry["name"] for entry in audit["named_modules"] if entry["type"] == "Linear"]
        matched = {name for name in names if re.fullmatch(modules, name)}
    else:
        names = [entry["name"] for entry in audit.get("named_modules", []) if entry["type"] == "Linear"] or list(expected)
        matched = {name for name in names if any(name == key or name.endswith("." + key) for key in modules)}
    if matched != expected:
        raise ValueError("Target audit does not match adapter_config.target_modules")


def _visual_feature_order(config):
    # PI05._preprocess_images iterates config.image_features without sorting.
    return [name for name, feature in (config.get("input_features") or {}).items()
            if feature.get("type") == "VISUAL"]


def load_task_bundle(path, expected_base=None):
    root = Path(path).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported task manifest version")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", manifest.get("task_id", "")):
        raise ValueError("Invalid task_id")
    if expected_base and manifest["base_fingerprint"] != expected_base:
        raise ValueError("Task bundle Base content fingerprint mismatch")
    required = {"adapter_config.json", "adapter_model.safetensors", "config.json",
                "policy_preprocessor.json", "policy_postprocessor.json", "targets.json",
                manifest["offline_observation"]}
    artifacts = manifest["artifacts"]
    if not required.issubset(artifacts):
        raise ValueError("Incomplete task bundle")
    for name, expected in artifacts.items():
        if digest_file(_local_file(root, name)) != expected:
            raise ValueError(f"Task bundle artifact checksum mismatch: {name}")
    tokenizer_files = {name: digest for name, digest in artifacts.items() if name.startswith("tokenizer/")}
    if not tokenizer_files or canonical_digest(tokenizer_files) != manifest["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer content fingerprint mismatch or absent local tokenizer")
    config = read_json(root / "config.json")
    if config.get("type") != "pi05":
        raise ValueError("Task bundle must describe pi05")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in read_json(root / filename)["steps"]:
            if step.get("state_file") and step["state_file"] not in artifacts:
                raise ValueError("Processor tensor state is not covered by manifest")
    adapter = read_json(root / "adapter_config.json")
    validate_target_audit(read_json(root / "targets.json"), adapter)
    if adapter.get("modules_to_save") or adapter.get("bias", "none") != "none":
        raise ValueError("Shared Base requires adapter-only weights and bias=none")
    if adapter_structure(adapter) != adapter_structure(manifest["adapter_structure"]):
        raise ValueError("Adapter structure differs from the exported task")
    if {k: config.get(k) for k in STRUCTURE_KEYS} != manifest["model_structure"]:
        raise ValueError("Policy structure differs from exported task")
    if _visual_feature_order(config) != _visual_feature_order(manifest["model_structure"]):
        raise ValueError("Policy camera order differs from exported task")
    schema = manifest["action_schema"]
    if len(schema["names"]) != len(set(schema["names"])) or len(schema["names"]) != len(schema["units"]):
        raise ValueError("Action schema must have unique ordered names and explicit units")
    if len(schema["names"]) != config["output_features"]["action"]["shape"][0]:
        raise ValueError("Action schema dimension mismatch")
    frequency = schema["control_hz"]
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Control frequency must be a finite positive number")
    if any(key != value for key, value in manifest["camera_map"].items()):
        raise ValueError("Only identity camera maps are supported in this first backend")
    camera_keys = {key for key in config.get("input_features", {}) if key.startswith("observation.images.")}
    if set(manifest["camera_map"]) != camera_keys:
        raise ValueError("Camera map must cover the policy's exact image feature schema")
    if manifest["tokenizer"].get("max_length") != config.get("tokenizer_max_length"):
        raise ValueError("Tokenizer length differs from policy config")
    rtc = manifest["rtc"]
    if rtc["mode"] not in ("async", "training_time"):
        raise ValueError("Legacy gradient-guided RTC is not supported by this backend")
    if not 0 <= rtc["prefix_steps"] <= rtc["max_delay"] < config["chunk_size"]:
        raise ValueError("Invalid deployment RTC prefix range")
    if rtc["max_delay"] != config.get("rtc_training_max_delay", 0):
        raise ValueError("RTC range differs from checkpoint training metadata")
    if rtc["mode"] == "async" and rtc["prefix_steps"]:
        raise ValueError("Async mode cannot use a clean prefix")
    acp = manifest["acp"]
    if acp.get("enabled") and acp.get("prompt_strategy") != "positive":
        raise ValueError("This backend currently supports ACP positive single branch only")
    if acp.get("enabled"):
        beta = acp.get("beta")
        if acp.get("cfg_enabled") is not False or type(beta) not in (int, float) or not math.isfinite(beta) or beta != 1:
            raise ValueError("ACP deployment must explicitly select CFG disabled and beta=1")
    if "Advantage:" in manifest["task_text"]:
        raise ValueError("Task text must be untagged; ACP is added by the backend exactly once")
    return root, manifest


def check_compatible(manifests):
    first = manifests[0]
    for task in manifests[1:]:
        if _visual_feature_order(task["model_structure"]) != _visual_feature_order(first["model_structure"]):
            raise ValueError("Incompatible task bundles: ordered VISUAL camera features")
        for key in ("base_fingerprint", "model_structure", "adapter_structure", "action_schema", "camera_map", "tokenizer"):
            left = adapter_structure(task[key]) if key == "adapter_structure" else task[key]
            right = adapter_structure(first[key]) if key == "adapter_structure" else first[key]
            if left != right:
                raise ValueError(f"Incompatible task bundles: {key}")


def export_task(checkpoint, base_path, destination, metadata, targets, offline_observation, tokenizer_path=None):
    """Export a checkpoint; all deployment semantics must be explicit metadata."""
    source = Path(checkpoint).resolve(strict=True)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    fingerprint = base_fingerprint(base_path)
    adapter = read_json(source / "adapter_config.json")
    base_reference = Path(adapter.get("base_model_name_or_path", ""))
    if not base_reference.is_dir() or (
        base_reference.resolve() != Path(base_path).resolve() and base_fingerprint(base_reference) != fingerprint
    ):
        raise ValueError("Adapter base_model_name_or_path is missing, nonlocal, or references a different Base")
    config = read_json(source / "config.json")
    target_list = read_json(targets)
    validate_target_audit(target_list, adapter)
    tokenizer_source = Path(tokenizer_path) if tokenizer_path else source / "tokenizer"
    if not tokenizer_source.is_dir():
        raise ValueError("Export requires a local tokenizer directory (checkpoint/tokenizer or --tokenizer-path)")
    _tokenizer_asset_files(tokenizer_source)
    destination.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        shutil.copy2(source / name, destination / name)
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in read_json(source / filename)["steps"]:
            if step.get("state_file"):
                relative = step["state_file"]
                shutil.copy2(_local_file(source, relative), destination / relative)
    shutil.copy2(targets, destination / "targets.json")
    shutil.copy2(offline_observation, destination / "offline_observation.safetensors")
    copy_tokenizer_assets(tokenizer_source, destination / "tokenizer")
    manifest = dict(metadata)
    manifest.update(schema_version=SCHEMA_VERSION, base_fingerprint=fingerprint,
                    base_reference=str(Path(base_path).resolve()),
                    adapter_structure=adapter_structure(adapter),
                    model_structure={k: config.get(k) for k in STRUCTURE_KEYS},
                    offline_observation="offline_observation.safetensors")
    manifest["artifacts"] = {p.relative_to(destination).as_posix(): digest_file(p)
                             for p in destination.rglob("*") if p.is_file()}
    manifest["tokenizer"] = dict(manifest["tokenizer"], sha256=canonical_digest({
        name: digest for name, digest in manifest["artifacts"].items() if name.startswith("tokenizer/")
    }))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_task_bundle(destination, fingerprint)
    return manifest
