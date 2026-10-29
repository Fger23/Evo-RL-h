"""Eager, resident PI05 adapters. All disk I/O is confined to initialization."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from .task_bundle import TaskBundle, audit_targets, content_fingerprint, validate_task_pair


def tokenizer_fingerprint(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special": tokenizer.special_tokens_map,
        "backend": backend.to_str() if backend is not None else None,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ResidentBackend:
    """Caller serializes activation and the whole prediction/postprocessing lifecycle."""

    def __init__(self, bundles, models, processors, *, mode="shared", device="cpu", load_counts=None):
        validate_task_pair(bundles)
        self.bundles = {b.task_id: b for b in bundles}
        self.models = models
        self.processors = processors
        self.mode, self.device = mode, device
        self.active_task = None
        self._ready = set()
        self.load_counts = dict(load_counts or {"base": 0, "adapters": 0})
        self.warmup_counts = {b.task_id: 0 for b in bundles}
        self.target_audits = {}
        self.load_seconds = None
        self.warmup_seconds = None

    @property
    def task_ids(self):
        return tuple(self.bundles)

    def metadata(self, task_id):
        meta = copy.deepcopy(self.bundles[task_id].manifest)
        meta.update(
            adapter_id=task_id,
            rtc_mode=meta["rtc"]["mode"],
            rtc_prefix_steps=meta["rtc"].get("prefix_steps", 0),
            rtc_training_max_delay=meta.get("policy_config", {}).get("rtc_training_max_delay", 0),
            control_hz=meta["action_schema"].get("fps"),
            fps=meta["action_schema"].get("fps"),
        )
        return meta

    def ready(self, task_id):
        return task_id in self._ready

    def _model(self, task_id):
        return self.models["shared"] if self.mode == "shared" else self.models[task_id]

    def activate(self, task_id):
        if task_id not in self.bundles:
            raise ValueError(f"Unregistered task {task_id}")
        model = self._model(task_id)
        if self.mode == "shared":
            model.set_adapter(task_id)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        config = self.processors[task_id][2]
        # RTC settings and processors move together under the runtime's lock.
        base.config = config
        base.model.config = config
        model.eval()
        self.active_task = task_id

    def reset(self, task_id):
        model = self._model(task_id)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.reset()
        pre, post, _ = self.processors[task_id]
        pre.reset()
        post.reset()

    def predict(self, task_id, observation, prefix_steps=0, action_prefix=None):
        import torch

        if self.active_task != task_id:
            raise RuntimeError("Task must be activated atomically before inference")
        meta = self.bundles[task_id].manifest
        model = self._model(task_id)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        pre, post, config = self.processors[task_id]
        observation = dict(observation)
        for old, new in meta["camera_mapping"].items():
            if old in observation:
                observation[new] = observation.pop(old)
        if "observation.state" not in observation:
            from .helpers import raw_observation_to_observation

            observation = raw_observation_to_observation(
                observation, meta["lerobot_features"], config.image_features
            )
        observation = {k: torch.as_tensor(v) if hasattr(v, "shape") else v for k, v in observation.items()}
        task = meta["task_text"]
        if meta["acp"].get("enabled", False):
            from lerobot.rl.acp_tags import build_acp_tagged_task

            task = build_acp_tagged_task(task, is_positive=True)
        observation["task"] = task
        maximum = config.rtc_training_max_delay
        if not isinstance(prefix_steps, int) or not 0 <= prefix_steps <= maximum:
            raise ValueError("RTC prefix outside checkpoint training range")
        if meta["rtc"]["mode"] == "async" and prefix_steps:
            raise ValueError("Async task cannot accept a clean prefix")
        prefix = None
        if prefix_steps:
            prefix = torch.as_tensor(action_prefix, dtype=torch.float32, device=self.device)
            if (
                prefix.shape != (prefix_steps, len(meta["action_schema"]["names"]))
                or not torch.isfinite(prefix).all()
            ):
                raise ValueError("Normalized prefix has invalid shape or values")
            prefix = prefix.unsqueeze(0)
        elif action_prefix is not None:
            raise ValueError("Bootstrap must have no previous prefix")
        batch = pre(observation)
        # A tag could otherwise disappear under silent tokenizer truncation.
        if meta["acp"].get("enabled", False):
            steps = [s for s in pre.steps if hasattr(s, "input_tokenizer")]
            if len(steps) != 1:
                raise ValueError("Exactly one tokenizer is required for ACP prompt validation")
            tokenizer = steps[0].input_tokenizer
            # Tokenize the exact PI05 task/state prompt rather than task text alone.
            state = batch["observation.state"].detach().cpu().numpy()
            import numpy as np

            padded = np.pad(state, ((0, 0), (0, config.max_state_dim - state.shape[-1])))
            bins = np.digitize(padded[0], np.linspace(-1, 1, 257)[:-1]) - 1
            cleaned = task.strip().replace("_", " ").replace("\n", " ")
            prompt = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
            if len(tokenizer(prompt, truncation=False)["input_ids"]) > steps[0].max_length:
                raise ValueError("ACP PI05 prompt would be truncated by the tokenizer")
        with torch.no_grad():
            raw = base.predict_action_chunk(batch, prefix_steps=prefix_steps, action_prefix=prefix)
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError("Hot-switch supports one robot, one observation batch")
            if raw.shape[1:] != (config.chunk_size, len(meta["action_schema"]["names"])):
                raise ValueError("Prediction does not match registered action schema")
            processed = torch.stack([post(raw[:, i]) for i in range(raw.shape[1])], dim=1)
        if not torch.isfinite(raw).all() or not torch.isfinite(processed).all():
            raise ValueError("Nonfinite model output")
        if prefix_steps and not torch.equal(raw[:, :prefix_steps].to(prefix), prefix):
            raise ValueError("Model failed the clean-prefix hard clamp")
        return raw[0].detach().cpu(), processed[0].detach().cpu()

    def warmup(self, observations, repeats=2):
        import torch

        start = time.perf_counter()
        for task_id in self.task_ids:
            if task_id not in observations:
                raise ValueError(f"Missing offline warmup observation for {task_id}")
            self.activate(task_id)
            self.reset(task_id)
            for _ in range(repeats):
                raw, _ = self.predict(task_id, observations[task_id])
                d = self.metadata(task_id)["rtc"].get("prefix_steps", 0)
                if d:
                    self.predict(task_id, observations[task_id], d, raw[:d])
                self.warmup_counts[task_id] += 1
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            self.reset(task_id)
            self._ready.add(task_id)
        self.active_task = None
        self.warmup_seconds = time.perf_counter() - start

    def storage_evidence(self):
        """Count real tensor storages rather than summing aliased parameter views."""
        storages = {"base": {}, "adapter": {}}
        devices = set()
        for model in self.models.values():
            for name, parameter in model.named_parameters():
                storage = parameter.untyped_storage()
                key = (str(parameter.device), storage.data_ptr())
                group = "adapter" if "lora_" in name else "base"
                storages[group][key] = storage.nbytes()
                devices.add(str(parameter.device))
        return {
            "base_storage_bytes": sum(storages["base"].values()),
            "adapter_storage_bytes": sum(storages["adapter"].values()),
            "model_instances": len({id(m) for m in self.models.values()}),
            "devices": sorted(devices),
            "load_counts": dict(self.load_counts),
        }


def load_backend(registry, device=None, mode=None, offline_observations=None):
    """Load a trusted local registry and warm all paths before advertising readiness.

    Registry: {tasks:[bundle paths], base_path, device, mode, offline_observations:{id:npz}}.
    Remote callers only ever select an already registered task_id.
    """
    started = time.perf_counter()
    import numpy as np
    import torch
    from peft import PeftConfig, PeftModel

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    registry_root = Path.cwd()
    if isinstance(registry, (str, Path)):
        path = Path(registry).resolve(strict=True)
        registry_root = path.parent
        registry = json.loads(path.read_text(encoding="utf-8"))
    device, mode = device or registry.get("device", "cpu"), mode or registry.get("mode", "shared")
    if mode not in {"shared", "dual"}:
        raise ValueError("mode must be shared or dual")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device unavailable; no automatic CPU fallback")
    bundles = [TaskBundle.load(registry_root / p) for p in registry["tasks"]]
    validate_task_pair(bundles)
    base_path = Path(registry.get("base_path", bundles[0].manifest["base_path"]))
    if not base_path.is_absolute():
        base_path = registry_root / base_path
    base_path = base_path.resolve(strict=True)
    fingerprint = content_fingerprint(base_path)
    if fingerprint != bundles[0].manifest["base_fingerprint"]:
        raise ValueError("Shared Base content fingerprint mismatch")
    policy_class = get_policy_class("pi05")
    models, processors, targets = {}, {}, {}
    load_counts = {"base": 0, "adapters": 0}
    for bundle in bundles:
        config = PreTrainedConfig.from_pretrained(bundle.adapter_path, local_files_only=True)
        config.device = str(device)
        config.compile_model = False
        config.pretrained_path = base_path
        if config.rtc_config is not None and config.rtc_config.enabled:
            raise ValueError("Legacy gradient-guided RTC must use its existing tested inference path")
        if getattr(config, "type", None) != "pi05":
            raise ValueError("Expected PI05 task")
        processors[bundle.task_id] = (
            *make_pre_post_processors(
                config,
                pretrained_path=bundle.adapter_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            ),
            config,
        )
        tokenizer_steps = [s for s in processors[bundle.task_id][0].steps if hasattr(s, "input_tokenizer")]
        if (
            len(tokenizer_steps) != 1
            or tokenizer_fingerprint(tokenizer_steps[0].input_tokenizer)
            != bundle.manifest["tokenizer_fingerprint"]
        ):
            raise ValueError("Actual tokenizer fingerprint differs from registered tokenizer")
        adapter_config = PeftConfig.from_pretrained(bundle.adapter_path, local_files_only=True)
        if content_fingerprint(adapter_config.base_model_name_or_path) != fingerprint:
            raise ValueError("Adapter base_model_name_or_path content mismatch")
        if mode == "shared" and "shared" in models:
            models["shared"].load_adapter(
                bundle.adapter_path, adapter_name=bundle.task_id, is_trainable=False
            )
            load_counts["adapters"] += 1
        else:
            policy = policy_class.from_pretrained(base_path, config=config, local_files_only=True).to(device)
            load_counts["base"] += 1
            targets[bundle.task_id] = audit_targets(policy, adapter_config.target_modules)
            if sorted(targets[bundle.task_id]["targets"]) != sorted(bundle.manifest["target_layers"]):
                raise ValueError("Actual LoRA targets differ from export audit")
            policy = PeftModel.from_pretrained(
                policy, bundle.adapter_path, adapter_name=bundle.task_id, is_trainable=False
            )
            load_counts["adapters"] += 1
            if mode == "dual":
                policy = policy.merge_and_unload()  # Dedicated full-model baseline only, never shared.
            models["shared" if mode == "shared" else bundle.task_id] = policy
    expected = torch.device(device)
    for model in models.values():
        if any(
            p.device.type != expected.type
            or (expected.index is not None and p.device.index != expected.index)
            for p in model.parameters()
        ):
            raise RuntimeError("Not all Base and Adapter parameters reside on the requested device")
    backend = ResidentBackend(bundles, models, processors, mode=mode, device=device, load_counts=load_counts)
    backend.target_audits = targets
    if offline_observations is None:
        offline_observations = {}
        for task_id, path in registry.get("offline_observations", {}).items():
            with np.load(registry_root / path, allow_pickle=False) as data:
                offline_observations[task_id] = {k: data[k] for k in data.files}
    backend.load_seconds = time.perf_counter() - started
    backend.warmup(offline_observations)
    return backend
