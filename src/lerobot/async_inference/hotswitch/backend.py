"""One eager PI05 Base with resident adapters and task-specific processors."""

from __future__ import annotations

import copy
from pathlib import Path

from .manifest import base_fingerprint, check_compatible, digest_file, load_task_bundle, read_json


def configure_inference_precision(dtype=None):
    """Explicit FP32 requests disable reduced-precision matmul; default is unchanged."""
    if dtype not in (None, "bfloat16", "float32"):
        raise ValueError("Inference dtype must be None, bfloat16, or float32")
    import torch

    if dtype == "float32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return inference_compute_precision()


def inference_compute_precision():
    import torch

    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


class SharedAdapterBackend:
    """Initialization performs all file I/O. Runtime activate never loads weights.

    Use through HotSwitchRuntime: its lock covers the full inference lifecycle.
    Observations use LeRobot observation.* keys, as in the offline safetensors.
    """

    @classmethod
    def from_registry(cls, path, device="cuda", dtype=None):
        config = read_json(path)
        return cls(config["base_path"], config["tasks"], device=device, dtype=dtype)

    def __init__(self, base_path, tasks, device="cuda", dtype=None, *, _dual=False, _merge=True):
        compute_precision = configure_inference_precision(dtype)
        from peft import PeftModel
        from safetensors.torch import load_file

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.device = device
        self.dtype_override = dtype
        self.compute_precision = compute_precision
        self.task_ids = tuple(tasks)
        if not self.task_ids:
            raise ValueError("At least one registered task is required")
        fingerprint = base_fingerprint(base_path)
        bundles = {task_id: load_task_bundle(path, fingerprint) for task_id, path in tasks.items()}
        check_compatible([manifest for _, manifest in bundles.values()])
        self.task_metadata = {}
        self.processors = {}
        self.observations = {}
        self.models = {}
        self.manifests = {task_id: manifest for task_id, (_, manifest) in bundles.items()}
        self.load_counts = {"base": 0, "adapter": 0, "warmup": 0}
        self.active_task = None
        self._dual = _dual
        self._merge = _merge
        self.comparison_mode = "dual_merged" if _dual and _merge else "dual_independent_peft" if _dual else "shared"
        base = None
        for task_id, (root, manifest) in bundles.items():
            if manifest["task_id"] != task_id:
                raise ValueError("Registry task_id must match its manifest")
            adapter_config = read_json(root / "adapter_config.json")
            reference = Path(adapter_config["base_model_name_or_path"])
            if not reference.is_dir() or (
                reference.resolve() != Path(base_path).resolve() and base_fingerprint(reference) != fingerprint
            ):
                raise ValueError("Adapter's declared Base content differs from registry Base")
            policy_config = copy.deepcopy(PreTrainedConfig.from_pretrained(root))
            if dtype is not None:
                policy_config.dtype = dtype
            policy_config.device = device
            policy_config.compile_model = False
            policy_config.rtc_config = None
            if base is None or _dual:
                # Per-task output schema is validated identical; normalization stays
                # in independent processors and never overwrites the shared Base.
                base_config = copy.deepcopy(policy_config)
                base_config.use_peft = False
                base_config.pretrained_path = Path(base_path)
                base = get_policy_class("pi05").from_pretrained(
                    base_path, config=base_config, local_files_only=True
                )
                self.load_counts["base"] += 1
                model = PeftModel.from_pretrained(base, root, adapter_name=task_id, is_trainable=False)
                if _dual and _merge:
                    # Merge only this independent offline comparison copy.
                    model = model.merge_and_unload()
                self.models[task_id if _dual else "shared"] = model
            else:
                model = self.models["shared"]
                model.load_adapter(root, adapter_name=task_id, is_trainable=False)
            self.load_counts["adapter"] += 1
            model.eval()
            model.to(device)
            self.processors[task_id] = make_pre_post_processors(
                policy_config, pretrained_path=root,
                preprocessor_overrides={"device_processor": {"device": device},
                                        "tokenizer_processor": {"tokenizer_name": str(root / "tokenizer")}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
            self.observations[task_id] = load_file(str(root / manifest["offline_observation"]))
            rtc = manifest["rtc"]
            self.task_metadata[task_id] = {
                "rtc_mode": rtc["mode"], "max_prefix_steps": rtc["max_delay"],
                "prefix_steps": rtc["prefix_steps"], "action_dim": len(manifest["action_schema"]["names"]),
                "chunk_size": policy_config.chunk_size, "task_text": manifest["task_text"],
                "action_names": manifest["action_schema"]["names"],
                "action_units": list(manifest["action_schema"]["units"]),
                "control_hz": manifest["action_schema"]["control_hz"],
                "inference_dtype": policy_config.dtype, "inference_dtype_override": dtype,
                "compute_precision": dict(compute_precision), "backend_mode": self.comparison_mode,
                "bundle_manifest_sha256": digest_file(root / "manifest.json"),
                "image_features": {name: list(feature.shape) for name, feature in policy_config.image_features.items()},
            }
        for model in self.models.values():
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        if not _dual and self.load_counts["base"] != 1:
            raise AssertionError("Shared backend must load precisely one Base")
        self.storage_report = self.parameter_storage_report()
        self.parameter_dtypes = self.parameter_dtype_inventory()
        if dtype == "float32" and any(
            parameter.is_floating_point() and str(parameter.dtype) != "torch.float32"
            for model in self.models.values() for parameter in model.parameters()
        ):
            raise RuntimeError("Explicit float32 inference retained a non-float32 parameter")
        self.activate(self.task_ids[0])

    @property
    def policy(self):
        return self.models[self.active_task if self._dual else "shared"]

    def offline_observation(self, task_id):
        return {key: tensor.clone() for key, tensor in self.observations[task_id].items()}

    def activate(self, task_id):
        if task_id not in self.task_ids:
            raise KeyError(task_id)
        if not self._dual or not self._merge:
            model = self.models[task_id if self._dual else "shared"]
            model.set_adapter(task_id)
            selected = model.active_adapters
            selected = [selected] if isinstance(selected, str) else list(selected)
            if selected != [task_id]:
                raise RuntimeError(f"PEFT did not activate the registered adapter: {task_id}")
        self.active_task = task_id
        self.policy.eval()
        # PEFT set_adapter may re-enable adapter gradients; serving stays frozen.
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    def reset(self, task_id):
        if task_id != self.active_task:
            raise RuntimeError("Reset must follow atomic task activation")
        self.policy.reset()
        for processor in self.processors[task_id]:
            if hasattr(processor, "reset"):
                processor.reset()

    def infer(self, task_id, observation, action_prefix=None, prefix_steps=0):
        import torch

        if self.active_task != task_id:
            raise RuntimeError("Task differs from active adapter")
        manifest = self.manifests[task_id]
        observation = dict(observation)
        task = manifest["task_text"]
        if manifest["acp"].get("enabled"):
            from lerobot.rl.acp_tags import build_acp_tagged_task

            task = build_acp_tagged_task(task, is_positive=True)
        observation["task"] = task
        preprocess, postprocess = self.processors[task_id]
        batch = preprocess(observation)
        # Verify the positive marker survives tokenization, rather than trusting
        # a pre-tokenization string which may have been truncated.
        if manifest["acp"].get("enabled"):
            tokenizer_step = next(s for s in preprocess.steps if hasattr(s, "input_tokenizer"))
            decoded = tokenizer_step.input_tokenizer.batch_decode(batch["observation.language.tokens"])
            if not all(text.count("Advantage: positive") == 1 and "Advantage: negative" not in text for text in decoded):
                raise ValueError("ACP positive condition was lost, duplicated, or contradicted during tokenization")
        kwargs = {"prefix_steps": prefix_steps, "action_prefix": None}
        if prefix_steps:
            if manifest["rtc"]["mode"] != "training_time":
                raise ValueError("Clean prefix supplied to non-TRC task")
            prefix = torch.as_tensor(action_prefix, device=self.device, dtype=torch.float32)
            kwargs["action_prefix"] = prefix.unsqueeze(0) if prefix.ndim == 2 else prefix
        with torch.no_grad():
            raw = self.policy.predict_action_chunk(batch, **kwargs)
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError("Serving requires one complete action chunk")
            # Some processors mutate tensors: clone each action so raw is retained.
            processed = torch.stack([postprocess(raw[:, i, :].clone()) for i in range(raw.shape[1])], dim=1)
        return raw[0].detach().cpu(), processed[0].detach().cpu()

    def warmup(self, task_id):
        import torch

        self.activate(task_id)
        self.reset(task_id)
        raw, _ = self.infer(task_id, self.observations[task_id])
        prefix_steps = self.task_metadata[task_id]["prefix_steps"]
        if prefix_steps:
            prefix = raw[:prefix_steps].clone()
            conditioned, _ = self.infer(task_id, self.observations[task_id], prefix, prefix_steps)
            if not torch.equal(conditioned[:prefix_steps], prefix):
                raise AssertionError("Training-Time RTC prefix is not hard-clamped")
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.reset(task_id)
        self.load_counts["warmup"] += 1

    def parameter_storage_report(self):
        storage = {}
        owners = {}
        adapter_bytes = 0
        for model_id, model in self.models.items():
            for name, parameter in model.named_parameters():
                key = (str(parameter.device), parameter.untyped_storage().data_ptr())
                if self._dual and parameter.numel() and key in owners and owners[key] != model_id:
                    raise RuntimeError("Independent resident models share parameter storage")
                owners[key] = model_id
                storage[key] = parameter.untyped_storage().nbytes()
                if "lora_" in name:
                    adapter_bytes += parameter.numel() * parameter.element_size()
        return {"base_instances": len(self.models), "unique_storage_bytes": sum(storage.values()),
                "adapter_bytes": adapter_bytes, "mode": self.comparison_mode,
                "independent_model_storages": True if self._dual else None}

    def parameter_dtype_inventory(self):
        inventory = {}
        for model_id, model in self.models.items():
            groups = {"base": {}, "adapter": {}}
            for name, parameter in model.named_parameters():
                role = "adapter" if "lora_" in name else "base"
                count = groups[role].setdefault(str(parameter.dtype), {"tensors": 0, "elements": 0, "bytes": 0})
                count["tensors"] += 1
                count["elements"] += parameter.numel()
                count["bytes"] += parameter.numel() * parameter.element_size()
            inventory[model_id] = groups
        return inventory


class DualModelBackend(SharedAdapterBackend):
    """Two separately loaded, merged, resident policy copies for a fair baseline."""

    def __init__(self, base_path, tasks, device="cuda", dtype=None):
        super().__init__(base_path, tasks, device=device, dtype=dtype, _dual=True)


class DualUnmergedBackend(SharedAdapterBackend):
    """Two independent full Bases with their own resident, unmerged PEFT adapters."""

    def __init__(self, base_path, tasks, device="cuda", dtype=None):
        super().__init__(base_path, tasks, device=device, dtype=dtype, _dual=True, _merge=False)
