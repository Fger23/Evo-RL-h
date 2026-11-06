"""Audited joint LoRA targets, real-data smoke training and complete task export.

Run ``python -m lerobot.scripts.hotswitch_training --help``. Imports are lazy so
the export and help commands do not initialize CUDA or robot hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from lerobot.async_inference.hotswitch.manifest import export_task, read_json


def joint_targets(policy, extended=False, include_vision=False, diagnostic_path=None):
    """Match real module paths and refuse missing branches/projections."""
    import torch

    targets = {"vlm": [], "expert": [], "projections": [], "vision": []}
    attention = r"(q|k|v|o)_proj" if extended else r"(q|v)_proj"
    transformer = rf"(self_attn\.{attention}|mlp\.(gate_proj|up_proj|down_proj))" if extended else rf"self_attn\.{attention}"
    named_modules = []
    for name, module in policy.named_modules():
        named_modules.append({"name": name, "type": type(module).__name__})
        if not isinstance(module, torch.nn.Linear):
            continue
        if re.fullmatch(rf"model\.paligemma_with_expert\.paligemma(?:\.model)?\.language_model(?:\.model)?\.layers\.\d+\.{transformer}", name):
            targets["vlm"].append(name)
        elif re.fullmatch(rf"model\.paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.{transformer}", name):
            targets["expert"].append(name)
        elif name in {"model.action_in_proj", "model.action_out_proj", "model.time_mlp_in", "model.time_mlp_out"}:
            targets["projections"].append(name)
        elif include_vision and ".vision_tower." in name:
            targets["vision"].append(name)
    targets["target_modules"] = sorted(sum((targets[key] for key in ("vlm", "expert", "projections", "vision")), []))
    targets["named_modules"] = named_modules
    if diagnostic_path:
        Path(diagnostic_path).write_text(json.dumps(targets, indent=2), encoding="utf-8")
    if not targets["vlm"] or not targets["expert"] or len(targets["projections"]) != 4:
        raise ValueError(f"Joint target audit failed: { {k:len(targets[k]) for k in ('vlm', 'expert', 'projections', 'vision')} }")
    return targets


def parameter_digest(model, *, adapters=False):
    """Content hash verifies all frozen tensors without retaining a second Base."""
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if ("lora_" in name) != adapters:
            continue
        name = name.replace(".base_layer.", ".")
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def collate_smoke_sample(sample):
    """Use the trainer's batched schema, including [B,H,A] action chunks."""
    from torch.utils.data import default_collate

    return default_collate([sample])


def predict_with_target_hooks(policy, batch, targets):
    """Prove the custom chunk method reaches every injected target, then detach hooks."""
    base = policy.get_base_model()
    counts = dict.fromkeys(targets["target_modules"], 0)
    handles = []

    def count_forward(name):
        def hook(_module, _inputs, _output):
            counts[name] += 1
        return hook

    try:
        for name in counts:
            handles.append(base.get_submodule(name).register_forward_hook(count_forward(name)))
        result = policy.predict_action_chunk(batch)
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise AssertionError(f"predict_action_chunk bypassed adapter targets: {missing}")
    return result, counts


def smoke(args):
    """Ordinary SFT mechanism test on real dataset; this does not claim ACP."""
    import copy

    import torch
    from safetensors.torch import save_file

    from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    config = PreTrainedConfig.from_pretrained(args.base_path)
    config.pretrained_path = Path(args.base_path)
    config.device = args.device
    config.dtype = args.dtype
    config.compile_model = False
    config.gradient_checkpointing = True
    config.use_peft = False
    config.rtc_config = None
    if args.tokenizer_path:
        if not Path(args.tokenizer_path).is_dir():
            raise ValueError("--tokenizer-path must be an existing local tokenizer directory")
        config.tokenizer_name = str(Path(args.tokenizer_path).resolve())
    if args.rtc_max_delay is not None:
        if not 0 <= args.rtc_max_delay < config.chunk_size:
            raise ValueError("--rtc-max-delay must satisfy 0 <= delay < chunk_size")
        config.rtc_training_max_delay = args.rtc_max_delay
    if not 0 <= args.prefix_steps <= config.rtc_training_max_delay:
        raise ValueError("--prefix-steps exceeds the smoke checkpoint's training RTC range")
    # Ordinary smoke can test an existing TRC checkpoint but does not assert it
    # learned a new capability from these few ordinary SFT updates.
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root,
                            delta_timestamps={"action": [i / dataset.fps for i in range(config.chunk_size)]})
    # Base camera identifiers are dataset schema, not learned weights. Derive
    # the real dataset's three input features and keep its identity camera map.
    dataset_features = dataset_to_policy_features(dataset.meta.features)
    config.output_features = {name: feature for name, feature in dataset_features.items() if name == "action"}
    config.input_features = {name: feature for name, feature in dataset_features.items() if name != "action"}
    # Resolve tokenizer and statistics before allocating/loading the large Base.
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    preflight_batch = preprocessor(collate_smoke_sample(copy.deepcopy(dataset[0])))
    expected_action_shape = (1, config.chunk_size, config.output_features["action"].shape[0])
    if tuple(preflight_batch["action"].shape) != expected_action_shape:
        raise ValueError(f"Smoke action batch has shape {preflight_batch['action'].shape}, expected {expected_action_shape}")
    policy = make_policy(config, ds_meta=dataset.meta)
    targets = joint_targets(policy, extended=args.extended, diagnostic_path=output / "targets.json")
    (output / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    before = parameter_digest(policy)
    sample = dataset[0]
    torch.manual_seed(args.seed)
    base_reference = policy.predict_action_chunk(preprocessor(collate_smoke_sample(copy.deepcopy(sample)))).detach().cpu()
    if not torch.isfinite(base_reference).all():
        raise AssertionError("Original PI05 inference produced a nonfinite reference")
    save_file({"action": base_reference.contiguous()}, str(output / "base_reference.safetensors"))
    policy = policy.wrap_with_peft(peft_cli_overrides={
        "r": args.rank, "lora_alpha": args.alpha, "lora_dropout": 0.0,
        "target_modules": targets["target_modules"], "full_training_modules": [],
    })
    torch.manual_seed(args.seed)
    initialized_reference = policy.predict_action_chunk(preprocessor(collate_smoke_sample(copy.deepcopy(sample)))).detach().cpu()
    if not torch.isfinite(initialized_reference).all():
        raise AssertionError("Zero-initialized adapter inference produced a nonfinite reference")
    initialized_error = float((initialized_reference - base_reference).abs().max())
    save_file({"action": initialized_reference.contiguous()}, str(output / "zero_adapter_reference.safetensors"))
    if not torch.allclose(initialized_reference, base_reference, atol=args.atol, rtol=args.rtol):
        raise AssertionError("Zero-initialized joint LoRA changed the original PI05 inference output")
    (output / "injection_evidence.json").write_text(json.dumps({
        "seed": args.seed, "atol": args.atol, "rtol": args.rtol,
        "base_vs_zero_adapter_max_abs": initialized_error, "base_reference": "base_reference.safetensors",
        "zero_adapter_reference": "zero_adapter_reference.safetensors",
    }, indent=2), encoding="utf-8")
    offline = {key: tensor.clone().contiguous() for key, tensor in sample.items()
               if key.startswith("observation.") and isinstance(tensor, torch.Tensor)}
    save_file(offline, str(output / "offline_observation.safetensors"))
    initial = {name: tensor.detach().cpu().clone() for name, tensor in policy.named_parameters() if tensor.requires_grad}
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    losses = []
    step_wall_times = []
    step_timing = {
        "clock": "time.perf_counter",
        "cuda_synchronized": args.device.startswith("cuda"),
        "scope": "Offline smoke step wall time including sample decode/preprocessing, forward/backward, optimizer and finite checks; excludes model loading/checkpoint hashes.",
    }
    mechanism_schedule = []
    training_max_delay = config.rtc_training_max_delay
    policy.train()
    for step in range(args.steps):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        step_started = time.perf_counter()
        batch = preprocessor(collate_smoke_sample(copy.deepcopy(dataset[step % len(dataset)])))
        optimizer.zero_grad(set_to_none=True)
        step_max_delay = 0 if step == 0 else training_max_delay
        config.rtc_training_max_delay = step_max_delay
        try:
            loss, step_metrics = policy(batch)
        finally:
            config.rtc_training_max_delay = training_max_delay
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite training loss")
        loss.backward()
        gradients = {name: p.grad.detach().float().norm().item() for name, p in policy.named_parameters()
                     if p.requires_grad and p.grad is not None}
        if not all(torch.isfinite(torch.tensor(value)) for value in gradients.values()):
            raise AssertionError("Non-finite adapter gradient")
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in policy.parameters() if parameter.requires_grad):
            raise AssertionError("Optimizer produced a nonfinite adapter parameter")
        losses.append(float(loss.detach()))
        mechanism_schedule.append({"step": step + 1, "rtc_training_max_delay": step_max_delay,
                                   "sampled_rtc_prefix_length": step_metrics.get("rtc_prefix_length", 0.0)})
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        step_wall_times.append(time.perf_counter() - step_started)
        print(json.dumps({"step": step + 1, "loss": losses[-1], "step_wall_time_s": step_wall_times[-1],
                          "cuda_synchronized": step_timing["cuda_synchronized"]}), flush=True)
    if training_max_delay and not any(step["sampled_rtc_prefix_length"] > 0 for step in mechanism_schedule):
        raise AssertionError("Smoke did not sample a nonzero training RTC prefix; use more smoke steps")
    updates = {branch: sum(float((p.detach().cpu() - initial[name]).float().abs().sum())
                          for name, p in policy.named_parameters() if name in initial
                          and any(f".{target}." in f".{name}" for target in targets[branch]))
               for branch in ("vlm", "expert", "projections")}
    if not all(value > 0 for value in updates.values()):
        raise AssertionError(f"Joint LoRA branches did not all update: {updates}")
    # Parameter names acquire the PEFT base_model prefix; hash the real policy.
    after = parameter_digest(policy.get_base_model())
    if after != before:
        raise AssertionError("Frozen Base changed during adapter training")
    training_evidence = {
        "phase": "ordinary_joint_lora_mechanism_only", "steps": args.steps, "losses": losses,
        "branch_update_l1": updates, "base_parameters_before_sha256": before,
        "base_parameters_after_sha256": after, "frozen_base_unchanged": True,
        "rtc_max_delay": config.rtc_training_max_delay, "prefix_steps": args.prefix_steps,
        "rank": args.rank, "alpha": args.alpha, "seed": args.seed, "dtype": args.dtype,
        "second_task_trained": False, "atol": args.atol, "rtol": args.rtol,
        "mechanism_schedule": mechanism_schedule,
        "step_wall_time_s": step_wall_times, "step_timing": step_timing,
    }
    (output / "training_evidence.json").write_text(json.dumps(training_evidence, indent=2), encoding="utf-8")
    checkpoint = output / "checkpoint"
    policy.eval()
    torch.manual_seed(args.seed)
    saved_output, target_forward_counts = predict_with_target_hooks(
        policy, preprocessor(collate_smoke_sample(copy.deepcopy(sample))), targets
    )
    saved_output = saved_output.detach().cpu()
    if not torch.isfinite(saved_output).all():
        raise AssertionError("Trained adapter inference produced nonfinite actions")
    save_file({"action": saved_output.contiguous()}, str(output / "saved_reference.safetensors"))
    torch.manual_seed(args.seed)
    with policy.disable_adapter():
        disabled_output = policy.predict_action_chunk(preprocessor(collate_smoke_sample(copy.deepcopy(sample)))).detach().cpu()
    if not torch.isfinite(disabled_output).all():
        raise AssertionError("Disabled adapter inference produced nonfinite actions")
    save_file({"action": disabled_output.contiguous()}, str(output / "disabled_adapter_reference.safetensors"))
    adapter_effect = saved_output - disabled_output
    effect_max_abs = float(adapter_effect.abs().max())
    effect_mse = float(adapter_effect.square().mean())
    disabled_base_error = float((disabled_output - base_reference).abs().max())
    if not effect_max_abs > 0 or not effect_mse > 0:
        raise AssertionError("The trained adapter has no measurable effect on predict_action_chunk")
    if not torch.allclose(disabled_output, base_reference, atol=args.atol, rtol=args.rtol):
        raise AssertionError("Disabling the adapter did not recover the original Base output")
    adapter_effect_evidence = {
        "seed": args.seed, "enabled_vs_disabled_max_abs": effect_max_abs,
        "enabled_vs_disabled_mse": effect_mse, "disabled_vs_base_max_abs": disabled_base_error,
        "base_vs_zero_adapter_max_abs": initialized_error,
        "target_forward_counts": target_forward_counts,
        "target_branch_call_counts": {branch: sum(target_forward_counts[name] for name in targets[branch])
                                      for branch in ("vlm", "expert", "projections")},
        "hooks_removed_after_inference": True, "atol": args.atol, "rtol": args.rtol,
    }
    (output / "adapter_effect_evidence.json").write_text(json.dumps(adapter_effect_evidence, indent=2), encoding="utf-8")
    policy.save_pretrained(checkpoint)
    config.use_peft = True
    config.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    tokenizer_step = next(step for step in preprocessor.steps if hasattr(step, "input_tokenizer"))
    tokenizer_step.input_tokenizer.save_pretrained(checkpoint / "tokenizer")
    # Save optimizer state for an explicit continuation check below.
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    metadata = {
        "task_id": "A", "task_text": sample["task"],
        "action_schema": {"names": dataset.meta.features["action"].get("names") or [f"action_{i}" for i in range(config.output_features["action"].shape[0])],
                          "units": [args.action_unit] * config.output_features["action"].shape[0],
                          "control_hz": dataset.fps},
        "camera_map": {name: name for name in config.image_features},
        "tokenizer": {"name": config.tokenizer_name, "max_length": config.tokenizer_max_length},
        "rtc": {"mode": "training_time" if config.rtc_training_max_delay else "async",
                "max_delay": config.rtc_training_max_delay, "prefix_steps": args.prefix_steps},
        "acp": {"enabled": False, "prompt_strategy": "none", "indicator_dropout_prob": 0.0},
        "training_source": {"dataset": args.dataset_root, "repo_id": args.repo_id, "steps": args.steps,
                            "status": "mechanism_only", "second_task_trained": False},
    }
    # B is explicitly the identical A adapter for residency and ABA regression;
    # it is never reported as an independently trained second task.
    for name in ("A", "A_copy"):
        metadata["task_id"] = name
        export_task(checkpoint, args.base_path, output / name, metadata,
                    output / "targets.json", output / "offline_observation.safetensors")
    registry = {"base_path": str(Path(args.base_path).resolve()),
                "tasks": {task: str(output / task) for task in ("A", "A_copy")}}
    (output / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    del policy, optimizer
    import gc
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    backend = SharedAdapterBackend.from_registry(output / "registry.json", args.device)
    backend.warmup("A")
    backend.warmup("A_copy")
    outputs = []
    for task in ("A", "A_copy", "A"):
        backend.activate(task)
        backend.reset(task)
        torch.manual_seed(args.seed)
        outputs.append(backend.infer(task, backend.offline_observation(task))[0])
    restored_max_abs = float((saved_output[0] - outputs[0]).abs().max())
    if not torch.allclose(saved_output[0], outputs[0], atol=args.atol, rtol=args.rtol):
        raise AssertionError("Saved/restored shared adapter differs from the trained adapter")
    max_abs = max(float((outputs[0] - other).abs().max()) for other in outputs[1:])
    mse = max(float((outputs[0] - other).square().mean()) for other in outputs[1:])
    if not all(torch.allclose(outputs[0], other, atol=args.atol, rtol=args.rtol) for other in outputs[1:]):
        raise AssertionError("A -> same-adapter copy -> A consistency failed")
    report = {"status": "real_gpu_passed" if args.device.startswith("cuda") else "real_cpu_passed",
              "scope": "ordinary_joint_lora_mechanism_only", "losses": losses, "branch_update_l1": updates,
              "frozen_base_unchanged": True, "aba_max_abs": max_abs, "aba_mse": mse,
              "saved_restored_max_abs": restored_max_abs,
              "adapter_effect": adapter_effect_evidence,
              "step_wall_time_s": step_wall_times, "step_timing": step_timing,
              "atol": args.atol, "rtol": args.rtol, "storage": backend.storage_report,
              "load_counts": backend.load_counts, "formal_training_completed": False,
              "acp_acceptance_completed": False, "robot_validation_completed": False}
    # Preserve the completed comparison before attempting another restore stage.
    (output / "shared_restore_evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Restore an independent trainable adapter and optimizer, then continue one
    # real backward/update step. Release the backend to keep one resident Base.
    del backend
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    from peft import PeftModel

    resume_config = copy.deepcopy(config)
    resume_config.use_peft = False
    resumed_base = make_policy(resume_config, ds_meta=dataset.meta)
    resumed = PeftModel.from_pretrained(resumed_base, checkpoint, is_trainable=True)
    resumed.eval()
    torch.manual_seed(args.seed)
    independent = resumed.predict_action_chunk(preprocessor(collate_smoke_sample(copy.deepcopy(sample)))).detach().cpu()
    independent_error = float((independent - saved_output).abs().max())
    if not torch.allclose(independent, saved_output, atol=args.atol, rtol=args.rtol):
        raise AssertionError("Independent adapter restore differs from saved output")
    resume_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=args.lr)
    resume_optimizer.load_state_dict(torch.load(output / "optimizer.pt", map_location=args.device, weights_only=True))
    adapter_before = parameter_digest(resumed.get_base_model(), adapters=True)
    resumed.train()
    resume_optimizer.zero_grad(set_to_none=True)
    resume_loss, _ = resumed(preprocessor(collate_smoke_sample(copy.deepcopy(sample))))
    if not torch.isfinite(resume_loss):
        raise AssertionError("Nonfinite loss after restoring training state")
    resume_loss.backward()
    resume_optimizer.step()
    if parameter_digest(resumed.get_base_model(), adapters=True) == adapter_before:
        raise AssertionError("Restored adapter is not updating")
    if parameter_digest(resumed.get_base_model()) != before:
        raise AssertionError("Base changed after resumed adapter training")
    resumed.save_pretrained(output / "continued_checkpoint")
    report.update(independent_restore_max_abs=independent_error, continuation_step_passed=True,
                  continuation_loss=float(resume_loss.detach()),
                  rtc_max_delay=config.rtc_training_max_delay, deployment_prefix_steps=args.prefix_steps)
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export_task")
    for option in ("checkpoint", "base-path", "output", "metadata", "targets", "offline-observation"):
        export.add_argument("--" + option, required=True)
    export.add_argument("--tokenizer-path")
    audit = sub.add_parser("audit_targets")
    audit.add_argument("--base-path", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--extended", action="store_true")
    audit.add_argument("--include-vision", action="store_true")
    audit.add_argument("--meta", action="store_true", help="Inspect real named_modules on meta without loading weights")
    smoke_parser = sub.add_parser("smoke")
    for option in ("base-path", "dataset-root", "repo-id", "output", "action-unit"):
        smoke_parser.add_argument("--" + option, required=True)
    smoke_parser.add_argument("--device", default="cuda")
    smoke_parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    smoke_parser.add_argument("--steps", type=int, default=3)
    smoke_parser.add_argument("--rank", type=int, default=32)
    smoke_parser.add_argument("--alpha", type=int, default=64)
    smoke_parser.add_argument("--lr", type=float, default=2.5e-5)
    smoke_parser.add_argument("--seed", type=int, default=1000)
    smoke_parser.add_argument("--prefix-steps", type=int, default=0)
    smoke_parser.add_argument("--rtc-max-delay", type=int)
    smoke_parser.add_argument("--tokenizer-path")
    smoke_parser.add_argument("--extended", action="store_true")
    smoke_parser.add_argument("--atol", type=float, default=1e-5)
    smoke_parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.command == "smoke":
        if args.steps < 2:
            parser.error("At least two steps are required to assess both LoRA factors")
        smoke(args)
    elif args.command == "audit_targets":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        config = PreTrainedConfig.from_pretrained(args.base_path)
        config.device = args.device
        config.compile_model = False
        if args.meta:
            import torch
            config.device = "meta"
            with torch.device("meta"):
                model = get_policy_class("pi05")(config)
        else:
            model = get_policy_class("pi05").from_pretrained(args.base_path, config=config, local_files_only=True)
        result = joint_targets(model, args.extended, args.include_vision, diagnostic_path=args.output)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: len(result[key]) for key in ("vlm", "expert", "projections", "vision")}))
    else:
        result = export_task(args.checkpoint, args.base_path, args.output, read_json(args.metadata),
                             args.targets, args.offline_observation, args.tokenizer_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
