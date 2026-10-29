"""Opt-in real PI05/PEFT smoke test; never imports or constructs a robot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path

from .task_bundle import audit_targets, content_fingerprint, joint_target_pattern


def require_pi05_environment():
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
        from transformers.models.siglip import check
    except ImportError as exc:
        raise RuntimeError(
            "PI05 requires this repository's custom Transformers branch (AdaRMSNorm and SigLIP check)"
        ) from exc

    if "cond" not in inspect.signature(GemmaRMSNorm.forward).parameters:
        raise RuntimeError("Installed Transformers lacks the PI05 AdaRMSNorm cond interface")
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError("PI05 custom Transformers compatibility check failed")


def run_real_smoke(
    base_path,
    observations_path,
    output,
    *,
    device="cuda",
    steps=3,
    task_text="mechanism selftest",
    atol=1e-4,
    rtol=1e-4,
    processor_path=None,
    rtc_max_delay=0,
):
    """Train two temporary adapters sequentially on one supplied offline labeled batch.

    This proves mechanics only. It does not provide ACP labels or establish a task
    success rate, and writes no formal-training claim. Caller confirms GPU availability.
    """
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model_state_dict
    from safetensors.torch import save_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    if steps < 3:
        raise ValueError("At least three updates are required to observe both LoRA factors")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real GPU smoke not executed")
    require_pi05_environment()
    base_path = Path(base_path).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fingerprint = content_fingerprint(base_path)
    config = PreTrainedConfig.from_pretrained(base_path, local_files_only=True)
    config.device, config.compile_model = str(device), False
    config.pretrained_path, config.train_expert_only = base_path, False
    if not isinstance(rtc_max_delay, int) or not 0 <= rtc_max_delay < config.chunk_size:
        raise ValueError("rtc_max_delay must satisfy 0 <= d < checkpoint chunk_size")
    config.rtc_training_max_delay = rtc_max_delay
    if config.rtc_config is not None and config.rtc_config.enabled:
        raise ValueError(
            "This mechanism test uses clean-prefix TRC; legacy gradient RTC requires its separate test"
        )
    cls = get_policy_class("pi05")
    base = cls.from_pretrained(base_path, config=config, local_files_only=True).to(device)
    report = audit_targets(base)
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=processor_path or base_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    with np.load(observations_path, allow_pickle=False) as data:
        observation = {k: torch.as_tensor(data[k]) for k in data.files}
    if "action" not in observation:
        raise ValueError("Offline batch requires real action targets; no labels are fabricated")
    observation["task"] = task_text
    batch = pre(observation)

    def digest_base(model):
        digest = hashlib.sha256()
        for name, tensor in model.named_parameters():
            if "lora_" in name:
                continue
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def predict(model, prefix=None):
        torch.manual_seed(101)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(101)
        model.eval()
        underlying = model.get_base_model() if hasattr(model, "get_base_model") else model
        underlying.reset()
        kwargs = {}
        if prefix is not None:
            kwargs = {"prefix_steps": prefix.shape[1], "action_prefix": prefix.to(device)}
        actual = underlying.predict_action_chunk(batch, **kwargs).detach().cpu()
        if not torch.isfinite(actual).all():
            raise ValueError("Nonfinite real PI05 inference output")
        if prefix is not None and not torch.equal(actual[:, : prefix.shape[1]], prefix.cpu()):
            raise AssertionError("Real PI05 clean prefix was not held exactly")
        return actual

    # Validate the original local Base before altering its layers with PEFT.
    base.eval()
    with torch.no_grad():
        base_loss, _ = base(batch)
    if not torch.isfinite(base_loss):
        raise ValueError("Unadapted Base forward produced a nonfinite loss")
    base_output = predict(base)
    if rtc_max_delay:
        predict(base, base_output[:, :rtc_max_delay])

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0, target_modules=joint_target_pattern(), modules_to_save=[]
    )
    lora.base_model_name_or_path = str(base_path)
    policy = base.wrap_with_peft(peft_config=lora)
    before = digest_base(policy)
    output.mkdir(parents=True)
    expected, update_report, losses, effects = {}, {}, {}, {}
    expected_rtc, rtc_prefixes = {}, {}
    sampled_prefix_lengths = {}
    for task in ("default", "B"):
        if task == "B":
            policy.add_adapter(task, lora)
        policy.set_adapter(task)
        initial = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=config.optimizer_lr
        )
        losses[task] = []
        sampled_prefix_lengths[task] = []
        policy.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_details = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite real PI05 training loss")
            loss.backward()
            for p in policy.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise ValueError("Nonfinite LoRA gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in policy.parameters() if p.requires_grad):
                raise ValueError("Nonfinite LoRA parameter after optimizer update")
            losses[task].append(loss.item())
            if rtc_max_delay:
                sampled_prefix_lengths[task].append(loss_details["rtc_prefix_length"])
        if rtc_max_delay and not any(length > 0 for length in sampled_prefix_lengths[task]):
            raise AssertionError(
                "No nonzero RTC prefix was sampled; repeat the mechanism test with more steps"
            )
        changed = [
            n
            for n, p in policy.named_parameters()
            if n in initial and not torch.equal(initial[n], p.detach().cpu())
        ]
        effective_updates = [n for n in changed if ".lora_B." in n]
        branches = {
            "language": any(".paligemma.language_model." in n for n in effective_updates),
            "expert": any(".gemma_expert." in n for n in effective_updates),
        }
        for projection in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            branches[projection] = any(f".{projection}.lora_B." in n for n in effective_updates)
        if not all(branches.values()):
            raise AssertionError(f"Joint LoRA branches failed to update: {branches}")
        update_report[task] = branches
        expected[task] = predict(policy)
        if rtc_max_delay:
            rtc_prefixes[task] = expected[task][:, :rtc_max_delay].clone()
            expected_rtc[task] = predict(policy, rtc_prefixes[task])
        with policy.disable_adapter():
            disabled = predict(policy)
        effects[task] = (expected[task] - disabled).abs().max().item()
        if effects[task] == 0:
            raise AssertionError("Inference did not change when the trained adapter was disabled")
        target = output / ("A" if task == "default" else "B")
        target.mkdir()
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in get_peft_model_state_dict(policy, adapter_name=task).items()
            },
            target / "adapter_model.safetensors",
        )
        policy.peft_config[task].save_pretrained(target)
        config.save_pretrained(target)
        pre.save_pretrained(target)
        post.save_pretrained(target)
        if digest_base(policy) != before:
            raise AssertionError("Shared Base values changed during adapter training")
    # Free the training model before independent checkpoint restoration.
    del optimizer, policy, base
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    consistency, merged_consistency, rtc_consistency = {}, {}, {}
    for task, name in (("default", "A"), ("B", "B")):
        restored = cls.from_pretrained(base_path, config=config, local_files_only=True).to(device)
        restored = PeftModel.from_pretrained(restored, output / name, is_trainable=True)
        actual = predict(restored)
        delta = actual - expected[task]
        consistency[name] = {"max_abs_error": delta.abs().max().item(), "mse": delta.square().mean().item()}
        torch.testing.assert_close(actual, expected[task], atol=atol, rtol=rtol)
        if rtc_max_delay:
            rtc_actual = predict(restored, rtc_prefixes[task])
            torch.testing.assert_close(rtc_actual, expected_rtc[task], atol=atol, rtol=rtol)
            rtc_delta = rtc_actual - expected_rtc[task]
            rtc_consistency[name] = {
                "max_abs_error": rtc_delta.abs().max().item(),
                "mse": rtc_delta.square().mean().item(),
            }
        # This independent comparison copy may be merged; the resident shared Base is never merged.
        restored.merge_adapter(safe_merge=True)
        merged_actual = predict(restored)
        torch.testing.assert_close(merged_actual, actual, atol=atol, rtol=rtol)
        merged_delta = merged_actual - actual
        merged_consistency[name] = {
            "max_abs_error": merged_delta.abs().max().item(),
            "mse": merged_delta.square().mean().item(),
        }
        if rtc_max_delay:
            torch.testing.assert_close(
                predict(restored, rtc_prefixes[task]), expected_rtc[task], atol=atol, rtol=rtol
            )
        restored.unmerge_adapter()
        # Continue training after restore and verify trainable optimizer membership.
        restored.train()
        if any(p.requires_grad and "lora_" not in n for n, p in restored.named_parameters()):
            raise AssertionError("Restore made shared Base parameters trainable")
        optimizer = torch.optim.AdamW(
            [p for p in restored.parameters() if p.requires_grad], lr=config.optimizer_lr
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = restored(batch)
        loss.backward()
        optimizer.step()
        del restored, optimizer
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    shared = cls.from_pretrained(base_path, config=config, local_files_only=True).to(device)
    shared = PeftModel.from_pretrained(shared, output / "A", adapter_name="A", is_trainable=False)
    shared.load_adapter(output / "B", adapter_name="B", is_trainable=False)
    switch_outputs, rtc_switch_outputs = [], []
    for task in ("A", "B", "A"):
        shared.set_adapter(task)
        shared.get_base_model().reset()
        actual = predict(shared)
        torch.testing.assert_close(actual, expected["default" if task == "A" else "B"], atol=atol, rtol=rtol)
        switch_outputs.append(actual)
        if rtc_max_delay:
            original = "default" if task == "A" else "B"
            rtc_actual = predict(shared, rtc_prefixes[original])
            torch.testing.assert_close(rtc_actual, expected_rtc[original], atol=atol, rtol=rtol)
            rtc_switch_outputs.append(rtc_actual)
    torch.testing.assert_close(switch_outputs[0], switch_outputs[2], atol=atol, rtol=rtol)
    if rtc_max_delay:
        torch.testing.assert_close(rtc_switch_outputs[0], rtc_switch_outputs[2], atol=atol, rtol=rtol)
    result = {
        "status": "real_gpu_passed" if str(device).startswith("cuda") else "real_cpu_passed",
        "formal_training": False,
        "acp_validated": False,
        "base_forward_passed": True,
        "base_forward_loss": base_loss.item(),
        "rtc_mechanism_passed": True if rtc_max_delay else None,
        "rtc_max_delay": rtc_max_delay,
        "rtc_tested_prefix_steps": [0, rtc_max_delay] if rtc_max_delay else [0],
        "rtc_restoration": rtc_consistency,
        "rtc_sampled_mean_prefix_lengths": sampled_prefix_lengths,
        "merged_restoration": merged_consistency,
        "device": str(device),
        "language_updated": True,
        "expert_updated": True,
        "save_restore_passed": True,
        "a_b_a_passed": True,
        "base_fingerprint": fingerprint,
        "steps_per_adapter": steps,
        "losses": losses,
        "joint_updates": update_report,
        "adapter_output_effect_max_abs": effects,
        "base_unchanged": True,
        "restoration": consistency,
        "a_b_a_max_abs": (switch_outputs[0] - switch_outputs[2]).abs().max().item(),
        "atol": atol,
        "rtol": rtol,
        "targets": report,
    }
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--observations", required=True, help="Offline NPZ with action targets and unbatched observations"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--task-text", default="mechanism selftest")
    parser.add_argument("--processor-path")
    parser.add_argument(
        "--rtc-max-delay",
        type=int,
        default=0,
        help="Opt-in clean-prefix training/inference mechanism range; does not provide ACP acceptance",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_real_smoke(
                args.base,
                args.observations,
                args.output,
                device=args.device,
                steps=args.steps,
                task_text=args.task_text,
                processor_path=args.processor_path,
                rtc_max_delay=args.rtc_max_delay,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
