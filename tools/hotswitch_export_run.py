"""Export a completed single-task value/ACP/TRC run using the existing bundle API.

A_copy contains exactly the A training result for residency/ABA measurements. It
is never represented as a second trained task. This tool loads no policy or GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from lerobot.async_inference.hotswitch.manifest import digest_file, export_task, read_json


def completed_stage(status, stage):
    evidence = status.get("stages", {}).get(stage, {})
    if evidence.get("status") != "passed" or evidence.get("exit_code") != 0:
        raise ValueError(f"Pipeline stage {stage} has not completed successfully")
    return evidence


def checkpoint_step(checkpoint, expected):
    state = checkpoint.resolve(strict=True).parent / "training_state/training_step.json"
    actual = int(read_json(state)["step"])
    if actual != expected:
        raise ValueError(f"Checkpoint step {actual} differs from configured budget {expected}: {checkpoint}")
    return actual


def frozen_base_evidence(run_root, profile, steps, training):
    if not training.get("verify_peft_base", False):
        if profile == "formal":
            raise ValueError("Formal export requires the shared trainer frozen Base audit")
        return {"status": "NOT_ENABLED", "frozen_base_unchanged": None,
                "scope": "Pilot snapshot did not enable full-duration shared-trainer Base hashing"}
    path = run_root / "adapter_A/peft_base_audit.json"
    audit = read_json(path)
    from lerobot.utils.peft_audit import validate_audit_coverage

    coverage = validate_audit_coverage(path, steps,
        required_groups=("vlm", "expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"),
        require_final=True, expected_checkpoint=run_root / "adapter_A/checkpoints/last")
    return {"status": "PASSED", "frozen_base_unchanged": True, "source_report": str(path),
            "source_report_sha256": digest_file(path), "steps_completed": steps, "coverage": coverage, "report": audit}


def source_code_evidence(run_root, profile):
    snapshot_path = run_root / "source_snapshot_manifest.json"
    if not snapshot_path.is_file():
        if profile == "formal":
            raise ValueError("Formal export requires the actual frozen source snapshot manifest")
        return {"status": "NOT_RECORDED"}
    from hotswitch_freeze_source import verify_frozen_source

    return verify_frozen_source(run_root, require_formal=profile == "formal")


def tokenizer_evidence(checkpoint, profile, configured_path):
    saved = checkpoint / "tokenizer"
    if saved.is_dir():
        return {"status": "CHECKPOINT_SAVED_TOKENIZER", "saved_actual_source": str(saved),
                "configured_source": str(configured_path)}
    if profile == "formal":
        raise ValueError("Formal export requires the tokenizer saved from the training process in checkpoint/tokenizer")
    return {"status": "PILOT_EXTERNAL_TOKENIZER", "configured_source": str(configured_path),
            "scope": "Older pilot snapshot did not persist the in-memory tokenizer"}


def shared_base_file_evidence(run_root, profile, base_path):
    from lerobot.async_inference.hotswitch.manifest import base_fingerprint

    path = run_root / "shared_base_fingerprint.json"
    if not path.is_file():
        if profile == "formal":
            raise ValueError("Formal export requires captured before/after Base file fingerprints")
        return {"status": "NOT_CAPTURED", "scope": "Old pilot did not capture a pre-training file fingerprint"}
    report = read_json(path)
    if (report.get("status") != "PASSED" or report.get("file_content_unchanged") is not True
            or not report.get("before") or report.get("before") != report.get("after")):
        raise ValueError("Shared Base file fingerprint evidence did not pass")
    if Path(report["base_reference"]).resolve() != base_path.resolve(strict=True):
        raise ValueError("Captured Base source differs from the exported Base")
    if base_fingerprint(base_path) != report["before"]:
        raise ValueError("Base file content changed after the completed adapter run")
    return {"status": "PASSED", "source_report_sha256": digest_file(path), "report": report}


def build_metadata(args):
    """Read actual persisted evidence; no training completion is inferred from paths."""
    import pandas as pd
    from hotswitch_evorl_pipeline import validate_labels, validate_run_root
    from safetensors import safe_open

    run_root = args.run_root.resolve(strict=True)
    status = read_json(run_root / "pipeline_status.json")
    profile = status["profile"]
    if profile not in ("pilot", "formal"):
        raise ValueError("Unknown run profile")
    settings = status["config"]
    budget = settings["profiles"][profile]
    for stage in ("value-train", "value-infer", "adapter"):
        completed_stage(status, stage)
    checkpoint = run_root / "adapter_A/checkpoints/last/pretrained_model"
    value_checkpoint = run_root / "value_train/checkpoints/last/pretrained_model"
    steps = checkpoint_step(checkpoint, int(budget["adapter_steps"]))
    value_steps = checkpoint_step(value_checkpoint, int(budget["value_steps"]))
    training = read_json(checkpoint / "train_config.json")
    base_evidence = frozen_base_evidence(run_root, profile, steps, training)
    code_provenance = source_code_evidence(run_root, profile)
    tokenizer_provenance = tokenizer_evidence(checkpoint, profile, args.tokenizer_path)
    base_file_provenance = shared_base_file_evidence(run_root, profile, args.base_path)
    value_training = read_json(value_checkpoint / "value_train_config.json")
    policy = read_json(checkpoint / "config.json")
    trained_tokenizer = Path(policy.get("tokenizer_name", ""))
    if not trained_tokenizer.is_dir() or trained_tokenizer.resolve() != args.tokenizer_path.resolve(strict=True):
        raise ValueError("Tokenizer path differs from the local tokenizer actually configured for training")
    source = Path(settings["source_dataset_root"]).resolve(strict=True)
    validate_run_root(source, args.output)
    labeled = run_root / "labeled_dataset"
    if Path(training["dataset"]["root"]).resolve() != labeled.resolve():
        raise ValueError("Adapter was not trained on this run's private labeled dataset")
    if Path(value_training["dataset"]["root"]).resolve() != source:
        raise ValueError("Value model used a different source dataset")
    if Path(settings["base_path"]).resolve() != args.base_path.resolve(strict=True):
        raise ValueError("Requested Base differs from the pipeline Base")
    if int(training["steps"]) != steps or int(value_training["steps"]) != value_steps:
        raise ValueError("Saved training budgets differ from actual checkpoint steps")
    acp = training["acp"]
    if not acp.get("enable") or acp.get("indicator_field") != "complementary_info.acp_indicator":
        raise ValueError("Export requires actual ACP training with the expected real indicator field")
    max_delay = int(policy.get("rtc_training_max_delay", 0))
    if not 0 < args.prefix_steps <= max_delay < int(policy["chunk_size"]):
        raise ValueError("Deployment prefix must be positive and within the checkpoint's real RTC range")
    infer_arguments = completed_stage(status, "value-infer")["argv"]
    if f"--dataset.root={labeled}" not in infer_arguments or f"--inference.checkpoint_path={value_checkpoint}" not in infer_arguments:
        raise ValueError("Value inference provenance differs from this run's dataset/checkpoint")
    if "--acp.enable=true" not in infer_arguments:
        raise ValueError("Value inference did not enable ACP label generation")
    label_summary = validate_labels(labeled)
    info = read_json(labeled / "meta/info.json")
    source_manifest = source / "prepare_manifest.json"
    if read_json(source_manifest).get("status") != "validated":
        raise ValueError("Source preparation was not validated")
    task_frame = pd.read_parquet(labeled / "meta/tasks.parquet")
    if len(task_frame) != 1:
        raise ValueError("This exporter requires exactly one genuine task")
    text = str(task_frame.index[0])
    if "Advantage:" in text:
        raise ValueError("ACP must be added by the backend once, rather than stored in task text")
    actions = info["features"]["action"]
    names = actions.get("names")
    dimension = int(policy["output_features"]["action"]["shape"][0])
    if not names or len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("Explicit ordered dataset action names must match the checkpoint dimension")
    cameras = {key: key for key in policy["input_features"] if key.startswith("observation.images.")}
    if set(cameras) != {key for key, value in info["features"].items() if value.get("dtype") == "video"}:
        raise ValueError("Dataset/checkpoint camera schemas differ")
    with safe_open(args.offline_observation, framework="np") as observation:
        available_keys = set(observation.keys())
        for key in ["observation.state", *cameras]:
            if key not in available_keys:
                raise ValueError(f"Offline observation is missing {key}")
            if tuple(observation.get_slice(key).get_shape()) != tuple(policy["input_features"][key]["shape"]):
                raise ValueError(f"Offline observation shape differs from checkpoint: {key}")
    metadata = {
        "task_id": "A", "task_text": text,
        "action_schema": {"names": names, "units": [args.action_unit] * dimension, "control_hz": info["fps"]},
        "camera_map": cameras,
        "tokenizer": {"name": str(args.tokenizer_path.resolve(strict=True)), "max_length": policy["tokenizer_max_length"]},
        "rtc": {"mode": "training_time", "max_delay": max_delay, "prefix_steps": args.prefix_steps},
        "acp": {"enabled": True, "prompt_strategy": "positive", "beta": 1.0, "cfg_enabled": False,
                "indicator_field": acp["indicator_field"], "indicator_dropout_prob": acp["indicator_dropout_prob"],
                "tag_owner": "SharedAdapterBackend.infer"},
        "training_source": {
            "status": "pilot_completed" if profile == "pilot" else "formal_training_completed",
            "profile": profile, "steps": steps, "configured_steps": int(budget["adapter_steps"]),
            "dataset": str(labeled), "repo_id": training["dataset"]["repo_id"],
            "source_dataset": str(source), "source_manifest_sha256": digest_file(source_manifest),
            "checkpoint": str(checkpoint.resolve()), "train_config_sha256": digest_file(checkpoint / "train_config.json"),
            "value_checkpoint": str(value_checkpoint.resolve()), "value_steps": value_steps,
            "value_train_config_sha256": digest_file(value_checkpoint / "value_train_config.json"),
            "label_summary": label_summary, "labels_generated_by": "lerobot.scripts.lerobot_value_infer",
            "label_inference_argv": infer_arguments, "second_task_trained": False, "mechanism_copy": False,
            "formal_training_completed": profile == "formal", "robot_validation_completed": False,
            "frozen_base_audit": base_evidence,
            "source_code": code_provenance,
            "tokenizer_provenance": tokenizer_provenance,
            "shared_base_file_fingerprint": base_file_provenance,
            "task_success_rate": "NOT_MEASURED: no physical rollout",
        },
        "deployment_validation": {"status": "NOT_EXECUTED: exported artifact requires GPU reload/warmup and RTC acceptance",
            "action_units": args.action_unit, "hardware_validation_completed": False},
    }
    return checkpoint, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "base-path", "targets", "offline-observation", "tokenizer-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, default=6)
    parser.add_argument("--action-unit", default="recorded_robot_units_unverified")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint, metadata = build_metadata(args)
    args.output.mkdir(parents=True)
    manifests = {}
    for task_id in ("A", "A_copy"):
        task_metadata = copy.deepcopy(metadata)
        task_metadata["task_id"] = task_id
        if task_id == "A_copy":
            task_metadata["training_source"].update(mechanism_copy=True, copy_of="A")
        tokenizer_override = None if (checkpoint / "tokenizer").is_dir() else args.tokenizer_path
        manifests[task_id] = export_task(checkpoint, args.base_path, args.output / task_id, task_metadata,
                                         args.targets, args.offline_observation, tokenizer_override)
    adapter_sha = manifests["A"]["artifacts"]["adapter_model.safetensors"]
    if manifests["A_copy"]["artifacts"]["adapter_model.safetensors"] != adapter_sha:
        raise RuntimeError("Mechanism copy differs from A's exported adapter")
    registry = {"base_path": str(args.base_path.resolve()),
                "tasks": {name: str((args.output / name).resolve()) for name in ("A", "A_copy")},
                "scope": "single_trained_task_plus_identical_mechanism_copy", "second_task_trained": False}
    (args.output / "registry.json").write_text(json.dumps(registry, indent=2))
    (args.output / "export_report.json").write_text(json.dumps({"status": "EXPORTED_NOT_GPU_VALIDATED", "registry": registry,
        "identical_adapter_sha256": adapter_sha, "training_source": metadata["training_source"]}, indent=2))
    print(json.dumps({"status": "EXPORTED_NOT_GPU_VALIDATED", "registry": str(args.output / "registry.json"),
        "profile": metadata["training_source"]["profile"], "second_task_trained": False}))


if __name__ == "__main__":
    main()
