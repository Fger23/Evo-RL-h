"""Serial, GPU-exclusive value -> genuine ACP labels -> joint LoRA/TRC workflow.

The default prints the exact commands. --execute requires a successful real GPU
mechanism report and a real GPU offline switch benchmark. Existing datasets and
models are never overwritten. A pilot is explicitly distinct from formal training.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def commands(config, profile, run_root, targets, python):
    budget = config["profiles"][profile]
    repo_id = "straighten_cash_registered_r2"
    source = config["source_dataset_root"]
    labeled = run_root / "labeled_dataset"
    checkpoint = run_root / "value_train/checkpoints/last/pretrained_model"
    workers = f"--num_workers={budget['num_workers']}"
    common = [f"--seed={config['seed']}", "--wandb.enable=false"]
    value = [python, "-u", "-m", "lerobot.scripts.lerobot_value_train",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={source}",
        "--value.type=pistar06", "--value.device=cuda", "--value.push_to_hub=false",
        f"--value.vision_repo_id={config['vision_repo_id']}",
        f"--value.language_repo_id={config['language_repo_id']}",
        "--value.dtype=float32", "--value.optimizer_lr=5e-5",
        "--targets.success_field=episode_success",
        f"--steps={budget['value_steps']}", f"--batch_size={budget['value_batch_size']}",
        f"--log_freq={20 if profile == 'pilot' else 200}",
        f"--save_freq={budget['value_save_freq']}", "--save_checkpoint=true", workers,
        f"--output_dir={run_root / 'value_train'}", f"--job_name=straighten_cash_value_{profile}", *common]
    infer = [python, "-u", "-m", "lerobot.scripts.lerobot_value_infer",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={labeled}",
        "--dataset.success_field=episode_success", f"--inference.checkpoint_path={checkpoint}",
        "--runtime.device=cuda", "--runtime.batch_size=32", "--runtime.num_workers=4",
        "--acp.enable=true", "--acp.n_step=50", "--acp.positive_ratio=0.3",
        f"--output_dir={run_root / 'value_infer'}", f"--job_name=straighten_cash_value_infer_{profile}",
        f"--seed={config['seed']}"]
    adapter = [python, "-u", "-m", "lerobot.scripts.lerobot_train_trc",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={labeled}",
        "--policy.type=pi05", f"--policy.pretrained_path={config['base_path']}",
        f"--policy.tokenizer_name={config['tokenizer_name']}",
        "--policy.device=cuda", "--policy.dtype=bfloat16", "--policy.push_to_hub=false",
        "--policy.compile_model=false", "--policy.gradient_checkpointing=true",
        "--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false",
        "--policy.optimizer_lr=2.5e-5", f"--policy.rtc_training_max_delay={config['rtc_training_max_delay']}",
        "--peft.r=32", "--peft.lora_alpha=64", "--peft.lora_dropout=0",
        "--peft.full_training_modules=[]", "--peft.target_modules=" + json.dumps(targets, separators=(",", ":")),
        "--acp.enable=true", "--acp.indicator_field=complementary_info.acp_indicator",
        f"--acp.indicator_dropout_prob={config['acp_indicator_dropout_prob']}",
        f"--steps={budget['adapter_steps']}", f"--batch_size={budget['adapter_batch_size']}",
        f"--log_freq={20 if profile == 'pilot' else 200}",
        f"--save_freq={budget['adapter_save_freq']}", "--save_checkpoint=true", workers,
        f"--output_dir={run_root / 'adapter_A'}", f"--job_name=straighten_cash_joint_lora_{profile}", *common]
    if profile == "formal":
        adapter.append("--verify_peft_base=true")
    return {"value-train": value, "value-infer": infer, "adapter": adapter}


def require_idle_gpu(gpu):
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    running = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    pids = [line.split(",")[1].strip() for line in running.splitlines() if line.split(",")[0].strip() == gpu_uuid]
    if pids:
        raise RuntimeError(f"GPU {gpu} already has compute PIDs {pids}; refusing to compete")
    return gpu_uuid


def require_gates(smoke_path, benchmark_path):
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("status") != "real_gpu_passed" or not smoke.get("frozen_base_unchanged"):
        raise ValueError("Real GPU LoRA mechanism gate has not passed")
    benchmark = json.loads(benchmark_path.read_text())
    if not {"shared", "dual"}.issubset(benchmark):
        raise ValueError("Offline benchmark must include both deployment modes")
    for mode in ("shared", "dual"):
        if benchmark[mode].get("status") != "GPU_OFFLINE_PASSED":
            raise ValueError(f"Offline switch gate did not pass for {mode}")
        if benchmark[mode].get("samples", 0) < 200:
            raise ValueError(f"Offline switch gate has fewer than 200 samples for {mode}")


def validate_run_root(original, run_root):
    if run_root.resolve().is_relative_to(original.resolve()):
        raise ValueError("Run output must be outside the source dataset")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def begin_shared_base_fingerprint(base_path, evidence_path):
    from lerobot.async_inference.hotswitch.manifest import base_fingerprint

    if evidence_path.exists():
        raise FileExistsError(f"Refusing to overwrite captured Base provenance: {evidence_path}")
    report = {"status": "STARTED", "base_reference": str(Path(base_path).resolve(strict=True)),
              "before": base_fingerprint(base_path),
              "before_captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(evidence_path, report)
    return report


def finish_shared_base_fingerprint(base_path, evidence_path):
    from lerobot.async_inference.hotswitch.manifest import base_fingerprint

    report = json.loads(evidence_path.read_text())
    if report.get("status") != "STARTED" or not report.get("before"):
        raise ValueError("Base file provenance must have an actual pre-training capture")
    report.update(after=base_fingerprint(base_path),
                  after_captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    unchanged = report["before"] == report["after"]
    report.update(status="PASSED" if unchanged else "FAILED", file_content_unchanged=unchanged)
    write_json(evidence_path, report)
    if not unchanged:
        raise ValueError("Shared Base file content changed during adapter training")
    return report


def validate_labels(root):
    import numpy as np
    import pyarrow.parquet as pq

    fields = ["complementary_info.value", "complementary_info.advantage", "complementary_info.acp_indicator"]
    count, positive = 0, 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=fields)
        for field in fields:
            values = np.asarray(table[field].to_pylist(), dtype=np.float64).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite or missing {field}")
            if field == fields[-1]:
                if not np.isin(values, [0, 1]).all():
                    raise ValueError("Invalid ACP indicator")
                positive += int(values.sum())
        count += len(table)
    expected = json.loads((root / "meta/info.json").read_text())["total_frames"]
    if count != expected or count == 0:
        raise ValueError("ACP labels do not cover every dataset row")
    return {"frames": count, "positive": positive, "negative": count - positive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/hotswitch/straighten_cash_value.json")
    parser.add_argument("--profile", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="Real named_modules audit JSON from the passing mechanism smoke")
    parser.add_argument("--stage", choices=["all", "value-train", "value-infer", "adapter"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    args = parser.parse_args()
    if args.execute and args.profile == "formal":
        from hotswitch_freeze_source import verify_frozen_source

        verify_frozen_source(args.run_root, execution_root=Path(__file__).resolve().parents[1],
            config_path=args.config, require_formal=True, check_package=True)
    config = json.loads(args.config.read_text())
    original = Path(config["source_dataset_root"])
    validate_run_root(original, args.run_root)
    target_report = json.loads(args.targets.read_text())
    targets = target_report["target_modules"]
    if not target_report.get("vlm") or not target_report.get("expert") or len(target_report.get("projections", [])) != 4:
        raise ValueError("Both Transformer branches and four action/time projections must be audited")
    plan = commands(config, args.profile, args.run_root, targets, sys.executable)
    stages = list(plan) if args.stage == "all" else [args.stage]
    if not args.execute:
        print(json.dumps({"profile": args.profile, "scope": config["profiles"][args.profile]["scope"],
                          "commands": {k: shlex.join(plan[k]) for k in stages}}, indent=2))
        return
    if not args.smoke_report or not args.benchmark_report:
        parser.error("Execution requires --smoke-report and --benchmark-report")
    require_gates(args.smoke_report, args.benchmark_report)
    manifest = json.loads((original / "prepare_manifest.json").read_text())
    if manifest.get("status") != "validated":
        raise ValueError("Registered-episode dataset validation must pass first")
    args.run_root.mkdir(parents=True, exist_ok=True)
    import fcntl
    initial_gpu_uuid = require_idle_gpu(config["gpu"])
    lock_dir = Path(config["coordination_root"]) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{initial_gpu_uuid}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(config["gpu"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                   HF_HUB_OFFLINE="1", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")
        env["PATH"] = str(Path(sys.base_prefix) / "bin") + os.pathsep + env.get("PATH", "")
        from hotswitch_prepare_data import fingerprints
        before = fingerprints(original)
        new_status = {"profile": args.profile, "scope": config["profiles"][args.profile]["scope"], "stages": {},
                  "config": config, "targets_sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest(),
                  "robot_connected": False, "task_success_rate": "not measured", "second_task_trained": False}
        status_path = args.run_root / "pipeline_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else new_status
        if any(status.get(key) != new_status[key] for key in ["profile", "config", "targets_sha256"]):
            raise ValueError("Existing run configuration differs; use a new run root")
        for stage in stages:
            stage_log = args.run_root / f"{stage}.log"
            model_dir = {"value-train": "value_train", "value-infer": "value_infer", "adapter": "adapter_A"}[stage]
            if stage in status["stages"] or stage_log.exists() or (args.run_root / model_dir).exists():
                raise FileExistsError(f"Refusing to overwrite previous {stage} artifacts; use a new run root or trainer resume")
            gpu_uuid = require_idle_gpu(config["gpu"])
            if gpu_uuid != initial_gpu_uuid:
                raise RuntimeError("GPU identity changed during the pipeline")
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            if stage == "value-infer":
                checkpoint = args.run_root / "value_train/checkpoints/last/pretrained_model"
                if not checkpoint.is_dir():
                    raise FileNotFoundError(checkpoint)
                labeled = args.run_root / "labeled_dataset"
                if labeled.exists():
                    raise FileExistsError(f"Refusing to relabel existing {labeled}")
                shutil.copytree(original, labeled, symlinks=True)
            if stage == "adapter":
                labels = validate_labels(args.run_root / "labeled_dataset")
                (args.run_root / "label_validation.json").write_text(json.dumps(labels, indent=2))
            if require_idle_gpu(config["gpu"]) != initial_gpu_uuid:
                raise RuntimeError("GPU identity changed before process launch")
            if stage == "adapter" and args.profile == "formal":
                begin_shared_base_fingerprint(config["base_path"], args.run_root / "shared_base_fingerprint.json")
                if require_idle_gpu(config["gpu"]) != initial_gpu_uuid:
                    raise RuntimeError("GPU identity changed after Base provenance capture")
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            stage_started = time.perf_counter()
            with stage_log.open("x") as log:
                process = subprocess.Popen(plan[stage], env=env, stdout=log, stderr=subprocess.STDOUT)
                status["stages"][stage] = {"status": "running", "pid": process.pid, "gpu_uuid": gpu_uuid,
                    "started_utc": started, "argv": plan[stage]}
                write_json(status_path, status)
                code = process.wait()
            if code == 0:
                try:
                    if fingerprints(original) != before:
                        raise RuntimeError("The unlabeled source copy changed unexpectedly")
                    if stage == "adapter" and args.profile == "formal":
                        finish_shared_base_fingerprint(config["base_path"], args.run_root / "shared_base_fingerprint.json")
                except Exception as error:
                    status["stages"][stage].update(status="failed", exit_code=code, validation_error=str(error),
                        elapsed_s=time.perf_counter() - stage_started,
                        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write_json(status_path, status)
                    raise
            status["stages"][stage].update(status="passed" if code == 0 else "failed", exit_code=code,
                elapsed_s=time.perf_counter() - stage_started,
                ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(status_path, status)
            if code:
                raise RuntimeError(f"{stage} exited {code}; see {args.run_root / (stage + '.log')}")
        if (args.run_root / "labeled_dataset").exists():
            labels = validate_labels(args.run_root / "labeled_dataset")
            (args.run_root / "label_validation.json").write_text(json.dumps(labels, indent=2))
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
