"""Offline-first hot-switch tooling. No command in this module connects to a robot.

Use ``python -m lerobot.scripts.lerobot_hotswitch --help`` without the PI stack.
Model-dependent imports are deliberately deferred until the selected command.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _write_json(value, output=None):
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def environment_report():
    """Read package metadata without importing CUDA or constructing a policy."""
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "draccus", "grpcio", "protobuf", "lerobot"):
        try:
            dist = importlib.metadata.distribution(name)
            direct = dist.read_text("direct_url.json")
            packages[name] = {
                "version": dist.version,
                "location": str(dist.locate_file("")),
                "direct_url": json.loads(direct) if direct else None,
            }
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "executable": sys.executable, "packages": packages}
    spec = importlib.util.find_spec("lerobot")
    report["lerobot_import"] = spec.origin if spec else None
    root = Path(__file__).resolve().parents[3]
    report["source_root"] = str(root)
    for label, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("worktree", ["git", "status", "--short"]),
        (
            "gpu",
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        ),
    ):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15)
            report[label] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report[label] = {"error": str(exc)}
    report["pi05_compatibility"] = "UNVERIFIED: versions and metadata are evidence, not a model forward test"
    return report


def _load_policy(base_path, device):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    return PI05Policy.from_pretrained(base_path, device=device, local_files_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env_check", help="Read environment/GPU usage; never allocate GPU memory")
    env.add_argument("--output")
    targets = sub.add_parser("audit_targets", help="Load local base and inventory actual joint LoRA modules")
    targets.add_argument("--base", required=True)
    targets.add_argument("--device", default="cpu")
    targets.add_argument("--extended", action="store_true")
    targets.add_argument("--vision", action="store_true")
    targets.add_argument("--output", required=True)
    export = sub.add_parser(
        "export_task", help="Export validated task package with processors and content hashes"
    )
    export.add_argument("--adapter", required=True)
    export.add_argument("--base", required=True)
    export.add_argument("--metadata", required=True)
    export.add_argument("--output", required=True)
    train = sub.add_parser("train_task", help="Validate a local task plan, then invoke the existing trainer")
    train.add_argument("--plan", required=True)
    train.add_argument("--task", required=True)
    train.add_argument(
        "--execute", action="store_true", help="Execute after validation; default only prints command"
    )
    smoke = sub.add_parser(
        "smoke", help="Real local PI05 joint LoRA mechanism test; requires compatible PI environment"
    )
    smoke.add_argument("--base", required=True)
    smoke.add_argument(
        "--observations", required=True, help="Offline NPZ with real observation and action tensors"
    )
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--steps", type=int, default=3)
    smoke.add_argument(
        "--rtc-max-delay",
        type=int,
        default=0,
        help="Positive maximum also validates real clean-prefix mechanics",
    )
    serve = sub.add_parser("serve", help="Preload, warm and serve trusted local task registry")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry")
    source.add_argument(
        "--simulate", action="store_true", help="Only deterministic simulated policies, no weight files"
    )
    serve.add_argument("--prefix-steps", type=int, default=0, help="Simulated TRC deployment prefix")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--device")
    serve.add_argument("--mode", choices=("shared", "dual"))
    switch = sub.add_parser(
        "switch_task", help="Request a control-boundary switch from an active offline client"
    )
    switch.add_argument("--control-file", required=True)
    switch.add_argument("--task", required=True)
    switch.add_argument("--operation-id")
    replay = sub.add_parser(
        "offline_client", help="Run a simulated robot against a running server; never creates hardware"
    )
    replay.add_argument("--server", default="127.0.0.1:8080")
    replay.add_argument("--task", required=True)
    inputs = replay.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--observations", help="JSON mapping task_id to offline NPZ files")
    inputs.add_argument("--simulated-observations", action="store_true")
    replay.add_argument("--control-file", required=True)
    replay.add_argument("--duration", type=float, default=30)
    replay.add_argument("--output", required=True)
    bench = sub.add_parser("benchmark", help="Offline observations and simulated control interface only")
    bench.add_argument("--output", required=True)
    bench.add_argument("--switches", type=int, default=200)
    bench.add_argument("--registry", help="Omit to benchmark explicitly labelled simulated policies")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--mode", choices=("both", "shared", "dual"), default="both")
    bench.add_argument("--prefix-steps", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "env_check":
            _write_json(environment_report(), args.output)
        elif args.command == "audit_targets":
            from lerobot.async_inference.task_bundle import audit_targets

            policy = _load_policy(args.base, args.device)
            result = audit_targets(policy, extended=args.extended, vision=args.vision)
            result["named_modules"] = [name for name, _ in policy.named_modules()]
            _write_json(result, args.output)
        elif args.command == "export_task":
            from lerobot.async_inference.task_bundle import export_task

            metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
            destination = export_task(args.adapter, args.output, metadata, args.base)
            print(destination)
        elif args.command == "train_task":
            from lerobot.async_inference.hotswitch_tools import train_task

            train_task(args.plan, args.task, execute=args.execute)
        elif args.command == "smoke":
            from lerobot.async_inference.hotswitch_smoke import run_real_smoke

            _write_json(
                run_real_smoke(
                    args.base,
                    args.observations,
                    args.output,
                    device=args.device,
                    steps=args.steps,
                    rtc_max_delay=args.rtc_max_delay,
                )
            )
        elif args.command == "serve":
            from lerobot.async_inference.hotswitch_tools import serve_registry

            serve_registry(
                args.registry,
                args.host,
                args.port,
                args.device,
                args.mode,
                simulate=args.simulate,
                prefix_steps=args.prefix_steps,
            )
        elif args.command == "switch_task":
            from lerobot.async_inference.hotswitch_tools import write_switch_request

            _write_json(write_switch_request(args.control_file, args.task, args.operation_id))
        elif args.command == "offline_client":
            from lerobot.async_inference.hotswitch_tools import offline_client

            _write_json(
                offline_client(
                    args.server,
                    args.task,
                    args.control_file,
                    args.output,
                    args.duration,
                    args.observations,
                    args.simulated_observations,
                )
            )
        elif args.command == "benchmark":
            from lerobot.async_inference.hotswitch_benchmark import benchmark

            _write_json(
                benchmark(
                    args.output, args.switches, args.registry, args.device, args.mode, args.prefix_steps
                )
            )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"hot-switch: {exc}\n")


if __name__ == "__main__":
    main()
