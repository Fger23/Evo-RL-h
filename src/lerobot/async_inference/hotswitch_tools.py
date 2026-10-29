"""Local operator tools; runtime requests contain registered task IDs only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def write_switch_request(control_file, task_id, operation_id=None):
    """Atomically publish to the active client's control-boundary mailbox."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ValueError("task_id must be a registered identifier, never a path")
    target = Path(control_file)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    payload = {"task_id": task_id, "operation_id": operation_id or str(uuid.uuid4())}
    fd, temporary = tempfile.mkstemp(prefix=".switch-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def train_task(plan_path, task_id, execute=False):
    """Resolve common Base and dispatch the existing SFT or ACP/TRC trainer.

    Missing real data/resources fail before importing a trainer. ``--execute``
    is explicit so inspecting a configuration never starts a long GPU job.
    """
    from .task_bundle import content_fingerprint, joint_target_pattern

    plan_file = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    base = plan.get("common_base")
    if not base:
        raise ValueError("Set common_base to one confirmed local PI05 checkpoint")
    if not Path(base).is_absolute():
        raise ValueError("common_base must be an absolute confirmed local path")
    base = Path(base).resolve(strict=True)
    fingerprint = content_fingerprint(base)
    tasks = plan.get("tasks", {})
    if task_id not in tasks or len(tasks) < 2:
        raise ValueError("The plan must register A and B (at least two tasks)")
    configs = {}
    for name, entry in tasks.items():
        path = Path(entry["config_path"])
        if not path.is_absolute():
            path = plan_file.parent / path
        config = json.loads(path.read_text(encoding="utf-8"))
        policy, peft = config.get("policy", {}), config.get("peft", {})
        if policy.get("type") != "pi05" or not peft:
            raise ValueError(f"{name}: explicit PI05 + PEFT configuration required")
        checkpoint = policy.get("pretrained_path")
        if not checkpoint:
            raise ValueError(f"{name}: policy.pretrained_path is required")
        if Path(checkpoint).resolve() != base and content_fingerprint(checkpoint) != fingerprint:
            raise ValueError(f"{name}: adapter training Base differs from common_base")
        # Resume is handled by the shared trainer, but this first task launcher
        # starts each task from the common unmodified base.
        if policy.get("use_peft", False) or config.get("resume", False):
            raise ValueError(
                "For continuation use the trainer's --resume=true with its saved train_config.json"
            )
        if peft.get("target_modules") not in (
            joint_target_pattern(),
            joint_target_pattern(extended=True),
            joint_target_pattern(vision=True),
            joint_target_pattern(extended=True, vision=True),
        ):
            raise ValueError(f"{name}: use a complete joint preset from audit_targets")
        if peft.get("full_training_modules", []) or peft.get("method_type", "LORA") != "LORA":
            raise ValueError("Shared Base v1 requires LoRA and no modules_to_save")
        if policy.get("compile_model", False):
            raise ValueError("Initial validation uses eager execution")
        if policy.get("push_to_hub") is not False or config.get("wandb", {}).get("enable", False):
            raise ValueError("Local task launcher requires push_to_hub=false and wandb.enable=false")
        dataset = config.get("dataset", {})
        if not dataset.get("repo_id") or not dataset.get("root"):
            raise ValueError(f"{name}: confirmed local dataset.repo_id and dataset.root are required")
        if not Path(dataset["root"]).is_absolute() or not Path(config.get("output_dir") or "").is_absolute():
            raise ValueError(f"{name}: dataset.root and output_dir must be absolute paths")
        info_path = Path(dataset["root"]) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        acp = config.get("acp", {})
        if acp.get("enable", False):
            info = json.loads(info_path.read_text(encoding="utf-8"))
            field = acp.get("indicator_field", "complementary_info.acp_indicator")
            if field not in info.get("features", {}):
                raise ValueError(f"{name}: real ACP indicator feature {field} missing")
            # Actual indicator values remain validated by compute_acp_indicator_stats
            # in lerobot_train.py; schema presence alone never passes ACP acceptance.
        trc = entry.get("training_time_rtc", False)
        maximum = policy.get("rtc_training_max_delay", 0)
        if not isinstance(maximum, int) or maximum < 0:
            raise ValueError(f"{name}: set an integer rtc_training_max_delay")
        if trc and (not acp.get("enable", False) or maximum <= 0):
            raise ValueError("ACP/TRC requires real indicators and positive rtc_training_max_delay")
        if not trc and policy.get("rtc_training_max_delay", 0) > 0:
            raise ValueError("Use training_time_rtc=true to retain the TRC entry-point validations")
        if not isinstance(config.get("steps"), int) or config["steps"] <= 0 or not config.get("output_dir"):
            raise ValueError(f"{name}: positive steps and explicit output_dir required")
        configs[name] = (path, config, trc)
    peft_keys = ("method_type", "target_modules", "r", "lora_alpha", "lora_dropout", "full_training_modules")
    structures = [{key: entry[1]["peft"].get(key) for key in peft_keys} for entry in configs.values()]
    if any(structure != structures[0] for structure in structures[1:]):
        raise ValueError("Task adapters must use the same LoRA structure before starting training")
    path, config, trc = configs[task_id]
    module = "lerobot.scripts.lerobot_train_trc" if trc else "lerobot.scripts.lerobot_train"
    command = [sys.executable, "-m", module, f"--config_path={path}"]
    print(
        json.dumps(
            {
                "task_id": task_id,
                "common_base_fingerprint": fingerprint,
                "command": command,
                "execute_requested": execute,
                "executed": False,
            },
            indent=2,
        )
    )
    if execute:
        report_path = plan.get("mechanism_validation_report")
        if not report_path:
            raise ValueError(
                "Run real PI05 mechanism validation before formal training; set mechanism_validation_report"
            )
        if not Path(report_path).is_absolute():
            raise ValueError("mechanism_validation_report must be an absolute path")
        evidence = json.loads(Path(report_path).read_text(encoding="utf-8"))
        required = (
            "base_forward_passed",
            "base_unchanged",
            "language_updated",
            "expert_updated",
            "save_restore_passed",
            "a_b_a_passed",
        )
        if (
            not all(evidence.get(key) is True for key in required)
            or evidence.get("base_fingerprint") != fingerprint
        ):
            raise ValueError("PI05 validation report incomplete or references another Base")
        if trc and not evidence.get("rtc_mechanism_passed", False):
            raise ValueError(
                "Run real PI05 nonzero-prefix mechanism validation before formal ACP/TRC training"
            )
        if trc and evidence.get("rtc_max_delay", 0) < config["policy"]["rtc_training_max_delay"]:
            raise ValueError("Real PI05 mechanism validation does not cover the configured training delay")
        device = config["policy"].get("device", "")
        if not device:
            raise ValueError("Select the confirmed available training device explicitly")
        if device.startswith("cuda"):
            if evidence.get("status") != "real_gpu_passed":
                raise ValueError("CUDA formal training requires a real_gpu_passed mechanism report")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch/device unavailable; no training started")
            index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(index)
            required_free = plan.get("required_free_gpu_bytes")
            if not required_free or free < required_free:
                raise RuntimeError(f"Require configured free GPU memory budget; free={free}, total={total}")
        subprocess.run(command, check=True)
        print(json.dumps({"task_id": task_id, "executed": True, "trainer_exit_code": 0}))
    return command


def serve_registry(
    registry=None, host="127.0.0.1", port=8080, device=None, mode=None, *, simulate=False, prefix_steps=0
):
    from concurrent import futures

    import grpc

    from lerobot.transport import services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options

    from .configs import PolicyServerConfig
    from .hotswitch import HotSwitchRuntime
    from .policy_server import PolicyServer

    if simulate:
        from .hotswitch_benchmark import SimulatedResidentBackend

        if not 0 <= prefix_steps <= 4:
            raise ValueError("Simulated prefix_steps must be in [0,4]")
        backend = SimulatedResidentBackend(mode or "shared", prefix_steps)
    else:
        from .hotswitch_backend import load_backend

        backend = load_backend(registry, device=device, mode=mode)
    runtime = HotSwitchRuntime(backend)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    service = PolicyServer(PolicyServerConfig(host=host, port=port, inference_latency=0), runtime)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    if not server.add_insecure_port(f"{host}:{port}"):
        raise RuntimeError(f"Cannot bind {host}:{port}")
    server.start()
    print(json.dumps({"address": f"{host}:{port}", "state": runtime.status()}, default=str))
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(5).wait()


def offline_client(
    address, task_id, control_file, output, duration=30, observations_path=None, simulated=False
):
    """An active switchable replay client with an in-memory control interface."""
    import pickle
    import time

    import grpc
    import numpy as np

    from lerobot.scripts.recording_remote_policy import RemotePolicyActionClient, RemotePolicyRecordConfig
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options

    from .hotswitch_benchmark import OfflineRobot

    if duration <= 0:
        raise ValueError("duration must be positive")
    channel = grpc.insecure_channel(address, options=grpc_channel_options())
    client = None
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        reply = stub.ControlPolicy(
            services_pb2.PolicySetup(
                data=pickle.dumps({"protocol": "lerobot.hotswitch", "version": 1, "op": "capabilities"})
            ),
            timeout=10,
        )
        info = pickle.loads(reply.data)  # nosec: same trusted protocol
        if not info.get("ok"):
            raise ValueError(info.get("error", "Hot-switch protocol unavailable"))
        tasks = info["state"]["tasks"]
        if task_id not in tasks:
            raise ValueError(f"Unregistered task {task_id}")
        metadata = tasks[task_id]
        names = metadata["action_schema"]["names"]
        fps = metadata.get("control_hz", metadata["action_schema"]["fps"])
        if simulated:
            if not all(task.get("simulation") is True for task in tasks.values()):
                raise ValueError("Synthetic observations are allowed only for explicitly simulated policies")
            observations = {name: dict.fromkeys(names, 0.0) for name in tasks}
        else:
            mapping_file = Path(observations_path).resolve(strict=True)
            mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
            observations = {}
            for name in tasks:
                with np.load(mapping_file.parent / mapping[name], allow_pickle=False) as archive:
                    observations[name] = {key: archive[key] for key in archive.files}

        class ReplayRobot(OfflineRobot):
            def get_observation(self):
                active = client._hotswitch.identity.task_id if client is not None else task_id
                return dict(observations[active])

        robot = ReplayRobot(names, observations[task_id])
        cfg = RemotePolicyRecordConfig(
            enable=True,
            hotswitch_task_id=task_id,
            hotswitch_simulation=True,
            hotswitch_control_file=control_file,
            actions_per_chunk=metadata.get("policy_config", {}).get("chunk_size", 8),
            chunk_size_threshold=0.5,
        )
        client = RemotePolicyActionClient(cfg, robot, fps=fps, stub=stub, channel=channel)
        client.start()
        print(json.dumps({"status": "ready", "task_id": task_id, "control_file": control_file}), flush=True)
        started = deadline = time.perf_counter()
        while time.perf_counter() - started < duration:
            action = client.get_action(robot.get_observation(), None, len(events))
            client.validate_action_before_send(action)
            client.acknowledge_action(robot.send_action(action))
            identity = client._hotswitch.identity
            events.append(
                {
                    "session_id": identity.session_id,
                    "task_id": identity.task_id,
                    "policy_epoch": identity.policy_epoch,
                    "accepted": robot.accepted[-1][0],
                    "action": action,
                }
            )
            deadline += 1 / fps
            time.sleep(max(0, deadline - time.perf_counter()))
        result = {
            "status": "simulated_control_passed",
            "physical_robot": False,
            "accepted": len(events),
            "tasks_seen": sorted({event["task_id"] for event in events}),
            "counters": client._hotswitch.counters,
        }
        (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if client is not None:
            client.stop()
        else:
            channel.close()
        with (destination / "actions.jsonl").open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event) + "\n")
