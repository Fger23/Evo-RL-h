"""Offline end-to-end RPC benchmark; control acceptance always uses a fake robot.

Simulated policies measure harness/protocol overhead only. Real registries use
the same client, control boundary, offline observations and resident backends.
No result from this module is a physical robot latency or rollout success rate.
"""

from __future__ import annotations

import gc
import json
import pickle
import statistics
import time
from pathlib import Path


class SimulatedResidentBackend:
    """A functional deterministic policy with visibly different task processors."""

    def __init__(self, mode="shared", prefix_steps=0):
        import torch

        self.mode = mode
        self.prefix_steps = prefix_steps
        self.active_task = None
        self.task_ids = ("A", "B")
        self.load_counts = {"base": 1 if mode == "shared" else 2, "adapters": 2}
        self.models = (
            {"shared": torch.ones(64, 64)}
            if mode == "shared"
            else {
                "A": torch.ones(64, 64),
                "B": torch.ones(64, 64),
            }
        )
        self.warmup_counts = {"A": 0, "B": 0}
        self._ready = set()
        for task in self.task_ids:
            self.activate(task)
            raw, _ = self.predict(task, {})
            if prefix_steps:
                self.predict(task, {}, prefix_steps, raw[:prefix_steps])
            self.reset(task)
            self.warmup_counts[task] += 1
            self._ready.add(task)

    def metadata(self, task):
        return {
            "simulation": True,
            "task_id": task,
            "adapter_id": task,
            "task_text": "simulated " + task,
            "action_schema": {
                "names": ["joint_0", "joint_1"],
                "units": ["simulation", "simulation"],
                "fps": 100,
            },
            "control_hz": 100,
            "rtc_mode": "training_time" if self.prefix_steps else "async",
            "rtc_prefix_steps": self.prefix_steps,
            "rtc_training_max_delay": 4,
        }

    def ready(self, task):
        return task in self._ready

    def activate(self, task):
        if task not in self.task_ids:
            raise ValueError(task)
        self.active_task = task

    def reset(self, task):
        pass

    def predict(self, task, observation, prefix_steps=0, action_prefix=None):
        import torch

        if self.active_task != task:
            raise RuntimeError("Mixed task activation")
        raw = torch.full((8, 2), 0.1 if task == "A" else 0.2)
        if prefix_steps:
            if action_prefix.shape != (prefix_steps, 2):
                raise ValueError("Invalid simulated clean prefix")
            raw[:prefix_steps] = action_prefix
        # Each task has its own normalization. Incorrect processor switching
        # cannot accidentally produce the same execution action.
        processed = raw * (10 if task == "A" else 100)
        return raw, processed


class OfflineRobot:
    """An in-memory acceptance interface; it has no connect/serial/camera methods."""

    def __init__(self, names, observation):
        self.action_features = dict.fromkeys(names, float)
        self.observation = observation
        self.accepted = []

    def get_observation(self):
        return dict(self.observation)

    def send_action(self, action):
        self.accepted.append((time.perf_counter(), dict(action)))
        return dict(action)


class TimedStub:
    """Timestamp RPC receipt on the client clock, with no cross-host subtraction."""

    def __init__(self, stub):
        self.stub = stub
        self.switch_ack = None
        self.switch_started = None
        self.chunk_received = None
        self.inference_seconds = []
        self.exceptions = 0
        self.control_exceptions = 0
        self.control_response_errors = 0
        self.response_errors = 0
        self.events = []

    def ControlPolicy(self, request, **kwargs):  # noqa: N802
        begin = time.perf_counter()
        try:
            reply = self.stub.ControlPolicy(request, **kwargs)
        except Exception:
            self.control_exceptions += 1
            raise
        data = pickle.loads(request.data)  # nosec: local trusted protocol
        control_result = pickle.loads(reply.data)  # nosec
        self.control_response_errors += int(not control_result.get("ok", False))
        received = time.perf_counter()
        self.events.append(
            {
                "kind": "control",
                "op": data.get("op"),
                "operation_id": data.get("operation_id"),
                "started": begin,
                "received": received,
                "server_operation_timing": control_result.get("state", {}).get("last_operation"),
            }
        )
        if data.get("op") == "switch":
            self.switch_started = begin
            self.switch_ack = received
        return reply

    def Predict(self, request, **kwargs):  # noqa: N802
        begin = time.perf_counter()
        try:
            reply = self.stub.Predict(request, **kwargs)
        except Exception:
            self.exceptions += 1
            raise
        self.chunk_received = time.perf_counter()
        result = pickle.loads(reply.data)  # nosec
        self.response_errors += int(result.error is not None)
        self.inference_seconds.append(result.inference_seconds)
        self.events.append(
            {
                "kind": "inference",
                "request_id": result.identity.request_id,
                "session_id": result.identity.session_id,
                "task_id": result.identity.task_id,
                "policy_epoch": result.identity.policy_epoch,
                "started": begin,
                "received": self.chunk_received,
                "server_inference_s": result.inference_seconds,
                "server_lock_wait_s": result.lock_wait_seconds,
                "prefix_steps": result.prefix_steps,
                "error": result.error,
            }
        )
        return reply


def _percentiles(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {
        "n": len(values),
        "p50_ms": percentile(0.5) * 1000,
        "p95_ms": percentile(0.95) * 1000,
        "p99_ms": percentile(0.99) * 1000,
    }


def _memory(device, backend):
    import torch

    storage = {}
    adapter_bytes = 0
    for model in backend.models.values():
        if hasattr(model, "named_parameters"):
            for name, param in model.named_parameters():
                storage[(str(param.device), param.untyped_storage().data_ptr())] = (
                    param.untyped_storage().nbytes()
                )
                if "lora_" in name:
                    adapter_bytes += param.numel() * param.element_size()
        elif isinstance(model, torch.Tensor):
            storage[(str(model.device), model.untyped_storage().data_ptr())] = (
                model.untyped_storage().nbytes()
            )
    value = {
        "unique_parameter_storage_bytes": sum(storage.values()),
        "adapter_bytes": adapter_bytes,
        "allocated": None,
        "reserved": None,
        "peak_allocated": None,
        "device_total_used_bytes": None,
        "process_used_bytes": None,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)  # benchmark boundary, never a control-loop synchronization
        value.update(
            allocated=torch.cuda.memory_allocated(device),
            reserved=torch.cuda.memory_reserved(device),
            peak_allocated=torch.cuda.max_memory_allocated(device),
        )
        import os
        import subprocess

        index = str(torch.cuda.current_device() if ":" not in str(device) else int(str(device).split(":")[1]))
        query = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if query.returncode == 0:
            value["device_total_used_bytes"] = int(query.stdout.strip()) * 1024 * 1024
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == str(os.getpid()) and fields[1].isdigit():
                value["process_used_bytes"] = int(fields[1]) * 1024 * 1024
    return value


def benchmark(output, switches=200, registry=None, device="cpu", mode="both", prefix_steps=0):
    from concurrent import futures

    import grpc
    import numpy as np
    import torch

    from lerobot.scripts.recording_remote_policy import RemotePolicyActionClient, RemotePolicyRecordConfig
    from lerobot.transport import services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options

    from .configs import PolicyServerConfig
    from .hotswitch import HotSwitchRuntime
    from .policy_server import PolicyServer

    if switches < 1 or (registry is None and not 0 <= prefix_steps <= 4):
        raise ValueError("Positive switches and simulation prefix_steps in [0,4] required")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    modes = ["shared", "dual"] if mode == "both" else [mode]
    report = {
        "label": "SIMULATED_POLICY_SIMULATED_ROBOT"
        if registry is None
        else "REAL_POLICY_OFFLINE_OBSERVATIONS_SIMULATED_ROBOT",
        "physical_robot_verified": False,
        "task_success_rate": None,
        "device": device,
        "switches_per_mode": switches,
        "tail_latency_note": "200 samples provide limited P99 precision" if switches <= 200 else "",
        "modes": {},
    }
    reference_outputs = {}
    for variant in modes:
        if registry and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        if registry:
            from .hotswitch_backend import load_backend

            path = Path(registry).resolve(strict=True)
            config = json.loads(path.read_text(encoding="utf-8"))
            observations = {}
            for task, obs_path in config["offline_observations"].items():
                with np.load(path.parent / obs_path, allow_pickle=False) as archive:
                    observations[task] = {key: archive[key] for key in archive.files}
            backend = load_backend(registry, device=device, mode=variant, offline_observations=observations)
        else:
            backend = SimulatedResidentBackend(variant, prefix_steps)
            observations = {task: {"joint_0": 0.0, "joint_1": 0.0} for task in backend.task_ids}
        load_and_warm_seconds = time.perf_counter() - load_started
        tasks = backend.task_ids
        if len(tasks) != 2:
            raise ValueError("Benchmark requires exactly two registered tasks")
        equivalence = {
            "status": "not_applicable_simulated_policy" if registry is None else "shared_reference_only"
        }
        if registry:
            equivalence["atol"], equivalence["rtol"] = 1e-4, 1e-4
            for task in tasks:
                backend.activate(task)
                backend.reset(task)
                torch.manual_seed(101)
                if str(device).startswith("cuda"):
                    torch.cuda.manual_seed_all(101)
                raw, processed = backend.predict(task, observations[task])
                outputs = (raw.clone(), processed.clone())
                if variant == "shared":
                    reference_outputs[task] = outputs
                elif task in reference_outputs:
                    errors = []
                    for actual, expected in zip(outputs, reference_outputs[task], strict=True):
                        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                        delta = actual - expected
                        errors.append(
                            {"max_abs": delta.abs().max().item(), "mse": delta.square().mean().item()}
                        )
                    equivalence[task] = errors
                    equivalence["status"] = "passed_shared_vs_merged_dual"
                else:
                    equivalence["status"] = "unverified_run_mode_both_for_comparison"
                backend.reset(task)
        meta = backend.metadata(tasks[0])
        names = meta["action_schema"]["names"]
        fps = meta.get("control_hz", meta["action_schema"]["fps"])
        runtime = HotSwitchRuntime(backend)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            PolicyServer(PolicyServerConfig(inference_latency=0), runtime), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}", options=grpc_channel_options())
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = TimedStub(services_pb2_grpc.AsyncInferenceStub(channel))
        robot = OfflineRobot(names, observations[tasks[0]])
        cfg = RemotePolicyRecordConfig(
            enable=True,
            hotswitch_task_id=tasks[0],
            hotswitch_simulation=True,
            actions_per_chunk=8 if registry is None else meta["policy_config"]["chunk_size"],
            chunk_size_threshold=0.5,
            obs_queue_timeout_s=30,
        )
        client = RemotePolicyActionClient(cfg, robot, fps=fps, stub=stub, channel=channel)
        rows, cycle_errors, control_events, bootstrap_ids = [], [], [], set()
        try:
            client.start()
            for sample in range(switches):
                target = tasks[(sample + 1) % 2]
                robot.observation = observations[target]
                stub.switch_ack = stub.chunk_received = None
                started = time.perf_counter()
                client.request_task_switch(target, operation_id=f"{variant}-{sample}")
                action = client.get_action(robot.get_observation(), None, 0)
                client.validate_action_before_send(action)
                accepted = robot.send_action(action)
                accepted_at = robot.accepted[-1][0]
                bootstrap_id = client._hotswitch.pending_action.identity.request_id
                client.acknowledge_action(accepted)
                bootstrap_ids.add(bootstrap_id)
                bootstrap_event = next(
                    event for event in stub.events if event.get("request_id") == bootstrap_id
                )
                received_at = bootstrap_event["received"]
                identity = client._hotswitch.identity
                row = {
                    "sample": sample,
                    "task_id": identity.task_id,
                    "session_id": identity.session_id,
                    "policy_epoch": identity.policy_epoch,
                    "switch_confirm_s": stub.switch_ack - started,
                    "first_valid_chunk_s": received_at - started,
                    "first_control_accept_s": accepted_at - started,
                    "started": started,
                    "switch_ack_timestamp": stub.switch_ack,
                    "first_chunk_received_timestamp": received_at,
                    "first_control_accept_timestamp": accepted_at,
                    "hold_including_first_action_s": accepted_at - started,
                    "before_switch_rpc_s": stub.switch_started - started,
                    "switch_rpc_s": stub.switch_ack - stub.switch_started,
                    "post_ack_to_chunk_s": received_at - stub.switch_ack,
                    "queue_and_safety_s": accepted_at - received_at,
                }
                steady = []
                # Drive enough accepted commands to exercise nonzero-prefix refill.
                # Sleeping only paces the simulated control interface.
                deadline = time.perf_counter()
                for timestep in range(7):
                    deadline += 1 / fps
                    time.sleep(max(0, deadline - time.perf_counter()))
                    begin = time.perf_counter()
                    action = client.get_action(robot.get_observation(), None, timestep + 1)
                    client.validate_action_before_send(action)
                    client.acknowledge_action(robot.send_action(action))
                    control_events.append(
                        {
                            "sample": sample,
                            "task_id": identity.task_id,
                            "policy_epoch": identity.policy_epoch,
                            "scheduled": deadline,
                            "accepted": robot.accepted[-1][0],
                        }
                    )
                    cycle_errors.append(max(0, time.perf_counter() - deadline))
                    steady.append(time.perf_counter() - begin)
                row["control_get_action_mean_s"] = statistics.mean(steady)
                rows.append(row)
            client._hotswitch._wait_request()
            memory = _memory(device, backend)
            state = runtime.status()
            periods = [
                abs((current[0] - previous[0]) - 1 / fps)
                for previous, current in zip(robot.accepted, robot.accepted[1:], strict=False)
            ]
            steady_inference = [
                event["server_inference_s"]
                for event in stub.events
                if event["kind"] == "inference" and event["request_id"] not in bootstrap_ids
            ]
            result = {
                "load_and_warm_s": load_and_warm_seconds,
                "model_load_s": getattr(backend, "load_seconds", None),
                "warmup_s": getattr(backend, "warmup_seconds", None),
                "first_switch": rows[0],
                "warm_switch_confirm": _percentiles([r["switch_confirm_s"] for r in rows[1:]]),
                "switch_confirm": _percentiles([r["switch_confirm_s"] for r in rows]),
                "first_valid_chunk": _percentiles([r["first_valid_chunk_s"] for r in rows]),
                "first_control_accept": _percentiles([r["first_control_accept_s"] for r in rows]),
                "server_inference": _percentiles(stub.inference_seconds),
                "control_cycle_lateness": _percentiles(cycle_errors),
                "steady_server_inference": _percentiles(steady_inference),
                "server_inference_lock_wait": _percentiles(
                    [event["server_lock_wait_s"] for event in stub.events if event["kind"] == "inference"]
                ),
                "server_switch_lock_wait": _percentiles(
                    [
                        event["server_operation_timing"]["lock_wait_seconds"]
                        for event in stub.events
                        if event["kind"] == "control" and event["op"] == "switch"
                    ]
                ),
                "server_switch_commit": _percentiles(
                    [
                        event["server_operation_timing"]["commit_seconds"]
                        for event in stub.events
                        if event["kind"] == "control" and event["op"] == "switch"
                    ]
                ),
                "acceptance_period_abs_error_including_switches": _percentiles(periods),
                "client_counters": dict(client._hotswitch.counters),
                "rpc_exceptions": stub.exceptions,
                "control_rpc_exceptions": stub.control_exceptions,
                "control_response_errors": stub.control_response_errors,
                "response_errors": stub.response_errors,
                "server_rejected": state["rejected_count"],
                "load_counts": backend.load_counts,
                "warmup_counts": backend.warmup_counts,
                "memory": memory,
                "merged_output_equivalence": equivalence,
            }
            if hasattr(backend, "storage_evidence"):
                result["storage_evidence"] = backend.storage_evidence()
            report["modes"][variant] = result
        finally:
            client.stop()
            server.stop(2).wait()
            with (destination / f"{variant}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            for suffix, events in (("rpc", stub.events), ("control", control_events)):
                with (destination / f"{variant}_{suffix}.jsonl").open("w", encoding="utf-8") as stream:
                    for event in events:
                        stream.write(json.dumps(event) + "\n")
            del client, backend, runtime, server
            gc.collect()
        (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
