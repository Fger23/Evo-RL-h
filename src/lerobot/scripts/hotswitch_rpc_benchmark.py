"""Measure hot switches over an existing gRPC service with offline observations.

This client never creates a policy/server/robot driver, seeds CUDA, or reports GPU
memory. Server loading, numeric equivalence and physical rollouts are separate.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path

from lerobot.async_inference.hotswitch.manifest import (
    check_compatible,
    digest_file,
    load_task_bundle,
    read_json,
)
from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore
from lerobot.async_inference.hotswitch.simulation import SimulatedRobot


class _ObservedTransport:
    """Count transport failures even when the core recovers an idempotent retry."""

    def __init__(self, transport):
        self.transport = transport
        self.exceptions = []

    def _call(self, method, *args):
        try:
            return getattr(self.transport, method)(*args)
        except BaseException as exc:
            self.exceptions.append({
                "method": method, "client_monotonic_s": time.perf_counter(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise

    def status(self):
        return self._call("status")

    def open_session(self, previous_session_id=None):
        return self._call("open_session", previous_session_id)

    def switch(self, task_id, switch_id, session_id):
        return self._call("switch", task_id, switch_id, session_id)

    def infer(self, request):
        return self._call("infer", request)

    def close(self):
        return self._call("close")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": statistics.mean(values),
    }


def _source_proof(manifest_path):
    """Reuse the delivered complete source validator before importing its code."""
    path = Path(manifest_path).resolve(strict=True)
    recorded = read_json(path)
    root = Path(recorded["snapshot"]).resolve(strict=True)
    current = Path(__file__).resolve()
    relative = "src/lerobot/scripts/hotswitch_rpc_benchmark.py"
    files = recorded["snapshot_files_sha256"]
    if current != root / relative or files.get(relative) != digest_file(current):
        raise ValueError("RPC client source differs from the supplied frozen source manifest")
    helper = root / "tools/hotswitch_freeze_source.py"
    if files.get("tools/hotswitch_freeze_source.py") != digest_file(helper):
        raise ValueError("Frozen source verifier bytes differ from the manifest")
    spec = importlib.util.spec_from_file_location("hotswitch_rpc_source_verifier", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proof = module.verify_frozen_source(path.parent, execution_root=root, check_package=True)
    if proof["manifest_sha256"] != digest_file(path):
        raise ValueError("Supplied source manifest differs from canonical source evidence")
    return {**proof, "path": str(path), "client_source_sha256": digest_file(current)}


def _capture_inputs(args):
    registry_path = Path(args.registry).resolve(strict=True)
    registry = read_json(registry_path)
    if not isinstance(registry.get("tasks"), dict) or len(registry["tasks"]) != 2:
        raise ValueError("RPC switch benchmark requires exactly two registered task IDs")
    bundles = {}
    manifests = {}
    for task_id, path in registry["tasks"].items():
        root, manifest = load_task_bundle(path)
        if manifest["task_id"] != task_id:
            raise ValueError("Registry task_id differs from its validated package")
        if abs(float(manifest["action_schema"]["control_hz"]) - args.control_hz) > 1e-9:
            raise ValueError("Requested control-hz differs from the registered task schema")
        bundles[task_id] = {
            "root": str(root),
            "manifest_path": str(root / "manifest.json"),
            "manifest_sha256": digest_file(root / "manifest.json"),
            "artifacts": manifest["artifacts"],
            "base_fingerprint": manifest["base_fingerprint"],
        }
        manifests[task_id] = manifest
    check_compatible(list(manifests.values()))
    source = _source_proof(args.source_manifest)
    return {
        "schema_version": 1,
        "kind": "offline_observations_real_grpc_simulated_control",
        "registry_path": str(registry_path),
        "registry_sha256": digest_file(registry_path),
        "bundle_manifests": bundles,
        "source_manifest": source,
        "source_manifest_path": source["path"],
        "source_manifest_sha256": source["manifest_sha256"],
        "client_source_sha256": source["client_source_sha256"],
        "registry_scope": registry.get("scope"),
        "second_task_trained": registry.get("second_task_trained"),
        "options": {
            name: getattr(args, name)
            for name in ("address", "server_backend", "switches", "control_hz", "timeout")
        },
        "argv": list(sys.argv),
        "python": platform.python_version(),
        "physical_robot": False,
    }, manifests


def _load_observations(evidence, manifests):
    import torch
    from safetensors.torch import load_file

    observations = {}
    for task_id, manifest in manifests.items():
        root = Path(evidence["bundle_manifests"][task_id]["root"])
        observation = load_file(str(root / manifest["offline_observation"]), device="cpu")
        for name, feature in (manifest["model_structure"].get("input_features") or {}).items():
            if name not in observation or tuple(observation[name].shape) != tuple(feature["shape"]):
                raise ValueError(f"Offline observation does not match registered input shape: {name}")
            if not torch.isfinite(observation[name]).all():
                raise ValueError(f"Offline observation has NaN/Inf: {name}")
        observations[task_id] = observation
    return observations


def _verify_server(status, args, evidence, manifests):
    if status.get("version") != 1 or status.get("stopped"):
        raise ValueError("Existing server is unavailable or uses an incompatible protocol")
    active = status.get("session_id")
    if active != args.previous_session_id:
        raise ValueError(
            "Server session requires its explicit current --previous-session-id; no automatic takeover"
        )
    for task_id, manifest in manifests.items():
        if task_id not in status.get("ready", []):
            raise ValueError(f"Task is not preloaded and warmed: {task_id}")
        meta = status["task_metadata"][task_id]
        schema, rtc = manifest["action_schema"], manifest["rtc"]
        expected = {
            "action_names": schema["names"],
            "action_units": schema["units"],
            "control_hz": schema["control_hz"],
            "chunk_size": manifest["model_structure"]["chunk_size"],
            "action_dim": len(schema["names"]),
            "task_text": manifest["task_text"],
            "rtc_mode": rtc["mode"],
            "max_prefix_steps": rtc["max_delay"],
            "prefix_steps": rtc["prefix_steps"],
            "bundle_manifest_sha256": evidence["bundle_manifests"][task_id]["manifest_sha256"],
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Live registered package/schema differs from benchmark input: {task_id}")


def _accept(core, robot, switch_index, phase, actions):
    processed = core.next_action()
    row = {
        "switch_index": switch_index,
        "phase": phase,
        "action_index": len(actions),
        "identity": copy.deepcopy(core.pending_identity),
        "raw": list(core.pending[0]),
        "processed": list(processed),
        "accepted": None,
        "confirmation": "NOT_CONFIRMED",
    }
    actions.append(row)
    try:
        actual = robot.send_action(processed)
        row["accepted"] = list(actual) if actual is not None else None
        core.confirm_action(actual)
        row.update(
            confirmation="ACCEPTED",
            identity=copy.deepcopy(core.last_accepted_identity),
            accepted_monotonic_s=core.last_acceptance_time,
            acceptance_interval_s=core.last_acceptance_interval_s,
            action_clock_wait_s=core.last_action_wait_s,
        )
        return core.last_acceptance_time
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise


def run(args, *, transport_factory=GrpcHotSwitchTransport, robot_factory=SimulatedRobot):
    """Run one existing server backend; the harness compares independently saved runs."""
    if args.switches < 1 or not math.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("switches and control-hz must be positive and finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RPC benchmark evidence: {output}")
    # Reject this before writing even RUNNING: a report inside the snapshot
    # would itself invalidate the immutable source inventory.
    declared_source = read_json(args.source_manifest)
    if "snapshot" in declared_source and output.resolve().is_relative_to(
        Path(declared_source["snapshot"]).resolve()
    ):
        raise ValueError("Benchmark output must be outside the frozen source snapshot")
    output.mkdir(parents=True)
    status_path = output / "run_status.json"
    _write_json(status_path, {"status": "RUNNING", "server_backend": args.server_backend})
    switches, actions, requests = [], [], []
    core = transport = robot = None
    server_before = server_after = None
    evidence = None
    success = False
    failure = None
    phase = "input_preload"
    started = time.perf_counter()
    try:
        evidence, manifests = _capture_inputs(args)
        if output.resolve().is_relative_to(Path(evidence["source_manifest"]["snapshot"]).resolve()):
            raise ValueError("Benchmark output must be outside the frozen source snapshot")
        observations = _load_observations(evidence, manifests)
        _write_json(output / "input_provenance.json", evidence)
        phase = "server_preflight"
        transport = _ObservedTransport(transport_factory(args.address, timeout=args.timeout))
        server_initial = transport.status()  # Establish the real channel before timing.
        _verify_server(server_initial, args, evidence, manifests)
        _write_json(output / "server_initial_status.json", server_initial)
        robot = robot_factory()
        core = HotSwitchClientCore(
            transport, safe_hold=robot.hold, minimum_action_interval_s=1 / args.control_hz
        )
        task_ids = list(manifests)
        core.connect(task_ids[0], previous_session_id=args.previous_session_id)
        phase = "untimed_rpc_warmup"
        # Warm both real RPC routes. Discard these actions without execution.
        for task_id in task_ids:
            core.switch(task_id)
            robot.hold()
            if not core.request_actions(observations[task_id]):
                raise RuntimeError("Untimed warmup response was rejected")
        core.switch(task_ids[0])
        current, _ = _capture_inputs(args)
        if current != evidence:
            raise ValueError("Benchmark input/source bytes changed during setup")
        server_before = transport.status()
        nonce = uuid.uuid4().hex
        action_written = request_written = 0
        for index in range(args.switches):
            task_id = task_ids[(index + 1) % 2]
            row = {
                "index": index,
                "task_id": task_id,
                "server_backend": args.server_backend,
                "status": "RUNNING",
            }
            switches.append(row)
            phase = "switch"
            beginning = time.perf_counter()
            activated = core.switch(task_id, f"rpc-benchmark-{nonce}-{index}")
            acknowledged = time.perf_counter()
            row.update(
                session_id=activated["session_id"],
                policy_epoch=activated["policy_epoch"],
                switch_id=activated["switch_id"],
                switch_ack_ms=(acknowledged - beginning) * 1000,
                switch_timings_s=dict(core.last_timings["switch"]),
            )
            phase = "bootstrap_inference"
            if not core.request_actions(observations[task_id]):
                raise RuntimeError("Bootstrap response was rejected")
            received = time.perf_counter()
            first_alignment = core.last_inference_alignment
            if first_alignment["prefix_steps"] != 0:
                raise RuntimeError("New task bootstrap reused an old task prefix")
            requests.append(
                {
                    "switch_index": index,
                    "phase": "bootstrap",
                    **first_alignment,
                    "client_timings_s": dict(core.last_timings["inference"]),
                }
            )
            row.update(
                first_valid_chunk_ms=(received - beginning) * 1000,
                bootstrap_request_id=first_alignment["request_id"],
            )
            phase = "bootstrap_action_acceptance"
            first_accepted = _accept(core, robot, index, "bootstrap", actions)
            row["first_simulated_action_accepted_ms"] = (first_accepted - beginning) * 1000
            phase = "steady_refill"
            robot.hold()
            refill_started = time.perf_counter()
            if not core.request_actions(observations[task_id]):
                raise RuntimeError("Steady refill response was rejected")
            refill_received = time.perf_counter()
            steady = core.last_inference_alignment
            expected_prefix = manifests[task_id]["rtc"]["prefix_steps"]
            if steady["prefix_steps"] != expected_prefix:
                raise RuntimeError("Steady refill did not exercise the registered prefix length")
            requests.append(
                {
                    "switch_index": index,
                    "phase": "steady_refill",
                    **steady,
                    "client_timings_s": dict(core.last_timings["inference"]),
                }
            )
            row.update(
                steady_inference_ms=(refill_received - refill_started) * 1000,
                steady_request_id=steady["request_id"],
                steady_prefix_steps=steady["prefix_steps"],
            )
            phase = "steady_action_acceptance"
            _accept(core, robot, index, "steady_refill", actions)
            row.update(status="PASSED", client_metrics=dict(core.metrics))
            _append(output / "switches.jsonl", row)
            for action in actions[action_written:]:
                _append(output / "actions.jsonl", action)
            for request in requests[request_written:]:
                _append(output / "inference_requests.jsonl", request)
            action_written, request_written = len(actions), len(requests)
            _write_json(
                status_path,
                {
                    "status": "RUNNING",
                    "completed_switches": index + 1,
                    "requested_switches": args.switches,
                    "server_backend": args.server_backend,
                },
            )
        phase = "post_run_integrity"
        current, _ = _capture_inputs(args)
        if current != evidence:
            raise ValueError("Benchmark input/source bytes changed during measurement")
        server_after = transport.status()
        if server_after["session_id"] != core.status["session_id"]:
            raise RuntimeError("Live session changed during measurement")
        if any(core.metrics.values()):
            raise RuntimeError("Client recorded anomalous responses during measurement")
        if transport.exceptions:
            raise RuntimeError("RPC transport failures occurred, including any recovered retries")
        if any(
            server_after["metrics"].get(key, 0) - server_before["metrics"].get(key, 0)
            for key in ("switch_errors", "stale_requests")
        ):
            raise RuntimeError("Server recorded anomalous requests during measurement")
        success = True
    except BaseException as exc:
        failure = {
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if core is not None:
            core.stop()
        if robot is not None:
            robot.hold()
        if transport is not None:
            transport.close()
    # Rebuild complete logs from the in-memory trace on failure so partial valid
    # bootstrap requests and rejected commands remain reviewable exactly once.
    if not success:
        for name, rows in (("switches", switches), ("actions", actions), ("inference_requests", requests)):
            (output / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
    complete = [row for row in switches if row["status"] == "PASSED"]
    accepted = [row for row in actions if row["confirmation"] == "ACCEPTED"]
    intervals = [
        later["accepted_monotonic_s"] - earlier["accepted_monotonic_s"]
        for earlier, later in zip(accepted, accepted[1:], strict=False)
    ]
    metrics = (
        "switch_ack_ms",
        "first_valid_chunk_ms",
        "first_simulated_action_accepted_ms",
        "steady_inference_ms",
    )
    summary = {
        "status": "RPC_BENCHMARK_PASSED" if success else "FAILED",
        "server_backend": args.server_backend,
        "backend_label_scope": "Caller-supplied; match to the preserved server launch argv",
        "samples": len(complete),
        "switches_requested": args.switches,
        "switches_attempted": len(switches),
        "accepted_actions": len(accepted),
        "inference_requests": len(requests),
        "failure": failure,
        "metrics": {key: _percentiles([row[key] for row in complete]) for key in metrics},
        "component_metrics_s": {
            "switch": {
                key: _percentiles([row["switch_timings_s"][key] for row in complete])
                for key in sorted({key for row in complete for key in row["switch_timings_s"]})
            },
            "inference": {
                key: _percentiles([row["client_timings_s"][key] for row in requests])
                for key in sorted({key for row in requests for key in row["client_timings_s"]})
            },
        },
        "control_timing": {
            "expected_interval_s": 1 / args.control_hz,
            "interval_ms": _percentiles([value * 1000 for value in intervals]),
            "signed_jitter_ms": _percentiles([(value - 1 / args.control_hz) * 1000 for value in intervals]),
            "minimum_interval_s": min(intervals) if intervals else None,
        },
        "prefix_steps_observed": sorted({row["prefix_steps"] for row in requests}),
        "client_metrics": dict(core.metrics) if core is not None else {},
        "client_exception_count": max(len(transport.exceptions) if transport else 0, int(not success)),
        "transport_exception_count": len(transport.exceptions) if transport else 0,
        "transport_exceptions": list(transport.exceptions) if transport else [],
        "terminal_failure_count": int(not success),
        "server_metrics_before_timing": server_before.get("metrics") if server_before else None,
        "server_metrics_after_timing": server_after.get("metrics") if server_after else None,
        "server_metrics_delta": {
            key: server_after["metrics"].get(key, 0) - server_before["metrics"].get(key, 0)
            for key in sorted(set(server_before["metrics"]) | set(server_after["metrics"]))
        }
        if server_before and server_after
        else None,
        "safe_holds": robot.holds if robot is not None else 0,
        "client_elapsed_s": time.perf_counter() - started,
        "clock_scope": "All end-to-end durations use this client's perf_counter; server local durations remain separate",
        "gpu_metrics": "NOT_MEASURED_BY_CLIENT",
        "process_cold_start": "NOT_MEASURED",
        "physical_robot": False,
        "physical_motion_latency": "NOT_MEASURED",
        "task_success_rate": "NOT_MEASURED",
        "numeric_equivalence": "Separate fixed-condition model validation; no client CUDA seed or quality claim",
        "tail_note": "Fewer than 200 samples; tail estimates uncertain"
        if len(complete) < 200
        else "200+ samples; one offline RPC run",
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        status_path,
        {"status": "PASSED" if success else "FAILED", "samples": len(complete), "failure": failure},
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument(
        "--server-backend", required=True, help="Label bound to the actual existing server launch evidence"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--switches", type=int, default=200)
    parser.add_argument("--control-hz", type=float, default=30)
    parser.add_argument("--previous-session-id")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "RPC_BENCHMARK_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
