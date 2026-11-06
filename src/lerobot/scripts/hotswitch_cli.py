"""Offline-first hot-switch operations. No command constructs a robot driver."""

import argparse
import copy
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from lerobot.async_inference.hotswitch.environment import collect_environment, save_json


def _percentiles(values):
    ordered = sorted(values)
    def at(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        return ordered[low] + (ordered[math.ceil(position)] - ordered[low]) * (position - low)
    return {"n": len(ordered), "p50": at(0.5), "p95": at(0.95), "p99": at(0.99),
            "mean": statistics.mean(ordered)}


def _control_timing_summary(rows, interval_s):
    """Local simulated-controller intervals; server clocks are never subtracted."""
    intervals = [later["accepted_monotonic_s"] - earlier["accepted_monotonic_s"]
                 for earlier, later in zip(rows, rows[1:], strict=False)]
    empty = {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    return {"scope": "Simulated control acceptance on the client perf_counter clock; no physical timing",
            "expected_interval_s": interval_s,
            "interval_ms": _percentiles([x * 1000 for x in intervals]) if intervals else dict(empty),
            "signed_jitter_ms": _percentiles([(x - interval_s) * 1000 for x in intervals]) if intervals else dict(empty),
            "absolute_jitter_ms": _percentiles([abs(x - interval_s) * 1000 for x in intervals]) if intervals else dict(empty),
            "minimum_interval_s": min(intervals) if intervals else None,
            "maximum_interval_s": max(intervals) if intervals else None}


def _benchmark_input_provenance(args):
    """Bind raw benchmark rows to verified bundle content outside timed sections."""
    from lerobot.async_inference.hotswitch.manifest import digest_file, load_task_bundle, read_json

    result = {"schema_version": 1, "kind": "gpu_offline" if args.registry else "simulation",
              "argv": list(sys.argv), "cli_source_sha256": digest_file(Path(__file__)),
              "options": {name: getattr(args, name, None) for name in
                  ("registry", "device", "switches", "seed", "atol", "rtol", "simulated_inference_ms", "dtype")},
              "registry_path": None, "registry_sha256": None, "bundle_manifests": {},
              "source_manifest_path": None, "source_manifest_sha256": None}
    result["options"]["dual_mode"] = getattr(args, "dual_mode", "merged")
    if args.registry or getattr(args, "dtype", None) is not None:
        from lerobot.async_inference.hotswitch.backend import inference_compute_precision
        result["options"]["compute_precision"] = inference_compute_precision()
    source_manifest = getattr(args, "source_manifest", None)
    if source_manifest:
        source_manifest = Path(source_manifest).resolve(strict=True)
        result.update(source_manifest_path=str(source_manifest),
                      source_manifest_sha256=digest_file(source_manifest))
    if args.registry:
        registry = Path(args.registry).resolve(strict=True)
        config = read_json(registry)
        result.update(registry_path=str(registry), registry_sha256=digest_file(registry))
        for task_id, path in config["tasks"].items():
            root, manifest = load_task_bundle(path)
            if manifest["task_id"] != task_id:
                raise ValueError("Benchmark registry task differs from its bundle manifest")
            result["bundle_manifests"][task_id] = {
                "path": str(root / "manifest.json"), "sha256": digest_file(root / "manifest.json"),
                "base_fingerprint": manifest["base_fingerprint"], "artifacts": manifest["artifacts"]}
    return result


def _benchmark_numeric_controls(backend, args, mode, reference):
    """Compare both routes with the same seed and shared raw clean prefix, untimed."""
    import torch

    def seed():
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    def difference(actual, expected):
        expected = torch.as_tensor(expected, dtype=actual.dtype)
        if actual.shape != expected.shape or not torch.isfinite(actual).all():
            raise RuntimeError("Numerical control produced invalid actions")
        delta = actual - expected
        return {"max_abs_error": float(delta.abs().max()), "mse": float(delta.square().mean()),
                "within_tolerance": bool(torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol))}

    report = {"mode": mode, "seed": args.seed, "atol": args.atol, "rtol": args.rtol,
              "scope": "Outside timed benchmark; common shared bootstrap prefix and identical per-call random seed",
              "tasks": {}, "status": "PASSED"}
    for task_id in backend.task_ids:
        backend.activate(task_id)
        backend.reset(task_id)
        observation = backend.offline_observation(task_id)
        seed()
        raw, processed = backend.infer(task_id, observation)
        if task_id not in reference:
            reference[task_id] = {"bootstrap_raw": raw.tolist(), "bootstrap_processed": processed.tolist()}
        target = reference[task_id]
        task = {"bootstrap_raw": raw.tolist(), "bootstrap_processed": processed.tolist(),
                "bootstrap_raw_error": difference(raw, target["bootstrap_raw"]),
                "bootstrap_processed_error": difference(processed, target["bootstrap_processed"])}
        steps = backend.task_metadata[task_id].get("prefix_steps", 0)
        task["prefix_steps"] = steps
        if steps:
            prefix = target["bootstrap_raw"][:steps]
            backend.reset(task_id)
            seed()
            conditioned, conditioned_processed = backend.infer(task_id, observation, prefix, steps)
            if "conditioned_raw" not in target:
                target.update(conditioned_raw=conditioned.tolist(), conditioned_processed=conditioned_processed.tolist())
            task.update(common_raw_prefix=prefix, common_processed_prefix=target["bootstrap_processed"][:steps],
                        conditioned_raw=conditioned.tolist(), conditioned_processed=conditioned_processed.tolist(),
                        conditioned_raw_error=difference(conditioned, target["conditioned_raw"]),
                        conditioned_processed_error=difference(conditioned_processed, target["conditioned_processed"]),
                        raw_prefix_exact=bool(torch.equal(conditioned[:steps], torch.as_tensor(prefix, dtype=conditioned.dtype))),
                        processed_prefix_max_abs_error=float((conditioned_processed[:steps] - torch.as_tensor(
                            target["bootstrap_processed"][:steps], dtype=conditioned_processed.dtype)).abs().max()))
            if not task["raw_prefix_exact"] or task["processed_prefix_max_abs_error"] > 1e-6:
                report["status"] = "FAILED"
        if any(not value["within_tolerance"] for name, value in task.items() if name.endswith("_error") and isinstance(value, dict)):
            report["status"] = "FAILED"
        report["tasks"][task_id] = task
        backend.reset(task_id)
    return report


def _memory(device):
    if device == "cpu":
        return {"cuda": False}
    import torch
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    process_memory = [line.strip() for line in result.stdout.splitlines()
                      if line.split(",")[0].strip() == str(os.getpid())]
    return {"cuda": True, "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_used_bytes": total - free, "device_total_bytes": total,
            "process_nvidia_smi_pid_uuid_memory_mib": process_memory}


def _storage_signature(backend):
    return tuple((id(model), name, str(p.device), p.untyped_storage().data_ptr(), p.numel())
                 for model in getattr(backend, "models", {}).values()
                 for name, p in model.named_parameters())


@contextmanager
def _resident_switch_guard(backend):
    """Reject both Python file reads and resident model loader calls in switches."""
    guarded = []
    with ExitStack() as stack:
        for target in ("builtins.open", "io.open"):
            stack.enter_context(patch(target, side_effect=AssertionError(f"Switch attempted {target}")))
            guarded.append(target)
        classes = set()
        for model in getattr(backend, "models", {}).values():
            classes.add(type(model))
            get_base = getattr(model, "get_base_model", None)
            if callable(get_base):
                classes.add(type(get_base()))
            inner = getattr(model, "base_model", None)
            if inner is not None:
                classes.add(type(inner))
        for cls in classes:
            for method in ("from_pretrained", "load_adapter", "merge_and_unload", "to", "cpu", "cuda"):
                if hasattr(cls, method):
                    label = f"{cls.__module__}.{cls.__name__}.{method}"
                    stack.enter_context(patch.object(cls, method, side_effect=AssertionError(f"Switch attempted {label}")))
                    guarded.append(label)
        yield guarded


def benchmark(args):
    from lerobot.async_inference.hotswitch.runtime import HotSwitchClientCore, HotSwitchRuntime
    from lerobot.async_inference.hotswitch.simulation import SimulatedBackend, SimulatedRobot

    if args.switches < 1:
        raise ValueError("--switches must be positive")
    dtype = getattr(args, "dtype", None)
    dual_mode = getattr(args, "dual_mode", "merged")
    if dual_mode not in ("merged", "unmerged"):
        raise ValueError("dual-mode must be merged or unmerged")
    if args.registry or dtype is not None:
        from lerobot.async_inference.hotswitch.backend import configure_inference_precision
        configure_inference_precision(dtype)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix benchmark runs in nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_status.json", {"status": "RUNNING", "registry": args.registry,
                                         "switches": args.switches, "atol": args.atol, "rtol": args.rtol})
    save_json(output / "environment.json", collect_environment(Path(__file__).resolve().parents[3]))
    input_provenance = _benchmark_input_provenance(args)
    save_json(output / "input_provenance.json", input_provenance)

    def verify_inputs():
        try:
            current = _benchmark_input_provenance(args)
        except Exception as exc:
            save_json(output / "run_status.json", {"status": "FAILED", "reason": "benchmark inputs invalid",
                                                   "error": str(exc)})
            raise RuntimeError("Benchmark input content became invalid during the run") from exc
        if current != input_provenance:
            save_json(output / "run_status.json", {"status": "FAILED", "reason": "benchmark inputs changed"})
            raise RuntimeError("Benchmark registry, task content or source evidence changed during the run")

    all_summaries = {}
    reference = {}
    numeric_reference = {}
    for mode in ("shared", "dual"):
        verify_inputs()
        cold_start = time.perf_counter()
        if args.registry:
            from lerobot.async_inference.hotswitch.backend import (
                DualModelBackend,
                DualUnmergedBackend,
                SharedAdapterBackend,
            )
            dual_factory = DualModelBackend if dual_mode == "merged" else DualUnmergedBackend
            factory = SharedAdapterBackend if mode == "shared" else dual_factory
            backend = factory.from_registry(args.registry, device=args.device, dtype=dtype)
        else:
            backend = SimulatedBackend(mode, delay_s=args.simulated_inference_ms / 1000)
        load_s = time.perf_counter() - cold_start
        if args.registry:
            save_json(output / f"{mode}_backend_metadata.json", {
                "backend_mode": backend.comparison_mode, "inference_dtype_override": dtype,
                "parameter_dtypes": backend.parameter_dtypes, "compute_precision": backend.compute_precision,
                "parameter_storage": backend.storage_report, "load_counts": dict(backend.load_counts)})
        verify_inputs()
        runtime = HotSwitchRuntime(backend)
        start = time.perf_counter()
        runtime.preload()
        warmup_s = time.perf_counter() - start
        ids = list(backend.task_ids)
        if len(ids) != 2:
            raise ValueError("Benchmark requires exactly two registered packages, including mechanism-only copies")
        numerical_control_s = None
        controls = None
        if args.registry:
            control_start = time.perf_counter()
            controls = _benchmark_numeric_controls(backend, args, mode, numeric_reference)
            numerical_control_s = time.perf_counter() - control_start
            save_json(output / f"{mode}_numeric_controls.json", controls)
            if controls["status"] != "PASSED":
                save_json(output / "run_status.json", {"status": "FAILED", "reason": "untimed numerical controls", "mode": mode})
                raise RuntimeError(f"{mode} common-prefix numerical controls exceeded declared tolerance")
        robot = SimulatedRobot()
        core = HotSwitchClientCore(runtime, safe_hold=robot.hold)
        initial_activation_start = time.perf_counter()
        core.connect(ids[0])
        initial_session_activation_s = time.perf_counter() - initial_activation_start
        rows = []
        raw_log = output / f"{mode}.jsonl"
        raw_log.write_text("", encoding="utf-8")
        consistency = {}
        resident_storage = _storage_signature(backend)
        # Validate forbidden I/O and stable storage outside the timed run so
        # interception and tensor inventory do not contaminate deployment times.
        with _resident_switch_guard(backend) as guarded_calls:
            for check_index in range(args.switches):
                core.switch(ids[(check_index + 1) % 2], f"residency-check-{check_index}")
                if _storage_signature(backend) != resident_storage:
                    raise RuntimeError("Switch replaced model parameter storage or moved devices")
        runtime.metrics.clear()
        core.metrics.clear()
        if args.registry and args.device != "cpu":
            import torch
            torch.cuda.reset_peak_memory_stats(args.device)
        for index in range(args.switches):
            task_id = ids[(index + 1) % 2]
            start = time.perf_counter()
            status = core.switch(task_id, f"benchmark-{index}")
            ack = time.perf_counter()
            switch_timings = dict(getattr(core, "last_timings", {}).get("switch", {}))
            observation = backend.offline_observation(task_id)
            if args.registry:
                import torch
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            inference_start = time.perf_counter()
            accepted = core.request_actions(observation)
            chunk_time = time.perf_counter()
            first_timings = dict(getattr(core, "last_timings", {}).get("inference", {}))
            if not accepted:
                raise RuntimeError("Current task response was discarded")
            raw = [list(pair[0]) for pair in core.queue]
            action = core.next_action()
            actual = robot.send_action(action)
            core.confirm_action(actual)
            action_time = time.perf_counter()
            key = task_id
            if key not in reference:
                reference[key] = raw
            errors = [a - b for x, y in zip(raw, reference[key], strict=True)
                      for a, b in zip(x, y, strict=True)]
            max_error = max(abs(x) for x in errors)
            mse = sum(x*x for x in errors) / len(errors)
            consistency[key] = {"max_abs_error": max(max_error, consistency.get(key, {}).get("max_abs_error", 0)),
                                "mse": max(mse, consistency.get(key, {}).get("mse", 0))}
            if any(abs(a-b) > args.atol + args.rtol * abs(b)
                   for x, y in zip(raw, reference[key], strict=True)
                   for a, b in zip(x, y, strict=True)):
                save_json(output / f"{mode}_equivalence_failure.json", {
                    "status": "FAILED", "mode": mode, "task_id": task_id, "index": index,
                    "atol": args.atol, "rtol": args.rtol, "max_abs_error": max_error, "mse": mse,
                    "expected": reference[key], "actual": raw})
                save_json(output / "run_status.json", {"status": "FAILED", "reason": "equivalence",
                                                       "mode": mode, "task_id": task_id})
                raise RuntimeError(f"{mode} {task_id} equivalence exceeded declared tolerance")
            validation_end = time.perf_counter()
            steady_start = time.perf_counter()
            core.request_actions(observation)
            steady_end = time.perf_counter()
            steady_timings = dict(getattr(core, "last_timings", {}).get("inference", {}))
            row = {"kind": "gpu_offline" if args.registry else "simulation", "mode": mode,
                   "index": index, "task_id": task_id, "session_id": status["session_id"],
                   "policy_epoch": status["policy_epoch"], "switch_ack_ms": (ack-start)*1000,
                   "first_valid_chunk_ms": (chunk_time-start)*1000,
                   "first_simulated_action_accepted_ms": (action_time-start)*1000,
                   "validation_overhead_ms": (validation_end-action_time)*1000,
                   "first_inference_ms": (chunk_time-inference_start)*1000,
                   "steady_inference_ms": (steady_end-steady_start)*1000,
                   "server_switch_ack_ms": status.get("switch_ack_s", 0)*1000,
                   "timings_s": {"switch": switch_timings, "first_inference": first_timings,
                                 "steady_inference": steady_timings},
                   "client_metrics": dict(core.metrics), "server_metrics": dict(runtime.metrics)}
            rows.append(row)
            with raw_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        summary = {"status": "GPU_OFFLINE_PASSED" if args.registry else "SIMULATION_PASSED",
                   "mode": mode, "samples": len(rows), "cold_load_s": load_s, "warmup_s": warmup_s,
                   "dual_mode": dual_mode, "inference_dtype_override": dtype,
                   "backend_mode": getattr(backend, "comparison_mode", mode),
                   "parameter_dtypes": getattr(backend, "parameter_dtypes", None),
                   "compute_precision": getattr(backend, "compute_precision", None),
                   "numerical_control_s": numerical_control_s,
                   "numeric_controls_status": controls["status"] if controls else "NOT_EXECUTED_SIMULATION",
                   "initialization_scope": "Existing process backend construction, including local weights; process cold start not measured",
                   "process_cold_start_s": None, "initial_session_activation_s": initial_session_activation_s,
                   "atol": args.atol, "rtol": args.rtol, "consistency": consistency,
                   "metrics": {name: _percentiles([r[name] for r in rows]) for name in
                       ("switch_ack_ms", "first_valid_chunk_ms", "first_simulated_action_accepted_ms", "steady_inference_ms")},
                   "memory": _memory(args.device) if args.registry else {"measured": False},
                   "memory_peak_scope": "Inference after warmup; load/warmup peak not measured",
                   "parameter_storage": getattr(backend, "storage_report", None),
                   "load_counts": getattr(backend, "load_counts", None),
                   "switch_file_io_intercepted": True, "switch_parameter_storage_unchanged": True,
                   "forbidden_switch_calls_guarded": guarded_calls,
                   "untimed_residency_switch_checks": args.switches,
                   "timing_components_ms": {phase: {key: _percentiles([r["timings_s"][phase][key] * 1000 for r in rows])
                       for key in rows[0]["timings_s"][phase]} for phase in ("switch", "first_inference", "steady_inference")},
                   "client_metrics": dict(core.metrics), "server_metrics": dict(runtime.metrics),
                   "control_jitter": "NOT_MEASURED: synchronous offline simulator has no real-time control clock",
                   "task_success_rate": "NOT_MEASURED: no physical rollout",
                   "tail_note": "At least 200 samples recommended" if len(rows) < 200 else "200+ samples; offline control interface"}
        all_summaries[mode] = summary
        save_json(output / f"{mode}_summary.json", summary)
        verify_inputs()
        del core, runtime, backend
        gc.collect()
        if args.registry and args.device != "cpu":
            import torch
            torch.cuda.empty_cache()
    save_json(output / "summary.json", all_summaries)
    save_json(output / "run_status.json", {"status": "PASSED", "scope": "gpu_offline" if args.registry else "simulation"})
    print(json.dumps(all_summaries, indent=2))


def serve(args):
    from concurrent import futures

    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.hotswitch.backend import (
        DualModelBackend,
        DualUnmergedBackend,
        SharedAdapterBackend,
    )
    from lerobot.async_inference.hotswitch.runtime import HotSwitchRuntime
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2_grpc

    if not 0 <= args.port <= 65535:
        raise ValueError("port must be 0 (automatic) or 1..65535")
    if args.ready_file and Path(args.ready_file).exists():
        raise FileExistsError(f"Refusing to overwrite ready evidence: {args.ready_file}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[
        ("grpc.so_reuseport", 0),
        ("grpc.max_receive_message_length", 64*1024*1024), ("grpc.max_send_message_length", 64*1024*1024)])
    bound_port = server.add_insecure_port(f"127.0.0.1:{args.port}")
    if not bound_port:
        raise RuntimeError("Could not bind loopback server")
    backend_class = {"shared": SharedAdapterBackend, "dual": DualModelBackend, "dual-unmerged": DualUnmergedBackend}[args.backend]
    backend = backend_class.from_registry(args.registry, device=args.device, dtype=getattr(args, "dtype", None))
    runtime = HotSwitchRuntime(backend)
    status = runtime.preload()
    servicer = PolicyServer(PolicyServerConfig(host="127.0.0.1", port=bound_port), hotswitch_runtime=runtime)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    server.start()
    ready = {"ready": status, "address": f"127.0.0.1:{bound_port}",
             "backend_mode": getattr(backend, "comparison_mode", args.backend),
             "parameter_dtypes": getattr(backend, "parameter_dtypes", None),
             "compute_precision": getattr(backend, "compute_precision", None),
             "inference_dtype_override": getattr(args, "dtype", None)}
    if args.ready_file:
        save_json(args.ready_file, ready)
    print(json.dumps(ready, indent=2), flush=True)
    server.wait_for_termination()


def offline_client(args):
    """Exercise the real RPC with saved observations and a simulated controller."""
    from safetensors.torch import load_file

    from lerobot.async_inference.hotswitch.manifest import load_task_bundle, read_json
    from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore
    from lerobot.async_inference.hotswitch.simulation import SimulatedRobot

    if not math.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control-hz must be finite and positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite offline RPC evidence: {output}")
    interval_s = 1 / args.control_hz
    registry = read_json(args.registry)
    observations = {}
    for task_id, path in registry["tasks"].items():
        root, manifest = load_task_bundle(path)
        observations[task_id] = load_file(str(root / manifest["offline_observation"]))
    transport = GrpcHotSwitchTransport(args.address, timeout=args.timeout)
    robot = SimulatedRobot()
    core = HotSwitchClientCore(transport, safe_hold=robot.hold,
                               minimum_action_interval_s=interval_s)
    rows = []
    inference_requests = []
    error = None
    try:
        core.connect(args.task_id, previous_session_id=getattr(args, "previous_session_id", None))
        print(json.dumps({"status": core.status, "hardware": False}), flush=True)
        for step in range(args.steps):
            start = time.perf_counter()
            core.control_boundary()
            task = core.status["task_id"]
            if args.switch_every and step and step % args.switch_every == 0:
                ids = list(observations)
                core.switch(ids[(ids.index(task) + 1) % len(ids)])
                task = core.status["task_id"]
            meta = core.status["task_metadata"][task]
            if len(core.queue) <= max(1, meta.get("prefix_steps", 0)):
                robot.hold()
                if not core.request_actions(observations[task]):
                    raise RuntimeError("Current offline RPC response was discarded")
                inference_requests.append(copy.deepcopy(core.last_inference_alignment))
            command = core.next_action()
            raw_action = list(core.pending[0])
            actual = robot.send_action(command)
            core.confirm_action(actual)
            rows.append({"step": step, "task_id": task, "session_id": core.status["session_id"],
                         "policy_epoch": core.status["policy_epoch"], "accepted_steps": core.accepted_steps,
                         "action_identity": dict(getattr(core, "last_accepted_identity", {}) or {}),
                         "raw_action": raw_action, "processed_action": command, "accepted_action": actual,
                         "accepted_monotonic_s": core.last_acceptance_time,
                         "accepted_interval_s": core.last_acceptance_interval_s,
                         "minimum_interval_wait_s": core.last_action_wait_s,
                         "timings_s": dict(getattr(core, "last_timings", {})),
                         "cycle_s": time.perf_counter() - start,
                         "cycle_scope": "Loop work including refill and minimum interval wait; excludes outer sleep",
                         "physical_robot": False})
            time.sleep(max(0, interval_s - (time.perf_counter() - start)))
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        robot.hold()
        transport.close()
        save_json(output, {"status": "OFFLINE_RPC_FAILED" if error else "OFFLINE_RPC_PASSED",
                           "rows": rows, "error": error, "control_hz": args.control_hz,
                           "inference_requests": inference_requests,
                           "control_timing": _control_timing_summary(rows, interval_s),
                           "metrics": dict(core.metrics), "safe_holds": robot.holds,
                           "physical_robot": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env-check", help="Read-only environment provenance; no robot imports")
    env.add_argument("--output")
    env.add_argument("--include-torch", action="store_true")
    env.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    bench = sub.add_parser("benchmark", help="Compare resident backends with a simulated control interface")
    bench.add_argument("--registry", help="Omit for pure protocol simulation")
    bench.add_argument("--source-manifest", help="Frozen source inventory to bind into input_provenance.json")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--dtype", choices=("bfloat16", "float32"), help="Runtime-only override before Base construction")
    bench.add_argument("--dual-mode", choices=("merged", "unmerged"), default="merged")
    bench.add_argument("--switches", type=int, default=200)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--atol", type=float, default=0.02)
    bench.add_argument("--rtol", type=float, default=0.01)
    bench.add_argument("--simulated-inference-ms", type=float, default=0.1)
    bench.add_argument("--output", required=True)
    server = sub.add_parser("serve", help="Load registry, warm every task, then serve on loopback")
    server.add_argument("--registry", required=True)
    server.add_argument("--backend", choices=("shared", "dual", "dual-unmerged"), default="shared")
    server.add_argument("--device", default="cuda")
    server.add_argument("--dtype", choices=("bfloat16", "float32"), help="Runtime-only override before Base construction")
    server.add_argument("--port", type=int, default=8099)
    server.add_argument("--ready-file", help="Write resolved loopback address only after successful preload and startup")
    client = sub.add_parser("offline-client", help="Saved observations + real RPC + simulated robot only")
    client.add_argument("--registry", required=True)
    client.add_argument("--task-id", required=True)
    client.add_argument("--previous-session-id", help="Explicit reconnect token from status; never auto-steals a session")
    client.add_argument("--address", default="127.0.0.1:8099")
    client.add_argument("--timeout", type=float, default=60)
    client.add_argument("--steps", type=int, default=200)
    client.add_argument("--control-hz", type=float, default=30)
    client.add_argument("--switch-every", type=int, default=0)
    client.add_argument("--output", required=True)
    for name in ("status", "switch-task"):
        command = sub.add_parser(name)
        command.add_argument("--address", default="127.0.0.1:8099")
        command.add_argument("--timeout", type=float, default=30)
        if name == "switch-task":
            command.add_argument("--session-id", required=True)
            command.add_argument("--task-id", required=True)
            command.add_argument("--switch-id", required=True)
    args = parser.parse_args()
    if args.command == "env-check":
        report = collect_environment(args.repo, args.include_torch)
        if args.output:
            save_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "offline-client":
        if args.steps < 1 or not math.isfinite(args.control_hz) or args.control_hz <= 0 or args.switch_every < 0:
            parser.error("steps must be positive; control-hz must be finite and positive; switch-every must be nonnegative")
        offline_client(args)
    else:
        from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport
        transport = GrpcHotSwitchTransport(args.address, timeout=args.timeout)
        try:
            if args.command == "status":
                result = transport.status()
            else:
                result = transport.request_switch(args.task_id, args.switch_id, args.session_id)
            print(json.dumps(result, indent=2))
        finally:
            transport.close()


if __name__ == "__main__":
    main()
