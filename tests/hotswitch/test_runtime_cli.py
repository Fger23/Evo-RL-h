"""CLI artifact and real loopback control-flow integration, with no hardware."""

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchRuntime
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.scripts.hotswitch_cli import _control_timing_summary, offline_client, serve
from lerobot.transport import services_pb2_grpc


class CliTests(unittest.TestCase):
    def test_serve_routes_explicit_independent_peft_and_dtype(self):
        from lerobot.async_inference.hotswitch.backend import DualUnmergedBackend

        with tempfile.TemporaryDirectory() as directory:
            server = Mock()
            server.add_insecure_port.return_value = 48001
            args = SimpleNamespace(port=0, backend="dual-unmerged", dtype="float32", device="cpu",
                                   registry="cpu-fixture", ready_file=str(Path(directory) / "ready.json"))
            with (patch("grpc.server", return_value=server),
                  patch.object(DualUnmergedBackend, "from_registry", return_value=SimulatedBackend(delay_s=0)) as factory):
                serve(args)
            factory.assert_called_once_with("cpu-fixture", device="cpu", dtype="float32")
            ready = json.loads(Path(args.ready_file).read_text(encoding="utf-8"))
            self.assertEqual(ready["inference_dtype_override"], "float32")
            self.assertEqual(ready["backend_mode"], "dual-unmerged")

    def test_untimed_numerical_controls_reuse_the_shared_prefix_and_seed(self):
        import torch

        from lerobot.scripts.hotswitch_cli import _benchmark_numeric_controls

        class Backend:
            task_ids = ("A", "A_copy")
            task_metadata = {task: {"prefix_steps": 2} for task in task_ids}

            def __init__(self, delta=0):
                self.delta = delta
                self.calls = []

            def activate(self, task):
                self.active = task

            def reset(self, task):
                self.assert_active = self.active == task

            def offline_observation(self, task):
                return {}

            def infer(self, task, observation, action_prefix=None, prefix_steps=0):
                raw = torch.randn(4, 1) + self.delta
                if prefix_steps:
                    raw[:prefix_steps] = torch.tensor(action_prefix)
                self.calls.append((task, action_prefix, prefix_steps))
                return raw, raw * 2

        args = SimpleNamespace(seed=7, atol=1e-5, rtol=1e-5)
        reference = {}
        with patch("torch.cuda.is_available", return_value=False):
            shared = _benchmark_numeric_controls(Backend(), args, "shared", reference)
            peer = Backend()
            dual = _benchmark_numeric_controls(peer, args, "dual", reference)
            bad = _benchmark_numeric_controls(Backend(delta=0.1), args, "dual", reference)
        self.assertEqual(shared["status"], "PASSED")
        self.assertEqual(dual["status"], "PASSED")
        self.assertEqual(bad["status"], "FAILED")
        for task_id in Backend.task_ids:
            self.assertEqual(dual["tasks"][task_id]["common_raw_prefix"], reference[task_id]["bootstrap_raw"][:2])
            self.assertTrue(dual["tasks"][task_id]["raw_prefix_exact"])
            self.assertEqual(dual["tasks"][task_id]["conditioned_raw_error"]["max_abs_error"], 0)

    def test_changed_source_evidence_aborts_before_timed_benchmark(self):
        from lerobot.async_inference.hotswitch import simulation
        from lerobot.scripts import hotswitch_cli

        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "source_snapshot_manifest.json"
            evidence.write_text('{"source": "before"}', encoding="utf-8")
            output = Path(directory) / "benchmark"
            args = SimpleNamespace(registry=None, source_manifest=str(evidence), device="cpu", switches=2,
                                   seed=7, atol=0.02, rtol=0.01, simulated_inference_ms=0, output=str(output))

            def changed_source_factory(mode, delay_s):
                evidence.write_text('{"source": "changed"}', encoding="utf-8")
                return SimulatedBackend(mode, delay_s)

            with (
                patch.object(simulation, "SimulatedBackend", side_effect=changed_source_factory),
                patch.object(hotswitch_cli, "collect_environment", return_value={"scope": "CPU_TEST_ONLY"}),
                self.assertRaisesRegex(RuntimeError, "source evidence changed"),
            ):
                hotswitch_cli.benchmark(args)
            status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse((output / "summary.json").exists())
            self.assertTrue((output / "input_provenance.json").exists())

    def test_control_timing_keeps_refill_stalls_in_local_interval_distribution(self):
        report = _control_timing_summary([
            {"accepted_monotonic_s": 10},
            {"accepted_monotonic_s": 10 + 1 / 30},
            {"accepted_monotonic_s": 10 + 1 / 30 + 0.1},
        ], 1 / 30)
        self.assertEqual(report["interval_ms"]["n"], 2)
        self.assertAlmostEqual(report["maximum_interval_s"], 0.1)
        self.assertAlmostEqual(report["minimum_interval_s"], 1 / 30)
        self.assertGreater(report["absolute_jitter_ms"]["p95"], 60)
        self.assertEqual(_control_timing_summary([], 1 / 30)["interval_ms"]["n"], 0)
        self.assertIsNone(_control_timing_summary([{"accepted_monotonic_s": 0}], 1 / 30)["minimum_interval_s"])

    def test_offline_client_refuses_nan_rate_and_existing_evidence_before_rpc(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("original", encoding="utf-8")
            args = SimpleNamespace(control_hz=float("nan"), output=str(output))
            with self.assertRaisesRegex(ValueError, "finite"):
                offline_client(args)
            args.control_hz = 30
            with self.assertRaises(FileExistsError):
                offline_client(args)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_residency_check_rejects_model_loader_calls_after_startup(self):
        from lerobot.async_inference.hotswitch import simulation
        from lerobot.scripts import hotswitch_cli

        class FakeResidentModel:
            calls = []

            @classmethod
            def from_pretrained(cls, task):
                cls.calls.append(("from_pretrained", task))
                return cls()

            def load_adapter(self, task):
                self.calls.append(("load_adapter", task))

            def named_parameters(self):
                return iter(())

        class BadResidentBackend(SimulatedBackend):
            def __init__(self, mode, delay_s, forbidden):
                super().__init__(mode, delay_s)
                self.forbidden = forbidden
                self.resident = FakeResidentModel.from_pretrained("startup-only-offline-fixture")
                self.models = {"resident": self.resident}

            def activate(self, task):
                getattr(self.resident, self.forbidden)(task)
                super().activate(task)

        for method in ("from_pretrained", "load_adapter"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                FakeResidentModel.calls = []
                output = Path(directory) / "benchmark"
                args = SimpleNamespace(
                    registry=None,
                    device="cpu",
                    switches=2,
                    seed=7,
                    atol=0.02,
                    rtol=0.01,
                    simulated_inference_ms=0,
                    output=str(output),
                )

                def factory(mode, delay_s, forbidden=method):
                    return BadResidentBackend(mode, delay_s, forbidden)

                with (
                    patch.object(simulation, "SimulatedBackend", side_effect=factory),
                    patch.object(
                        hotswitch_cli, "collect_environment", return_value={"scope": "CPU_TEST_ONLY"}
                    ),
                    self.assertRaisesRegex(AssertionError, method),
                ):
                    hotswitch_cli.benchmark(args)
                # The fake startup/warmup loaders really ran before interception.
                self.assertTrue(any(name == method for name, _ in FakeResidentModel.calls))
                self.assertFalse((output / "summary.json").exists())
                status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
                self.assertNotEqual(status["status"], "PASSED")

    def test_serve_port_zero_publishes_ready_only_after_warmup_and_start(self):
        from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend

        with tempfile.TemporaryDirectory() as directory:
            ready_path = Path(directory) / "ready.json"
            backend = SimulatedBackend(delay_s=0)
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
            original_warmup, original_start = backend.warmup, server.start
            events = []

            def checked_factory(*args, **kwargs):
                self.assertFalse(ready_path.exists())
                return backend

            def checked_warmup(task_id):
                self.assertFalse(ready_path.exists())
                original_warmup(task_id)
                events.append(f"warm:{task_id}")

            def checked_start():
                self.assertEqual(backend.warmed, {"A", "B"})
                self.assertFalse(ready_path.exists())
                original_start()
                events.append("started")
                self.assertFalse(ready_path.exists())

            def check_ready_then_stop():
                self.assertTrue(ready_path.is_file())
                self.assertEqual(events, ["warm:A", "warm:B", "started"])
                evidence = json.loads(ready_path.read_text(encoding="utf-8"))
                self.assertTrue(evidence["address"].startswith("127.0.0.1:"))
                port = int(evidence["address"].rsplit(":", 1)[1])
                self.assertGreater(port, 0)
                transport = GrpcHotSwitchTransport(evidence["address"], timeout=3)
                try:
                    status = transport.status()
                    self.assertEqual(status["ready"], ["A", "B"])
                    self.assertEqual(evidence["ready"]["ready"], status["ready"])
                finally:
                    transport.close()
                    server.stop(0).wait(3)

            args = SimpleNamespace(
                port=0,
                backend="shared",
                device="cpu",
                registry="simulated-fixture",
                ready_file=str(ready_path),
            )
            try:
                with (
                    patch("grpc.server", return_value=server),
                    patch.object(SharedAdapterBackend, "from_registry", side_effect=checked_factory),
                    patch.object(backend, "warmup", side_effect=checked_warmup),
                    patch.object(server, "start", side_effect=checked_start),
                    patch.object(server, "wait_for_termination", side_effect=check_ready_then_stop),
                ):
                    serve(args)
            finally:
                server.stop(0).wait(3)

    def test_occupied_port_fails_before_constructing_backend(self):
        from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend

        with (
            tempfile.TemporaryDirectory() as directory,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied,
        ):
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
            ready_path = Path(directory) / "must-not-exist.json"
            args = SimpleNamespace(
                port=port,
                backend="shared",
                device="cpu",
                registry="must-not-load",
                ready_file=str(ready_path),
            )
            try:
                with (
                    patch("grpc.server", return_value=server),
                    patch.object(
                        SharedAdapterBackend,
                        "from_registry",
                        side_effect=AssertionError("Must not load a model"),
                    ) as factory,
                    self.assertRaises(RuntimeError),
                ):
                    serve(args)
                factory.assert_not_called()
                self.assertFalse(ready_path.exists())
            finally:
                server.stop(0).wait(3)

    def test_existing_grpc_listener_cannot_share_port_or_trigger_model_load(self):
        from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend

        original_server_factory = grpc.server
        occupied = original_server_factory(futures.ThreadPoolExecutor(max_workers=1))
        port = occupied.add_insecure_port("127.0.0.1:0")
        occupied.start()
        attempted_servers = []

        def track_server(*args, **kwargs):
            self.assertIn(("grpc.so_reuseport", 0), kwargs["options"])
            server = original_server_factory(*args, **kwargs)
            attempted_servers.append(server)
            return server

        try:
            with tempfile.TemporaryDirectory() as directory:
                ready_path = Path(directory) / "must-not-be-ready.json"
                args = SimpleNamespace(
                    port=port,
                    backend="shared",
                    device="cpu",
                    registry="must-not-load",
                    ready_file=str(ready_path),
                )
                with (
                    patch("grpc.server", side_effect=track_server),
                    patch.object(
                        SharedAdapterBackend,
                        "from_registry",
                        side_effect=AssertionError("Must not load a model"),
                    ) as factory,
                    self.assertRaises(RuntimeError),
                ):
                    serve(args)
                factory.assert_not_called()
                self.assertFalse(ready_path.exists())
        finally:
            occupied.stop(0).wait(3)
            for server in attempted_servers:
                server.stop(0).wait(3)

    def test_benchmark_artifacts_are_complete_and_cannot_mix_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            command = [
                sys.executable,
                "-m",
                "lerobot.scripts.hotswitch_cli",
                "benchmark",
                "--switches",
                "3",
                "--output",
                str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            provenance = json.loads((output / "input_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["kind"], "simulation")
            self.assertEqual(provenance["options"]["switches"], 3)
            for mode in ("shared", "dual"):
                rows = [
                    json.loads(line)
                    for line in (output / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 3)
                self.assertEqual(summary[mode]["samples"], 3)
                self.assertEqual(summary[mode]["status"], "SIMULATION_PASSED")
                self.assertEqual([row["task_id"] for row in rows], ["B", "A", "B"])
                for row in rows:
                    self.assertLessEqual(row["switch_ack_ms"], row["first_valid_chunk_ms"])
                    self.assertLessEqual(
                        row["first_valid_chunk_ms"], row["first_simulated_action_accepted_ms"]
                    )
                self.assertEqual(len({row["session_id"] for row in rows}), 1)
                self.assertEqual(len({row["policy_epoch"] for row in rows}), 3)
            original_summary = (output / "summary.json").read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(
                rerun.returncode, 0, "A reused output directory can leave stale success artifacts"
            )
            self.assertEqual((output / "summary.json").read_bytes(), original_summary)

    def test_benchmark_parser_records_explicit_dtype_and_independent_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            command = [sys.executable, "-m", "lerobot.scripts.hotswitch_cli", "benchmark",
                       "--switches", "2", "--dual-mode", "unmerged", "--dtype", "float32", "--output", str(output)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            header = json.loads((output / "input_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(header["options"]["dtype"], "float32")
            self.assertEqual(header["options"]["dual_mode"], "unmerged")
            self.assertEqual(header["options"]["compute_precision"], {
                "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False})

    def test_offline_client_uses_real_rpc_and_reports_simulated_acceptance(self):
        runtime = HotSwitchRuntime(SimulatedBackend(delay_s=0))
        runtime.preload()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            PolicyServer(PolicyServerConfig(), hotswitch_runtime=runtime), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "rpc.json"
                args = SimpleNamespace(
                    registry="offline-fixture",
                    address=f"127.0.0.1:{port}",
                    timeout=3,
                    task_id="A",
                    steps=21,
                    switch_every=7,
                    control_hz=1000,
                    output=str(output),
                )
                # Only disk loading is substituted. The CLI, generated transport,
                # server lock, identities, queues and simulator all run normally.
                with (
                    patch(
                        "lerobot.async_inference.hotswitch.manifest.read_json",
                        return_value={"tasks": {"A": "A", "B": "B"}},
                    ),
                    patch(
                        "lerobot.async_inference.hotswitch.manifest.load_task_bundle",
                        return_value=(Path(directory), {"offline_observation": "unused"}),
                    ),
                    patch("safetensors.torch.load_file", return_value={}),
                ):
                    offline_client(args)
                    from lerobot.async_inference.hotswitch.simulation import SimulatedRobot

                    original_send = SimulatedRobot.send_action
                    sends = []

                    def reject_third_command(robot, action):
                        sends.append(action)
                        if len(sends) == 3:
                            raise RuntimeError("simulated controller rejection")
                        return original_send(robot, action)

                    args.output = str(Path(directory) / "failed_rpc.json")
                    args.previous_session_id = runtime.status()["session_id"]
                    with (
                        patch.object(SimulatedRobot, "send_action", new=reject_third_command),
                        self.assertRaisesRegex(RuntimeError, "simulated controller rejection"),
                    ):
                        offline_client(args)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "OFFLINE_RPC_PASSED")
                self.assertFalse(report["physical_robot"])
                self.assertEqual(len(report["rows"]), 21)
                self.assertEqual(
                    [row["task_id"] for row in report["rows"]], ["A"] * 7 + ["B"] * 7 + ["A"] * 7
                )
                self.assertGreaterEqual(report["safe_holds"], 4)
                self.assertEqual(report["control_timing"]["interval_ms"]["n"], 20)
                self.assertGreaterEqual(report["control_timing"]["minimum_interval_s"], 1 / 1000)
                self.assertTrue(all(row["action_identity"] for row in report["rows"]))
                self.assertEqual({request["prefix_steps"] for request in report["inference_requests"]}, {0, 2})
                for request in report["inference_requests"]:
                    if request["prefix_steps"]:
                        self.assertEqual(request["input_raw_prefix"], request["returned_raw_prefix"])
                        self.assertEqual(request["input_processed_prefix"], request["returned_processed_prefix"])
                    else:
                        self.assertIsNone(request["input_raw_prefix"])
                        self.assertIsNone(request["input_processed_prefix"])
                        self.assertEqual(request["returned_raw_prefix"], [])
                        self.assertEqual(request["returned_processed_prefix"], [])
                failed = json.loads(Path(args.output).read_text(encoding="utf-8"))
                self.assertEqual(failed["status"], "OFFLINE_RPC_FAILED")
                self.assertEqual(len(failed["rows"]), 2)
                self.assertEqual(failed["error"]["type"], "RuntimeError")
                self.assertFalse(failed["physical_robot"])
        finally:
            server.stop(0).wait(3)


if __name__ == "__main__":
    unittest.main()
