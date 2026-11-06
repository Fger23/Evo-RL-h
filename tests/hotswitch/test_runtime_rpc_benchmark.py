"""Actual gRPC benchmark integration with CPU tensors and a simulated controller."""

import copy
import gc
import json
import tempfile
import unittest
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import grpc
import torch
from safetensors.torch import save_file

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.hotswitch.manifest import digest_file, export_task
from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchRuntime
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend, SimulatedRobot
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.scripts import hotswitch_rpc_benchmark as benchmark
from lerobot.transport import services_pb2_grpc


class RpcBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        config = {
            "type": "pi05", "chunk_size": 8, "rtc_training_max_delay": 3,
            "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
            "output_features": {"action": {"shape": [2]}}, "tokenizer_max_length": 200,
        }
        (self.base / "config.json").write_text(json.dumps(config))
        (self.base / "model.safetensors").write_bytes(b"CPU fixture; never loaded as a model")
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps(config))
        (checkpoint / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": str(self.base), "r": 32, "lora_alpha": 64,
            "peft_type": "LORA", "target_modules": ["v", "e", "a"], "bias": "none",
            "modules_to_save": [],
        }))
        (checkpoint / "adapter_model.safetensors").write_bytes(b"CPU fixture adapter")
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (checkpoint / name).write_text(json.dumps({"steps": []}))
        (checkpoint / "tokenizer").mkdir()
        (checkpoint / "tokenizer/tokenizer.json").write_text('{"fixture": true}')
        (checkpoint / "tokenizer/tokenizer_config.json").write_text('{"tokenizer_class": "FakeTokenizer"}')
        targets = self.root / "targets.json"
        targets.write_text(json.dumps({
            "vlm": ["v"], "expert": ["e"], "projections": ["a"], "target_modules": ["v", "e", "a"],
        }))
        observation = self.root / "observation.safetensors"
        save_file({"observation.state": torch.zeros(2)}, str(observation))
        self.manifests = {}
        for task_id in ("A", "B"):
            metadata = {
                "task_id": task_id, "task_text": f"Task {task_id}",
                "action_schema": {"names": ["left", "right"],
                                  "units": ["recorded_robot_units_unverified"] * 2, "control_hz": 1000},
                "rtc": {"mode": "training_time", "max_delay": 3, "prefix_steps": 2},
                "acp": {"enabled": False}, "camera_map": {},
                "tokenizer": {"name": "local", "max_length": 200},
            }
            self.manifests[task_id] = export_task(
                checkpoint, self.base, self.root / task_id, metadata, targets, observation
            )
        self.registry = self.root / "registry.json"
        self.registry.write_text(json.dumps({
            "base_path": str(self.base), "tasks": {name: str(self.root / name) for name in ("A", "B")},
            "scope": "CPU_ONLY_SIMULATION", "second_task_trained": False,
        }))
        self.source_manifest = self.root / "source_snapshot_manifest.json"
        self.source_manifest.write_text(json.dumps({"snapshot": str(self.root / "source_snapshot")}))
        # Complete frozen inventory validation has its own adversarial suite.
        # This fixture replaces only that independent proof, retaining real
        # bundle/artifact hashes, observation tensors, transport, server and core.
        self.source_patch = patch.object(benchmark, "_source_proof", side_effect=lambda path: {
            "status": "CPU_TEST_FIXTURE", "path": str(self.source_manifest),
            "manifest_sha256": digest_file(self.source_manifest),
            "client_source_sha256": digest_file(Path(benchmark.__file__)),
            "snapshot": str(self.root / "source_snapshot"),
        })
        self.source_patch.start()
        backend = SimulatedBackend(delay_s=0)
        for name, manifest in self.manifests.items():
            backend.task_metadata[name].update(
                action_names=manifest["action_schema"]["names"],
                action_units=manifest["action_schema"]["units"], control_hz=1000,
                task_text=manifest["task_text"],
                bundle_manifest_sha256=digest_file(self.root / name / "manifest.json"),
            )
        self.runtime = HotSwitchRuntime(backend)
        self.runtime.preload()
        self.executor = futures.ThreadPoolExecutor(max_workers=4)
        self.server = grpc.server(self.executor)
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            PolicyServer(PolicyServerConfig(), hotswitch_runtime=self.runtime), self.server
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.args = SimpleNamespace(
            registry=str(self.registry), source_manifest=str(self.source_manifest),
            address=f"127.0.0.1:{port}", output=str(self.root / "results"),
            server_backend="simulation", switches=3, control_hz=1000, timeout=3,
            previous_session_id=None,
        )

    def tearDown(self):
        self.source_patch.stop()
        self.server.stop(0).wait(3)
        self.executor.shutdown(wait=True, cancel_futures=True)
        # Failure-path traceback cycles can retain safetensors mmap views.
        # NFS cannot remove their .nfs* files until those views are released.
        gc.collect()
        self.temporary.cleanup()

    def rows(self, name):
        return [json.loads(line) for line in (Path(self.args.output) / f"{name}.jsonl").read_text().splitlines()]

    def test_real_loopback_three_switches_measures_local_clocks_and_prefixes(self):
        result = benchmark.run(self.args)
        self.assertEqual(result["status"], "RPC_BENCHMARK_PASSED", result.get("failure"))
        self.assertEqual((result["samples"], result["accepted_actions"], result["inference_requests"]), (3, 6, 6))
        self.assertEqual(result["prefix_steps_observed"], [0, 2])
        switches, actions, requests = self.rows("switches"), self.rows("actions"), self.rows("inference_requests")
        self.assertEqual([row["task_id"] for row in switches], ["B", "A", "B"])
        self.assertEqual(len({row["policy_epoch"] for row in switches}), 3)
        for row in switches:
            self.assertLessEqual(row["switch_ack_ms"], row["first_valid_chunk_ms"])
            self.assertLessEqual(row["first_valid_chunk_ms"], row["first_simulated_action_accepted_ms"])
        self.assertGreaterEqual(result["control_timing"]["minimum_interval_s"], 1 / 1000)
        for row in actions:
            request = next(item for item in requests if item["request_id"] == row["identity"]["request_id"])
            for key in ("session_id", "task_id", "policy_epoch"):
                self.assertEqual(row["identity"][key], request[key])
            self.assertEqual(row["processed"], row["accepted"])
        for row in requests:
            self.assertEqual(row["accepted_steps_on_response"], 0 if row["phase"] == "bootstrap" else 1)
            self.assertEqual(row["consumed_prefix_steps"], 0)
            if row["prefix_steps"]:
                self.assertEqual(row["input_raw_prefix"], row["returned_raw_prefix"])
                self.assertEqual(row["input_processed_prefix"], row["returned_processed_prefix"])
            else:
                self.assertIsNone(row["input_raw_prefix"])
                self.assertEqual(row["returned_raw_prefix"], [])
        self.assertEqual(result["server_metrics_delta"], {"inferences": 6, "switches": 3})
        for metric in result["metrics"].values():
            self.assertEqual(metric["n"], 3)
            self.assertLessEqual(metric["p50"], metric["p95"])
            self.assertLessEqual(metric["p95"], metric["p99"])
        self.assertEqual(result["component_metrics_s"]["inference"]["transport_roundtrip_s"]["n"], 6)
        self.assertFalse(result["physical_robot"])
        self.assertEqual(result["gpu_metrics"], "NOT_MEASURED_BY_CLIENT")
        self.assertEqual(json.loads((Path(self.args.output) / "run_status.json").read_text())["status"], "PASSED")

    def test_rejected_command_preserves_partial_failed_trace(self):
        class RejectedRobot(SimulatedRobot):
            def send_action(self, action):
                accepted = super().send_action(action)
                return [value + 1 for value in accepted] if len(self.accepted) == 3 else accepted

        result = benchmark.run(self.args, robot_factory=RejectedRobot)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual((result["samples"], result["accepted_actions"], result["inference_requests"]), (1, 2, 3))
        self.assertEqual(result["switches_attempted"], 2)
        self.assertEqual(result["failure"]["phase"], "bootstrap_action_acceptance")
        self.assertEqual([row["confirmation"] for row in self.rows("actions")], ["ACCEPTED", "ACCEPTED", "NOT_CONFIRMED"])
        self.assertEqual(len(self.rows("inference_requests")), 3)
        self.assertEqual(len(self.rows("switches")), 2)
        self.assertEqual(result["metrics"]["switch_ack_ms"]["n"], 1)

    def test_active_session_requires_explicit_token_before_takeover(self):
        previous = self.runtime.open_session()["session_id"]
        self.runtime.switch("A", "existing-session", previous)
        result = benchmark.run(self.args)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("explicit current --previous-session-id", result["failure"]["error"])
        self.assertEqual(self.runtime.status()["session_id"], previous)
        self.args.previous_session_id = previous
        self.args.output = str(self.root / "explicit-reconnect")
        result = benchmark.run(self.args)
        self.assertEqual(result["status"], "RPC_BENCHMARK_PASSED", result.get("failure"))
        self.assertNotEqual(self.runtime.status()["session_id"], previous)

    def test_wrong_live_package_digest_fails_before_session_creation(self):
        self.runtime.backend.task_metadata["B"]["bundle_manifest_sha256"] = "0" * 64
        result = benchmark.run(self.args)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("Live registered package/schema differs", result["failure"]["error"])
        self.assertIsNone(self.runtime.session_id)

    def test_existing_output_and_snapshot_output_never_overwritten(self):
        output = Path(self.args.output)
        output.mkdir()
        original = output / "summary.json"
        original.write_text("keep historical failed evidence")
        with self.assertRaises(FileExistsError):
            benchmark.run(self.args)
        self.assertEqual(original.read_text(), "keep historical failed evidence")
        self.args.output = str(self.root / "source_snapshot/results")
        with self.assertRaisesRegex(ValueError, "outside the frozen source"):
            benchmark.run(self.args)
        self.assertFalse((self.root / "source_snapshot").exists())

    def test_source_manifest_changes_during_run_cannot_pass(self):
        manifest_file = self.source_manifest

        class ChangingRobot(SimulatedRobot):
            def send_action(self, action):
                result = super().send_action(action)
                if len(self.accepted) == 1:
                    value = json.loads(manifest_file.read_text())
                    manifest_file.write_text(json.dumps({**value, "changed_after_setup": True}))
                return result

        result = benchmark.run(self.args, robot_factory=ChangingRobot)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure"]["phase"], "post_run_integrity")
        self.assertEqual(result["samples"], 3)  # Valid rows remain; overall integrity gate fails.

    def test_invalid_control_options_rejected_before_network(self):
        for frequency in (float("nan"), float("inf"), 0, -1):
            args = copy.copy(self.args)
            args.control_hz = frequency
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                benchmark.run(args)
        self.assertFalse(Path(self.args.output).exists())

    def test_recovered_idempotent_timeout_is_still_reported_as_anomaly(self):
        class RecoveredTimeoutTransport(GrpcHotSwitchTransport):
            calls = 0

            def switch(self, *args):
                self.calls += 1
                response = super().switch(*args)
                if self.calls == 5:  # First timed switch committed; its reply was lost.
                    raise TimeoutError("CPU test injected lost committed switch response")
                return response

        result = benchmark.run(self.args, transport_factory=RecoveredTimeoutTransport)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["samples"], 3)  # Valid retry/identity rows remain preserved.
        self.assertEqual(result["transport_exception_count"], 1)
        self.assertEqual(result["client_exception_count"], 1)
        self.assertEqual(result["transport_exceptions"][0]["method"], "switch")
        self.assertIn("recovered retries", result["failure"]["error"])
        self.assertEqual(self.runtime.metrics["switches"] - result["server_metrics_before_timing"]["switches"], 3)


if __name__ == "__main__":
    unittest.main()
