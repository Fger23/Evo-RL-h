"""Real CPU RPC with fake bimanual hardware; never import/connect a robot SDK."""
import copy
from concurrent import futures
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchRuntime
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.scripts import hotswitch_so101_client as client
from lerobot.scripts import hotswitch_server as server
from lerobot.transport import services_pb2_grpc

ROOT = Path(__file__).resolve().parents[2]


def configured():
    value = json.loads((ROOT / "configs/hotswitch/client_so101.json").read_text())
    value.update(address="127.0.0.1:1", training_units="normalized", steps=9, max_delta_per_step=2.0,
                 bundle_manifest_sha256="cpu-fixture-only")
    for side in ("left", "right"):
        value["robot"][f"{side}_arm_config"]["port"] = "/dev/cpu_fixture_" + side
        for camera in value["robot"][f"{side}_arm_config"]["cameras"].values():
            if camera["type"] == "opencv":
                camera["index_or_path"] = 0
            else:
                camera["serial_number_or_name"] = "cpu-fixture"
    return value


class Backend(SimulatedBackend):
    def __init__(self):
        super().__init__(delay_s=0)
        self.task_ids = ("A", "A_copy")
        self.task_metadata["A_copy"] = self.task_metadata.pop("B")
        for meta in self.task_metadata.values():
            meta.update(action_dim=12, action_names=client.NAMES,
                        action_units=["recorded_robot_units_unverified"] * 12,
                        control_hz=30, bundle_manifest_sha256="cpu-fixture-only",
                        rtc_mode="training_time", prefix_steps=2, max_prefix_steps=37,
                        image_features={"observation.images." + x: [3, 480, 640] for x in client.CAMERAS})

    def infer(self, task_id, observation, action_prefix=None, prefix_steps=0):
        raw = torch.full((8, 12), .2)
        if prefix_steps:
            raw[:prefix_steps] = torch.tensor(action_prefix, dtype=raw.dtype)
        return raw, raw * 2

    def warmup(self, task_id):
        self.infer(task_id, {})
        self.infer(task_id, {}, [[0.0] * 12, [0.1] * 12], 2)


class FakeDriver:
    def __init__(self, config):
        self.connected = False
        self.closed = False
        self.held = 0
        self.sent = []

    def connect(self):
        self.connected = True

    def state(self):
        return [0.0] * 12

    def observe(self):
        return {**dict.fromkeys(client.NAMES, 0.0),
                **{name: torch.zeros((480, 640, 3), dtype=torch.uint8) for name in client.CAMERAS}}

    def send(self, vector):
        self.sent.append(list(vector))
        return list(vector)

    def hold(self):
        self.held += 1

    def close(self):
        self.closed = True


class OneCommandClientTests(unittest.TestCase):
    def setUp(self):
        self.runtime = HotSwitchRuntime(Backend())
        self.runtime.preload()
        self.pool = futures.ThreadPoolExecutor(max_workers=2)
        self.server = grpc.server(self.pool, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024),
                                                     ("grpc.max_send_message_length", 64 * 1024 * 1024)])
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        with patch.dict(os.environ, LEROBOT_ASYNC_LOG_TO_FILE="0"):
            servicer = PolicyServer(PolicyServerConfig(), hotswitch_runtime=self.runtime)
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, self.server)
        self.server.start()
        self.config = configured()
        self.config["address"] = f"127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.stop(0).wait()
        self.pool.shutdown(wait=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_rpc_controls_fake12axis_and_preserves_prefixes(self):
        drivers = []
        def factory(config):
            driver = FakeDriver(config)
            drivers.append(driver)
            return driver
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(client.run_client(self.config, driver_factory=factory), 9)
        self.assertEqual(len(drivers[0].sent), 9)
        self.assertTrue(drivers[0].closed)
        self.assertGreater(drivers[0].held, 0)
        self.assertGreater(self.runtime.metrics["inferences"], 1)
        self.assertTrue(all(len(action) == 12 for action in drivers[0].sent))

    def test_wrong_model_is_rejected_before_driver_factory(self):
        self.config["bundle_manifest_sha256"] = "wrong"
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "different model"):
            client.run_client(self.config, driver_factory=factory)
        factory.assert_not_called()

    def test_existing_session_not_taken_over_before_hardware(self):
        self.runtime.open_session()
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "Another session"):
            client.run_client(self.config, driver_factory=factory)
        factory.assert_not_called()

    def test_action_jump_never_reaches_send_and_cleanup_runs(self):
        self.config["max_delta_per_step"] = .01
        driver = FakeDriver(self.config)
        with self.assertRaisesRegex(ValueError, "Action jump"), contextlib.redirect_stdout(io.StringIO()):
            client.run_client(self.config, driver_factory=lambda cfg: driver)
        self.assertFalse(driver.sent)
        self.assertTrue(driver.closed)
        self.assertGreater(driver.held, 0)

    def test_partial_connect_failure_is_closed(self):
        driver = FakeDriver(self.config)
        driver.connect = lambda: (_ for _ in ()).throw(RuntimeError("partial camera failure"))
        with self.assertRaisesRegex(RuntimeError, "partial camera"), contextlib.redirect_stdout(io.StringIO()):
            client.run_client(self.config, driver_factory=lambda cfg: driver)
        self.assertTrue(driver.closed)
        self.assertFalse(driver.sent)

    def test_missing_client_sdk_does_not_claim_a_server_session(self):
        def missing(config):
            raise ImportError("CPU fixture: missing camera SDK")
        with self.assertRaisesRegex(ImportError, "missing camera SDK"):
            client.run_client(self.config, driver_factory=missing)
        self.assertIsNone(self.runtime.status()["session_id"])

    def test_sensor_mapping_is_exact_cpu_float_chw(self):
        value = client.policy_observation(FakeDriver(self.config).observe())
        self.assertEqual(tuple(value["observation.state"].shape), (12,))
        for name in client.CAMERAS:
            self.assertEqual(tuple(value["observation.images." + name].shape), (3, 480, 640))
            self.assertEqual(value["observation.images." + name].dtype, torch.float32)

    def test_bad_camera_never_becomes_a_policy_observation(self):
        raw = FakeDriver(self.config).observe()
        raw["left_fixed"] = torch.zeros((3, 480, 640))
        with self.assertRaisesRegex(ValueError, "RGB"):
            client.policy_observation(raw)


class ConfigurationTests(unittest.TestCase):
    def test_gpu_lock_refuses_occupied_card_without_stopping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "compute process"):
                with server.gpu_lock({"lock_root": directory}, "GPU-fixture",
                                     lambda *a, **k: SimpleNamespace(stdout="123456\n")):
                    self.fail("Busy GPU must not launch")

    def test_gpu_lock_blocks_second_launcher_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"lock_root": directory}
            empty = lambda *a, **k: SimpleNamespace(stdout="")
            with server.gpu_lock(config, "GPU-fixture", empty):
                with self.assertRaisesRegex(RuntimeError, "coordinated job"):
                    with server.gpu_lock(config, "GPU-fixture", empty):
                        self.fail("Second launcher must not acquire the same GPU")
            with server.gpu_lock(config, "GPU-fixture", empty):
                pass

    def test_server_entry_passes_explicit_host_and_has_no_ready_directory(self):
        config = {"host": "0.0.0.0", "port": 8099, "registry": "cpu-registry", "backend": "shared", "dtype": "bfloat16"}
        with patch.dict(os.environ), patch.object(server, "options", return_value=(SimpleNamespace(check=False), config)), \
             patch.object(server, "selected_uuid", return_value="GPU-fixture"), \
             patch.object(server, "gpu_lock", return_value=contextlib.nullcontext()), \
             patch("lerobot.scripts.hotswitch_cli.serve") as serve, contextlib.redirect_stdout(io.StringIO()):
            server.main([])
        args = serve.call_args.args[0]
        self.assertEqual(args.host, "0.0.0.0")
        self.assertIsNone(args.ready_file)
        self.assertEqual(args.registry, "cpu-registry")

    def test_missing_collection_units_fail_before_robot_import(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = configured()
            cfg["training_units"] = None
            path = Path(directory) / "client.json"
            path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable, "-B", str(ROOT / "client.py"), "--config", str(path), "--check"],
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no hardware connected", result.stderr)

    def test_missing_training_units_is_not_assumed_from_driver_default(self):
        cfg = configured()
        cfg["training_units"] = None
        with self.assertRaisesRegex(ValueError, "training_units"):
            client.validate_config(cfg, check_local_files=False)

    def test_degree_switch_must_agree_on_both_arms(self):
        cfg = configured()
        cfg["robot"]["left_arm_config"]["use_degrees"] = True
        with self.assertRaisesRegex(ValueError, "use_degrees"):
            client.validate_config(cfg, check_local_files=False)

    def test_wrong_camera_set_is_rejected(self):
        cfg = configured()
        cfg["robot"]["left_arm_config"]["cameras"].pop("fixed")
        with self.assertRaisesRegex(ValueError, "Camera names"):
            client.validate_config(cfg, check_local_files=False)

    def test_bad_step_limit_is_rejected(self):
        cfg = configured()
        cfg["max_delta_per_step"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            client.validate_config(cfg, check_local_files=False)

    def test_selected_gpu_is_verified_without_cuda_import(self):
        cfg = {"gpu": "1", "allowed_gpu_uuid": "GPU-test"}
        def run(argv, **kwargs):
            self.assertEqual(argv[:3], ["nvidia-smi", "-i", "1"])
            return SimpleNamespace(stdout="GPU-test\n")
        self.assertEqual(server.selected_uuid(cfg, run), "GPU-test")

    def test_wrong_gpu_uuid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "authorized GPU"):
            server.selected_uuid({"gpu": "1", "allowed_gpu_uuid": "GPU-expected"},
                                 lambda *a, **k: SimpleNamespace(stdout="GPU-other\n"))

    def test_hold_attempts_both_arms_after_one_bus_error(self):
        calls = []
        def bad_read(*args):
            raise OSError("left read failed")
        left = SimpleNamespace(bus=SimpleNamespace(is_connected=True, is_calibrated=True, sync_read=bad_read))
        right = SimpleNamespace(bus=SimpleNamespace(is_connected=True, is_calibrated=True,
            sync_read=lambda *a: dict.fromkeys(client.MOTORS, 0), sync_write=lambda *a: calls.append("right held")))
        driver = object.__new__(client.SO101Driver)
        driver.robot = SimpleNamespace(left_arm=left, right_arm=right)
        with self.assertRaisesRegex(RuntimeError, "left read failed"):
            driver.hold()
        self.assertEqual(calls, ["right held"])


if __name__ == "__main__":
    unittest.main()
