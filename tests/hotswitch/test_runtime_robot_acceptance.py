"""Pure mocked-driver acceptance tests. No robot constructor, SDK or hardware runs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lerobot.async_inference.hotswitch.runtime import HotSwitchRuntime
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend

SCRIPT = Path(__file__).resolve().parents[2] / "tools/hotswitch_robot_acceptance.py"
SPEC = importlib.util.spec_from_file_location("robot_acceptance_tool", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class FakeTransport:
    def __init__(self, runtime):
        self.runtime, self.closed = runtime, False

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def close(self):
        self.closed = True


class FakeDriver:
    def __init__(self, schema, rewrite=False):
        self.schema, self.rewrite = schema, rewrite
        self.calls = []
        self.current = [0.0, 0.0]

    def connect(self):
        self.calls.append("connect")

    def close(self):
        self.calls.append("close")

    def safe_hold(self):
        self.calls.append("hold")
        return {"held": True, "action_schema": self.schema, "reference_action": list(self.current)}

    def observe_policy(self):
        self.calls.append("observe")
        return {}

    def validate_action(self, action):
        self.calls.append("validate")
        return True

    def send_action(self, action):
        self.calls.append("send")
        self.current = list(action)
        if self.rewrite:
            self.current[0] += 0.1
        return list(self.current)


class RobotAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.schema = {
            "names": ["joint.pos", "gripper.pos"],
            "units": ["deg", "dimensionless"],
            "control_hz": 1000,
        }
        (self.root / "hold_evidence.txt").write_text(
            "Mock evidence for a simulated driver only", encoding="utf-8"
        )
        self.operator = {
            "operator_id": "mock-operator",
            "robot_id": "sim-only",
            "control_mode": "simulated-hold",
            "hardware_authorized": True,
            "action_schema": self.schema,
            "task_sequence": ["A", "B", "A"],
            "cycles_per_task": 1,
            "limits": {
                name: {"min": -10, "max": 10, "max_delta_per_step": 2} for name in self.schema["names"]
            },
            "hold_validation": {
                "passed": True,
                "robot_id": "sim-only",
                "control_mode": "simulated-hold",
                "evidence_path": "hold_evidence.txt",
                "max_hold_s": 5,
            },
            "driver": {"simulated": True},
        }
        self.manifests = {
            task: {
                "task_id": task,
                "action_schema": self.schema,
                "rtc": {"mode": "async", "max_delay": 0, "prefix_steps": 0},
                **{
                    key: {}
                    for key in (
                        "base_fingerprint",
                        "model_structure",
                        "adapter_structure",
                        "camera_map",
                        "tokenizer",
                    )
                },
            }
            for task in ("A", "B")
        }
        self.registry = self.root / "registry.json"
        self.registry.write_text(json.dumps({"tasks": {"A": "A", "B": "B"}}), encoding="utf-8")
        self.operator_path = self.root / "operator.json"
        self.operator_path.write_text(json.dumps(self.operator), encoding="utf-8")

    def preflight(self):
        with patch(
            "lerobot.async_inference.hotswitch.manifest.load_task_bundle",
            side_effect=lambda task: (self.root, self.manifests[task]),
        ):
            return acceptance.build_preflight(self.registry, self.operator_path, "mock_driver:factory")

    def transport(self):
        backend = SimulatedBackend(delay_s=0, rtc=False)
        for meta in backend.task_metadata.values():
            meta.update(
                action_names=self.schema["names"],
                action_units=self.schema["units"],
                control_hz=self.schema["control_hz"],
            )
        runtime = HotSwitchRuntime(backend)
        runtime.preload()
        return FakeTransport(runtime)

    def test_default_dry_run_never_imports_driver_or_invokes_hardware_path(self):
        output = self.root / "dry_run.json"
        with (
            patch(
                "lerobot.async_inference.hotswitch.manifest.load_task_bundle",
                side_effect=lambda task: (self.root, self.manifests[task]),
            ),
            patch.object(acceptance, "run_hardware_acceptance") as execute,
            patch.object(
                acceptance.importlib, "import_module", side_effect=AssertionError("Driver import forbidden")
            ),
        ):
            code = acceptance.main(
                [
                    "--registry",
                    str(self.registry),
                    "--operator-config",
                    str(self.operator_path),
                    "--driver-factory",
                    "mock_driver:factory",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 0)
        execute.assert_not_called()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "NOT_EXECUTED")
        self.assertFalse(report["hardware_connected"])
        self.assertFalse(report["driver_constructed"])
        self.assertFalse(report["server_checked"])

    def test_unverified_units_block_even_the_driver_factory(self):
        self.operator["action_schema"]["units"] = ["recorded_robot_units_unverified"] * 2
        with self.assertRaisesRegex(ValueError, "Verified action units"):
            acceptance.validate_operator(self.operator, self.manifests, self.root)

    def test_mock_execution_connects_once_and_confirms_a_b_a(self):
        report, manifests, operator = self.preflight()
        driver, transport = FakeDriver(self.schema), self.transport()
        loader = Mock(return_value=lambda config: driver)
        result = acceptance.run_hardware_acceptance(
            report,
            manifests,
            operator,
            "mock_driver:factory",
            "unused",
            1,
            transport=transport,
            factory_loader=loader,
        )
        self.assertEqual(result["status"], "HARDWARE_PROTOCOL_ACCEPTANCE_PASSED")
        self.assertEqual(driver.calls.count("connect"), 1)
        self.assertEqual(driver.calls.count("close"), 1)
        self.assertEqual(driver.calls.count("send"), 3)
        self.assertEqual([row["task_id"] for row in result["rows"]], ["A", "B", "A"])
        self.assertTrue(result["final_hold_confirmed"])
        self.assertTrue(transport.closed)
        self.assertEqual(result["task_success_rate"], "NOT_MEASURED")

    def test_live_schema_mismatch_prevents_factory_import(self):
        report, manifests, operator = self.preflight()
        transport = self.transport()
        transport.runtime.backend.task_metadata["A"]["action_units"] = ["unverified", "unverified"]
        loader = Mock(side_effect=AssertionError("Must not construct driver"))
        result = acceptance.run_hardware_acceptance(
            report,
            manifests,
            operator,
            "mock_driver:factory",
            "unused",
            1,
            transport=transport,
            factory_loader=loader,
        )
        self.assertEqual(result["status"], "HARDWARE_ACCEPTANCE_FAILED")
        self.assertFalse(result["driver_constructed"])
        loader.assert_not_called()
        self.assertTrue(transport.closed)

    def test_mock_command_rewrite_is_recorded_and_stops_in_hold(self):
        report, manifests, operator = self.preflight()
        driver, transport = FakeDriver(self.schema, rewrite=True), self.transport()
        result = acceptance.run_hardware_acceptance(
            report,
            manifests,
            operator,
            "mock_driver:factory",
            "unused",
            1,
            transport=transport,
            factory_loader=lambda spec: lambda config: driver,
        )
        self.assertEqual(result["status"], "HARDWARE_ACCEPTANCE_FAILED")
        self.assertEqual(driver.calls.count("connect"), 1)
        self.assertEqual(driver.calls.count("send"), 1)
        self.assertEqual(driver.calls.count("close"), 1)
        self.assertTrue(result["final_hold_confirmed"])
        self.assertEqual(result["rows"][0]["confirmation"], "NOT_CONFIRMED")
        self.assertNotEqual(result["rows"][0]["processed"], result["rows"][0]["accepted"])


if __name__ == "__main__":
    unittest.main()
