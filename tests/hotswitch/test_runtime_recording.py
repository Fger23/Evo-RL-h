"""Physical recording boundary checks use a fake client and never construct a robot."""

import ast
import logging
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from lerobot.scripts.recording_remote_policy import RemotePolicyActionClient
from lerobot.utils.recording_annotations import resolve_collector_policy_id


class RecordingBoundaryTests(unittest.TestCase):
    def make_client(self, **overrides):
        client = object.__new__(RemotePolicyActionClient)
        meta = {
            "action_names": ["joint.pos", "gripper.pos"],
            "action_units": ["deg", "dimensionless"],
            "control_hz": 30,
        }
        meta.update(overrides)
        client.robot = SimpleNamespace(action_features=["joint.pos", "gripper.pos"], send_action=Mock())
        client.environment_dt = 1 / 30
        client._safe_hold = Mock()
        client._hot = SimpleNamespace(
            status={"task_id": "A", "task_metadata": {"A": meta}},
            queue=deque([([0, 0], [0, 0])]),
            lock=threading.RLock(),
            stopped=False,
            control_boundary=Mock(),
        )
        return client

    def assert_held_without_command(self, client):
        self.assertTrue(client._hot.stopped)
        self.assertFalse(client._hot.queue)
        client._safe_hold.assert_called_once_with()
        client.robot.send_action.assert_not_called()

    def test_active_task_text_is_untagged_and_legacy_uses_callers_text(self):
        client = self.make_client()
        client._hot.status["task_metadata"]["A"]["task_text"] = "整理纸币 A"
        self.assertEqual(client.active_task_text, "整理纸币 A")
        self.assertNotIn("Advantage", client.active_task_text)
        client._hot = None
        self.assertIsNone(client.active_task_text)

    def test_unverified_units_cannot_be_authorized_by_callbacks(self):
        client = self.make_client(action_units=["recorded_robot_units_unverified"] * 2)
        with self.assertRaisesRegex(ValueError, "verified units"):
            client.control_boundary()
        self.assert_held_without_command(client)

    def test_missing_unit_list_stops_before_action_selection(self):
        client = self.make_client()
        del client._hot.status["task_metadata"]["A"]["action_units"]
        with self.assertRaisesRegex(ValueError, "verified units"):
            client._validate_hot_schema()
        self.assert_held_without_command(client)

    def test_control_frequency_mismatch_or_nan_holds_and_clears(self):
        for frequency in (20, float("nan"), float("inf"), 0):
            with self.subTest(frequency=frequency):
                client = self.make_client(control_hz=frequency)
                with self.assertRaisesRegex(ValueError, "control frequency"):
                    client.control_boundary()
                self.assert_held_without_command(client)

    def test_action_order_mismatch_holds_and_clears(self):
        client = self.make_client(action_names=["gripper.pos", "joint.pos"])
        with self.assertRaisesRegex(ValueError, "action order"):
            client.control_boundary()
        self.assert_held_without_command(client)

    def test_verified_schema_preserves_valid_queue(self):
        client = self.make_client()
        client._validate_hot_schema()
        self.assertFalse(client._hot.stopped)
        self.assertEqual(len(client._hot.queue), 1)
        client._safe_hold.assert_not_called()
        client.robot.send_action.assert_not_called()


def load_function_without_hardware_imports(filename, function_name, namespace):
    """Run the actual source body with doubles, without importing robot SDKs."""
    source = Path(__file__).resolve().parents[2] / "src/lerobot/scripts" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[function_name]


class RecordingAnnotationTests(unittest.TestCase):
    def run_frames(self, *, hot=True, explicit_id=None):
        events = {"exit_early": False}
        frames = []
        tasks = []

        class FakeRemote:
            index = -1

            def reset(self):
                pass

            def control_boundary(self):
                self.index += 1

            @property
            def active_task_text(self):
                return ("整理任务 A", "整理任务 B")[self.index] if hot else None

            @property
            def policy_id(self):
                return ("A", "B")[self.index] if hot else "legacy-model"

            def get_action(self, observation, task, timestep):
                tasks.append(task)
                return {"joint.pos": float(self.index)}

            def confirm_sent_action(self, actual):
                pass

        def add_frame(frame):
            frames.append(frame)
            if len(frames) == 2:
                events["exit_early"] = True

        namespace = {
            "logging": logging,
            "time": time,
            "ACTION": "action",
            "OBS_STR": "observation",
            "ACPInferenceConfig": lambda: SimpleNamespace(enable=False, use_cfg=False),
            "Teleoperator": type("UnconstructedTeleoperator", (), {}),
            "INTERVENTION_STATE_POLICY": 0,
            "INTERVENTION_STATE_ACTIVE": 1,
            "INTERVENTION_STATE_RELEASE": 2,
            "build_dataset_frame": lambda features, values, prefix: {prefix: values},
            "resolve_collector_policy_id": resolve_collector_policy_id,
            "precise_sleep": lambda duration: None,
        }
        loop = load_function_without_hardware_imports("recording_loop.py", "record_loop", namespace)
        robot = SimpleNamespace(
            action_features=["joint.pos"],
            get_observation=lambda: {"joint.pos": 0.0},
            send_action=lambda action: dict(action),
        )
        dataset = SimpleNamespace(
            fps=30,
            features={"action": {"names": ["joint.pos"]}, "complementary_info.collector_policy_id": {}},
            add_frame=add_frame,
        )
        loop(
            robot=robot,
            events=events,
            fps=30,
            teleop_action_processor=lambda pair: pair[0],
            robot_action_processor=lambda pair: pair[0],
            robot_observation_processor=lambda observation: observation,
            dataset=dataset,
            remote_policy_client=FakeRemote(),
            single_task="旧的固定文本",
            control_time_s=1,
            collector_policy_id_policy=explicit_id,
        )
        return tasks, frames

    def test_frames_and_remote_requests_use_current_untagged_task(self):
        tasks, frames = self.run_frames()
        self.assertEqual(tasks, ["整理任务 A", "整理任务 B"])
        self.assertEqual([frame["task"] for frame in frames], tasks)
        self.assertEqual([frame["complementary_info.collector_policy_id"] for frame in frames], ["A", "B"])

    def test_explicit_collector_id_is_never_replaced_by_task_id(self):
        _, frames = self.run_frames(explicit_id="user-selected-version")
        self.assertEqual(
            [frame["complementary_info.collector_policy_id"] for frame in frames],
            ["user-selected-version"] * 2,
        )

    def test_legacy_recording_retains_single_task(self):
        tasks, frames = self.run_frames(hot=False)
        self.assertEqual(tasks, ["旧的固定文本"] * 2)
        self.assertEqual([frame["task"] for frame in frames], tasks)
        self.assertEqual(
            [frame["complementary_info.collector_policy_id"] for frame in frames], ["legacy-model"] * 2
        )

    def test_human_inloop_keeps_hot_default_dynamic_but_preserves_explicit_id(self):
        namespace = {
            "logging": logging,
            "record": lambda cfg: cfg,
            "_HumanInloopFailureResetController": lambda cfg: SimpleNamespace(
                on_record_connected=lambda: None, on_episode_outcome=lambda: None
            ),
            "infer_collector_policy_version": lambda cfg: "local-version",
        }
        entry = load_function_without_hardware_imports(
            "lerobot_human_inloop_record.py", "human_inloop_record", namespace
        )
        for supplied, task_id, expected in (
            (None, "A", None),
            ("policy", "A", "policy"),
            (None, "", "legacy-model"),
        ):
            with self.subTest(supplied=supplied, task_id=task_id):
                cfg = SimpleNamespace(
                    teleop=object(),
                    policy=None,
                    remote_policy=SimpleNamespace(
                        enable=True, task_id=task_id, pretrained_name_or_path="legacy-model"
                    ),
                    collector_policy_id_policy=supplied,
                    intervention_toggle_key="i",
                    episode_success_key="s",
                    episode_failure_key="f",
                    acp_inference=SimpleNamespace(enable=False, use_cfg=False, cfg_beta=1.0),
                )
                self.assertEqual(entry(cfg).collector_policy_id_policy, expected)

    def test_record_entry_does_not_freeze_inferred_remote_collector_id(self):
        source = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_record.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "collector_policy_id_policy"
                for target in node.targets
            )
        )
        expression = compile(ast.Expression(body=assignment.value), str(source), "eval")
        for supplied, remote, expected in (
            (None, SimpleNamespace(policy_id="initial-A"), None),
            ("policy", SimpleNamespace(policy_id="initial-A"), "policy"),
            (None, None, "local-policy"),
        ):
            actual = eval(
                expression,
                {
                    "cfg": SimpleNamespace(collector_policy_id_policy=supplied, policy=object()),
                    "remote_policy_client": remote,
                    "infer_collector_policy_id": lambda cfg: "local-policy",
                },
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
