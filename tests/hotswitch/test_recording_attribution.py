"""CPU-only: real client core and record_loop body, no robot SDK/connection."""

import ast
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.async_inference.hotswitch.runtime import HotSwitchError, HotSwitchRuntime
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend
from lerobot.datasets.utils import load_info, validate_feature_dtype_and_shape
from lerobot.utils.recording_annotations import resolve_collector_policy_id

HERE = Path(__file__).resolve().parent
PATCH_SOURCE = Path(os.environ.get("RECORDING_ATTRIBUTION_SOURCE", HERE.parents[1] / "src"))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


attribution = load_module("lerobot.utils.recording_attribution", PATCH_SOURCE / "lerobot/utils/recording_attribution.py")
remote = load_module("lerobot.scripts.recording_remote_policy", PATCH_SOURCE / "lerobot/scripts/recording_remote_policy.py")

class InProcessTransport:
    def __init__(self, runtime):
        self.runtime = runtime
        self.switch_calls = []
        self.intent_calls = []

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def switch(self, task_id, switch_id, session_id):
        self.switch_calls.append(task_id)
        return self.runtime.switch(task_id, switch_id, session_id)

    def request_switch(self, task_id, switch_id, session_id):
        self.intent_calls.append(task_id)
        return self.runtime.request_switch(task_id, switch_id, session_id)

    def close(self):
        pass


class CpuRobot:
    action_features = ["joint.pos", "gripper.pos"]

    def __init__(self):
        self.holds = 0
        self.hold_threads = []
        self.accepted = []

    def hold(self):
        self.holds += 1
        self.hold_threads.append(threading.get_ident())

    def get_observation(self):
        return dict.fromkeys(self.action_features, 0.0)

    def send_action(self, action):
        self.accepted.append(dict(action))
        return dict(action)


class CpuTeleoperator:
    def __init__(self):
        self.manual_modes = []

    def get_action(self):
        return {"joint.pos": 0.7, "gripper.pos": 0.8}

    def set_manual_control(self, value):
        self.manual_modes.append(value)


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("CPU recording tests must not initialize CUDA")

    @classmethod
    def tearDownClass(cls):
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("CPU recording tests initialized CUDA")

    def make_client(self):
        backend = SimulatedBackend(delay_s=0)
        for task, meta in backend.task_metadata.items():
            meta.update(action_names=CpuRobot.action_features, action_units=["deg", "dimensionless"],
                        control_hz=30, task_text=f"Task {task}")
        runtime = HotSwitchRuntime(backend)
        runtime.preload()
        transport = InProcessTransport(runtime)
        robot = CpuRobot()
        cfg = remote.RemotePolicyRecordConfig(enable=True, task_id="A")
        with patch("grpc.insecure_channel", return_value=Mock()):
            client = remote.RemotePolicyActionClient(cfg, robot, 30, safe_hold=robot.hold,
                                                    validate_action=lambda action: True)
        with patch("lerobot.async_inference.hotswitch.runtime.GrpcHotSwitchTransport", return_value=transport):
            client.start()
        client._hot.minimum_action_interval_s = 0  # CPU tests do not measure timing.
        return client, runtime, transport, robot

    def accept(self, client, robot):
        client._hot.request_actions({})
        action = dict(zip(robot.action_features, client._hot.next_action(), strict=True))
        client.confirm_sent_action(robot.send_action(action))
        return client.recording_frame_attribution(selected_from_policy=True)

    def assert_failed_closed(self, client, robot):
        self.assertTrue(client._hot.stopped)
        self.assertFalse(client._hot.queue)
        self.assertTrue(client._recording_attribution.failed)
        self.assertGreater(robot.holds, 0)

    def test_stock_cli_cannot_bypass_required_hold_or_limits(self):
        cfg = remote.RemotePolicyRecordConfig(enable=True, task_id="A")
        with patch("grpc.insecure_channel") as channel:
            with self.assertRaisesRegex(ValueError, "safe_hold"):
                remote.RemotePolicyActionClient(cfg, CpuRobot(), 30)
            with self.assertRaisesRegex(ValueError, "limits"):
                remote.RemotePolicyActionClient(cfg, CpuRobot(), 30, safe_hold=lambda: None)
        channel.assert_not_called()

    def test_no_columns_refuses_direct_intent_before_any_activation_rpc(self):
        client, runtime, transport, robot = self.make_client()
        self.assertEqual(client.begin_recording_episode({}), "single_task_only")
        before = runtime.status()
        with self.assertRaisesRegex(attribution.RecordingAttributionError, "Finish this episode"):
            client.request_task_switch("B")
        self.assertEqual(transport.intent_calls, [])
        self.assertEqual(transport.switch_calls, ["A"])
        self.assertEqual(runtime.status()["policy_epoch"], before["policy_epoch"])
        self.assertEqual(runtime.status()["task_id"], "A")
        self.assertFalse(client._hot.stopped)
        self.assertFalse(client._recording_attribution.failed)
        self.assertTrue(client._recording_attribution.active)

    def test_external_intent_is_guarded_at_actual_core_switch(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        client._hot.request_actions({})
        before = runtime.status()
        runtime.request_switch("B", "external", before["session_id"])
        with self.assertRaises(attribution.RecordingAttributionError):
            client.control_boundary()
        self.assertEqual(transport.switch_calls, ["A"])
        self.assertEqual(runtime.status()["task_id"], "A")
        self.assertEqual(runtime.status()["policy_epoch"], before["policy_epoch"])
        self.assert_failed_closed(client, robot)
        with self.assertRaises(HotSwitchError):
            client._hot.next_action()

    def test_direct_core_switch_cannot_bypass_episode_guard(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        with self.assertRaises(attribution.RecordingAttributionError):
            client._hot.switch("B")
        self.assertEqual(transport.switch_calls, ["A"])
        self.assertEqual(runtime.status()["task_id"], "A")

    def test_intent_thread_rejection_during_send_does_not_hold_or_mutate_queue(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        client._hot.request_actions({})
        command = dict(zip(robot.action_features, client._hot.next_action(), strict=True))
        entered_send, intent_done = threading.Event(), threading.Event()
        observed = {}
        owner = threading.get_ident()

        def intent_thread():
            if not entered_send.wait(2):
                return
            before = (list(client._hot.queue), dict(client._hot.pending_identity), robot.holds)
            for name, call in [("intent", lambda: client.request_task_switch("B")),
                               ("core_switch", lambda: client._hot.switch("B")),
                               ("boundary", client.control_boundary), ("reset", client.reset)]:
                try:
                    call()
                except attribution.RecordingAttributionError:
                    observed[name] = "rejected"
            observed["unchanged"] = before == (list(client._hot.queue), dict(client._hot.pending_identity), robot.holds)
            intent_done.set()

        worker = threading.Thread(target=intent_thread)
        worker.start()
        original_send = robot.send_action

        def paused_cpu_send(action):
            entered_send.set()
            if not intent_done.wait(2):
                raise AssertionError("CPU intent did not finish while send was in progress")
            return original_send(action)

        robot.send_action = paused_cpu_send
        actual = robot.send_action(command)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(observed, {"intent": "rejected", "core_switch": "rejected", "boundary": "rejected",
                                    "reset": "rejected", "unchanged": True})
        self.assertFalse(transport.intent_calls)
        self.assertEqual(transport.switch_calls, ["A"])
        self.assertTrue(all(thread == owner for thread in robot.hold_threads))
        client.confirm_sent_action(actual)
        self.assertEqual(client.recording_frame_attribution(selected_from_policy=True), {})
        self.assertFalse(client._hot.stopped)

    def test_valid_intent_thread_only_submits_and_owner_commits_after_confirmation(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode(attribution.resident_recording_features())
        client._hot.request_actions({})
        command = dict(zip(robot.action_features, client._hot.next_action(), strict=True))
        owner = threading.get_ident()
        holds_before = robot.holds
        worker = threading.Thread(target=lambda: client.request_task_switch("B", "thread-intent"))
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(robot.holds, holds_before)
        self.assertEqual(runtime.status()["task_id"], "A")
        client.confirm_sent_action(robot.send_action(command))
        first = client.recording_frame_attribution(selected_from_policy=True)
        client.control_boundary()
        second = self.accept(client, robot)
        self.assertEqual(first[attribution.FIELDS["task_id"]], "A")
        self.assertEqual(second[attribution.FIELDS["task_id"]], "B")
        self.assertTrue(all(thread == owner for thread in robot.hold_threads))

    def test_same_task_reset_and_intent_preserve_single_task_episode(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        session = client._hot.status["session_id"]
        client.reset()
        self.assertNotEqual(client._hot.status["session_id"], session)
        self.assertTrue(client._recording_attribution.active)
        client.request_task_switch("A", "same")
        client.control_boundary()
        self.assertEqual(self.accept(client, robot), {})
        self.assertFalse(client._recording_attribution.failed)

    def test_partial_or_wrong_dataset_schema_refused_before_first_command(self):
        for mode in ("partial", "dtype", "shape"):
            with self.subTest(mode=mode):
                client, runtime, transport, robot = self.make_client()
                features = attribution.resident_recording_features()
                if mode == "partial":
                    features.pop(attribution.FIELDS["request_id"])
                elif mode == "dtype":
                    features[attribution.FIELDS["policy_epoch"]]["dtype"] = "float32"
                else:
                    features[attribution.FIELDS["policy_epoch"]]["shape"] = (2,)
                with self.assertRaises(attribution.RecordingAttributionError):
                    client.begin_recording_episode(features)
                self.assertFalse(robot.accepted)
                self.assert_failed_closed(client, robot)

    def test_json_roundtrip_and_actual_dataset_load_info_shape_normalization(self):
        raw = json.loads(json.dumps({"features": attribution.resident_recording_features()}))
        self.assertEqual(raw["features"][attribution.FIELDS["policy_epoch"]]["shape"], [1])
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode(raw["features"])
        frame = self.accept(client, robot)
        self.assertEqual(raw["features"][attribution.FIELDS["policy_epoch"]]["shape"], [1])
        with patch("lerobot.datasets.utils.load_json", return_value=raw):
            actual_loaded = load_info(Path("CPU_fixture_no_disk_read"))
        for name, value in frame.items():
            self.assertEqual(actual_loaded["features"][name]["shape"], (1,))
            self.assertEqual(validate_feature_dtype_and_shape(name, actual_loaded["features"][name], value), "")

    def test_schema_does_not_accept_bool_string_or_multiaxis_shapes(self):
        for shape in ([True], ["1"], [1, 1], [], 1, None):
            with self.subTest(shape=shape):
                client, runtime, transport, robot = self.make_client()
                features = attribution.resident_recording_features()
                features[attribution.FIELDS["policy_epoch"]]["shape"] = shape
                with self.assertRaises(attribution.RecordingAttributionError):
                    client.begin_recording_episode(features)
                self.assertFalse(robot.accepted)

    def test_full_schema_records_actual_confirmed_identity_across_aba(self):
        client, runtime, transport, robot = self.make_client()
        features = attribution.resident_recording_features()
        self.assertEqual(client.begin_recording_episode(features), "full_identity")
        frames = []
        for i, task in enumerate(("A", "B", "A")):
            if i:
                client.request_task_switch(task)
                client.control_boundary()
            frame = self.accept(client, robot)
            frames.append(frame)
            accepted = client._hot.last_accepted_identity
            self.assertEqual(frame[attribution.FIELDS["request_id"]], accepted["request_id"])
            self.assertEqual(frame[attribution.FIELDS["task_id"]], task)
            self.assertEqual(frame[attribution.FIELDS["adapter_id"]], task)
            self.assertEqual(frame[attribution.FIELDS["session_id"]], accepted["session_id"])
            self.assertEqual(frame[attribution.FIELDS["policy_epoch"]].tolist(), [accepted["policy_epoch"]])
            self.assertEqual(frame[attribution.FIELDS["action_source"]], "policy")
            for name, value in frame.items():
                self.assertEqual(validate_feature_dtype_and_shape(name, features[name], value), "")
        self.assertEqual(len({f[attribution.FIELDS["request_id"]] for f in frames}), 3)

    def test_human_frame_does_not_claim_previous_policy_request(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode(attribution.resident_recording_features())
        self.accept(client, robot)
        frame = client.recording_frame_attribution(selected_from_policy=False)
        self.assertEqual(frame[attribution.FIELDS["request_id"]], "")
        self.assertEqual(frame[attribution.FIELDS["action_source"]], "human")
        self.assertEqual(frame[attribution.FIELDS["task_id"]], "A")

    def test_policy_frame_without_confirmation_fails_closed(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode(attribution.resident_recording_features())
        client._hot.request_actions({})
        client._hot.next_action()
        with self.assertRaisesRegex(attribution.RecordingAttributionError, "unconfirmed"):
            client.recording_frame_attribution(selected_from_policy=True)
        self.assert_failed_closed(client, robot)

    def test_pending_command_cannot_reuse_previous_accepted_identity(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode(attribution.resident_recording_features())
        self.accept(client, robot)
        client._hot.next_action()
        with self.assertRaisesRegex(attribution.RecordingAttributionError, "unconfirmed"):
            client.recording_frame_attribution(selected_from_policy=True)
        self.assert_failed_closed(client, robot)

    def test_legacy_client_has_no_resident_annotation_or_episode_side_effect(self):
        client = object.__new__(remote.RemotePolicyActionClient)
        client._hot = None
        self.assertIsNone(client.begin_recording_episode({}))
        self.assertEqual(client.recording_frame_attribution(selected_from_policy=True), {})
        self.assertEqual(client.recording_frame_attribution(selected_from_policy=False), {})
        self.assertIsNone(client.end_recording_episode(failed=True))

    def test_failed_episode_cannot_be_reopened_by_reset_or_begin(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        client.end_recording_episode(failed=True)
        count = len(transport.switch_calls)
        for call in (client.reset, lambda: client.begin_recording_episode({}), lambda: client._hot.switch("A")):
            with self.assertRaises(attribution.RecordingAttributionError):
                call()
        self.assertEqual(len(transport.switch_calls), count)
        self.assert_failed_closed(client, robot)

    def test_normal_episode_end_allows_switch_before_new_episode(self):
        client, runtime, transport, robot = self.make_client()
        client.begin_recording_episode({})
        self.accept(client, robot)
        client.end_recording_episode()
        client.request_task_switch("B")
        client.control_boundary()
        client.begin_recording_episode({})
        self.assertEqual(client._recording_attribution.task_id, "B")
        self.assertEqual(self.accept(client, robot), {})

    def run_loop(self, *, full=True, human=False, external_switch=False, write_fails=False, modified=False):
        client, runtime, transport, robot = self.make_client()
        events = {"exit_early": False}
        frames = []
        features = {"action": {"names": robot.action_features},
                    "complementary_info.collector_policy_id": {},
                    "complementary_info.is_intervention": {}, "complementary_info.state": {}}
        if full:
            features.update(attribution.resident_recording_features())

        def add_frame(frame):
            if write_fails:
                raise OSError("injected CPU dataset write failure")
            frames.append(frame)
            if len(frames) == 1:
                if external_switch:
                    runtime.request_switch("B", "external-loop", client._hot.status["session_id"])
                elif human:
                    events["toggle_intervention"] = True
                elif full:
                    client.request_task_switch("B", "inside-loop")
            if len(frames) == 2 and human:
                events["toggle_intervention"] = True
            if len(frames) == (3 if human else 2):
                events["exit_early"] = True

        def get_action(observation, task, timestep):
            if not client._hot.queue:
                client._hot.request_actions({})
            return dict(zip(robot.action_features, client._hot.next_action(), strict=True))

        client.get_action = get_action  # Only observation conversion is replaced.
        tree = ast.parse((PATCH_SOURCE / "lerobot/scripts/recording_loop.py").read_text(encoding="utf-8"))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record_loop")
        func.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), func], type_ignores=[])
        ns = {"logging": logging, "time": time, "ACTION": "action", "OBS_STR": "observation",
              "ACPInferenceConfig": lambda: SimpleNamespace(enable=False, use_cfg=False),
              "Teleoperator": CpuTeleoperator, "INTERVENTION_STATE_POLICY": 0,
              "INTERVENTION_STATE_ACTIVE": 1, "INTERVENTION_STATE_RELEASE": 2,
              "build_dataset_frame": lambda features, values, prefix: {prefix: values},
              "resolve_collector_policy_id": resolve_collector_policy_id, "precise_sleep": lambda _: None}
        import numpy as np
        ns["np"] = np
        exec(compile(ast.fix_missing_locations(module), str(PATCH_SOURCE / "lerobot/scripts/recording_loop.py"), "exec"), ns)
        teleop = CpuTeleoperator() if human else None
        processor = (lambda pair: {k: v + 1 for k, v in pair[0].items()}) if modified else (lambda pair: pair[0])
        error = None
        try:
            ns["record_loop"](robot=robot, events=events, fps=30, teleop_action_processor=lambda pair: pair[0],
                              robot_action_processor=processor, robot_observation_processor=lambda obs: obs,
                              dataset=SimpleNamespace(fps=30, features=features, add_frame=add_frame),
                              remote_policy_client=client, teleop=teleop, single_task="old-task",
                              control_time_s=2, collector_policy_id_policy="explicit-collector")
        except Exception as exc:
            error = exc
        return client, runtime, transport, robot, frames, teleop, error

    def test_real_record_loop_full_schema_mixed_episode_and_explicit_collector(self):
        client, runtime, transport, robot, frames, teleop, error = self.run_loop()
        self.assertIsNone(error)
        self.assertEqual([f[attribution.FIELDS["task_id"]] for f in frames], ["A", "B"])
        self.assertEqual([f["task"] for f in frames], ["Task A", "Task B"])
        self.assertEqual([f["complementary_info.collector_policy_id"] for f in frames], ["explicit-collector"] * 2)
        self.assertFalse(client._recording_attribution.active)
        self.assertFalse(client._recording_attribution.failed)

    def test_real_record_loop_external_intent_without_schema_never_activates(self):
        client, runtime, transport, robot, frames, teleop, error = self.run_loop(full=False, external_switch=True)
        self.assertIsInstance(error, attribution.RecordingAttributionError)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(robot.accepted), 1)
        self.assertNotIn("B", transport.switch_calls)
        self.assertEqual(runtime.status()["task_id"], "A")
        self.assertFalse(client._recording_attribution.active)
        self.assert_failed_closed(client, robot)

    def test_real_record_loop_human_takeover_release_reset_and_frame_writes(self):
        client, runtime, transport, robot, frames, teleop, error = self.run_loop(human=True)
        self.assertIsNone(error)
        self.assertEqual([f[attribution.FIELDS["action_source"]] for f in frames], ["policy", "human", "policy"])
        self.assertEqual(frames[1][attribution.FIELDS["request_id"]], "")
        self.assertNotEqual(frames[0][attribution.FIELDS["session_id"]], frames[2][attribution.FIELDS["session_id"]])
        self.assertNotEqual(frames[0][attribution.FIELDS["request_id"]], frames[2][attribution.FIELDS["request_id"]])
        self.assertEqual([f["complementary_info.collector_policy_id"] for f in frames], ["explicit-collector", "human", "explicit-collector"])
        self.assertEqual(teleop.manual_modes, [False, True, False])
        self.assertEqual([float(f["complementary_info.state"][0]) for f in frames], [0, 1, 2])

    def test_real_record_loop_dataset_write_exception_closes_and_latches_failure(self):
        client, runtime, transport, robot, frames, teleop, error = self.run_loop(write_fails=True)
        self.assertIsInstance(error, OSError)
        self.assertFalse(client._recording_attribution.active)
        self.assert_failed_closed(client, robot)

    def test_real_record_loop_modified_command_cannot_write_policy_attribution(self):
        client, runtime, transport, robot, frames, teleop, error = self.run_loop(modified=True)
        self.assertIsInstance(error, HotSwitchError)
        self.assertEqual(frames, [])
        self.assert_failed_closed(client, robot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
