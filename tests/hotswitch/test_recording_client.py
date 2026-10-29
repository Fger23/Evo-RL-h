"""Offline controller tests: no robot construction, ports, or GPU calls."""

import json
import pickle
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.hotswitch import HotSwitchRuntime
from lerobot.scripts.recording_remote_policy import (
    RemotePolicyActionClient,
    RemotePolicyRecordConfig,
    remote_policy_episode,
)


class FakeRobot:
    action_features = {"joint.pos": float}

    def __init__(self):
        self.observations = 0
        self.commands = []

    def get_observation(self):
        self.observations += 1
        return {"joint.pos": 0.0}

    def send_action(self, action):
        self.commands.append(action)
        return action


class FakeBackend:
    task_ids = ["A", "B"]

    def __init__(self, rtc=False):
        self.task = None
        self.rtc = rtc
        self.requests = []

    def metadata(self, task):
        return {"action_names": ["joint.pos"], "control_hz": 100, "rtc_mode":
                "training_time" if self.rtc else "async", "rtc_prefix_steps": 2,
                "rtc_training_max_delay": 2}

    def ready(self, task):
        return True

    def activate(self, task):
        self.task = task

    def reset(self, task):
        pass

    def predict(self, task, observation, prefix_steps=0, action_prefix=None):
        self.requests.append((task, prefix_steps, action_prefix))
        raw = torch.full((4, 1), 1.0 if task == "A" else 2.0)
        if prefix_steps:
            raw[:prefix_steps] = action_prefix
        # Distinct normalization: the raw/processed queues cannot be confused.
        return raw, raw * (10 if task == "A" else 20)


class FakeStub:
    def __init__(self, runtime):
        self.runtime = runtime
        self.fail_switch_once = False
        self.corrupt_response = None

    def ControlPolicy(self, request, timeout=None):  # noqa: N802
        value = pickle.loads(request.data)
        result = self.runtime.dispatch_control(value)
        if value["op"] == "switch" and self.fail_switch_once:
            self.fail_switch_once = False
            raise TimeoutError("Reply lost after commit")
        return SimpleNamespace(data=pickle.dumps(result))

    def Predict(self, request, timeout=None):  # noqa: N802
        response = self.runtime.infer(pickle.loads(request.data))
        if self.corrupt_response:
            response = self.corrupt_response(response)
        return SimpleNamespace(data=pickle.dumps(response))


def make_client(*, rtc=False, cfg_extra=None):
    backend = FakeBackend(rtc)
    stub = FakeStub(HotSwitchRuntime(backend))
    robot = FakeRobot()
    cfg = RemotePolicyRecordConfig(enable=True, hotswitch_task_id="A", hotswitch_simulation=True,
                                  actions_per_chunk=4, obs_queue_timeout_s=0.2, **(cfg_extra or {}))
    client = RemotePolicyActionClient(cfg, robot, 100, stub=stub)
    client.start()
    return client, backend, stub


def take_action(client):
    action = client.get_action(client.robot.get_observation(), None, 0)
    client.validate_action_before_send(action)
    client.acknowledge_action(client.robot.send_action(action))
    return action


def test_a_b_a_uses_task_normalization_and_fresh_observation():
    client, backend, stub = make_client()
    assert take_action(client) == {"joint.pos": 10}
    first = client._hotswitch.identity
    client.request_task_switch("B")
    observations = client.robot.observations
    assert take_action(client) == {"joint.pos": 40}
    assert client.robot.observations == observations + 2
    assert client._hotswitch.identity.policy_epoch > first.policy_epoch
    client.request_task_switch("A")
    assert take_action(client) == {"joint.pos": 10}
    assert backend.requests == [("A", 0, None), ("B", 0, None), ("A", 0, None)]


def test_timeout_after_commit_retries_same_operation_and_reconciles():
    client, _, stub = make_client()
    before = stub.runtime.status()["switch_count"]
    stub.fail_switch_once = True
    operation = client.request_task_switch("B", "switch-1")
    assert operation == "switch-1"
    assert take_action(client) == {"joint.pos": 40}
    assert stub.runtime.status()["switch_count"] == before + 1


def test_historical_switch_reply_cannot_reactivate_a_superseded_epoch():
    client, _, stub = make_client()
    original = stub.ControlPolicy

    def superseded(request, timeout=None):
        result = original(request, timeout)
        operation = pickle.loads(request.data)
        if operation["op"] == "switch":
            stub.runtime.switch("A", "later-operation", stub.runtime.status()["session_id"])
        return result

    stub.ControlPolicy = superseded
    client.request_task_switch("B")
    with pytest.raises(RuntimeError, match="superseded"):
        take_action(client)
    assert client._hotswitch.blocked
    assert not client.robot.commands


def test_unknown_task_holds_and_cannot_use_old_queue():
    client, _, _ = make_client()
    take_action(client)
    client.request_task_switch("unknown")
    with pytest.raises(RuntimeError, match="safe hold"):
        take_action(client)
    assert client.action_queue == []
    assert client._hotswitch.blocked
    assert len(client.robot.commands) == 1


def test_old_session_response_fails_closed():
    client, _, stub = make_client()
    stub.corrupt_response = lambda response: replace(
        response, identity=replace(response.identity, session_id="old-session")
    )
    with pytest.raises(RuntimeError, match="identity"):
        take_action(client)
    assert not client.robot.commands


def test_processor_modification_rejected_before_sending():
    client, _, _ = make_client(rtc=True)
    client.get_action({}, None, 0)
    with pytest.raises(RuntimeError, match="changed a command"):
        client.validate_action_before_send({"joint.pos": 12})
    assert client._hotswitch.blocked
    assert not client.robot.commands


def test_get_action_requires_actual_control_acknowledgement():
    client, _, _ = make_client()
    client.get_action({}, None, 0)
    assert client._hotswitch.accepted_steps == 0
    with pytest.raises(RuntimeError, match="not been acknowledged"):
        client.get_action({}, None, 1)


def test_switch_inside_recorded_episode_is_explicitly_rejected():
    client, _, _ = make_client()

    @remote_policy_episode
    def episode(remote_policy_client=None, dataset=None):
        remote_policy_client.request_task_switch("B")

    with pytest.raises(RuntimeError, match="recorded episode"):
        episode(client, dataset=object())
    assert not client._hotswitch.episode_active


def test_mailbox_switch_commits_at_recorded_episode_entry(tmp_path):
    path = tmp_path / "switch.json"
    client, _, _ = make_client(cfg_extra={"hotswitch_control_file": str(path)})
    observed_tasks = []

    @remote_policy_episode
    def episode(remote_policy_client=None, dataset=None, during=None):
        assert remote_policy_client._hotswitch.episode_active
        observed_tasks.append(remote_policy_client._hotswitch.identity.task_id)
        if during is not None:
            path.write_text(json.dumps(during), encoding="utf-8")
        return take_action(remote_policy_client)

    assert episode(client, dataset=object()) == {"joint.pos": 10}
    path.write_text(json.dumps({"task_id": "B", "operation_id": "between-episodes"}), encoding="utf-8")
    assert client._hotswitch.identity.task_id == "A"
    assert episode(client, dataset=object(), during={"task_id": "A", "operation_id": "during-episode"}) == {
        "joint.pos": 40
    }
    assert client._hotswitch.identity.task_id == "B"
    assert client._hotswitch.last_control_operation == "between-episodes"
    assert episode(client, dataset=object()) == {"joint.pos": 10}
    assert observed_tasks == ["A", "B", "A"]
    assert client._hotswitch.last_control_operation == "during-episode"


def test_pending_api_request_commits_before_first_recorded_frame():
    client, _, _ = make_client()
    client.request_task_switch("B", "before-episode")

    @remote_policy_episode
    def episode(remote_policy_client=None, dataset=None):
        assert remote_policy_client._hotswitch.identity.task_id == "B"
        return take_action(remote_policy_client)

    assert episode(client, dataset=object()) == {"joint.pos": 40}


def test_teleop_reset_without_dataset_does_not_mark_recorded_episode():
    client, _, _ = make_client()

    @remote_policy_episode
    def teleop_reset(remote_policy_client=None, dataset=None):
        assert not remote_policy_client._hotswitch.episode_active
        return remote_policy_client.request_task_switch("B", "teleop-boundary")

    assert teleop_reset(client) == "teleop-boundary"
    assert client._hotswitch.identity.task_id == "A"
    assert client._hotswitch.pending_switch == ("B", "teleop-boundary")


def test_local_control_file_is_consumed_once_at_control_boundary(tmp_path):
    path = tmp_path / "switch.json"
    client, _, _ = make_client(cfg_extra={"hotswitch_control_file": str(path)})
    path.write_text(json.dumps({"task_id": "B", "operation_id": "local-1"}), encoding="utf-8")
    take_action(client)
    epoch = client._hotswitch.identity.policy_epoch
    take_action(client)
    assert client._hotswitch.identity.policy_epoch == epoch


def test_hardware_mode_requires_explicit_hold_and_complete_safety_limits():
    cfg = RemotePolicyRecordConfig(enable=True, hotswitch_task_id="A")
    with pytest.raises(ValueError, match="hold callback"):
        RemotePolicyActionClient(cfg, FakeRobot(), 100, stub=object())
    with pytest.raises(ValueError, match="per-action"):
        RemotePolicyActionClient(cfg, FakeRobot(), 100, stub=object(), hold_callback=lambda: None)


def test_hardware_jump_limit_holds_before_any_command_is_sent():
    client, _, _ = make_client()
    client.cfg.hotswitch_simulation = False
    client.cfg.hotswitch_max_jump = {"joint.pos": 1.0}
    client.cfg.hotswitch_max_velocity = {"joint.pos": 10000.0}
    holds = []
    client._hotswitch.hold = lambda: holds.append(True)
    with pytest.raises(RuntimeError, match="jump limit"):
        take_action(client)
    assert holds == [True]
    assert not client.robot.commands


def test_hardware_velocity_limit_holds_before_any_command_is_sent():
    client, _, _ = make_client()
    client.cfg.hotswitch_simulation = False
    client.cfg.hotswitch_max_jump = {"joint.pos": 100.0}
    client.cfg.hotswitch_max_velocity = {"joint.pos": 1.0}
    holds = []
    client._hotswitch.hold = lambda: holds.append(True)
    with pytest.raises(RuntimeError, match="velocity limit"):
        take_action(client)
    assert holds == [True]
    assert not client.robot.commands


def test_generic_robot_client_can_use_injected_robot_without_connect(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient

    class ConnectionGuardRobot(FakeRobot):
        observation_features = {"joint.pos": float}
        robot_type = "fake"

        def connect(self):
            raise AssertionError("No constructor hardware connections allowed")

        def disconnect(self):
            raise AssertionError("Injected existing connection must not be disconnected")

    robot = ConnectionGuardRobot()
    monkeypatch.setattr("lerobot.async_inference.robot_client.make_robot_from_config",
                        lambda _: pytest.fail("Should not create another robot"))
    cfg = RobotClientConfig(policy_type="pi05", pretrained_name_or_path="unused", robot=None,
                            actions_per_chunk=4)
    client = RobotClient(cfg, robot=robot, connect_robot=False, stub=object())
    assert client.robot is robot
    client.stop()


def test_training_rtc_refill_uses_raw_prefix_and_accepted_steps():
    client, backend, _ = make_client(rtc=True)
    take_action(client)
    hot = client._hotswitch
    # Request the refill before executing the queued actions, as a controller
    # would. It must send raw 1.0, not task-A processed 10.0.
    hot._request(client.robot.get_observation())
    assert backend.requests[-1][1] == 2
    assert torch.equal(backend.requests[-1][2], torch.ones((2, 1)))
    take_action(client)
    client.request_task_switch("B")
    take_action(client)
    assert backend.requests[-1] == ("B", 0, None)
    hot._request(client.robot.get_observation())
    assert torch.equal(backend.requests[-1][2], torch.full((2, 1), 2.0))


def test_delayed_training_rtc_response_rejected_by_actual_executed_count():
    client, _, stub = make_client(rtc=True)
    take_action(client)
    original = stub.Predict

    def delayed(request, timeout=None):
        result = original(request, timeout)
        # Three accepted controller commands exceed the two conditioned steps.
        client._hotswitch.accepted_steps += 3
        return result

    stub.Predict = delayed
    with pytest.raises(RuntimeError, match="prefix coverage"):
        client._hotswitch._request({})
    assert client._hotswitch.counters["late_chunks"] == 1


def test_late_old_error_cannot_latch_error_in_new_generation():
    client, _, stub = make_client()
    started, finish = threading.Event(), threading.Event()

    def delayed_error(request, timeout=None):
        started.set()
        assert finish.wait(1)
        raise TimeoutError("old request failure")

    stub.Predict = delayed_error
    thread = threading.Thread(target=lambda: client._hotswitch._request({}))
    thread.start()
    assert started.wait(1)
    client._hotswitch._clear()
    finish.set()
    thread.join(1)
    assert client._hotswitch.error is None
    assert client._hotswitch.counters["stale_errors"] == 1


def test_late_old_success_cannot_replace_new_generation_queue():
    client, _, stub = make_client()
    started, finish = threading.Event(), threading.Event()
    original = stub.Predict

    def delayed(request, timeout=None):
        result = original(request, timeout)
        started.set()
        assert finish.wait(1)
        return result

    stub.Predict = delayed
    thread = threading.Thread(target=lambda: client._hotswitch._request({}))
    thread.start()
    assert started.wait(1)
    client._hotswitch._clear()
    finish.set()
    thread.join(1)
    assert client.action_queue == []
    assert client._hotswitch.counters["stale_responses"] == 1


def test_background_refill_keeps_raw_processed_pair_aligned_during_execution():
    client, backend, stub = make_client(rtc=True, cfg_extra={"chunk_size_threshold": 1.0})
    original = stub.Predict
    started, finish = threading.Event(), threading.Event()
    calls = 0

    def block_refill(request, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
            assert finish.wait(1)
        return original(request, timeout)

    stub.Predict = block_refill
    take_action(client)
    assert started.wait(1)
    take_action(client)
    finish.set()
    client._hotswitch._wait_request()
    assert client._hotswitch.error is None
    assert [entry.timestep for entry in client.action_queue] == [2, 3]
    assert all(torch.equal(entry.raw * 10, entry.processed) for entry in client.action_queue)
    assert backend.requests[-1][1] == 2


def test_concurrent_task_switch_request_is_explicitly_rejected():
    client, _, _ = make_client()
    client.request_task_switch("B")
    with pytest.raises(RuntimeError, match="already pending"):
        client.request_task_switch("A")


def test_control_thread_ownership_prevents_queue_mutation_from_another_thread():
    client, _, _ = make_client()
    take_action(client)
    errors = []

    def other_thread():
        try:
            client.commit_task_switch()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(1)
    assert len(errors) == 1
    assert "control thread" in str(errors[0])


def test_nonfinite_actions_rejected_before_control_interface():
    client, _, stub = make_client()
    stub.corrupt_response = lambda response: replace(response, processed_actions=torch.full((4, 1), float("nan")))
    with pytest.raises(RuntimeError, match="finite"):
        take_action(client)
    assert not client.robot.commands
