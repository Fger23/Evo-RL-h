# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote policy client used by recording loops.

This adapter talks to ``lerobot.async_inference.policy_server`` but leaves robot
control and dataset writing inside ``recording_loop.py`` so human-in-loop
takeover semantics stay in one place.
"""

import logging
import math
import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.policies.rtc.delay_telemetry import RTCDelayTelemetrySender, parse_udp_address

if TYPE_CHECKING:
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.processor import RobotAction
    from lerobot.robots import Robot
else:
    RobotAction = dict[str, Any]


AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class RemotePolicyRecordConfig:
    """Configuration for using a remote async policy while recording."""

    enable: bool = False
    server_address: str = "localhost:8080"
    policy_type: str = ""
    pretrained_name_or_path: str = ""
    policy_device: str = "cpu"
    client_device: str = "cpu"
    actions_per_chunk: int = 1
    chunk_size_threshold: float = 0.0
    aggregate_fn_name: str = "latest_only"
    obs_queue_timeout_s: float = 10.0
    rtc_d_monitor_address: str = ""
    rename_map: dict[str, str] = field(default_factory=dict)
    # Resident-task mode uses a server registry; no model path is sent remotely.
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.enable:
            return
        if not self.server_address:
            raise ValueError("`remote_policy.server_address` cannot be empty.")
        if not self.policy_type and not self.task_id:
            raise ValueError("`remote_policy.policy_type` is required when remote policy is enabled.")
        if not self.pretrained_name_or_path and not self.task_id:
            raise ValueError(
                "`remote_policy.pretrained_name_or_path` is required when remote policy is enabled."
            )
        if self.actions_per_chunk <= 0:
            raise ValueError("`remote_policy.actions_per_chunk` must be positive.")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("`remote_policy.chunk_size_threshold` must be between 0 and 1.")
        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"Unknown `remote_policy.aggregate_fn_name={self.aggregate_fn_name}`. "
                f"Available: {sorted(AGGREGATE_FUNCTIONS)}"
            )
        if self.obs_queue_timeout_s < 0:
            raise ValueError("`remote_policy.obs_queue_timeout_s` must be non-negative.")
        if self.rtc_d_monitor_address:
            parse_udp_address(self.rtc_d_monitor_address)


class RemotePolicyActionClient:
    """Small synchronous client for remote policy actions during recording."""

    _hot = None
    _hot_transport = None

    def __init__(
        self, cfg: RemotePolicyRecordConfig, robot: "Robot", fps: int, *, safe_hold=None, validate_action=None
    ):
        if not cfg.enable:
            raise ValueError("RemotePolicyActionClient requires `cfg.enable=true`.")

        self.cfg = cfg
        self.robot = robot
        self.environment_dt = 1 / fps
        self._hot = None
        self._hot_transport = None
        self._safe_hold = safe_hold
        self._validate_action = validate_action
        from lerobot.utils.recording_attribution import RecordingEpisodeAttribution

        self._recording_attribution = RecordingEpisodeAttribution()
        if cfg.task_id and not callable(safe_hold):
            raise ValueError("Resident recording requires an explicitly validated robot safe_hold callback")
        if cfg.task_id and not callable(validate_action):
            raise ValueError("Resident recording requires configured action limits and continuity validation")
        import grpc

        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        self.services_pb2 = services_pb2
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.action_queue: list[TimedAction] = []
        self.latest_action_timestep = -1
        self.latest_action = None
        self.aggregate_fn = AGGREGATE_FUNCTIONS[cfg.aggregate_fn_name]
        self._state_lock = threading.Lock()
        self._request_thread: threading.Thread | None = None
        self._generation = 0
        self.rtc_d_telemetry = (
            RTCDelayTelemetrySender(
                cfg.rtc_d_monitor_address,
                control_hz=fps,
                source="recording_remote_policy",
            )
            if cfg.rtc_d_monitor_address
            else None
        )

    @property
    def _rpc_timeout_s(self) -> float:
        return max(self.cfg.obs_queue_timeout_s, self.environment_dt)

    @property
    def policy_id(self) -> str:
        if self._hot is not None:
            return self._hot.status["task_id"]
        return self.cfg.pretrained_name_or_path or self.cfg.policy_type or "remote_policy"

    @property
    def active_task_text(self) -> str | None:
        """Untagged registry text for frame annotations; legacy callers supply their own text."""
        if self._hot is None:
            return None
        status = self._hot.status
        return status["task_metadata"][status["task_id"]]["task_text"]

    def start(self) -> None:
        if self.cfg.task_id:
            from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore

            owner = self

            class RecordingClientCore(HotSwitchClientCore):
                def switch(self, task_id, switch_id=None):
                    # Covers direct switches AND intents discovered inside
                    # control_boundary, under the same lock as actual commit.
                    with self.control_lock:
                        owner._assert_recording_control_thread()
                        owner._check_recording_task_switch(task_id)
                        return super().switch(task_id, switch_id)

            self._hot_transport = GrpcHotSwitchTransport(self.cfg.server_address, self._rpc_timeout_s)
            self._hot = RecordingClientCore(
                self._hot_transport,
                self._safe_hold,
                self._validate_action,
                minimum_action_interval_s=self.environment_dt,
            )
            self._hot.connect(self.cfg.task_id)
            self._validate_hot_schema()
            return
        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features

        self.stub.Ready(self.services_pb2.Empty(), timeout=self._rpc_timeout_s)
        policy_config = RemotePolicyConfig(
            policy_type=self.cfg.policy_type,
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy_device,
            rename_map=self.cfg.rename_map,
        )
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        logging.info("Remote policy recording client connected to %s.", self.cfg.server_address)

    def stop(self) -> None:
        if self._hot_transport is not None:
            self._hot_transport.close()
        self.channel.close()
        if self.rtc_d_telemetry is not None:
            self.rtc_d_telemetry.close()

    def reset(self) -> None:
        if self._hot is not None:
            with self._hot.control_lock:
                self._assert_recording_control_thread()
                guard = getattr(self, "_recording_attribution", None)
                if guard is not None:
                    guard.ensure_usable()
                self._safe_hold()
                task_id, session = self._hot.status["task_id"], self._hot.status["session_id"]
                self._hot.connect(task_id, previous_session_id=session)
            return
        self._wait_for_background_request()
        with self._state_lock:
            self._generation += 1
            self.action_queue.clear()
            self.latest_action_timestep = -1
            self.latest_action = None
        # Sync server dedup / policy internal state when a new episode starts or
        # control returns to the policy after human takeover.
        self.stub.Ready(self.services_pb2.Empty(), timeout=self._rpc_timeout_s)

    def _ready_to_send_observation(self) -> bool:
        with self._state_lock:
            if not self.action_queue:
                return True
            return len(self.action_queue) / self.cfg.actions_per_chunk <= self.cfg.chunk_size_threshold

    def _send_observation(self, observation: dict, timestep: int, task: str | None, must_go: bool) -> None:
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        raw_observation = dict(observation)
        if task is not None:
            raw_observation["task"] = task

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(timestep, 0),
            observation=raw_observation,
            must_go=must_go,
        )
        observation_bytes = pickle.dumps(timed_observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            self.services_pb2.Observation,
            log_prefix="[RECORD_REMOTE_POLICY] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator, timeout=self._rpc_timeout_s)

    def _receive_actions(self, generation: int) -> None:
        deadline_t = time.perf_counter() + self.cfg.obs_queue_timeout_s
        while True:
            if self.cfg.obs_queue_timeout_s == 0:
                rpc_timeout_s = self._rpc_timeout_s
            else:
                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0:
                    raise TimeoutError("Timed out waiting for remote policy actions.")
                rpc_timeout_s = remaining_s

            actions_chunk = self.stub.GetActions(
                self.services_pb2.Empty(),
                timeout=rpc_timeout_s,
            )
            if actions_chunk.data:
                receive_time = time.time()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                telemetry = getattr(self, "rtc_d_telemetry", None)
                if timed_actions and telemetry is not None:
                    with self._state_lock:
                        latest_action_timestep = self.latest_action_timestep
                    telemetry.emit(
                        observation_timestamp=timed_actions[0].get_timestamp(),
                        receive_timestamp=receive_time,
                        first_action_timestep=timed_actions[0].get_timestep(),
                        latest_action_timestep=latest_action_timestep,
                        metadata={"server_address": self.cfg.server_address},
                    )
                self._merge_actions(timed_actions, generation=generation)
                return

            if self.cfg.obs_queue_timeout_s == 0 or time.perf_counter() >= deadline_t:
                raise TimeoutError("Timed out waiting for remote policy actions.")

    def _merge_actions(self, incoming_actions: list["TimedAction"], generation: int) -> None:
        with self._state_lock:
            if generation != self._generation:
                logging.info("Discarding remote actions from stale generation %d.", generation)
                return

            current_actions = {action.get_timestep(): action for action in self.action_queue}
            merged = {}

            for action in self.action_queue:
                if action.get_timestep() > self.latest_action_timestep:
                    merged[action.get_timestep()] = action

            for action in incoming_actions:
                timestep = action.get_timestep()
                if timestep <= self.latest_action_timestep:
                    continue
                if timestep in current_actions:
                    from lerobot.async_inference.helpers import TimedAction

                    action = TimedAction(
                        timestamp=action.get_timestamp(),
                        timestep=timestep,
                        action=self.aggregate_fn(current_actions[timestep].get_action(), action.get_action()),
                    )
                merged[timestep] = action

            if incoming_actions and not merged:
                logging.warning(
                    "Remote policy returned %d actions but all were discarded because their timesteps "
                    "<= latest_action_timestep=%d. Incoming range: %d..%d.",
                    len(incoming_actions),
                    self.latest_action_timestep,
                    incoming_actions[0].get_timestep(),
                    incoming_actions[-1].get_timestep(),
                )

            self.action_queue = [merged[timestep] for timestep in sorted(merged)]

    def _request_actions(
        self,
        observation: dict,
        task: str | None,
        timestep: int,
        generation: int,
    ) -> None:
        self._send_observation(
            observation=observation,
            timestep=timestep,
            task=task,
            must_go=True,
        )
        self._receive_actions(generation=generation)

    def _background_request(
        self,
        observation: dict,
        task: str | None,
        timestep: int,
        generation: int,
    ) -> None:
        try:
            self._request_actions(observation, task, timestep, generation)
        except Exception:
            logging.exception("Remote policy background request failed.")
        finally:
            current_thread = threading.current_thread()
            with self._state_lock:
                if self._request_thread is current_thread:
                    self._request_thread = None

    def _start_background_request(self, observation: dict, task: str | None, timestep: int) -> None:
        with self._state_lock:
            if self._request_thread is not None:
                return
            if (
                self.action_queue
                and len(self.action_queue) / self.cfg.actions_per_chunk > self.cfg.chunk_size_threshold
            ):
                return
            generation = self._generation
            request_thread = threading.Thread(
                target=self._background_request,
                args=(observation, task, timestep, generation),
                daemon=True,
                name="remote-policy-request",
            )
            self._request_thread = request_thread
        request_thread.start()

    def _wait_for_background_request(self) -> None:
        with self._state_lock:
            request_thread = self._request_thread
        if request_thread is None or request_thread is threading.current_thread():
            return

        request_thread.join(timeout=2 * self._rpc_timeout_s + self.environment_dt)
        if request_thread.is_alive():
            raise TimeoutError("Timed out waiting for the in-flight remote policy request to stop.")

    def _wait_for_actions(self, observation: dict, task: str | None, timestep: int) -> None:
        while True:
            with self._state_lock:
                if self.action_queue:
                    return
                request_thread = self._request_thread
                generation = self._generation

            if request_thread is not None:
                request_thread.join(timeout=self._rpc_timeout_s + self.environment_dt)
                if request_thread.is_alive():
                    logging.warning(
                        "Remote policy action queue is empty; still waiting for in-flight inference."
                    )
                continue

            try:
                self._request_actions(observation, task, timestep, generation)
            except Exception:
                logging.exception(
                    "Remote policy action queue is empty and inference failed; retrying in %.3fs.",
                    self.environment_dt,
                )
                time.sleep(self.environment_dt)

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> RobotAction:
        if self.cfg.client_device != "cpu" and action_tensor.device.type != self.cfg.client_device:
            action_tensor = action_tensor.to(self.cfg.client_device)
        else:
            action_tensor = action_tensor.cpu()
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def get_action(self, observation: dict, task: str | None, timestep: int) -> RobotAction:
        if self._hot is not None:
            self._assert_recording_control_thread()
            from types import SimpleNamespace

            from lerobot.async_inference.helpers import (
                map_robot_keys_to_lerobot_features,
                raw_observation_to_observation,
            )

            meta = self._hot.status["task_metadata"][self._hot.status["task_id"]]
            # Synchronous first implementation explicitly holds at refill boundaries.
            # A keep-alive/servo implementation must be supplied by safe_hold.
            if len(self._hot.queue) <= meta.get("prefix_steps", 0):
                self._safe_hold()
                formatted = raw_observation_to_observation(
                    observation,
                    map_robot_keys_to_lerobot_features(self.robot),
                    {key: SimpleNamespace(shape=shape) for key, shape in meta["image_features"].items()},
                )
                self._hot.request_actions(formatted)
            action = self._hot.next_action()
            return dict(zip(self.robot.action_features, action, strict=True))
        if self._ready_to_send_observation():
            with self._state_lock:
                queue_is_empty = not self.action_queue
            if queue_is_empty:
                self._wait_for_actions(observation, task, timestep)
            else:
                self._start_background_request(observation, task, timestep)

        with self._state_lock:
            timed_action = self.action_queue.pop(0)
            self.latest_action = timed_action
            self.latest_action_timestep = timed_action.get_timestep()
        return self._tensor_to_action(timed_action.get_action())

    def _validate_hot_schema(self):
        try:
            meta = self._hot.status["task_metadata"][self._hot.status["task_id"]]
            if list(self.robot.action_features) != meta["action_names"]:
                raise ValueError("Robot action order differs from registered task schema")
            frequency = float(meta["control_hz"])
            if (
                not math.isfinite(frequency)
                or frequency <= 0
                or abs(frequency - 1 / self.environment_dt) > 1e-6
            ):
                raise ValueError("Robot control frequency differs from registered task schema")
            units = meta.get("action_units")
            unknown = ("unverified", "unknown", "unspecified", "unconfirmed", "n/a")
            if (
                not isinstance(units, list)
                or len(units) != len(meta["action_names"])
                or any(
                    not isinstance(unit, str)
                    or not unit.strip()
                    or any(marker in unit.casefold() for marker in unknown)
                    for unit in units
                )
            ):
                raise ValueError("Physical recording requires verified units for every registered action")
        except Exception:
            # Schema validation runs on the control thread. A missing/invalid
            # field must fail closed before any further command is selected.
            with self._hot.lock:
                self._hot.stopped = True
                self._hot.queue.clear()
                self._hot.queue_identity = None
                self._hot.pending_identity = None
                clock_wakeup = getattr(self._hot, "_action_clock_wakeup", None)
                if clock_wakeup is not None:
                    clock_wakeup.set()
            self._safe_hold()
            raise

    def request_task_switch(self, task_id, switch_id=None):
        """May be called from another thread; commit happens at a control boundary."""
        import uuid

        if self._hot is None:
            raise RuntimeError("Legacy protocol does not support resident task switching")
        with self._hot.control_lock:
            # This method is explicitly callable by an intent/UI thread. A
            # rejected intent is a pure error: never call SDK hold, stop or
            # mutate execution queues from that thread.
            guard = getattr(self, "_recording_attribution", None)
            if guard is not None:
                guard.check_switch(task_id)
            return self._hot_transport.request_switch(
                task_id, switch_id or uuid.uuid4().hex, self._hot.status["session_id"]
            )

    def _assert_recording_control_thread(self):
        owner = getattr(self, "_recording_control_thread", None)
        if owner is not None and threading.get_ident() != owner:
            from lerobot.utils.recording_attribution import RecordingAttributionError

            raise RecordingAttributionError("Only the recording control thread may execute this operation")

    def _recording_failure(self):
        self._assert_recording_control_thread()
        guard = getattr(self, "_recording_attribution", None)
        if guard is not None:
            guard.end(failed=True)
        self._hot.stop()
        self._safe_hold()

    def _check_recording_task_switch(self, task_id):
        guard = getattr(self, "_recording_attribution", None)
        if guard is not None:
            try:
                guard.check_switch(task_id)
            except Exception:
                self._recording_failure()
                raise

    def begin_recording_episode(self, features):
        """Bind the real dataset schema before selecting this episode's actions."""
        if self._hot is None:
            return None
        with self._hot.control_lock:
            self._assert_recording_control_thread()
            self._recording_control_thread = threading.get_ident()
            try:
                return self._recording_attribution.begin(features, self._hot.status)
            except Exception:
                self._recording_failure()
                raise

    def end_recording_episode(self, *, failed=False):
        if self._hot is None:
            return
        with self._hot.control_lock:
            self._assert_recording_control_thread()
            if failed:
                self._recording_failure()
            else:
                self._recording_attribution.end()
                self._recording_control_thread = None

    def recording_frame_attribution(self, *, selected_from_policy):
        if self._hot is None:
            return {}
        with self._hot.control_lock, self._hot.lock:
            self._assert_recording_control_thread()
            try:
                if selected_from_policy and self._hot.pending is not None:
                    from lerobot.utils.recording_attribution import RecordingAttributionError

                    raise RecordingAttributionError("Policy frame has an unconfirmed pending command")
                return self._recording_attribution.frame(
                    self._hot.status, self._hot.last_accepted_identity,
                    selected_from_policy=selected_from_policy,
                )
            except Exception:
                self._recording_failure()
                raise

    def control_boundary(self):
        """Recording control thread calls this before taking a fresh observation."""
        if self._hot is not None:
            self._assert_recording_control_thread()
            self._hot.control_boundary()
            self._validate_hot_schema()

    def confirm_sent_action(self, actual):
        if self._hot is not None:
            self._assert_recording_control_thread()
            values = None if actual is None else [actual[key] for key in self.robot.action_features]
            self._hot.confirm_action(values)
