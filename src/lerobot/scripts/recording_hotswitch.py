"""Control-thread hot switching with paired normalized/execution action state.

This module never connects a robot. The caller owns the control interface and
must acknowledge commands only after that interface accepts them.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class QueuedPolicyAction:
    identity: Any
    timestep: int
    raw: torch.Tensor
    processed: torch.Tensor

    def get_timestep(self):
        return self.timestep

    def get_action(self):
        return self.processed


class HotSwitchRecordingSession:
    """Single controller and one active task; all task changes fail closed."""

    def __init__(self, owner, *, hold=None):
        self.owner = owner
        self.hold = hold
        self.lock = threading.Lock()
        self.control_thread = None
        self.identity = None
        self.metadata = {}
        self.generation = 0
        self.accepted_steps = 0
        self.pending_action = None
        self.last_accepted = None
        self.pending_switch = None
        self.request_thread = None
        self.error = None
        self.blocked = True
        self.episode_active = False
        self.last_control_operation = None
        self.needs_fresh_observation = False
        self.bootstrap_pending = True
        self.hold_started = None
        self.counters = {
            "stale_responses": 0,
            "stale_errors": 0,
            "late_chunks": 0,
            "holds": 0,
            "accepted": 0,
            "bootstrap_requests": 0,
            "queue_underruns": 0,
            "safety_hold_seconds": 0.0,
        }
        if not owner.cfg.hotswitch_simulation:
            if hold is None:
                raise ValueError("Hardware hot switching requires an explicitly validated hold callback.")
            names = set(owner.robot.action_features)
            for limits in (owner.cfg.hotswitch_max_jump, owner.cfg.hotswitch_max_velocity):
                if set(limits) != names or any(not math.isfinite(v) or v <= 0 for v in limits.values()):
                    raise ValueError(
                        "Provide positive per-action jump and velocity limits for every robot action."
                    )

    def _claim_control(self):
        current = threading.get_ident()
        if self.control_thread is None:
            self.control_thread = current
        elif self.control_thread != current:
            raise RuntimeError("Only the control thread may modify execution state or commit a switch.")

    def _control(self, op, **values):
        request = {"protocol": "lerobot.hotswitch", "version": 1, "op": op, **values}
        reply = self.owner.stub.ControlPolicy(
            self.owner.services_pb2.PolicySetup(data=pickle.dumps(request)), timeout=self.owner._rpc_timeout_s
        )
        result = pickle.loads(reply.data)  # nosec: same trusted transport as legacy RPC
        if result.get("protocol") != request["protocol"] or result.get("version") != 1:
            raise RuntimeError("Server did not negotiate hot-switch protocol version 1.")
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Hot-switch control request rejected."))
        return result

    def _install_status(self, state):
        from lerobot.async_inference.hotswitch import PolicyIdentity

        if not state.get("healthy", False) or state.get("task_id") not in state.get("ready_tasks", []):
            raise RuntimeError("Active task is not healthy and warmed up.")
        metadata = state.get("metadata", {})
        names = metadata.get("action_names") or metadata.get("action_schema", {}).get("names")
        if names is not None and list(names) != list(self.owner.robot.action_features):
            raise ValueError("Task action order differs from the control interface.")
        fps = metadata.get("control_hz", metadata.get("fps"))
        if fps is not None and not math.isclose(float(fps), 1 / self.owner.environment_dt):
            raise ValueError("Task control frequency differs from the recording loop.")
        mode = metadata.get("rtc_mode", "async")
        if mode not in ("async", "none", "training_time"):
            raise ValueError(f"Recording hot switching does not support RTC mode {mode!r}.")
        if mode == "training_time":
            prefix_steps = metadata.get("rtc_prefix_steps", 0)
            maximum = metadata.get("rtc_training_max_delay", 0)
            if (
                not isinstance(prefix_steps, int)
                or isinstance(prefix_steps, bool)
                or not isinstance(maximum, int)
                or not 0 < prefix_steps <= maximum
            ):
                raise ValueError("Deployment RTC prefix must lie inside the checkpoint training range.")
        self.identity = PolicyIdentity(state["session_id"], state["task_id"], state["policy_epoch"], "")
        self.metadata = metadata
        self.blocked = False
        self.error = None

    def start(self):
        result = self._control("capabilities")
        capabilities = result.get("capabilities", [])
        if "hotswitch_v1" not in capabilities:
            raise RuntimeError("Server does not support hot switching.")
        self._install_status(
            self._control("connect", task_id=self.owner.cfg.hotswitch_task_id, operation_id=uuid.uuid4().hex)[
                "state"
            ]
        )
        if self.metadata.get("rtc_mode") == "training_time" and "training_time_rtc" not in capabilities:
            self.blocked = True
            raise RuntimeError("Server has not validated the Training-Time RTC path.")

    def _clear(self):
        with self.lock:
            self.generation += 1
            self.owner._generation += 1
            self.owner.action_queue.clear()
            self.owner.latest_action = None
            self.owner.latest_action_timestep = -1
            self.pending_action = None
            self.accepted_steps = 0
            self.bootstrap_pending = True

    def reset(self):
        self._claim_control()
        self.blocked = True
        self._clear()
        self._wait_request()
        task_id = self.identity.task_id if self.identity is not None else self.owner.cfg.hotswitch_task_id
        self._install_status(
            self._control("connect", task_id=task_id, operation_id=uuid.uuid4().hex)["state"]
        )

    def request_switch(self, task_id, operation_id=None):
        """Queue a switch; this method never touches the robot or action queue."""
        with self.lock:
            if self.episode_active:
                raise RuntimeError(
                    "Finish the recorded episode before changing task; mixed attribution is unsupported."
                )
            if self.pending_switch is not None:
                raise RuntimeError("A task switch is already pending.")
            self.pending_switch = (task_id, operation_id or uuid.uuid4().hex)
            return self.pending_switch[1]

    def _hold(self):
        self.counters["holds"] += 1
        if self.hold_started is None:
            self.hold_started = time.perf_counter()
        if self.hold is not None:
            self.hold()

    def _fail(self, message):
        self.blocked = True
        self._clear()
        self._hold()
        raise RuntimeError(message)

    def _wait_request(self):
        with self.lock:
            thread = self.request_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2 * self.owner._rpc_timeout_s + self.owner.environment_dt)
            if thread.is_alive():
                self._fail("Inference is still in flight; safe hold remains active.")

    def commit_switch(self):
        self._claim_control()
        with self.lock:
            pending = self.pending_switch
            if pending is None:
                return False
            if self.episode_active:
                raise RuntimeError("Task changes are prohibited inside a recorded episode.")
            self.pending_switch = None
        self.blocked = True
        self._hold()
        old_identity = self.identity
        self._clear()
        self._wait_request()
        task_id, operation_id = pending
        try:
            try:
                result = self._control(
                    "switch", task_id=task_id, operation_id=operation_id, session_id=old_identity.session_id
                )
            except Exception:
                # Query is diagnostic only; idempotent retry of the SAME operation
                # is the authority after a timeout, even when task_id is unchanged.
                self._control("status")
                result = self._control(
                    "switch", task_id=task_id, operation_id=operation_id, session_id=old_identity.session_id
                )
            state = result["state"]
            if state["session_id"] != old_identity.session_id or state["task_id"] != task_id:
                raise RuntimeError("Switch acknowledgement has an unexpected session or task.")
            current = self._control("status")["state"]
            fields = ("session_id", "task_id", "policy_epoch")
            if tuple(current[k] for k in fields) != tuple(state[k] for k in fields):
                raise RuntimeError("A later control operation superseded the acknowledged switch.")
            state = current
            self._install_status(state)
            self.needs_fresh_observation = True
        except Exception as exc:
            self._fail(f"Switch outcome is not confirmed; remain in safe hold: {exc}")
        return True

    def _request_snapshot(self):
        with self.lock:
            identity = replace(self.identity, request_id=uuid.uuid4().hex)
            generation = self.generation
            accepted_before = self.accepted_steps
            queue = list(self.owner.action_queue)
            steps = 0
            prefix = None
            if self.metadata.get("rtc_mode") == "training_time" and queue:
                configured = int(self.metadata.get("rtc_prefix_steps", 0))
                maximum = int(self.metadata.get("rtc_training_max_delay", 0))
                if not 0 < configured <= maximum:
                    raise ValueError("Deployment RTC prefix must lie inside the checkpoint training range.")
                steps = min(configured, len(queue))
                prefix = torch.stack([entry.raw for entry in queue[:steps]])
            return identity, generation, accepted_before, steps, prefix

    def _request(self, observation, snapshot=None):
        from lerobot.async_inference.hotswitch import InferenceRequest, InferenceResponse

        identity, generation, accepted_before, steps, prefix = snapshot or self._request_snapshot()
        request = InferenceRequest(
            identity, dict(observation), timestep=accepted_before, prefix_steps=steps, action_prefix=prefix
        )
        if self.bootstrap_pending:
            self.counters["bootstrap_requests"] += 1
        try:
            result = self.owner.stub.Predict(
                self.owner.services_pb2.PolicySetup(data=pickle.dumps(request)),
                timeout=self.owner._rpc_timeout_s,
            )
            response = pickle.loads(result.data)  # nosec
            with self.lock:
                if generation != self.generation or self.identity.session_id != identity.session_id:
                    self.counters["stale_responses"] += 1
                    return
                if not isinstance(response, InferenceResponse) or response.identity != identity:
                    raise ValueError("Inference response identity does not match the request.")
                if response.error:
                    raise RuntimeError(response.error)
                raw, processed = response.raw_actions, response.processed_actions
                if (
                    not isinstance(raw, torch.Tensor)
                    or not isinstance(processed, torch.Tensor)
                    or raw.ndim != 2
                    or raw.shape != processed.shape
                    or raw.shape[1] != len(self.owner.robot.action_features)
                    or not torch.isfinite(raw).all()
                    or not torch.isfinite(processed).all()
                ):
                    raise ValueError("Raw and processed actions must be aligned finite [H,D] tensors.")
                if response.timestep != accepted_before or response.prefix_steps != steps:
                    raise ValueError("Response action alignment differs from the request.")
                if steps and not torch.equal(raw[:steps].cpu(), prefix.cpu()):
                    raise ValueError("Training-Time RTC response did not hard clamp the normalized prefix.")
                consumed = self.accepted_steps - accepted_before
                if self.metadata.get("rtc_mode") == "training_time" and consumed > steps:
                    self.counters["late_chunks"] += 1
                    raise RuntimeError("Executed commands exceeded the conditioned prefix coverage.")
                if consumed >= len(raw):
                    self.counters["late_chunks"] += 1
                    raise RuntimeError("Inference chunk expired before reception.")
                first = consumed
                if self.pending_action is not None:
                    first = max(first, self.pending_action.timestep + 1 - accepted_before)
                self.owner.action_queue = [
                    QueuedPolicyAction(
                        identity,
                        accepted_before + i,
                        raw[i].detach().cpu().clone(),
                        processed[i].detach().cpu().clone(),
                    )
                    for i in range(first, len(raw))
                ]
                self.bootstrap_pending = False
        except Exception as exc:
            with self.lock:
                if generation != self.generation:
                    self.counters["stale_errors"] += 1
                    return
                self.error = exc
            raise

    def _background(self, observation, snapshot):
        try:
            self._request(observation, snapshot)
        except Exception:
            pass  # Error is latched for the control thread, which owns hold.
        finally:
            with self.lock:
                if self.request_thread is threading.current_thread():
                    self.request_thread = None

    def get_action(self, observation):
        self._claim_control()
        self._poll_control_file()
        self.commit_switch()
        if self.needs_fresh_observation:
            if not hasattr(self.owner.robot, "get_observation"):
                self._fail("A fresh observation callback is required after a task switch.")
            observation = self.owner.robot.get_observation()
            self.needs_fresh_observation = False
        if self.blocked:
            self._fail("Policy is blocked; reconnect/reset explicitly before resuming.")
        if self.pending_action is not None:
            self._fail("The previous command has not been acknowledged by the control interface.")
        if self.error is not None:
            self._fail(f"Remote inference failed: {self.error}")
        if not self.owner.action_queue:
            if not self.bootstrap_pending:
                self.counters["queue_underruns"] += 1
            self._wait_request()
            if self.error is not None:
                self._fail(f"Remote inference failed: {self.error}")
            if not self.owner.action_queue:
                try:
                    self._request(observation)
                except Exception as exc:
                    self._fail(f"Remote inference failed: {exc}")
        # Snapshot while the next command is still queued. The raw prefix then
        # covers exactly the commands that may be accepted during inference.
        try:
            snapshot = self._request_snapshot()
        except Exception as exc:
            self._fail(f"Cannot prepare an aligned RTC refill: {exc}")
        with self.lock:
            entry = self.owner.action_queue.pop(0)
            self.pending_action = entry
            self.owner.latest_action = entry
            self.owner.latest_action_timestep = entry.timestep
        action = self.owner._tensor_to_action(entry.processed)
        self._validate_limits(action, observation)
        with self.lock:
            refill = (
                self.owner.action_queue
                and self.request_thread is None
                and len(self.owner.action_queue) / self.owner.cfg.actions_per_chunk
                <= self.owner.cfg.chunk_size_threshold
            )
            if refill:
                self.request_thread = threading.Thread(
                    target=self._background,
                    args=(dict(observation), snapshot),
                    daemon=True,
                    name="hotswitch-refill",
                )
                thread = self.request_thread
            else:
                thread = None
        if thread is not None:
            thread.start()
        return action

    def _poll_control_file(self):
        with self.lock:
            if self.episode_active:
                # A mailbox update cannot change the policy attribution of an
                # existing episode. Leave it unconsumed until the next entry.
                return
        filename = self.owner.cfg.hotswitch_control_file
        if not filename or not Path(filename).is_file():
            return
        request = json.loads(Path(filename).read_text(encoding="utf-8"))
        if set(request) != {"task_id", "operation_id"} or not all(
            isinstance(request[key], str) and request[key] for key in request
        ):
            self._fail("Invalid local switch request; task_id and operation_id are required.")
        if request["operation_id"] == self.last_control_operation:
            return
        self.request_switch(request["task_id"], request["operation_id"])
        self.last_control_operation = request["operation_id"]

    def _validate_limits(self, action, observation):
        reference = self.last_accepted if self.last_accepted is not None else observation
        for name, value in action.items():
            if not math.isfinite(value):
                self._fail("Non-finite action rejected.")
            if self.owner.cfg.hotswitch_simulation:
                continue
            if name not in reference or not math.isfinite(float(reference[name])):
                self._fail(f"Missing finite safety reference for {name}.")
            difference = abs(value - float(reference[name]))
            if difference > self.owner.cfg.hotswitch_max_jump[name]:
                self._fail(f"Action jump limit exceeded for {name}.")
            if difference / self.owner.environment_dt > self.owner.cfg.hotswitch_max_velocity[name]:
                self._fail(f"Action velocity limit exceeded for {name}.")

    def validate_sent_action(self, command):
        self._claim_control()
        if self.pending_action is None:
            self._fail("No pending policy command to validate.")
        expected = self.owner._tensor_to_action(self.pending_action.processed)
        if set(command) != set(expected) or any(
            not math.isfinite(float(command[k])) or float(command[k]) != expected[k] for k in expected
        ):
            self._fail(
                "Control processor or hardware changed a command; normalized action history is invalid."
            )

    def acknowledge(self, accepted):
        self.validate_sent_action(accepted)
        with self.lock:
            self.accepted_steps += 1
            self.counters["accepted"] += 1
            self.last_accepted = dict(accepted)
            self.pending_action = None
            if self.hold_started is not None:
                self.counters["safety_hold_seconds"] += time.perf_counter() - self.hold_started
                self.hold_started = None
