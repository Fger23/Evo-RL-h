"""Single robot, resident-policy switching with explicit session and execution identity.

No hardware imports occur here. ``safe_hold`` and action acceptance belong to the
control thread; a transport timeout never means the server rolled back a switch.
"""

import math
import threading
import time
import uuid
from collections import Counter, deque

PROTOCOL_VERSION = 1


class HotSwitchError(RuntimeError):
    pass


class StaleIdentityError(HotSwitchError):
    pass


StaleIdentity = StaleIdentityError


def _matrix(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise HotSwitchError(f"{name} must be a nonempty [H,A] matrix")
    result = []
    width = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise HotSwitchError(f"{name} must have two dimensions")
        width = width or len(row)
        if len(row) != width or any(not math.isfinite(float(x)) for x in row):
            raise HotSwitchError(f"{name} has invalid shape or NaN/Inf")
        result.append([float(x) for x in row])
    return result


def _identity(value):
    return tuple(value.get(k) for k in ("session_id", "task_id", "policy_epoch"))


class HotSwitchRuntime:
    """Serializes activation, complete inference and task-specific postprocessing."""

    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.RLock()
        self.ready = set()
        self.session_id = None
        self.task_id = None
        self.policy_epoch = 0
        self.stopped = False
        self.switches = {}
        self.requests = set()
        self.pending_switch = None
        self.metrics = Counter()

    def preload(self):
        with self.lock:
            for task_id in self.backend.task_ids:
                self.backend.warmup(task_id)
                self.backend.reset(task_id)
                self.ready.add(task_id)
            return self.status()

    def status(self):
        with self.lock:
            return {
                "version": PROTOCOL_VERSION,
                "capabilities": ["resident_switch", "identity_v1", "raw_processed_v1"],
                "session_id": self.session_id,
                "task_id": self.task_id,
                "policy_epoch": self.policy_epoch,
                "ready": sorted(self.ready),
                "stopped": self.stopped,
                "task_metadata": getattr(self.backend, "task_metadata", {}),
                "metrics": dict(self.metrics),
                "pending_switch": self.pending_switch,
            }

    def open_session(self, previous_session_id=None):
        with self.lock:
            if self.session_id is not None and previous_session_id != self.session_id:
                raise HotSwitchError("Active session exists; reconnect requires its current session_id")
            if self.task_id is not None:
                self.backend.reset(self.task_id)
            self.session_id = uuid.uuid4().hex
            self.policy_epoch += 1
            self.task_id = None
            self.stopped = False
            self.switches.clear()
            self.requests.clear()
            self.pending_switch = None
            return self.status()

    def request_switch(self, task_id, switch_id, session_id):
        """Queue a control-thread intent. Only that thread commits the switch."""
        with self.lock:
            if session_id != self.session_id or not session_id:
                raise StaleIdentity("Wrong switch session")
            if task_id not in self.ready or not switch_id:
                raise HotSwitchError("Task not ready or switch_id missing")
            if switch_id in self.switches:
                previous_task, result = self.switches[switch_id]
                if previous_task != task_id:
                    raise HotSwitchError("switch_id reused for a different task")
                if _identity(result) != (self.session_id, self.task_id, self.policy_epoch):
                    raise HotSwitchError("Switch intent was superseded")
                return self.status()
            if self.pending_switch and self.pending_switch["switch_id"] != switch_id:
                raise HotSwitchError("A different switch is already pending")
            if self.pending_switch and self.pending_switch["task_id"] != task_id:
                raise HotSwitchError("switch_id reused for a different pending task")
            self.pending_switch = {"task_id": task_id, "switch_id": switch_id}
            return self.status()

    def switch(self, task_id, switch_id, session_id):
        start = time.perf_counter()
        if not switch_id:
            raise HotSwitchError("switch_id is required for timeout recovery")
        with self.lock:
            lock_acquired = time.perf_counter()
            lock_wait_s = lock_acquired - start
            if session_id != self.session_id or not session_id:
                raise StaleIdentity("Wrong switch session")
            if switch_id in self.switches:
                old_task, old_result = self.switches[switch_id]
                if old_task != task_id:
                    raise HotSwitchError("switch_id reused for a different task")
                if _identity(old_result) != (self.session_id, self.task_id, self.policy_epoch):
                    raise HotSwitchError(
                        "Idempotent switch was superseded; query current status and remain held"
                    )
                return {
                    **old_result,
                    "lock_wait_s": lock_wait_s,
                    "activation_s": 0.0,
                    "reset_s": 0.0,
                    "activation_reset_s": 0.0,
                    "idempotent": True,
                    "switch_ack_s": time.perf_counter() - start,
                }
            if self.stopped:
                raise HotSwitchError("Runtime is stopped")
            if task_id not in self.ready:
                raise HotSwitchError(f"Task is unknown or not warmed: {task_id}")
            old_task = self.task_id
            activation_s = reset_s = 0.0
            if task_id != old_task:
                try:
                    activation_start = time.perf_counter()
                    self.backend.activate(task_id)
                    activation_end = time.perf_counter()
                    activation_s = activation_end - activation_start
                    self.backend.reset(task_id)
                    reset_s = time.perf_counter() - activation_end
                except Exception:
                    if old_task is not None:
                        try:
                            self.backend.activate(old_task)
                            self.backend.reset(old_task)
                        except Exception:
                            self.stopped = True
                    else:
                        self.stopped = True
                    self.metrics["switch_errors"] += 1
                    raise
                self.task_id = task_id
                self.policy_epoch += 1
                self.metrics["switches"] += 1
            result = self.status()
            self.pending_switch = None
            result["pending_switch"] = None
            result["switch_id"] = switch_id
            result.update(
                lock_wait_s=lock_wait_s,
                activation_s=activation_s,
                reset_s=reset_s,
                activation_reset_s=activation_s + reset_s,
                idempotent=False,
            )
            result["switch_ack_s"] = time.perf_counter() - start
            self.switches[switch_id] = (task_id, dict(result))
            return result

    def infer(self, request):
        start = time.perf_counter()
        with self.lock:
            lock_acquired = time.perf_counter()
            lock_wait_s = lock_acquired - start
            if request.get("version") != PROTOCOL_VERSION:
                raise HotSwitchError("Unsupported protocol version")
            if _identity(request) != (self.session_id, self.task_id, self.policy_epoch):
                self.metrics["stale_requests"] += 1
                raise StaleIdentity("Inference identity is no longer active")
            if self.stopped or self.task_id is None:
                raise HotSwitchError("No active task")
            request_id = request.get("request_id")
            if not request_id or request_id in self.requests:
                raise HotSwitchError("request_id missing or reused")
            self.requests.add(request_id)
            meta = getattr(self.backend, "task_metadata", {}).get(self.task_id, {})
            steps = request.get("prefix_steps", 0)
            if not isinstance(steps, int) or steps < 0:
                raise HotSwitchError("Invalid prefix_steps")
            mode = meta.get("rtc_mode", "async")
            if mode == "legacy":
                raise HotSwitchError("Legacy inference RTC requires its separate verified client")
            if steps and (mode != "training_time" or steps > meta.get("max_prefix_steps", 0)):
                raise HotSwitchError("Prefix exceeds checkpoint training range")
            prefix = request.get("action_prefix")
            if steps:
                prefix = _matrix(prefix, "action_prefix")
                if len(prefix) != steps:
                    raise HotSwitchError("prefix_steps does not match prefix length")
            elif prefix is not None:
                raise HotSwitchError("Bootstrap must have action_prefix=None")
            backend_start = time.perf_counter()
            raw, processed = self.backend.infer(
                self.task_id, request["observation"], action_prefix=prefix, prefix_steps=steps
            )
            backend_inference_s = time.perf_counter() - backend_start
            raw, processed = _matrix(raw, "raw_actions"), _matrix(processed, "processed_actions")
            if len(raw) != len(processed) or len(raw[0]) != len(processed[0]):
                raise HotSwitchError("Raw and processed action matrices are not aligned")
            if meta.get("action_dim") and len(raw[0]) != meta["action_dim"]:
                raise HotSwitchError("Action dimension differs from task schema")
            if meta.get("chunk_size") and len(raw) != meta["chunk_size"]:
                raise HotSwitchError("Action horizon differs from task schema")
            if steps >= len(raw):
                raise HotSwitchError("Prefix leaves no new predicted action")
            if steps and any(
                abs(a - b) > 1e-5
                for x, y in zip(prefix, raw[:steps], strict=True)
                for a, b in zip(x, y, strict=True)
            ):
                raise HotSwitchError("Training-time RTC prefix is not hard clamped")
            self.metrics["inferences"] += 1
            return {
                **{k: request[k] for k in ("version", "session_id", "task_id", "policy_epoch", "request_id")},
                "observation_step": request["observation_step"],
                "prefix_steps": steps,
                "raw_actions": raw,
                "processed_actions": processed,
                "server_inference_s": time.perf_counter() - start,
                "lock_wait_s": lock_wait_s,
                "backend_inference_s": backend_inference_s,
            }


class HotSwitchClientCore:
    """Action pairs stay pending until the control interface confirms exact acceptance.

    ``safe_hold`` must implement and validate the robot's holding behavior. An
    offline simulator may explicitly provide a no-op; hardware callers may not.
    """

    def __init__(self, transport, safe_hold, validate_action=None, *, minimum_action_interval_s=0.0):
        if not callable(safe_hold):
            raise ValueError("An explicit validated safe_hold callback is required")
        try:
            interval = float(minimum_action_interval_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("minimum_action_interval_s must be finite and nonnegative") from exc
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("minimum_action_interval_s must be finite and nonnegative")
        self.transport = transport
        self.safe_hold = safe_hold
        self.validate_action = validate_action
        self.lock = threading.RLock()
        self.control_lock = threading.RLock()
        self._action_clock_wakeup = threading.Event()
        self.minimum_action_interval_s = interval
        # This is the controller clock, independent of task/model state. A task
        # switch or session reset must never reset the last physical acceptance.
        self.last_acceptance_time = None
        self.last_acceptance_interval_s = None
        self.last_action_wait_s = 0.0
        self.status = None
        self.queue = deque()
        self.queue_identity = None
        self.pending = None
        self.pending_identity = None
        self.last_accepted_identity = None
        self.last_inference_alignment = None
        self.accepted_steps = 0
        self.stopped = False
        self.metrics = Counter()
        self._inflight = set()
        self._generation = 0
        self.last_timings = {"switch": {}, "inference": {}}

    def _clear_queue(self):
        """Caller holds self.lock. Preserve a physically pending command's identity."""
        self.queue.clear()
        self.queue_identity = None
        self.last_inference_alignment = None
        self._action_clock_wakeup.set()

    def stop(self):
        """Request a control-thread stop without blocking on its pacing lock.

        An already pending command remains pending until its acceptance is
        known. The waiting control thread performs the hold before returning.
        """
        with self.lock:
            self.stopped = True
            self._clear_queue()

    def connect(self, task_id, previous_session_id=None):
        self.status = self.transport.open_session(previous_session_id)
        return self.switch(task_id)

    def switch(self, task_id, switch_id=None):
        start = time.perf_counter()
        with self.control_lock:
            control_acquired = time.perf_counter()
            self.safe_hold()
            held = time.perf_counter()
            queue_clear_start = time.perf_counter()
            with self.lock:
                if self.pending is not None:
                    raise HotSwitchError("Control interface has not confirmed its pending command")
                self.stopped = True
                self._generation += 1
                self._clear_queue()
                session = self.status["session_id"]
            queue_clear_s = time.perf_counter() - queue_clear_start
            switch_id = switch_id or uuid.uuid4().hex
            transport_start = time.perf_counter()
            try:
                status = self.transport.switch(task_id, switch_id, session)
            except (TimeoutError, OSError) as exc:
                # An acknowledged transport failure can occur after commit. Retry the
                # same idempotency key; if confirmation also fails remain in hold.
                try:
                    status = self.transport.switch(task_id, switch_id, session)
                except Exception:
                    raise HotSwitchError("Switch outcome unknown; remain in safe hold") from exc
            transport_end = time.perf_counter()
            queue_clear_start = time.perf_counter()
            with self.lock:
                self.status = status
                self.accepted_steps = 0
                self.stopped = False
                self._clear_queue()
                self.pending = None
                self.pending_identity = None
                self.last_accepted_identity = None
                # The runtime commit waited for old model work. Delayed network
                # delivery may remain, but is isolated by local generation.
                self._inflight.clear()
                queue_clear_s += time.perf_counter() - queue_clear_start
                self.last_timings["switch"] = {
                    "control_lock_wait_s": control_acquired - start,
                    "safe_hold_s": held - control_acquired,
                    "queue_clear_s": queue_clear_s,
                    "transport_switch_s": transport_end - transport_start,
                    "control_boundary_s": time.perf_counter() - start,
                    "server_lock_wait_s": status.get("lock_wait_s"),
                    "activation_s": status.get("activation_s"),
                    "reset_s": status.get("reset_s"),
                    "activation_reset_s": status.get("activation_reset_s"),
                }
            return status

    def request_actions(self, observation):
        start = time.perf_counter()
        with self.lock:
            if self.stopped or self.status is None:
                raise HotSwitchError("Client is not active")
            if self._inflight:
                raise HotSwitchError("Only one inference may be in flight")
            if self.pending is not None:
                raise HotSwitchError("Submit observations at an accepted control boundary")
            meta = self.status.get("task_metadata", {}).get(self.status["task_id"], {})
            prefix_steps = min(meta.get("prefix_steps", 0), len(self.queue))
            request = {
                **{k: self.status[k] for k in ("version", "session_id", "task_id", "policy_epoch")},
                "request_id": uuid.uuid4().hex,
                "observation": observation,
                "observation_step": self.accepted_steps,
                "prefix_steps": prefix_steps,
                "action_prefix": [entry[0] for entry in list(self.queue)[:prefix_steps]]
                if prefix_steps
                else None,
            }
            processed_prefix = [entry[1] for entry in list(self.queue)[:prefix_steps]]
            self._inflight.add(request["request_id"])
            generation = self._generation
        transport_start = time.perf_counter()
        try:
            result = self.transport.infer(request)
            transport_end = time.perf_counter()
        except Exception:
            with self.lock:
                if generation != self._generation or _identity(request) != _identity(self.status):
                    self.metrics["stale_errors"] += 1
                    return False
                self.stopped = True
                self._clear_queue()
            self.safe_hold()
            raise
        finally:
            with self.lock:
                self._inflight.discard(request["request_id"])
        with self.lock:
            if self.stopped or generation != self._generation or _identity(result) != _identity(self.status):
                self.metrics["stale_responses"] += 1
                return False
            if result.get("request_id") != request["request_id"]:
                self.stopped = True
                raise HotSwitchError("Response request_id mismatch")
            raw, processed = (
                _matrix(result["raw_actions"], "raw_actions"),
                _matrix(result["processed_actions"], "processed_actions"),
            )
            if len(raw) != len(processed) or len(raw[0]) != len(processed[0]):
                self.stopped = True
                raise HotSwitchError("Raw/processed alignment lost")
            if (
                result.get("observation_step") != request["observation_step"]
                or result.get("prefix_steps") != prefix_steps
            ):
                self.stopped = True
                raise HotSwitchError("Response control alignment metadata differs from request")
            if prefix_steps and any(
                len(a) != len(b) or any(abs(x - y) > 1e-5 for x, y in zip(a, b, strict=True))
                for a, b in zip(raw[:prefix_steps], request["action_prefix"], strict=True)
            ):
                self.stopped = True
                raise HotSwitchError("Response changed the known normalized prefix")
            if prefix_steps and any(
                len(a) != len(b) or any(abs(x - y) > 1e-5 for x, y in zip(a, b, strict=True))
                for a, b in zip(processed[:prefix_steps], processed_prefix, strict=True)
            ):
                self.stopped = True
                raise HotSwitchError("Response postprocessing changed the known execution prefix")
            consumed = self.accepted_steps - request["observation_step"]
            if consumed < 0 or consumed > prefix_steps or self.pending is not None:
                self.metrics["prefix_overruns"] += 1
                self.stopped = True
                self._clear_queue()
                self.safe_hold()
                raise HotSwitchError("Accepted commands exceeded RTC prefix coverage")
            self.queue = deque(zip(raw[consumed:], processed[consumed:], strict=True))
            self.queue_identity = {
                **{
                    key: result[key]
                    for key in ("session_id", "task_id", "policy_epoch", "request_id", "observation_step")
                },
                "prefix_steps": prefix_steps,
                "next_action_step": self.accepted_steps,
                "next_chunk_index": consumed,
            }
            self.last_inference_alignment = {
                **{
                    key: result[key]
                    for key in ("version", "session_id", "task_id", "policy_epoch", "request_id")
                },
                "observation_step": request["observation_step"],
                "prefix_steps": prefix_steps,
                "input_raw_prefix": [list(row) for row in request["action_prefix"]] if prefix_steps else None,
                "input_processed_prefix": [list(row) for row in processed_prefix] if prefix_steps else None,
                "returned_raw_prefix": [list(row) for row in raw[:prefix_steps]],
                "returned_processed_prefix": [list(row) for row in processed[:prefix_steps]],
                "accepted_steps_on_response": self.accepted_steps,
                "consumed_prefix_steps": consumed,
            }
            self._action_clock_wakeup.set()
            end = time.perf_counter()
            self.last_timings["inference"] = {
                "transport_roundtrip_s": transport_end - transport_start,
                "client_queue_s": end - transport_end,
                "request_total_s": end - start,
                "server_lock_wait_s": result.get("lock_wait_s"),
                "backend_inference_s": result.get("backend_inference_s"),
                "server_inference_s": result.get("server_inference_s"),
            }
            return True

    def control_boundary(self):
        """Called by the control thread, before capturing a new observation."""
        try:
            status = self.transport.status()
        except Exception:
            self.safe_hold()
            with self.lock:
                self.stopped = True
            raise
        if _identity(status) != _identity(self.status):
            self.safe_hold()
            with self.lock:
                self._clear_queue()
                self.stopped = True
                self._generation += 1
            raise StaleIdentity("Server activation changed outside this control thread; held safely")
        intent = status.get("pending_switch")
        if intent:
            return self.switch(intent["task_id"], intent["switch_id"])
        return self.status

    def next_action(self):
        with self.control_lock:
            with self.lock:
                self.last_action_wait_s = 0.0
            while True:
                with self.lock:
                    self._action_clock_wakeup.clear()
                    stopped = self.stopped
                    if not stopped:
                        remaining = (
                            0.0
                            if self.last_acceptance_time is None
                            else (
                                self.minimum_action_interval_s
                                - (time.perf_counter() - self.last_acceptance_time)
                            )
                        )
                        if remaining <= 0:
                            return self._next_action_locked()
                if stopped:
                    # Hold belongs to this control thread, not the thread that
                    # requested stop while it was waiting for the next period.
                    self.safe_hold()
                    raise HotSwitchError("Client is stopped in safe hold")
                waiting = time.perf_counter()
                self._action_clock_wakeup.wait(timeout=remaining)
                with self.lock:
                    self.last_action_wait_s += time.perf_counter() - waiting

    def _next_action_locked(self):
        """Called by the control thread only after its minimum period has elapsed."""
        with self.lock:
            if self.stopped:
                raise HotSwitchError("Client is stopped in safe hold")
            if self.pending is not None:
                raise HotSwitchError("Previous command acceptance has not been confirmed")
            if not self.queue:
                self.metrics["queue_underruns"] += 1
                raise HotSwitchError("No validated action is queued")
            raw, processed = self.queue.popleft()
            if self.validate_action is not None:
                try:
                    if self.validate_action(processed) is False:
                        raise HotSwitchError("Configured action validation rejected command")
                except Exception:
                    self.stopped = True
                    self._clear_queue()
                    self.safe_hold()
                    raise
            self.pending = (raw, processed)
            if self.queue_identity is not None:
                self.pending_identity = {
                    **{
                        key: self.queue_identity[key]
                        for key in (
                            "session_id",
                            "task_id",
                            "policy_epoch",
                            "request_id",
                            "observation_step",
                            "prefix_steps",
                        )
                    },
                    "action_step": self.queue_identity["next_action_step"],
                    "chunk_index": self.queue_identity["next_chunk_index"],
                }
                self.queue_identity["next_action_step"] += 1
                self.queue_identity["next_chunk_index"] += 1
            else:
                self.pending_identity = None
            if not self.queue:
                self.queue_identity = None
            return list(processed)

    def confirm_action(self, actual):
        with self.control_lock, self.lock:
            if self.pending is None:
                raise HotSwitchError("No pending action to confirm")
            expected = self.pending[1]
            if (
                actual is None
                or len(actual) != len(expected)
                or any(
                    not math.isfinite(float(a)) or abs(float(a) - b) > 1e-6
                    for a, b in zip(actual, expected, strict=True)
                )
            ):
                self.stopped = True
                self._clear_queue()
                self.safe_hold()
                raise HotSwitchError("Control interface modified or did not confirm the policy command")
            self.pending = None
            self.accepted_steps += 1
            accepted_at = time.perf_counter()
            self.last_acceptance_interval_s = (
                accepted_at - self.last_acceptance_time if self.last_acceptance_time is not None else None
            )
            self.last_acceptance_time = accepted_at
            self.last_accepted_identity = (
                {
                    **self.pending_identity,
                    "accepted_step": self.accepted_steps,
                    "accepted_monotonic_s": accepted_at,
                }
                if self.pending_identity is not None
                else None
            )
            self.pending_identity = None
            self._action_clock_wakeup.set()


class GrpcHotSwitchTransport:
    """Versioned extension to the repository's existing trusted pickle RPC."""

    def __init__(self, address, timeout=10.0):
        import grpc

        from lerobot.transport import services_pb2, services_pb2_grpc

        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ],
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.pb = services_pb2
        self.timeout = timeout

    def _call(self, method, payload=None):
        import pickle

        request = (
            self.pb.Empty()
            if payload is None
            else self.pb.HotSwitchMessage(protocol_version=PROTOCOL_VERSION, data=pickle.dumps(payload))
        )
        try:
            response = method(request, timeout=self.timeout)
        except Exception as exc:
            import grpc

            if isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError("Hot-switch RPC deadline exceeded") from exc
            raise
        if response.protocol_version != PROTOCOL_VERSION:
            raise HotSwitchError("Server does not support hot-switch protocol v1")
        return pickle.loads(response.data)

    def status(self):
        return self._call(self.stub.HotSwitchStatus)

    def open_session(self, previous_session_id=None):
        return self._call(self.stub.HotSwitchSession, {"previous_session_id": previous_session_id})

    def switch(self, task_id, switch_id, session_id):
        return self._call(
            self.stub.HotSwitchTask, {"task_id": task_id, "switch_id": switch_id, "session_id": session_id}
        )

    def request_switch(self, task_id, switch_id, session_id):
        return self._call(
            self.stub.HotSwitchTask,
            {"task_id": task_id, "switch_id": switch_id, "session_id": session_id, "prepare_only": True},
        )

    def infer(self, request):
        return self._call(self.stub.HotSwitchInfer, request)

    def close(self):
        self.channel.close()
