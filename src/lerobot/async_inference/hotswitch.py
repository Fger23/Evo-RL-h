"""Versioned, single-client hot switching without model loads on the switch path.

This module deliberately imports no robot or model packages. The lock covers the
whole backend call, including preprocessing, denoising and postprocessing. A
cancelled RPC therefore cannot cause an adapter to change under CUDA work.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

PROTOCOL = "lerobot.hotswitch"
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class PolicyIdentity:
    session_id: str
    task_id: str
    policy_epoch: int
    request_id: str


@dataclass(frozen=True)
class InferenceRequest:
    identity: PolicyIdentity
    observation: dict[str, Any]
    timestep: int = 0
    prefix_steps: int = 0
    action_prefix: Any = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class InferenceResponse:
    identity: PolicyIdentity
    raw_actions: Any
    processed_actions: Any
    timestep: int
    prefix_steps: int
    error: str | None = None
    inference_seconds: float = 0.0
    lock_wait_seconds: float = 0.0


class HotSwitchError(ValueError):
    """A request was rejected without performing inference or committing a task."""


class HotSwitchRuntime:
    """Own the active bundle, session, and epoch for one robot connection.

    Backend construction/preloading/warmup happens before this object is made.
    Runtime operations only call activate/reset/predict and never the loader.
    Operations are serialized; old requests awaiting the lock are rejected by
    identity after a switch. Every error response retains the request identity.
    """

    def __init__(self, backend, initial_task: str | None = None):
        self.backend = backend
        self._lock = threading.RLock()
        self._session_id = str(uuid.uuid4())
        self._epoch = 0
        self._task_id: str | None = None
        self._healthy = True
        self._operations: dict[str, tuple[tuple, dict]] = {}
        self._seen_requests: set[str] = set()
        self._last_timestep = -1
        self._bootstrap = True
        self._inference_count = 0
        self._rejected_count = 0
        self._switch_count = 0
        self._last_operation = None
        self._metadata = {task: copy.deepcopy(backend.metadata(task)) for task in backend.task_ids}
        # Parameter traversal belongs to startup, never to the measured switch
        # path. Loader counters remain visible in every health/status reply.
        storage_audit = getattr(backend, "storage_evidence", None)
        self._storage_evidence = storage_audit() if callable(storage_audit) else None
        if not self._metadata:
            raise HotSwitchError("The startup task registry is empty")
        if initial_task is not None:
            self._commit_task(initial_task)

    def _validate_task(self, task_id: str):
        if task_id not in self._metadata:
            raise HotSwitchError(f"Task is not registered: {task_id}")
        if not self.backend.ready(task_id):
            raise HotSwitchError(f"Task has not completed warmup: {task_id}")

    def _commit_task(self, task_id: str):
        self._validate_task(task_id)
        previous = self._task_id
        try:
            self.backend.activate(task_id)
            self.backend.reset(task_id)
        except Exception as error:
            if previous is not None:
                try:
                    self.backend.activate(previous)
                    self.backend.reset(previous)
                    # Resetting shared model state invalidates previous in-flight
                    # prefixes even when the old adapter has been restored.
                    self._epoch += 1
                    self._seen_requests.clear()
                    self._last_timestep = -1
                    self._bootstrap = True
                except Exception:
                    self._healthy = False
            else:
                self._healthy = False
            raise HotSwitchError(f"Task commit failed; healthy={self._healthy}: {error}") from error
        self._task_id = task_id
        self._epoch += 1
        self._seen_requests.clear()
        self._last_timestep = -1
        self._bootstrap = True
        self._switch_count += 1

    def _status_unlocked(self):
        return {
            "session_id": self._session_id,
            "task_id": self._task_id,
            "policy_epoch": self._epoch,
            "healthy": self._healthy,
            "ready_tasks": [task for task in self._metadata if self.backend.ready(task)],
            "metadata": copy.deepcopy(self._metadata.get(self._task_id, {})),
            "tasks": copy.deepcopy(self._metadata),
            "inference_count": self._inference_count,
            "rejected_count": self._rejected_count,
            "switch_count": self._switch_count,
            "last_operation": copy.deepcopy(self._last_operation),
            "backend": {
                "mode": getattr(self.backend, "mode", None),
                "device": str(getattr(self.backend, "device", "unknown")),
                "load_counts": copy.deepcopy(getattr(self.backend, "load_counts", None)),
                "startup_storage": copy.deepcopy(self._storage_evidence),
                "load_seconds": getattr(self.backend, "load_seconds", None),
                "warmup_seconds": getattr(self.backend, "warmup_seconds", None),
            },
        }

    def status(self):
        with self._lock:
            return self._status_unlocked()

    def _operation(self, operation_id: str, signature: tuple, execute, *, lock_wait_seconds=0.0):
        if not isinstance(operation_id, str) or not operation_id:
            raise HotSwitchError("A nonempty operation_id is required for retry safety")
        existing = self._operations.get(operation_id)
        if existing:
            if existing[0] != signature:
                raise HotSwitchError("operation_id was already used with different arguments")
            # Return the exact result, even if a later operation has changed state.
            # Callers must query status after an ambiguous timeout before resuming.
            return copy.deepcopy(existing[1])
        started = time.perf_counter()
        execute()
        self._last_operation = {
            "operation_id": operation_id,
            "op": signature[0],
            "lock_wait_seconds": lock_wait_seconds,
            "commit_seconds": time.perf_counter() - started,
        }
        result = self._status_unlocked()
        self._operations[operation_id] = (signature, copy.deepcopy(result))
        return result

    def connect(self, operation_id: str, task_id: str | None = None):
        queued = time.perf_counter()
        with self._lock:
            lock_wait_seconds = time.perf_counter() - queued
            def execute():
                if not self._healthy:
                    raise HotSwitchError("Runtime is stopped after an unrecoverable failure")
                target = task_id or self._task_id
                if target is None:
                    raise HotSwitchError("Initial connect must select a registered task_id")
                self._commit_task(target)
                self._session_id = str(uuid.uuid4())

            return self._operation(
                operation_id, ("connect", task_id), execute, lock_wait_seconds=lock_wait_seconds
            )

    def switch(self, task_id: str, operation_id: str, session_id: str):
        queued = time.perf_counter()
        with self._lock:
            lock_wait_seconds = time.perf_counter() - queued
            def execute():
                if session_id != self._session_id:
                    raise HotSwitchError("Stale session_id")
                if not self._healthy:
                    raise HotSwitchError("Runtime is stopped after an unrecoverable failure")
                self._commit_task(task_id)

            return self._operation(
                operation_id, ("switch", session_id, task_id), execute,
                lock_wait_seconds=lock_wait_seconds,
            )

    def _validate_identity(self, identity: PolicyIdentity):
        if not isinstance(identity, PolicyIdentity):
            raise HotSwitchError("Inference requires PolicyIdentity")
        if not self._healthy:
            raise HotSwitchError("Runtime is stopped after an unrecoverable failure")
        if (identity.session_id, identity.task_id, identity.policy_epoch) != (
            self._session_id, self._task_id, self._epoch
        ):
            raise HotSwitchError("Stale session/task/epoch")
        if not isinstance(identity.request_id, str) or not identity.request_id:
            raise HotSwitchError("request_id must be a nonempty string")
        if identity.request_id in self._seen_requests:
            raise HotSwitchError("Duplicate request_id")

    def _validate_prefix(self, request: InferenceRequest):
        if isinstance(request.prefix_steps, bool) or not isinstance(request.prefix_steps, int):
            raise HotSwitchError("prefix_steps must be an integer")
        if request.prefix_steps < 0:
            raise HotSwitchError("prefix_steps must be nonnegative")
        if self._bootstrap and (request.prefix_steps or request.action_prefix is not None):
            raise HotSwitchError("New task/session must bootstrap without a previous prefix")
        if request.prefix_steps == 0 and request.action_prefix is not None:
            raise HotSwitchError("A zero-length prefix must be None")
        if request.prefix_steps and request.action_prefix is None:
            raise HotSwitchError("A nonzero prefix requires raw action_prefix")
        metadata = self._metadata[request.identity.task_id]
        rtc = metadata.get("rtc", {})
        mode = metadata.get("rtc_mode", rtc.get("mode", "async"))
        if request.prefix_steps and mode not in ("training_time", "training_time_rtc", "trc"):
            raise HotSwitchError("This task is not configured for Training-Time RTC")
        maximum = metadata.get(
            "rtc_training_max_delay",
            rtc.get("training_max_delay", metadata.get("policy_config", {}).get("rtc_training_max_delay", 0)),
        )
        if request.prefix_steps and request.prefix_steps > maximum:
            raise HotSwitchError("prefix_steps exceeds the checkpoint training delay range")

    def infer(self, request: InferenceRequest):
        if not isinstance(request, InferenceRequest):
            raise HotSwitchError("Expected version 1 InferenceRequest")
        queued = time.perf_counter()
        with self._lock:
            lock_wait_seconds = time.perf_counter() - queued
            started = None
            try:
                self._validate_identity(request.identity)
                if request.protocol_version != PROTOCOL_VERSION:
                    raise HotSwitchError("Unsupported inference protocol version")
                self._validate_prefix(request)
                if not isinstance(request.timestep, int) or request.timestep < self._last_timestep:
                    raise HotSwitchError("Observation timestep must not go backwards in an epoch")
                self._seen_requests.add(request.identity.request_id)
                self._last_timestep = request.timestep
                started = time.perf_counter()
                raw, processed = self.backend.predict(
                    request.identity.task_id, request.observation,
                    prefix_steps=request.prefix_steps, action_prefix=request.action_prefix,
                )
                # Moving responses to CPU completes their CUDA dependencies before
                # releasing the adapter lock, and avoids device RPC pickles.
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu()
                if hasattr(processed, "detach"):
                    processed = processed.detach().cpu()
                self._bootstrap = False
                self._inference_count += 1
                return InferenceResponse(
                    request.identity, raw, processed, request.timestep,
                    request.prefix_steps, inference_seconds=time.perf_counter() - started,
                    lock_wait_seconds=lock_wait_seconds,
                )
            except Exception as error:
                self._rejected_count += 1
                return InferenceResponse(
                    request.identity, None, None, request.timestep, request.prefix_steps,
                    error=f"{type(error).__name__}: {error}",
                    inference_seconds=0.0 if started is None else time.perf_counter() - started,
                    lock_wait_seconds=lock_wait_seconds,
                )

    def dispatch_control(self, request: dict):
        try:
            if request.get("protocol") != PROTOCOL or request.get("version") != PROTOCOL_VERSION:
                raise HotSwitchError("Unsupported hot-switch protocol or version")
            operation = request.get("op")
            if operation == "connect":
                state = self.connect(request.get("operation_id"), request.get("task_id"))
            elif operation == "switch":
                state = self.switch(
                    request.get("task_id"), request.get("operation_id"), request.get("session_id")
                )
            elif operation in ("status", "capabilities"):
                state = self.status()
            else:
                raise HotSwitchError(f"Unsupported control operation: {operation}")
            capabilities = ["hotswitch_v1", "identity", "raw_processed_actions"]
            if any(
                meta.get("rtc_mode", meta.get("rtc", {}).get("mode"))
                in ("training_time", "training_time_rtc", "trc")
                for meta in self._metadata.values()
            ):
                capabilities.append("training_time_rtc")
            return {
                "protocol": PROTOCOL, "version": PROTOCOL_VERSION, "ok": True, "state": state,
                "capabilities": capabilities,
            }
        except Exception as error:
            return {
                "protocol": PROTOCOL, "version": PROTOCOL_VERSION, "ok": False,
                "state": self.status(), "error": f"{type(error).__name__}: {error}",
            }
