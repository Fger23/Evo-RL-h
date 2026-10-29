"""CPU-only, no hardware tests; runnable directly with unittest."""

import threading
import time
import unittest

from lerobot.async_inference.hotswitch import (
    PROTOCOL,
    HotSwitchError,
    HotSwitchRuntime,
    InferenceRequest,
    PolicyIdentity,
)


class FakeBackend:
    task_ids = ("A", "B", "cold")

    def __init__(self):
        self.active = None
        self.fail_activate = None
        self.fail_reset = None
        self.fail_infer = False
        self.entered = None
        self.release = None
        self.calls = []

    def metadata(self, task):
        return {
            "task_text": f"task {task}", "normalization": task,
            "action_names": ["joint"], "rtc_mode": "training_time",
            "rtc_training_max_delay": 2,
        }

    def ready(self, task):
        return task != "cold"

    def activate(self, task):
        self.calls.append(("activate", task))
        if self.fail_activate == task:
            raise RuntimeError("injected activation failure")
        self.active = task

    def reset(self, task):
        self.calls.append(("reset", task))
        if self.fail_reset == task:
            raise RuntimeError("injected reset failure")

    def predict(self, task, observation, prefix_steps=0, action_prefix=None):
        before = self.active
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(timeout=3):
                raise RuntimeError("test synchronization timeout")
        if self.fail_infer:
            raise RuntimeError("injected inference failure")
        self.calls.append(("predict", task, before, self.active))
        if task != self.active or before != self.active:
            raise RuntimeError("adapter changed during forward/postprocess")
        raw = [[1 if task == "A" else 2]] * 4
        processed = [[10 if task == "A" else 20]] * 4
        return raw, processed


class HotSwitchTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.runtime = HotSwitchRuntime(self.backend)
        self.state = self.runtime.connect("initial-connect", "A")

    def request(self, request_id="request", state=None, **kwargs):
        state = state or self.runtime.status()
        return InferenceRequest(
            PolicyIdentity(state["session_id"], state["task_id"], state["policy_epoch"], request_id),
            {"offline": True}, **kwargs,
        )

    def switch(self, task, operation="switch"):
        return self.runtime.switch(task, operation, self.state["session_id"])

    def test_a_b_a_output_and_metadata(self):
        first = self.runtime.infer(self.request("a1"))
        b = self.switch("B", "b")
        self.assertEqual(b["metadata"]["normalization"], "B")
        second = self.runtime.infer(self.request("b1"))
        self.switch("A", "a")
        third = self.runtime.infer(self.request("a2"))
        self.assertIsNone(third.error)
        self.assertEqual(first.raw_actions, third.raw_actions)
        self.assertNotEqual(first.processed_actions, second.processed_actions)

    def test_retry_is_idempotent_and_collision_rejected(self):
        first = self.switch("B", "b")
        calls = list(self.backend.calls)
        second = self.switch("B", "b")
        self.assertEqual(first, second)
        self.assertEqual(calls, self.backend.calls)
        with self.assertRaisesRegex(HotSwitchError, "different arguments"):
            self.switch("A", "b")

    def test_connect_retry_preserves_session(self):
        state = self.runtime.connect("initial-connect", "A")
        self.assertEqual(state["session_id"], self.state["session_id"])

    def test_same_task_new_operation_creates_epoch(self):
        state = self.switch("A")
        self.assertGreater(state["policy_epoch"], self.state["policy_epoch"])

    def test_invalid_unready_tasks_leave_state_unchanged(self):
        for task in ("unknown", "cold"):
            with self.assertRaises(HotSwitchError):
                self.switch(task, task)
            self.assertEqual(self.runtime.status(), self.state)

    def test_late_old_response_identity_and_error_isolation(self):
        old = self.request("old")
        self.switch("B")
        result = self.runtime.infer(old)
        self.assertEqual(result.identity, old.identity)
        self.assertIn("Stale", result.error)
        self.assertIsNone(result.raw_actions)
        self.assertEqual(self.runtime.status()["task_id"], "B")

    def test_reconnect_invalidates_old_session(self):
        old = self.request()
        new = self.runtime.connect("reconnect", "A")
        self.assertNotEqual(new["session_id"], self.state["session_id"])
        self.assertIn("Stale", self.runtime.infer(old).error)

    def test_duplicate_and_decreasing_timestep(self):
        request = self.request(timestep=5)
        self.assertIsNone(self.runtime.infer(request).error)
        self.assertIn("Duplicate", self.runtime.infer(request).error)
        self.assertIn("backwards", self.runtime.infer(self.request("next", timestep=4)).error)

    def test_bootstrap_and_training_range(self):
        self.assertIn("bootstrap", self.runtime.infer(self.request(prefix_steps=1, action_prefix=[[1]])).error)
        self.assertIsNone(self.runtime.infer(self.request("bootstrap")).error)
        self.assertIsNone(self.runtime.infer(self.request("refill", prefix_steps=2, action_prefix=[[1], [1]])).error)
        self.assertIn("training delay", self.runtime.infer(self.request("range", prefix_steps=3, action_prefix=[[1]] * 3)).error)
        self.switch("B")
        self.assertIn("bootstrap", self.runtime.infer(self.request("new", prefix_steps=1, action_prefix=[[1]])).error)

    def test_commit_failure_rolls_back_with_fresh_epoch(self):
        self.backend.fail_reset = "B"
        with self.assertRaisesRegex(HotSwitchError, "commit failed"):
            self.switch("B")
        state = self.runtime.status()
        self.assertTrue(state["healthy"])
        self.assertEqual(state["task_id"], "A")
        self.assertEqual(self.backend.active, "A")
        self.assertGreater(state["policy_epoch"], self.state["policy_epoch"])
        self.assertIn("Stale", self.runtime.infer(self.request(state=self.state)).error)

    def test_rollback_failure_stops_runtime(self):
        self.backend.fail_activate = "B"
        self.backend.fail_reset = "A"
        with self.assertRaises(HotSwitchError):
            self.switch("B")
        self.assertFalse(self.runtime.status()["healthy"])
        self.assertIn("stopped", self.runtime.infer(self.request()).error)

    def test_complete_inference_lifecycle_serializes_switch(self):
        self.backend.entered = threading.Event()
        self.backend.release = threading.Event()
        responses = []
        inferred = threading.Thread(target=lambda: responses.append(self.runtime.infer(self.request("old"))))
        inferred.start()
        self.assertTrue(self.backend.entered.wait(timeout=2))
        switched = threading.Event()
        switcher = threading.Thread(target=lambda: (self.switch("B"), switched.set()))
        switcher.start()
        time.sleep(0.02)
        self.assertFalse(switched.is_set())
        self.backend.release.set()
        inferred.join(timeout=2)
        switcher.join(timeout=2)
        self.assertFalse(inferred.is_alive())
        self.assertFalse(switcher.is_alive())
        self.assertIsNone(responses[0].error)
        self.assertEqual(responses[0].identity.task_id, "A")
        self.assertEqual(self.runtime.status()["task_id"], "B")

    def test_concurrent_switches_get_distinct_epochs(self):
        epochs = []
        def change(index):
            epochs.append(self.switch("B" if index % 2 else "A", f"op-{index}")["policy_epoch"])
        threads = [threading.Thread(target=change, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(len(epochs), 20)
        self.assertEqual(len(set(epochs)), 20)

    def test_inference_exception_returns_original_identity(self):
        self.backend.fail_infer = True
        request = self.request()
        response = self.runtime.infer(request)
        self.assertEqual(response.identity, request.identity)
        self.assertIn("inference failure", response.error)

    def test_capability_negotiation_and_unsupported_version(self):
        reply = self.runtime.dispatch_control({"protocol": PROTOCOL, "version": 1, "op": "capabilities"})
        self.assertTrue(reply["ok"])
        self.assertIn("training_time_rtc", reply["capabilities"])
        rejected = self.runtime.dispatch_control({"protocol": PROTOCOL, "version": 2, "op": "status"})
        self.assertFalse(rejected["ok"])
        request = self.request(protocol_version=2)
        self.assertIn("version", self.runtime.infer(request).error)


if __name__ == "__main__":
    unittest.main()
