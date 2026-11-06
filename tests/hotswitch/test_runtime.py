"""Offline concurrency/RTC checks; runnable without repository hardware fixtures.

PYTHONPATH=src python -m unittest discover -s tests/hotswitch -p test_runtime.py -v
"""

import threading
import time
import unittest
from unittest.mock import patch

from lerobot.async_inference.hotswitch.runtime import (
    HotSwitchClientCore,
    HotSwitchError,
    HotSwitchRuntime,
    StaleIdentity,
)
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend, SimulatedRobot


class DelayedTransport:
    def __init__(self, runtime, *, fail=False):
        self.runtime = runtime
        self.ready = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def __getattr__(self, key):
        return getattr(self.runtime, key)

    def infer(self, request):
        result = self.runtime.infer(request)
        self.ready.set()
        if not self.release.wait(5):
            raise TimeoutError("test release missing")
        if self.fail:
            raise RuntimeError("late old inference failure")
        return result


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatedBackend(delay_s=0)
        self.runtime = HotSwitchRuntime(self.backend)
        self.runtime.preload()
        self.robot = SimulatedRobot()
        self.client = HotSwitchClientCore(self.runtime, self.robot.hold)
        self.client.connect("A")

    def execute(self, client=None):
        client = client or self.client
        action = client.next_action()
        client.confirm_action(self.robot.send_action(action))
        return action

    def paced_client(self, period):
        client = HotSwitchClientCore(self.runtime, self.robot.hold, minimum_action_interval_s=period)
        client.connect("A", previous_session_id=self.client.status["session_id"])
        self.client = client
        return client

    def test_slow_refill_never_causes_a_catch_up_action_burst(self):
        period = 0.02
        client = self.paced_client(period)
        client.request_actions({})
        self.execute(client)
        first = client.last_acceptance_time
        self.backend.delay_s = period * 1.5
        client.request_actions({})
        self.execute(client)
        second = client.last_acceptance_time
        self.execute(client)
        third = client.last_acceptance_time
        self.assertGreaterEqual(second - first, period)
        self.assertGreaterEqual(third - second, period)
        self.assertEqual(client.last_acceptance_interval_s, third - second)
        self.assertEqual(client.last_accepted_identity["accepted_monotonic_s"], third)

    def test_task_switch_preserves_control_clock_while_resetting_policy_state(self):
        period = 0.03
        client = self.paced_client(period)
        client.request_actions({})
        self.execute(client)
        previous = client.last_acceptance_time
        client.switch("B")
        self.assertEqual(client.last_acceptance_time, previous)
        self.assertIsNone(client.last_accepted_identity)
        client.request_actions({})
        self.execute(client)
        self.assertGreaterEqual(client.last_acceptance_time - previous, period)
        self.assertEqual(client.accepted_steps, 1)
        previous = client.last_acceptance_time
        client.connect("A", previous_session_id=client.status["session_id"])
        self.assertEqual(client.last_acceptance_time, previous)
        client.request_actions({})
        self.execute(client)
        self.assertGreaterEqual(client.last_acceptance_time - previous, period)

    def wait_for_paced_selection(self, client):
        waiting = threading.Event()
        original_wait = client._action_clock_wakeup.wait
        results, errors = [], []

        def observed_wait(timeout=None):
            waiting.set()
            return original_wait(timeout)

        def select():
            try:
                results.append(client.next_action())
            except Exception as exc:
                errors.append(exc)

        interception = patch.object(client._action_clock_wakeup, "wait", side_effect=observed_wait)
        interception.start()
        self.addCleanup(interception.stop)
        worker = threading.Thread(target=select)
        worker.start()
        self.assertTrue(waiting.wait(1), "Control thread did not enter its paced wait")
        return worker, results, errors

    def test_stop_can_wake_waiting_control_thread_without_state_lock_deadlock(self):
        client = self.paced_client(0.2)
        client.request_actions({})
        self.execute(client)
        hold_threads = []
        client.safe_hold = lambda: hold_threads.append(threading.get_ident())
        worker, results, errors = self.wait_for_paced_selection(client)
        client.stop()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(results)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], HotSwitchError)
        self.assertEqual(hold_threads, [worker.ident])
        self.assertIsNone(client.pending)

    def test_active_inference_error_during_paced_wait_cannot_return_an_action(self):
        client = self.paced_client(0.2)
        client.request_actions({})
        self.execute(client)
        worker, results, errors = self.wait_for_paced_selection(client)
        # The synchronous production path does not run a background driver. This
        # injected CPU failure specifically proves the waiting thread released
        # state.lock so an inference failure can stop it before action selection.
        with (
            patch.object(self.runtime, "infer", side_effect=ValueError("late active error")),
            self.assertRaisesRegex(ValueError, "late active error"),
        ):
            client.request_actions({})
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(results)
        self.assertEqual(len(errors), 1)
        self.assertTrue(client.stopped)
        self.assertIsNone(client.pending)

    def test_old_exception_during_new_task_wait_does_not_reset_its_clock(self):
        client = self.paced_client(0.2)
        client.request_actions({})
        self.execute(client)
        previous = client.last_acceptance_time
        delayed, old_thread, old_errors = self._delayed_request(fail=True)
        client.switch("B")
        client.transport = self.runtime
        client.request_actions({})
        worker, results, errors = self.wait_for_paced_selection(client)
        delayed.release.set()
        old_thread.join(1)
        worker.join(1)
        self.assertFalse(old_errors)
        self.assertFalse(errors)
        self.assertEqual(client.metrics["stale_errors"], 1)
        self.assertFalse(client.stopped)
        self.assertEqual(len(results), 1)
        client.confirm_action(self.robot.send_action(results[0]))
        self.assertGreaterEqual(client.last_acceptance_time - previous, client.minimum_action_interval_s)

    def test_action_interval_rejects_nonfinite_negative_or_missing_values(self):
        for period in (float("nan"), float("inf"), float("-inf"), -0.1, None, "not-a-number"):
            with self.subTest(period=period), self.assertRaises(ValueError):
                HotSwitchClientCore(self.runtime, self.robot.hold, minimum_action_interval_s=period)

    def test_normal_a_b_a_and_no_files_or_reloads(self):
        loads = self.backend.base_load_count, self.backend.adapter_load_count
        self.client.request_actions({})
        expected = self.execute()
        with patch("builtins.open", side_effect=AssertionError("switch performed file I/O")):
            self.client.switch("B")
            self.client.request_actions({})
            self.assertNotEqual(self.execute(), expected)
            self.client.switch("A")
            self.client.request_actions({})
            self.assertEqual(self.execute(), expected)
        self.assertEqual(loads, (self.backend.base_load_count, self.backend.adapter_load_count))
        self.assertEqual(self.client.accepted_steps, 1)

    def test_same_task_is_idempotent(self):
        status = self.client.status
        result = self.runtime.switch("A", "same", status["session_id"])
        self.assertEqual(result["policy_epoch"], status["policy_epoch"])
        repeated = self.runtime.switch("A", "same", status["session_id"])
        self.assertEqual(result["policy_epoch"], repeated["policy_epoch"])
        self.assertEqual(result["switch_id"], repeated["switch_id"])
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["activation_reset_s"], 0)

    def test_unknown_and_not_ready_leave_old_task(self):
        with self.assertRaises(HotSwitchError):
            self.runtime.switch("missing", "bad", self.client.status["session_id"])
        self.runtime.ready.remove("B")
        with self.assertRaises(HotSwitchError):
            self.runtime.switch("B", "bad2", self.client.status["session_id"])
        self.assertEqual(self.runtime.status()["task_id"], "A")

    def test_control_intent_waits_for_control_thread(self):
        self.client.request_actions({})
        self.runtime.request_switch("B", "intent", self.client.status["session_id"])
        self.assertEqual(self.runtime.status()["task_id"], "A")
        self.client.control_boundary()
        self.assertEqual(self.client.status["task_id"], "B")
        self.assertFalse(self.client.queue)
        self.assertFalse(self.runtime.status()["pending_switch"])

    def test_outside_control_commit_stops_old_queue(self):
        self.client.request_actions({})
        self.runtime.switch("B", "external", self.client.status["session_id"])
        with self.assertRaises(StaleIdentity):
            self.client.control_boundary()
        self.assertTrue(self.client.stopped)
        self.assertFalse(self.client.queue)

    def test_pending_control_command_cannot_be_retracted(self):
        self.client.request_actions({})
        self.client.next_action()
        with self.assertRaises(HotSwitchError):
            self.client.switch("B")
        self.assertEqual(self.runtime.status()["task_id"], "A")

    def test_acceptance_must_match_without_clipping(self):
        self.client.request_actions({})
        action = self.client.next_action()
        with self.assertRaises(HotSwitchError):
            self.client.confirm_action([action[0] + 0.1, action[1]])
        self.assertTrue(self.client.stopped)
        self.assertFalse(self.client.queue)
        self.assertEqual(self.client.accepted_steps, 0)

    def test_raw_and_processed_prefix_stay_paired(self):
        self.client.request_actions({})
        bootstrap = self.client.last_inference_alignment
        self.assertEqual(bootstrap["prefix_steps"], 0)
        self.assertIsNone(bootstrap["input_raw_prefix"])
        self.assertIsNone(bootstrap["input_processed_prefix"])
        self.assertEqual(bootstrap["returned_raw_prefix"], [])
        self.execute()
        prefix = list(self.client.queue)[:2]
        self.client.request_actions({})
        self.assertEqual(list(self.client.queue)[:2], prefix)
        self.assertNotEqual(prefix[0][0], prefix[0][1])
        self.assertEqual(self.client.accepted_steps, 1)
        alignment = self.client.last_inference_alignment
        self.assertEqual(alignment["prefix_steps"], 2)
        self.assertEqual(alignment["input_raw_prefix"], [row[0] for row in prefix])
        self.assertEqual(alignment["input_processed_prefix"], [row[1] for row in prefix])
        self.assertEqual(alignment["returned_raw_prefix"], alignment["input_raw_prefix"])
        self.assertEqual(alignment["returned_processed_prefix"], alignment["input_processed_prefix"])
        self.assertEqual(alignment["observation_step"], 1)
        self.assertEqual(alignment["accepted_steps_on_response"], 1)
        self.assertEqual(alignment["consumed_prefix_steps"], 0)
        self.assertEqual(alignment["request_id"], self.client.queue_identity["request_id"])
        for key in ("session_id", "task_id", "policy_epoch"):
            self.assertEqual(alignment[key], self.client.status[key])
        accepted_at = self.client.last_acceptance_time
        self.client.switch("B")
        self.assertIsNone(self.client.last_inference_alignment)
        self.assertEqual(self.client.last_acceptance_time, accepted_at)
        self.client.request_actions({})
        fresh = self.client.last_inference_alignment
        self.assertEqual(fresh["task_id"], "B")
        self.assertEqual(fresh["prefix_steps"], 0)
        self.assertIsNone(fresh["input_raw_prefix"])
        self.assertNotEqual(fresh["request_id"], alignment["request_id"])

    def test_new_task_bootstrap_has_no_old_prefix(self):
        self.client.request_actions({})
        self.client.switch("B")
        with patch.object(self.backend, "infer", wraps=self.backend.infer) as call:
            self.client.request_actions({})
        self.assertEqual(call.call_args.kwargs["prefix_steps"], 0)
        self.assertIsNone(call.call_args.kwargs["action_prefix"])

    def _delayed_request(self, *, fail=False):
        transport = DelayedTransport(self.runtime, fail=fail)
        self.client.transport = transport
        errors = []

        def run():
            try:
                self.client.request_actions({})
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(transport.ready.wait(3))
        return transport, thread, errors

    def test_late_old_result_is_discarded(self):
        transport, thread, errors = self._delayed_request()
        self.client.switch("B")
        transport.release.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertFalse(self.client.queue)
        self.assertEqual(self.client.metrics["stale_responses"], 1)
        self.assertFalse(self.client.stopped)

    def test_late_old_exception_does_not_stop_new_task(self):
        transport, thread, errors = self._delayed_request(fail=True)
        self.client.switch("B")
        transport.release.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(self.client.metrics["stale_errors"], 1)
        self.assertFalse(self.client.stopped)

    def test_same_task_repeat_also_invalidates_late_response(self):
        transport, thread, _ = self._delayed_request()
        self.client.switch("A")
        transport.release.set()
        thread.join(3)
        self.assertFalse(self.client.queue)
        self.assertEqual(self.client.metrics["stale_responses"], 1)

    def test_timeout_after_commit_recovers_idempotently(self):
        runtime = self.runtime

        class TimeoutOnce:
            failed = False

            def switch(self, *args):
                result = runtime.switch(*args)
                if not self.failed:
                    self.failed = True
                    raise TimeoutError("response lost after commit")
                return result

        self.client.transport = TimeoutOnce()
        self.client.switch("B", "timeout-once")
        self.assertEqual(self.client.status["task_id"], "B")
        self.assertEqual(self.runtime.metrics["switches"], 2)

    def test_unconfirmed_timeout_remains_in_hold(self):
        class TimeoutAlways:
            def switch(self, *args):
                raise TimeoutError("unreachable")

        self.client.transport = TimeoutAlways()
        with self.assertRaises(HotSwitchError):
            self.client.switch("B")
        self.assertTrue(self.client.stopped)

    def test_accepted_delay_cannot_exceed_prefix_coverage(self):
        self.client.request_actions({})
        transport, thread, errors = self._delayed_request()
        self.execute()
        self.execute()
        self.execute()
        transport.release.set()
        thread.join(3)
        self.assertEqual(len(errors), 1)
        self.assertTrue(self.client.stopped)
        self.assertEqual(self.client.metrics["prefix_overruns"], 1)

    def test_reconnect_rejects_old_session_request(self):
        request = {
            **{k: self.client.status[k] for k in ("version", "session_id", "task_id", "policy_epoch")},
            "request_id": "old",
            "observation": {},
            "observation_step": 0,
        }
        self.client.connect("A", self.client.status["session_id"])
        with self.assertRaises(StaleIdentity):
            self.runtime.infer(request)

    def test_failure_rolls_back_complete_activation(self):
        activate = self.backend.activate

        def fail_b(task):
            activate(task)
            if task == "B":
                raise RuntimeError("activation failed")

        with patch.object(self.backend, "activate", fail_b), self.assertRaises(RuntimeError):
            self.runtime.switch("B", "rollback", self.client.status["session_id"])
        self.assertEqual(self.runtime.status()["task_id"], "A")
        self.assertEqual(self.backend.active, "A")
        self.assertFalse(self.runtime.stopped)

    def test_switch_id_cannot_be_reused_or_resurrected(self):
        session = self.client.status["session_id"]
        self.runtime.switch("B", "id", session)
        with self.assertRaises(HotSwitchError):
            self.runtime.switch("A", "id", session)
        self.runtime.switch("A", "next", session)
        with self.assertRaises(HotSwitchError):
            self.runtime.switch("B", "id", session)

    def test_inference_and_activation_are_mutually_exclusive(self):
        entered, release, switched = threading.Event(), threading.Event(), threading.Event()
        infer = self.backend.infer

        def slow_infer(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return infer(*args, **kwargs)

        request = {
            **{k: self.client.status[k] for k in ("version", "session_id", "task_id", "policy_epoch")},
            "request_id": "slow",
            "observation": {},
            "observation_step": 0,
        }
        with patch.object(self.backend, "infer", slow_infer):
            first = threading.Thread(target=lambda: self.runtime.infer(request))
            second = threading.Thread(
                target=lambda: (self.runtime.switch("B", "after", request["session_id"]), switched.set())
            )
            first.start()
            self.assertTrue(entered.wait(3))
            second.start()
            self.assertFalse(switched.wait(0.02))
            release.set()
            first.join(3)
            second.join(3)
        self.assertTrue(switched.is_set())

    def test_nan_and_broken_prefix_are_rejected(self):
        self.client.request_actions({})
        with (
            patch.object(self.backend, "infer", return_value=([[float("nan"), 0]] * 8, [[0, 0]] * 8)),
            self.assertRaises(HotSwitchError),
        ):
            self.client.request_actions({})
        self.assertTrue(self.client.stopped)

    def test_missing_safety_callback_is_rejected(self):
        with self.assertRaises(ValueError):
            HotSwitchClientCore(self.runtime, None)

    def test_segment_timings_measure_callbacks_and_complete_roundtrip(self):
        activate, reset = self.backend.activate, self.backend.reset

        def delayed_activate(task):
            time.sleep(0.002)
            activate(task)

        def delayed_reset(task):
            time.sleep(0.002)
            reset(task)

        def delayed_hold():
            time.sleep(0.002)
            self.robot.hold()

        self.client.safe_hold = delayed_hold
        with (
            patch.object(self.backend, "activate", delayed_activate),
            patch.object(self.backend, "reset", delayed_reset),
        ):
            status = self.client.switch("B")
        timings = self.client.last_timings["switch"]
        self.assertGreaterEqual(timings["safe_hold_s"], 0.002)
        self.assertGreaterEqual(status["activation_s"], 0.002)
        self.assertGreaterEqual(status["reset_s"], 0.002)
        self.assertGreaterEqual(
            timings["control_boundary_s"],
            timings["safe_hold_s"] + timings["queue_clear_s"] + timings["transport_switch_s"],
        )
        self.client.request_actions({})
        inference = self.client.last_timings["inference"]
        self.assertGreaterEqual(inference["server_lock_wait_s"], 0)
        self.assertGreaterEqual(inference["backend_inference_s"], 0)
        self.assertGreaterEqual(inference["transport_roundtrip_s"], inference["server_inference_s"])
        self.assertGreaterEqual(
            inference["request_total_s"], inference["transport_roundtrip_s"] + inference["client_queue_s"]
        )

    def test_each_accepted_action_retains_response_identity(self):
        self.client.request_actions({})
        identity = dict(self.client.queue_identity)
        self.assertTrue(identity["request_id"])
        self.assertEqual(len(self.client.queue[0]), 2)
        action = self.client.next_action()
        self.assertEqual(self.client.pending_identity["request_id"], identity["request_id"])
        self.assertEqual(self.client.pending_identity["action_step"], 0)
        self.client.confirm_action(self.robot.send_action(action))
        accepted = self.client.last_accepted_identity
        self.assertEqual(accepted["observation_step"], 0)
        self.assertEqual(accepted["chunk_index"], 0)
        self.assertEqual(accepted["accepted_step"], 1)
        self.assertIsNone(self.client.pending_identity)
        self.client.switch("B")
        self.assertIsNone(self.client.queue_identity)
        self.assertIsNone(self.client.pending_identity)
        self.assertIsNone(self.client.last_accepted_identity)

    def test_inflight_refill_identity_tracks_consumed_prefix_steps(self):
        self.client.request_actions({})
        old_request_id = self.client.queue_identity["request_id"]
        transport, thread, errors = self._delayed_request()
        self.execute()
        self.execute()
        transport.release.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertNotEqual(self.client.queue_identity["request_id"], old_request_id)
        alignment = self.client.last_inference_alignment
        self.assertEqual(alignment["observation_step"], 0)
        self.assertEqual(alignment["accepted_steps_on_response"], 2)
        self.assertEqual(alignment["consumed_prefix_steps"], 2)
        self.execute()
        accepted = self.client.last_accepted_identity
        self.assertEqual(accepted["observation_step"], 0)
        self.assertEqual(accepted["chunk_index"], 2)
        self.assertEqual(accepted["action_step"], 2)
        self.assertEqual(accepted["accepted_step"], 3)

    def test_late_delivery_cannot_replace_new_task_queue_identity(self):
        transport, thread, errors = self._delayed_request()
        self.client.switch("B")
        self.client.transport = self.runtime
        self.client.request_actions({})
        expected = dict(self.client.queue_identity)
        transport.release.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(self.client.queue_identity, expected)
        self.assertEqual(expected["task_id"], "B")


if __name__ == "__main__":
    unittest.main()
