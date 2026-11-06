"""CPU-only controls. Linux cases own inert subprocesses, never a model/service.

For deployed Linux tests put temporary artifacts in a new personal NFS directory
with FORMAL_GUARDIAN_TEST_ARTIFACT_ROOT. Code stays beside this HOME test file.
"""

import copy
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SOURCE = (Path(__file__).resolve().parents[2] / "scripts/hotswitch/server_guardian.py")
spec = importlib.util.spec_from_file_location("tested_birth_guardian", SOURCE)
guardian = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guardian)


def private_write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(value)


def wait_state(path, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if path.exists():
            last = json.loads(path.read_text())
            if predicate(last):
                return last
            if last.get("status") == "FAILED":
                raise AssertionError(last)
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for own CPU fixture: {last}")


def launch_fixture(root, *, nondumpable=False):
    root = Path(root)
    work = root / "server_work"
    work.mkdir()
    token = secrets.token_hex(24)
    token_file = root / "token"
    private_write(token_file, token)
    code = "import time; time.sleep(120)"
    if nondumpable:
        code = (
            "import ctypes,time; function=ctypes.CDLL(None,use_errno=True).prctl; "
            "function.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]; "
            "function.restype=ctypes.c_int; assert function(4,0,0,0,0)==0; time.sleep(120)"
        )
    selected = {
        "CUDA_VISIBLE_DEVICES": "",
        "HOTSWITCH_RPC_RUN_TOKEN": token,
        "PYTHONPATH": str(SOURCE.parent),
    }
    value = {
        "schema_version": 1,
        "argv": [sys.executable, "-B", "-c", code],
        "cwd": str(work),
        "stdout": str(root / "child.stdout"),
        "stderr": str(root / "child.stderr"),
        "expected_uid": os.getuid(),
        "harness_pid": os.getpid(),
        "harness_identity": guardian.process_identity(os.getpid()),
        "environment": selected,
    }
    private_write(root / "spec.json", json.dumps(value))
    state = root / "state.json"
    stderr = (root / "guardian.stderr").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(SOURCE),
            "serve",
            "--spec",
            str(root / "spec.json"),
            "--state",
            str(state),
            "--token-file",
            str(token_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
    )
    stderr.close()
    return process, state, token_file


class IdentityTests(unittest.TestCase):
    def identity(self):
        return {
            "pid": 11,
            "parent_pid": 7,
            "start_ticks": "123",
            "uid": 42,
            "uids": [42] * 4,
            "state": "S",
            "argv": ["/trusted/python", "-m", "owned"],
        }

    def test_every_kernel_identity_component_is_required(self):
        with mock.patch.object(guardian.os, "getuid", return_value=42, create=True):
            expected = self.identity()
            guardian.validate_identity(expected, expected)
            for key, value in [
                ("pid", 12),
                ("parent_pid", 8),
                ("start_ticks", "124"),
                ("uid", 43),
                ("uids", [42, 0, 42, 42]),
                ("argv", ["/foreign/python"]),
            ]:
                with self.subTest(key=key):
                    actual = copy.deepcopy(expected)
                    actual[key] = value
                    with self.assertRaises(ValueError):
                        guardian.validate_identity(actual, expected)

    def test_wrong_identity_never_reaches_pidfd_signal(self):
        owner = object.__new__(guardian.Guardian)
        owner.child = mock.Mock(pid=11)
        owner.child.poll.return_value = None
        owner.pidfd = 99
        owner.identity = self.identity()
        owner.birth = None
        owner.last_cleanup = None
        actual = self.identity()
        actual["start_ticks"] = "999"
        with (
            mock.patch.object(guardian, "process_identity", return_value=actual),
            mock.patch.object(guardian, "send_pidfd") as send,
            mock.patch.object(guardian.os, "getuid", return_value=42, create=True),
        ):
            with self.assertRaises(ValueError):
                owner.cleanup()
            send.assert_not_called()

    def test_runtime_has_no_numeric_pid_signal_fallback(self):
        self.assertNotIn("os.kill(", SOURCE.read_text())
        self.assertNotIn("libc.kill", SOURCE.read_text())


@unittest.skipUnless(sys.platform == "linux", "Actual birth pidfds and non-dumpable /proc require Linux")
class LinuxOwnedChildTests(unittest.TestCase):
    def setUp(self):
        artifact_root = os.environ.get("FORMAL_GUARDIAN_TEST_ARTIFACT_ROOT")
        self.evidence_root = Path(artifact_root) if artifact_root else None
        self.temporary = tempfile.TemporaryDirectory(prefix="guardian_cpu_", dir=artifact_root)
        self.root = Path(self.temporary.name)
        self.process = None

    def persist(self, proof, **facts):
        if self.evidence_root is not None:
            private_write(
                self.evidence_root / (self._testMethodName + ".json"),
                json.dumps(
                    {
                        "kind": "CPU_ONLY_GUARDIAN_FIXTURE",
                        "real_policy_server": False,
                        "real_gpu_tested": False,
                        "physical_robot": False,
                        "guardian_source_sha256": guardian.sha(SOURCE),
                        "proof": proof,
                        **facts,
                    },
                    indent=2,
                ),
            )

    def tearDown(self):
        if self.process is not None:
            state, token = self.root / "state.json", self.root / "token"
            if state.exists() and self.process.poll() is None:
                guardian.request(state, token, "stop")
            self.process.wait(timeout=15)
        self.temporary.cleanup()

    def launch(self, **options):
        self.process, state, token = launch_fixture(self.root, **options)
        proof = wait_state(state, lambda item: item["status"] == "RUNNING" and item.get("control_address"))
        self.assertTrue(proof["birth_env_verified"])
        self.assertTrue(proof["pidfd_opened_before_go"])
        self.assertTrue(proof["pidfd_held"])
        self.assertEqual(proof["server_identity"]["parent_pid"], self.process.pid)
        self.assertEqual(proof["server_identity"]["uids"], [os.getuid()] * 4)
        return state, token, proof

    def test_real_birth_then_term_and_idempotent_stop(self):
        state, token, proof = self.launch()
        own_pid = proof["server_identity"]["pid"]
        current = guardian.request(state, token, "status")
        self.assertEqual(current["server_identity"]["pid"], own_pid)
        stopped = guardian.request(state, token, "stop")
        self.assertEqual(stopped["cleanup"]["signals"], ["SIGTERM"])
        self.assertEqual(self.process.wait(timeout=15), 0)
        again = guardian.request(state, token, "stop")
        self.assertTrue(again["reaped"])
        self.assertFalse(again["pidfd_held"])
        self.assertEqual(again["cleanup"]["signals"], ["SIGTERM"])
        self.assertFalse(Path(f"/proc/{own_pid}").exists())
        self.persist(again, idempotent_stop=True)

    def test_nondumpable_child_late_environ_denied_but_born_fd_stop_succeeds(self):
        state, token, proof = self.launch(nondumpable=True)
        pid = proof["server_identity"]["pid"]
        # Do not require timing of Python startup/prctl; wait for actual denial.
        deadline = time.monotonic() + 5
        while True:
            try:
                Path(f"/proc/{pid}/environ").read_bytes()
            except PermissionError:
                break
            self.assertLess(time.monotonic(), deadline, "Fixture did not actually become non-dumpable")
            time.sleep(0.02)
        self.assertTrue(guardian.request(state, token, "status")["identity_verified"])
        stopped = guardian.request(state, token, "stop")
        self.assertFalse(stopped["late_environ_read_required"])
        self.assertTrue(stopped["birth_env_verified"])
        self.assertEqual(stopped["cleanup"]["signals"], ["SIGTERM"])
        self.assertEqual(self.process.wait(timeout=15), 0)

        self.persist(json.loads(state.read_text()), late_environ_permission_error_observed=True)

    def test_wrong_bearer_and_partial_control_request_do_not_destroy_guardian(self):
        state, token_file, proof = self.launch()
        host, port = proof["control_address"]
        with socket.create_connection((host, port), timeout=3) as connection:
            connection.sendall(json.dumps({"action": "stop", "token": "0" * 48}).encode() + b"\n")
            reply = json.loads(connection.makefile("rb").readline())
        self.assertEqual(reply["status"], "REFUSED")
        with socket.create_connection((host, port), timeout=3) as connection:
            connection.sendall(b"{")
        time.sleep(0.05)
        current = guardian.request(state, token_file, "status")
        self.assertEqual(current["status"], "RUNNING")
        self.persist(current, wrong_bearer_refused=True, partial_message_survived=True)

    def test_private_control_files_reject_group_access(self):
        self.process, state, token = launch_fixture(self.root)
        wait_state(state, lambda item: item["status"] == "RUNNING" and item.get("control_address"))
        token.chmod(0o644)
        try:
            with self.assertRaises(ValueError):
                guardian.read_token(token)
        finally:
            token.chmod(0o600)

    def test_parent_exit_cleans_only_its_born_child(self):
        # This fixture mode is code in this HOME test file, not a new NFS script.
        stderr = (self.root / "harness.stderr").open("wb")
        harness = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--fixture-harness", str(self.root)],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        )
        stderr.close()
        state = self.root / "state.json"
        proof = wait_state(state, lambda item: item["status"] == "RUNNING" and item.get("control_address"))
        pid = proof["server_identity"]["pid"]
        (self.root / "exit_harness").touch()
        self.assertEqual(harness.wait(timeout=10), 0)
        final = wait_state(
            state, lambda item: item.get("pidfd_held") is False and item["status"] == "STOPPED"
        )
        self.assertIn("HARNESS_LOST_OR_CHANGED", final["automatic_cleanup_reason"])
        self.assertEqual(final["cleanup"]["signals"], ["SIGTERM"])
        self.assertTrue(final["reaped"])
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.persist(final, parent_exit_cleanup=True)


def fixture_harness(root):
    process, _, _ = launch_fixture(root)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (Path(root) / "exit_harness").exists():
        time.sleep(0.02)
    # No ordinary PID signal. Guardian observes this direct owner's disappearance.
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fixture-harness":
        raise SystemExit(fixture_harness(sys.argv[2]))
    unittest.main()
