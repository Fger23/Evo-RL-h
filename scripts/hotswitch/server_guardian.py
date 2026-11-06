"""Own one direct child through its birth pidfd; no model imports or PID signals.

The child shim waits for GO before exec. Actual birth environment is verified
while readable; subsequent ownership uses the held pidfd and kernel identity.
The loopback control bearer is read from a private file, never from argv.
"""

import argparse
import contextlib
import ctypes
import datetime
import errno
import hashlib
import hmac
import json
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    uids = list(
        map(
            int,
            next(
                line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:")
            ).split()[1:],
        )
    )
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "uid": uids[0],
        "uids": uids,
        "state": fields[0],
        "argv": [item.decode() for item in (root / "cmdline").read_bytes().split(b"\0") if item],
    }


def validate_identity(actual, expected):
    require(actual["state"] != "Z", "Child is already a zombie")
    for key in ("pid", "parent_pid", "start_ticks", "uid", "uids", "argv"):
        require(actual[key] == expected[key], f"Refusing signal/control: child {key} differs")
    require(
        len(actual["uids"]) == 4 and all(uid == os.getuid() for uid in actual["uids"]),
        "Every child UID must equal the guardian UID",
    )


def private_file(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "Private control files must not be symlinks")
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0,
        "Private control file must be owned by this UID with no group/other access",
    )
    return path


def read_token(path):
    token = private_file(path).read_text().strip()
    require(
        len(token) >= 48 and all(char in "0123456789abcdef" for char in token),
        "Expected a private random lowercase hex bearer of at least192 bits",
    )
    return token


def open_pidfd(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, 0), "native"
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor, "libc"


def send_pidfd(descriptor, value):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(descriptor, value, None, 0)
        return
    function = ctypes.CDLL(None, use_errno=True).pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(descriptor, value, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def read_line(stream, limit=32768):
    data = stream.readline(limit + 1)
    require(len(data) <= limit and data.endswith(b"\n"), "Missing or oversized control message")
    return data


def child_shim(args):
    # This mode imports only this stdlib file, then blocks before any model code.
    spec = json.loads(private_file(args.spec).read_text())
    with os.fdopen(args.ack_fd, "wb", buffering=0) as stream:
        stream.write(b"BIRTH\n")
    with os.fdopen(args.go_fd, "rb", buffering=0) as stream:
        ready, _, _ = select.select([stream], [], [], 30)
        require(ready, "Guardian did not authorize child exec within30s")
        message = read_line(stream).decode().strip()
    expected = "GO " + os.environ["HOTSWITCH_RPC_RUN_TOKEN"]
    require(hmac.compare_digest(message, expected), "Child GO bearer differs")
    os.chdir(spec["cwd"])
    os.execvpe(spec["argv"][0], spec["argv"], os.environ.copy())


class Guardian:
    def __init__(self, spec_path, state_path, token_file):
        self.spec_path = private_file(spec_path)
        self.spec = json.loads(self.spec_path.read_text())
        self.token = read_token(token_file)
        self.token_sha = hashlib.sha256(self.token.encode()).hexdigest()
        self.state_path = Path(state_path).absolute()
        require(not self.state_path.exists(), "Refusing an existing guardian state")
        self.root = self.spec_path.parent.resolve(strict=True)
        require(
            self.state_path.parent.resolve(strict=True) == self.root,
            "Guardian state must be beside its private launch spec",
        )
        self.child = None
        self.pidfd = None
        self.birth = self.identity = None
        self.stop_requested = False
        self.last_cleanup = None
        self.state = {
            "schema_version": 1,
            "status": "STARTING",
            "guardian_identity": process_identity(os.getpid()),
            "spec_sha256": sha(self.spec_path),
            "guardian_source_sha256": sha(__file__),
            "token_sha256": self.token_sha,
            "birth_env_verified": False,
            "pidfd_held": False,
            "late_environ_read_required": False,
            "signals": [],
        }
        self.harness_identity = process_identity(os.getppid())
        self.state["harness_identity"] = self.harness_identity

    def save(self):
        self.state.update(
            observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            heartbeat_monotonic_s=time.monotonic(),
        )
        temporary = self.state_path.with_name(self.state_path.name + f".{os.getpid()}.tmp")
        with temporary.open("x") as stream:
            json.dump(self.state, stream, indent=2)
        os.replace(temporary, self.state_path)

    def launch(self):
        spec = self.spec
        require(
            spec.get("schema_version") == 1 and spec.get("expected_uid") == os.getuid(),
            "Wrong launch spec UID/version",
        )
        require(spec.get("harness_pid", os.getppid()) == os.getppid(), "Guardian belongs to another harness")
        if "harness_identity" in spec:
            validate_identity(self.harness_identity, spec["harness_identity"])
        require(
            isinstance(spec["argv"], list) and spec["argv"] and all(isinstance(x, str) for x in spec["argv"]),
            "Expected an exact argv array",
        )
        require(Path(spec["argv"][0]).is_absolute(), "Child executable must be an absolute path")
        cwd = Path(spec["cwd"]).absolute()
        require(
            not cwd.is_symlink()
            and cwd.resolve(strict=True).is_relative_to(self.root)
            and cwd.is_dir()
            and not list(cwd.iterdir()),
            "Use a new empty artifact working directory",
        )
        selected = spec["environment"]
        require(
            set(selected) == {"CUDA_VISIBLE_DEVICES", "HOTSWITCH_RPC_RUN_TOKEN", "PYTHONPATH"}
            and all(isinstance(value, str) for value in selected.values()),
            "Unexpected selected environment schema",
        )
        require(
            hmac.compare_digest(selected["HOTSWITCH_RPC_RUN_TOKEN"], self.token),
            "Spec/control bearer differs",
        )
        env = dict(os.environ, **selected)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        read_go, write_go = os.pipe()
        read_ack, write_ack = os.pipe()
        shim = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "_child",
            "--spec",
            str(self.spec_path),
            "--go-fd",
            str(read_go),
            "--ack-fd",
            str(write_ack),
        ]
        output_paths = [Path(spec[key]).absolute() for key in ("stdout", "stderr")]
        for path in output_paths:
            require(
                path.parent.resolve(strict=True) == self.root and not path.exists(),
                "Child logs must be new files beside spec",
            )
        try:
            with (
                output_paths[0].open("xb", buffering=0) as stdout,
                output_paths[1].open("xb", buffering=0) as stderr,
            ):
                self.child = subprocess.Popen(
                    shim,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    pass_fds=(read_go, write_ack),
                )
            os.close(read_go)
            read_go = None
            os.close(write_ack)
            write_ack = None
            self.pidfd, backend = open_pidfd(self.child.pid)
            self.state.update(pidfd_held=True, pidfd_backend=backend)
            ready, _, _ = select.select([read_ack], [], [], 10)
            require(ready and os.read(read_ack, 32) == b"BIRTH\n", "Child did not reach its birth barrier")
            birth = process_identity(self.child.pid)
            require(
                birth["parent_pid"] == os.getpid() and birth["argv"] == shim,
                "Birth child is not the exact direct shim",
            )
            require(
                birth["uid"] == os.getuid() and birth["uids"] == [os.getuid()] * 4,
                "Wrong birth UID credentials",
            )
            self.birth = birth
            environment = (Path("/proc") / str(self.child.pid) / "environ").read_bytes().split(b"\0")
            for key, value in selected.items():
                require((key + "=" + value).encode() in environment, f"Birth environment differs: {key}")
            self.state.update(
                birth_identity=birth,
                birth_env_verified=True,
                birth_environment={
                    "CUDA_VISIBLE_DEVICES": selected["CUDA_VISIBLE_DEVICES"],
                    "PYTHONPATH": selected["PYTHONPATH"],
                    "run_token_sha256": self.token_sha,
                },
                pidfd_opened_before_go=True,
            )
            validate_identity(process_identity(self.harness_identity["pid"]), self.harness_identity)
            self.save()
            os.write(write_go, ("GO " + self.token + "\n").encode())
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                require(self.child.poll() is None, "Child exited before its final argv was verified")
                actual = process_identity(self.child.pid)
                if actual["argv"] == spec["argv"]:
                    expected = dict(birth, argv=spec["argv"])
                    validate_identity(actual, expected)
                    self.identity = expected
                    self.state.update(server_identity=expected, status="RUNNING", child_exec_verified=True)
                    return
                require(actual["argv"] in (shim, []), "Child execed an unexpected command")
                time.sleep(0.01)
            raise TimeoutError("Child did not exec its exact server argv")
        finally:
            for descriptor in (read_go, write_go, read_ack, write_ack):
                if descriptor is not None:
                    os.close(descriptor)

    def current(self):
        require(self.child is not None and self.pidfd is not None, "Guardian has no born child/pidfd")
        code = self.child.poll()
        if code is not None:
            self.state.update(status="EXITED", child_exit_code=code, reaped=True)
        else:
            actual = process_identity(self.child.pid)
            if actual["state"] == "Z":
                self.state.update(status="EXITED", child_exit_code=self.child.wait(timeout=1), reaped=True)
                self.save()
                return self.state
            validate_identity(actual, self.identity)
            self.state.update(status="RUNNING", server_identity=actual, identity_verified=True)
        self.save()
        return self.state

    def cleanup(self):
        require(self.child is not None and self.pidfd is not None, "No owned Popen/pidfd to clean")
        if self.last_cleanup:
            return self.last_cleanup
        sent = []
        if self.child.poll() is None:
            expected = self.identity or self.birth
            require(expected is not None, "Birth proof unavailable: refusing signal")
            actual = process_identity(self.child.pid)
            if actual["state"] == "Z":
                self.child.wait(timeout=1)
                return self.cleanup()
            # Startup may fail after GO but before the final identity was stored.
            if self.identity is None and actual["argv"] == self.spec["argv"]:
                expected = dict(expected, argv=self.spec["argv"])
            validate_identity(actual, expected)
            try:
                send_pidfd(self.pidfd, signal.SIGTERM)
                sent.append("SIGTERM")
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                validate_identity(process_identity(self.child.pid), expected)
                send_pidfd(self.pidfd, signal.SIGKILL)
                sent.append("SIGKILL")
                self.child.wait(timeout=10)
        else:
            self.child.wait()
        self.last_cleanup = {
            "status": "STOPPED",
            "signals": sent,
            "reaped": True,
            "child_exit_code": self.child.returncode,
            "signal_backend": self.state["pidfd_backend"],
            "birth_env_verified": self.state["birth_env_verified"],
            "pidfd_opened_before_go": True,
            "late_environ_read_required": False,
            "ownership": "DIRECT_POPEN_AND_HELD_BIRTH_PIDFD",
        }
        failed = self.state.get("status") == "FAILED"
        self.state.update(self.last_cleanup, cleanup=self.last_cleanup)
        if failed:
            self.state["status"] = "FAILED"
        self.save()
        return self.state

    def serve(self):
        original_handlers = {}

        def stop(signum, frame):
            self.stop_requested = True

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                original_handlers[number] = signal.signal(number, stop)
            self.launch()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(4)
                listener.settimeout(0.2)
                self.state["control_address"] = list(listener.getsockname())
                self.save()
                last_heartbeat = time.monotonic()
                while not self.stop_requested:
                    try:
                        validate_identity(
                            process_identity(self.harness_identity["pid"]), self.harness_identity
                        )
                    except (OSError, ValueError, KeyError) as error:
                        self.state["automatic_cleanup_reason"] = f"HARNESS_LOST_OR_CHANGED: {error}"
                        break
                    if self.child.poll() is not None:
                        break
                    if time.monotonic() - last_heartbeat >= 1:
                        self.save()
                        last_heartbeat = time.monotonic()
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connection.settimeout(3)
                        try:
                            with connection.makefile("rb") as stream:
                                query = json.loads(read_line(stream))
                            require(
                                hmac.compare_digest(query.get("token", ""), self.token),
                                "Control bearer differs",
                            )
                            require(query.get("action") in ("status", "stop"), "Unknown control action")
                            result = self.current() if query["action"] == "status" else self.cleanup()
                            connection.sendall(json.dumps(result).encode() + b"\n")
                            if query["action"] == "stop":
                                self.stop_requested = True
                        except Exception as error:
                            # A timed-out/malformed caller cannot destroy ownership.
                            with contextlib.suppress(OSError):
                                connection.sendall(
                                    json.dumps({"status": "REFUSED", "error": str(error)}).encode() + b"\n"
                                )
            if not self.last_cleanup:
                self.cleanup()
            return 0
        except BaseException as error:
            self.state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            self.save()
            raise
        finally:
            if self.child is not None and self.pidfd is not None and not self.last_cleanup:
                self.cleanup()
            if self.pidfd is not None:
                os.close(self.pidfd)
                self.state["pidfd_held"] = False
                self.save()
            for number, handler in original_handlers.items():
                signal.signal(number, handler)


def request(state_path, token_file, action):
    token = read_token(token_file)
    state = json.loads(Path(state_path).read_text())
    require(
        hmac.compare_digest(state["token_sha256"], hashlib.sha256(token.encode()).hexdigest()),
        "State/control bearer differs",
    )
    if action == "stop" and state.get("status") == "STOPPED" and state.get("reaped") is True:
        return state
    host, port = state["control_address"]
    require(host == "127.0.0.1" and 0 < int(port) <= 65535, "Non-loopback guardian endpoint")
    with socket.create_connection((host, port), timeout=25) as connection:
        connection.settimeout(25)
        connection.sendall(json.dumps({"action": action, "token": token}).encode() + b"\n")
        with connection.makefile("rb") as stream:
            result = json.loads(read_line(stream))
    require(result["status"] != "REFUSED", result.get("error", "Guardian refused control"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("_child")
    child.add_argument("--spec", required=True)
    child.add_argument("--go-fd", type=int, required=True)
    child.add_argument("--ack-fd", type=int, required=True)
    server = sub.add_parser("serve")
    server.add_argument("--spec", required=True)
    server.add_argument("--state", required=True)
    server.add_argument("--token-file", required=True)
    client = sub.add_parser("request")
    client.add_argument("--state", required=True)
    client.add_argument("--token-file", required=True)
    client.add_argument("--action", choices=("status", "stop"), required=True)
    args = parser.parse_args()
    if args.command == "_child":
        child_shim(args)
        return 0
    if args.command == "serve":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Guardian parent must keep CUDA hidden")
        return Guardian(args.spec, args.state, args.token_file).serve()
    result = request(args.state, args.token_file, args.action)
    print(json.dumps(result))
    return 0 if result["status"] in ("RUNNING", "STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
