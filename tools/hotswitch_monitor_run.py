"""Read-only Linux training monitor; emits identities, progress and terminal state.

This tool never starts/stops a job, imports a model, writes a dataset or touches a
robot. Optional JSONL output is a separate monitoring ledger, appended on resume.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import time
from pathlib import Path


def tail(path, size=16384):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors="replace")


def process_identity(pid, expected=None):
    try:
        root = Path("/proc") / str(int(pid))
        command = [part.decode() for part in (root / "cmdline").read_bytes().split(b"\0") if part]
        state = (root / "stat").read_text().rsplit(")", 1)[1].split()[0]
        return {"pid": int(pid), "live": state != "Z" and bool(command),
                "identity_matches": command == expected if expected else True, "state": state}
    except (OSError, ValueError):
        return {"pid": pid, "live": False, "identity_matches": False}


def snapshot(run_root, gpu=None):
    status = json.loads((run_root / "pipeline_status.json").read_text())
    summary = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "run_root": str(run_root), "profile": status["profile"], "stages": {}, "state": "RUNNING"}
    expected_stages = ["value-train", "value-infer", "adapter"]
    for name, stage in status["stages"].items():
        entry = {key: stage.get(key) for key in ["status", "pid", "exit_code", "elapsed_s", "gpu_uuid"]}
        if stage.get("status") == "running":
            entry["process"] = process_identity(stage["pid"], stage["argv"])
        text = tail(run_root / f"{name}.log")
        if name == "value-infer":
            progress = re.findall(r"(\d+)/(\d+) \[([^\]]+)\]", text)
            if progress:
                current, total, timing = progress[-1]
                entry["progress"] = {"batches": int(current), "total_batches": int(total), "tqdm_timing": timing}
        else:
            steps = [line for line in text.splitlines() if "step:" in line]
            if steps:
                entry["latest_training_log"] = steps[-1]
        entry["traceback_in_log_tail"] = "Traceback (most recent call last)" in text
        summary["stages"][name] = entry
    if any(s.get("status") == "failed" for s in status["stages"].values()):
        summary["state"] = "FAILED"
    elif all(status["stages"].get(name, {}).get("status") == "passed" for name in expected_stages):
        summary["state"] = "PILOT_COMPLETED" if status["profile"] == "pilot" else "FORMAL_TRAINING_COMPLETED"
    else:
        pid_file = run_root / "supervisor.pid"
        if pid_file.exists():
            supervisor = process_identity(pid_file.read_text().strip())
            summary["supervisor"] = supervisor
            if not supervisor["live"]:
                summary["state"] = "FAILED_SUPERVISOR_EXITED"
    if gpu is not None:
        try:
            fields = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
            summary["gpu"] = {"index": gpu, "uuid": fields[0].strip(), "used_MiB": int(fields[1]), "utilization_percent": int(fields[2])}
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            summary["gpu_query_error"] = str(error)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=55)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        report = snapshot(args.run_root.resolve(strict=True), args.gpu)
        line = json.dumps(report)
        if args.ledger:
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            with args.ledger.open("a") as stream:
                stream.write(line + "\n")
        print(line, flush=True)
        if report["state"].startswith("FAILED"):
            raise SystemExit(1)
        if args.once or report["state"] != "RUNNING":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
