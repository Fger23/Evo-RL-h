"""Reviewable physical-interface acceptance; defaults to a hardware-free dry run.

The local driver factory must have a side-effect-free constructor. Its explicit
connect/close methods own one connection. No robot SDK or driver is imported by
--help or dry-run. Protocol acceptance does not measure task success or motion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_units(units, count):
    unknown = ("unverified", "unknown", "unspecified", "unconfirmed", "n/a")
    return (
        isinstance(units, list)
        and len(units) == count
        and all(
            isinstance(unit, str)
            and unit.strip()
            and not any(marker in unit.casefold() for marker in unknown)
            for unit in units
        )
    )


def validate_operator(operator, manifests, config_directory):
    for name in ("operator_id", "robot_id", "control_mode"):
        require(isinstance(operator.get(name), str) and operator[name].strip(), f"Missing {name}")
    require(
        operator.get("hardware_authorized") is True,
        "Operator configuration must explicitly authorize hardware",
    )
    sequence = operator.get("task_sequence", [])
    require(
        isinstance(sequence, list)
        and len(sequence) >= 3
        and sequence[0] == sequence[-1]
        and len(set(sequence)) >= 2,
        "task_sequence must contain an explicit A -> B -> A acceptance",
    )
    require(all(task in manifests for task in sequence), "Task sequence contains an unregistered task_id")
    schema = operator["action_schema"]
    names, units = schema["names"], schema["units"]
    require(names and len(set(names)) == len(names), "Action names must be explicit, ordered and unique")
    require(
        verified_units(units, len(names)), "Verified action units are required before hardware construction"
    )
    hz = float(schema["control_hz"])
    require(math.isfinite(hz) and hz > 0, "A finite positive control_hz is required")
    for task in sequence:
        require(manifests[task]["action_schema"] == schema, f"Action schema differs for task {task}")
    limits = operator["limits"]
    require(set(limits) == set(names), "Every action needs its own limits and continuity threshold")
    for name in names:
        lower, upper, delta = (float(limits[name][key]) for key in ("min", "max", "max_delta_per_step"))
        require(
            all(math.isfinite(x) for x in (lower, upper, delta)) and lower < upper and delta > 0,
            f"Invalid limits for {name}",
        )
    hold = operator["hold_validation"]
    require(
        hold.get("passed") is True
        and hold.get("robot_id") == operator["robot_id"]
        and hold.get("control_mode") == operator["control_mode"],
        "Holding must have robot/mode-specific validation",
    )
    evidence = (config_directory / hold["evidence_path"]).resolve(strict=True)
    require(evidence.is_file(), "A real operator-reviewed hold validation evidence file is required")
    maximum_hold = float(hold["max_hold_s"])
    require(math.isfinite(maximum_hold) and maximum_hold > 0, "A validated maximum hold duration is required")
    cycles = operator.get("cycles_per_task")
    require(
        isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0,
        "cycles_per_task must be positive",
    )


def build_preflight(registry_path, operator_path=None, driver_factory=None):
    """Local file checks only; never imports a driver or contacts hardware/service."""
    from lerobot.async_inference.hotswitch.manifest import check_compatible, load_task_bundle, read_json

    report = {
        "status": "NOT_EXECUTED",
        "hardware_connected": False,
        "driver_constructed": False,
        "server_checked": False,
        "task_success_rate": "NOT_MEASURED",
        "blockers": [],
        "checklist": [
            "Validated local task bundles and common Base",
            "Verified ordered action units and frequency",
            "Per-action limits and continuity thresholds",
            "Robot/mode-specific hold validation evidence",
            "Live server ready/session/schema checks before driver import",
            "Single connect; control-thread switch; exact acceptance confirmation",
        ],
    }
    manifests, operator = {}, None
    try:
        registry = read_json(registry_path)
        for task, path in registry["tasks"].items():
            _, manifest = load_task_bundle(path)
            require(manifest["task_id"] == task, "Registry and manifest task_id differ")
            manifests[task] = manifest
        require(bool(manifests), "Registry is empty")
        check_compatible(list(manifests.values()))
        report["registered_tasks"] = list(manifests)
    except Exception as exc:
        report["blockers"].append(f"Task bundles: {exc}")
    try:
        require(operator_path is not None, "An operator configuration is required")
        operator_path = Path(operator_path).resolve(strict=True)
        operator = read_json(operator_path)
        validate_operator(operator, manifests, operator_path.parent)
        report["operator_id"], report["robot_id"] = operator["operator_id"], operator["robot_id"]
    except Exception as exc:
        report["blockers"].append(f"Operator configuration: {exc}")
    if not driver_factory or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", driver_factory
    ):
        report["blockers"].append(
            "A trusted local --driver-factory module:factory is required; none is imported in dry-run"
        )
    report["driver_factory"] = driver_factory
    return report, manifests, operator


def load_driver_factory(spec):
    module, factory = spec.split(":", 1)
    return getattr(importlib.import_module(module), factory)


def run_hardware_acceptance(
    report,
    manifests,
    operator,
    driver_factory,
    address,
    timeout,
    previous_session_id=None,
    *,
    transport=None,
    factory_loader=load_driver_factory,
):
    """Called only by explicit execution or hardware-free tests using injected doubles."""
    from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore

    require(not report["blockers"], "Hardware execution blocked by local preflight")
    require(
        0 < timeout <= float(operator["hold_validation"]["max_hold_s"]),
        "RPC timeout exceeds validated holding range",
    )
    transport = transport or GrpcHotSwitchTransport(address, timeout=timeout)
    driver = core = None
    rows = []
    schema = operator["action_schema"]
    reference = None
    hold_started = None
    holding = False

    def validate_vector(action, *, continuity):
        require(
            isinstance(action, (list, tuple)) and len(action) == len(schema["names"]),
            "Driver action shape differs",
        )
        for name, value, index in zip(schema["names"], action, range(len(action)), strict=True):
            value = float(value)
            limit = operator["limits"][name]
            require(
                math.isfinite(value) and float(limit["min"]) <= value <= float(limit["max"]),
                f"Action outside verified limits: {name}",
            )
            if continuity:
                require(
                    reference is not None
                    and abs(value - reference[index]) <= float(limit["max_delta_per_step"]),
                    f"Action discontinuity exceeds verified limit: {name}",
                )

    def hold():
        nonlocal reference, hold_started, holding
        acknowledgement = driver.safe_hold()
        require(
            isinstance(acknowledgement, dict) and acknowledgement.get("held") is True,
            "Driver did not confirm its validated hold mode",
        )
        require(
            acknowledgement.get("action_schema") == schema,
            "Driver hold schema differs from operator/server schema",
        )
        action = acknowledgement.get("reference_action")
        validate_vector(action, continuity=False)
        reference = [float(x) for x in action]
        hold_started, holding = time.perf_counter(), True

    def validate_action(action):
        validate_vector(action, continuity=True)
        require(
            driver.validate_action(action) is True,
            "Driver rejected action under its real control constraints",
        )

    try:
        status = transport.status()
        require(
            status.get("version") == 1 and not status.get("stopped"), "Server protocol is unavailable/stopped"
        )
        require(
            status.get("session_id") is None or status["session_id"] == previous_session_id,
            "Active server session requires its explicit reconnect identity",
        )
        for task in set(operator["task_sequence"]):
            require(task in status["ready"], f"Task is not preloaded and warmed: {task}")
            meta = status["task_metadata"][task]
            require(
                meta.get("action_names") == schema["names"]
                and meta.get("action_units") == schema["units"]
                and meta.get("control_hz") == schema["control_hz"],
                f"Live server schema differs for task {task}",
            )
            rtc = manifests[task]["rtc"]
            require(
                meta.get("rtc_mode") == rtc["mode"]
                and meta.get("max_prefix_steps") == rtc["max_delay"]
                and meta.get("prefix_steps") == rtc["prefix_steps"],
                f"Live server RTC range differs for task {task}",
            )
        report["server_checked"] = True
        # This is the first driver import/construction. All local/server checks
        # above must finish before a factory is allowed to see hardware config.
        driver = factory_loader(driver_factory)(operator.get("driver", {}))
        report["driver_constructed"] = True
        for method in ("connect", "close", "observe_policy", "safe_hold", "validate_action", "send_action"):
            require(callable(getattr(driver, method, None)), f"Driver must implement {method}")
        driver.connect()
        report["hardware_connected"] = True
        core = HotSwitchClientCore(
            transport,
            safe_hold=hold,
            validate_action=validate_action,
            minimum_action_interval_s=1 / float(schema["control_hz"]),
        )
        core.connect(operator["task_sequence"][0], previous_session_id=previous_session_id)
        for phase, task in enumerate(operator["task_sequence"]):
            if task != core.status["task_id"]:
                core.switch(task)
            for cycle in range(operator["cycles_per_task"]):
                started = time.perf_counter()
                live = transport.status()
                intent = live.get("pending_switch")
                require(
                    not intent or intent["task_id"] in operator["task_sequence"],
                    "Switch intent is outside authorized tasks",
                )
                core.control_boundary()
                observation = driver.observe_policy()
                task_id = core.status["task_id"]
                meta = core.status["task_metadata"][task_id]
                if len(core.queue) <= max(1, meta["prefix_steps"]):
                    hold()
                    core.request_actions(observation)
                require(
                    not holding
                    or time.perf_counter() - hold_started <= float(operator["hold_validation"]["max_hold_s"]),
                    "Inference/switch exceeded the validated holding duration",
                )
                expected = core.next_action()
                raw = list(core.pending[0])
                accepted = driver.send_action(expected)
                row = {
                    "phase": phase,
                    "cycle": cycle,
                    "task_id": task_id,
                    "session_id": core.status["session_id"],
                    "policy_epoch": core.status["policy_epoch"],
                    "raw": raw,
                    "processed": expected,
                    "action_identity": dict(core.pending_identity)
                    if core.pending_identity is not None
                    else None,
                    "accepted": list(accepted) if accepted is not None else None,
                    "confirmation": "NOT_CONFIRMED",
                }
                rows.append(row)
                core.confirm_action(accepted)
                reference, holding = [float(x) for x in accepted], False
                row.update(
                    confirmation="ACCEPTED",
                    action_identity=getattr(core, "last_accepted_identity", None),
                    accepted_steps=core.accepted_steps,
                    control_interface_cycle_s=time.perf_counter() - started,
                )
                time.sleep(max(0, 1 / float(schema["control_hz"]) - (time.perf_counter() - started)))
        report.update(
            status="HARDWARE_PROTOCOL_ACCEPTANCE_PASSED",
            rows=rows,
            task_success_rate="NOT_MEASURED",
            physical_motion_latency="NOT_MEASURED",
            client_metrics=dict(core.metrics),
        )
    except Exception as exc:
        report.update(status="HARDWARE_ACCEPTANCE_FAILED", rows=rows, error=f"{type(exc).__name__}: {exc}")
    finally:
        if driver is not None:
            try:
                hold()
                report["final_hold_confirmed"] = True
            except Exception as exc:
                report.update(
                    status="HARDWARE_ACCEPTANCE_FAILED", final_hold_confirmed=False, hold_error=str(exc)
                )
            try:
                driver.close()
            except Exception as exc:
                report.update(status="HARDWARE_ACCEPTANCE_FAILED", close_error=str(exc))
        transport.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--operator-config", type=Path)
    parser.add_argument("--driver-factory", help="Trusted local module:factory; not imported in dry-run")
    parser.add_argument("--address", default="127.0.0.1:8099")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--previous-session-id")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Default: local review only, no driver/server contact"
    )
    mode.add_argument(
        "--execute-hardware",
        action="store_true",
        help="Explicitly authorize one hardware connection after all gates",
    )
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    report, manifests, operator = build_preflight(args.registry, args.operator_config, args.driver_factory)
    if args.execute_hardware:
        require(args.output is not None, "Hardware execution requires a new --output report path")
        if report["blockers"]:
            report["status"] = "HARDWARE_PREFLIGHT_BLOCKED"
        else:
            report = run_hardware_acceptance(
                report,
                manifests,
                operator,
                args.driver_factory,
                args.address,
                args.timeout,
                args.previous_session_id,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.execute_hardware and report["status"] != "HARDWARE_PROTOCOL_ACCEPTANCE_PASSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
