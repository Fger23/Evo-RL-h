"""Bimanual SO101 control using the existing resident-policy RPC and drivers.

No model is loaded on the client. No SDK is imported until configuration and
the live server contract have passed. Calibration is never started implicitly.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NAMES = [f"{side}_{motor}.pos" for side in ("left", "right") for motor in MOTORS]
CAMERAS = ("left_fixed", "left_wrist", "right_wrist")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_positive(value, label):
    require(not isinstance(value, bool), f"{label} must be positive")
    number = float(value)
    require(math.isfinite(number) and number > 0, f"{label} must be finite and positive")
    return number


def concrete(value):
    return isinstance(value, str) and bool(value.strip()) and "<" not in value and ">" not in value


def calibration_directory(robot):
    if robot.get("calibration_dir"):
        return Path(robot["calibration_dir"]).expanduser()
    # Match RTC_run.txt and Robot's real default, including an overridden HF_HOME.
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
    return HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower"


def validate_config(config, *, check_local_files=True):
    require(config.get("robot", {}).get("type") == "bi_so_follower", "This client expects two SO101 follower arms")
    robot = config["robot"]
    require(concrete(config.get("address")) and ":" in config["address"], "Fill the server address HOST:PORT")
    require(concrete(robot.get("id")), "Set the same robot.id used during calibration")
    # This assertion concerns the original training data, not just today's SDK.
    require(config.get("training_units") in ("normalized", "degrees"),
            "Set training_units to normalized or degrees after checking the collection settings; the old model bundle records unverified units")
    units = config["training_units"]
    require(all(robot[f"{side}_arm_config"].get("use_degrees") is (units == "degrees") for side in ("left", "right")),
            "Both arms' use_degrees must match the training units (grippers always use 0..100)")
    ports = [robot[f"{side}_arm_config"].get("port") for side in ("left", "right")]
    require(all(concrete(port) for port in ports) and ports[0] != ports[1], "Fill the two distinct follower ports from the original robot setup")
    calibration = calibration_directory(robot)
    expected_cameras = {}
    for side in ("left", "right"):
        for name, camera in robot[f"{side}_arm_config"].get("cameras", {}).items():
            expected_cameras[f"{side}_{name}"] = camera
    require(set(expected_cameras) == set(CAMERAS), "Camera names must map exactly to left_fixed, left_wrist and right_wrist")
    for name, camera in expected_cameras.items():
        require(camera.get("type") in ("opencv", "intelrealsense"), f"Unsupported camera type: {name}")
        source = camera.get("index_or_path") if camera["type"] == "opencv" else camera.get("serial_number_or_name")
        require((isinstance(source, int) and not isinstance(source, bool) and source >= 0) or concrete(source), f"Fill camera device for {name}")
        require((camera.get("width"), camera.get("height"), camera.get("fps")) == (640, 480, 30), "Current model expects RGB640x480 at30Hz")
    require(float(config.get("control_hz", 0)) == 30, "Current policy uses30Hz")
    finite_positive(config["timeout_s"], "timeout_s")
    finite_positive(config["max_delta_per_step"], "max_delta_per_step")
    require(isinstance(config["steps"], int) and not isinstance(config["steps"], bool) and config["steps"] >= 0, "steps must be0 (until Ctrl+C) or positive")
    if units == "degrees":
        require(set(config.get("limits", {})) == set(NAMES), "Degree mode requires explicit calibrated limits for all12 actions")
    for name in NAMES:
        lower, upper, delta = action_limit(config, name)
        require(all(math.isfinite(v) for v in (lower, upper, delta)) and lower < upper and delta > 0, f"Invalid limit for {name}")
    if check_local_files:
        require(calibration.is_dir(), f"Existing calibration directory not found: {calibration}")
        for side in ("left", "right"):
            file = calibration / f"{robot['id']}_{side}.json"
            require(file.is_file(), f"Use the existing calibration file: {file}; this command will not calibrate")
            saved = json.loads(file.read_text(encoding="utf-8"))
            require(set(saved) == set(MOTORS), f"Calibration motor set differs: {file}")
            for index, name in enumerate(MOTORS, 1):
                require(saved[name].get("id") == index and saved[name]["range_min"] < saved[name]["range_max"], f"Invalid calibration for {name}")
        require(all(Path(port).exists() for port in ports), "Follower serial device is not present on this client")
        require(len({str(Path(port).resolve()) for port in ports}) == 2, "Both ports resolve to the same device")
    return config


def action_limit(config, name):
    if name in config.get("limits", {}):
        limit = config["limits"][name]
        return tuple(float(limit[k]) for k in ("min", "max", "max_delta_per_step"))
    return (0.0 if "gripper" in name else -100.0, 100.0, float(config["max_delta_per_step"]))


def check_server(config, status):
    task = config["task_id"]
    require(status.get("version") == 1 and task in status.get("ready", []), "Selected task is not ready on this server")
    meta = status["task_metadata"][task]
    require(meta.get("action_names") == NAMES and meta.get("action_dim") == 12, "Server action order/schema differs from the dual SO101")
    require(float(meta.get("control_hz", 0)) == float(config["control_hz"]), "Server control frequency differs")
    images = meta.get("image_features", {})
    require(set(images) == {"observation.images." + name for name in CAMERAS}, "Server camera views differ")
    require(all(list(shape) == [3, 480, 640] for shape in images.values()), "Server image dimensions differ")
    expected = config.get("bundle_manifest_sha256")
    require(concrete(expected) and meta.get("bundle_manifest_sha256") == expected, "Server is serving a different model bundle")
    expected_units = ["deg" if config["training_units"] == "degrees" and "gripper" not in name else
                      "normalized_0_100" if "gripper" in name else "normalized_-100_100" for name in NAMES]
    units = meta.get("action_units")
    require(units == ["recorded_robot_units_unverified"] * 12 or units == expected_units,
            "Server action units conflict with the explicitly configured training units")
    # The old manifest stays immutable. The client config carries the explicit
    # collection-unit assertion and pins it to this exact bundle fingerprint.
    require(meta.get("rtc_mode") == "training_time" and 0 <= meta.get("prefix_steps", -1) <= meta.get("max_prefix_steps", -1), "Unsupported RTC metadata")
    return meta


def policy_observation(raw):
    import torch
    state = torch.tensor([float(raw[name]) for name in NAMES], dtype=torch.float32)
    require(bool(torch.isfinite(state).all()), "Robot state is not finite")
    result = {"observation.state": state}
    for camera in CAMERAS:
        image = torch.as_tensor(raw[camera])
        require(image.dtype == torch.uint8 and list(image.shape) == [480, 640, 3], f"Expected RGB uint8 HWC from {camera}")
        result["observation.images." + camera] = image.permute(2, 0, 1).contiguous().float() / 255.0
    return result


class SO101Driver:
    """Thin adapter around the repository's original BiSOFollower/SOFollower.

    Existing calibration and position mode are required. We do not run motor
    setup, calibration, PID configuration, or change normalization at runtime.
    """
    def __init__(self, config):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
        from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower import SOFollowerConfig

        robot = config["robot"]
        def arm(side):
            settings = robot[f"{side}_arm_config"]
            cameras = {}
            for name, camera in settings["cameras"].items():
                camera = dict(camera)
                kind = camera.pop("type")
                if kind == "opencv":
                    cameras[name] = OpenCVCameraConfig(**camera)
                else:
                    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
                    cameras[name] = RealSenseCameraConfig(**camera)
            return SOFollowerConfig(port=settings["port"], cameras=cameras, use_degrees=settings["use_degrees"],
                                    max_relative_target=None, disable_torque_on_disconnect=False)
        self.robot = BiSOFollower(BiSOFollowerConfig(id=robot["id"], calibration_dir=calibration_directory(robot),
                                                   left_arm_config=arm("left"), right_arm_config=arm("right")))
        require(list(self.robot.action_features) == NAMES, "Original driver action order differs")

    def connect(self):
        # Explicit bus ownership makes a partial left/right/camera failure
        # cleanable even when BiSOFollower.is_connected is still false.
        for arm in (self.robot.left_arm, self.robot.right_arm):
            arm.bus.connect()
            require(arm.bus.is_calibrated, "Stored and motor calibration differ; fix with the original setup workflow")
            require(all(int(v) == 0 for v in arm.bus.sync_read("Operating_Mode").values()), "SO101 must already be in position mode")
            positions = arm.bus.sync_read("Present_Position")
            require(set(positions) == set(MOTORS) and all(math.isfinite(float(v)) for v in positions.values()), "Invalid initial motor positions")
            # Replace any stale power-on goal BEFORE enabling torque.
            arm.bus.sync_write("Goal_Position", positions)
            arm.bus.enable_torque()
        for arm in (self.robot.left_arm, self.robot.right_arm):
            for camera in arm.cameras.values():
                camera.connect()

    def state(self):
        values = {}
        for side, arm in (("left", self.robot.left_arm), ("right", self.robot.right_arm)):
            values.update({f"{side}_{name}.pos": float(value) for name, value in arm.bus.sync_read("Present_Position").items()})
        return [values[name] for name in NAMES]

    def observe(self):
        return self.robot.get_observation()

    def send(self, vector):
        actual = self.robot.send_action(dict(zip(NAMES, vector, strict=True)))
        return [float(actual[name]) for name in NAMES]

    def hold(self):
        # Also handles partial connection without a complete observation/camera.
        failures = []
        for arm in (self.robot.left_arm, self.robot.right_arm):
            try:
                if arm.bus.is_connected and arm.bus.is_calibrated:
                    positions = arm.bus.sync_read("Present_Position")
                    require(set(positions) == set(MOTORS) and all(math.isfinite(float(v)) for v in positions.values()), "Holding state is invalid")
                    arm.bus.sync_write("Goal_Position", positions)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Holding failed: " + "; ".join(failures))

    def close(self):
        failures = []
        for arm in (self.robot.left_arm, self.robot.right_arm):
            for camera in arm.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception as exc:
                    failures.append(str(exc))
            try:
                if arm.bus.is_connected:
                    arm.bus.disconnect(disable_torque=False)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Client cleanup: " + "; ".join(failures))


def run_client(config, *, transport=None, driver_factory=SO101Driver):
    from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore

    validate_config(config, check_local_files=driver_factory is SO101Driver)
    transport = transport or GrpcHotSwitchTransport(config["address"], timeout=float(config["timeout_s"]))
    driver = core = None
    count = 0
    started = time.monotonic()
    try:
        status = transport.status()
        check_server(config, status)
        previous = config.get("previous_session_id")
        require(status.get("session_id") is None or status["session_id"] == previous,
                "Another session exists; explicitly provide your previous session id, never auto-take over")
        # Resolve SDK/configuration before claiming a server session. The
        # original driver constructors do not open serial ports or cameras.
        driver = driver_factory(config)
        session = transport.open_session(previous)
        check_server(config, session)
        print(json.dumps({"status": "RPC_SESSION_OPENED", "session_id": session["session_id"],
                          "hardware_connected": False}), flush=True)
        driver.connect()
        def validate_action(vector):
            require(len(vector) == 12, "Invalid action dimension")
            current = driver.state()
            require(len(current) == 12, "Invalid measured state dimension")
            for name, value, measured in zip(NAMES, vector, current, strict=True):
                lower, upper, delta = action_limit(config, name)
                require(math.isfinite(float(value)) and lower <= float(value) <= upper, f"Action outside limits: {name}")
                require(math.isfinite(float(measured)) and abs(float(value) - float(measured)) <= delta, f"Action jump exceeds max_delta_per_step: {name}")
        core = HotSwitchClientCore(transport, safe_hold=driver.hold, validate_action=validate_action,
                                  minimum_action_interval_s=1 / float(config["control_hz"]))
        core.status = session
        core.switch(config["task_id"])
        print(json.dumps({"status": "CONTROLLING_DUAL_SO101", "session_id": core.status["session_id"],
                          "task_id": config["task_id"], "training_units_asserted_in_client": config["training_units"]}), flush=True)
        while config["steps"] == 0 or count < config["steps"]:
            core.control_boundary()
            require(core.status["task_id"] == config["task_id"], "Unexpected task change; stopping")
            meta = check_server(config, core.status)
            if len(core.queue) <= meta["prefix_steps"]:
                driver.hold()
                require(core.request_actions(policy_observation(driver.observe())), "Inference response was discarded")
            vector = core.next_action()
            require(vector is not None, "No valid action available")
            # Recheck fresh measured positions at the actual send boundary.
            validate_action(vector)
            core.confirm_action(driver.send(vector))
            count += 1
            if count % 30 == 0:
                print(json.dumps({"status": "RUNNING", "commands_accepted": count, "elapsed_s": time.monotonic() - started}), flush=True)
    finally:
        if core is not None:
            core.stop()
        try:
            if driver is not None:
                try:
                    driver.hold()
                finally:
                    driver.close()
        finally:
            transport.close()
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hotswitch/client_so101.json")
    parser.add_argument("--address", "--remote_policy.server_address", dest="address", help="RPC server address HOST:PORT")
    parser.add_argument("--task-id")
    parser.add_argument("--previous-session-id")
    parser.add_argument("--steps", type=int, help="0 runs until Ctrl+C")
    parser.add_argument("--check", action="store_true", help="Configuration checks only; no RPC, SDK or hardware")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LEROBOT_ASYNC_LOG_TO_FILE"] = "0"
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("address", "task_id", "previous_session_id", "steps"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    try:
        validate_config(config)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(f"Client configuration: {exc}; no hardware connected")
    if args.check:
        print(json.dumps({"status": "CLIENT_CONFIG_PASSED", "mode": "dual_so101_hardware", "hardware_connected": False}))
        return
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        run_client(config)
    except KeyboardInterrupt:
        print("Stopped: held current joint goals and closed client connections.", flush=True)


if __name__ == "__main__":
    main()
