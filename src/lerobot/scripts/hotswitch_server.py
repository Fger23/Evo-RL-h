"""Load the resident model in the foreground, with no per-start directory."""
import argparse
from contextlib import contextmanager
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

NODE3_UUID = "GPU-4e74582b-8cca-a9c3-8d95-73f7babc3c2a"
ROOT = Path(__file__).resolve().parents[3]


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hotswitch/server.json")
    parser.add_argument("--registry")
    parser.add_argument("--gpu", help="Physical GPU index or UUID, before CUDA visibility remapping")
    parser.add_argument("--host", help="Listen address; node profile uses its private LAN address")
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend", choices=("shared", "dual-unmerged"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--check", action="store_true", help="Check configuration/UUID only, without loading a model")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("registry", "gpu", "host", "port", "backend", "dtype"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    ipaddress.ip_address(config["host"])
    if not isinstance(config["port"], int) or isinstance(config["port"], bool) or not 0 <= config["port"] <= 65535:
        raise ValueError("port must be 0..65535")
    if config["backend"] not in ("shared", "dual-unmerged") or config["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Unsupported backend or precision")
    registry = Path(config["registry"]).expanduser().resolve(strict=True)
    config["registry"] = str(registry)
    # Preserve this chat's explicit GPU authorization on the target host.
    if socket.gethostname() == "uucloud-bare-h200-node3":
        if str(config["gpu"]) not in ("1", NODE3_UUID) or config["allowed_gpu_uuid"] != NODE3_UUID:
            raise ValueError("This node is authorized for physical GPU1 only")
    return args, config


def selected_uuid(config, runner=subprocess.run):
    result = runner(["nvidia-smi", "-i", str(config["gpu"]), "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True, timeout=15)
    uuid = result.stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid or uuid != config["allowed_gpu_uuid"]:
        raise ValueError("Selected GPU UUID does not match the configured authorized GPU")
    return uuid


@contextmanager
def gpu_lock(config, uuid, runner=subprocess.run):
    import fcntl

    path = Path(config["lock_root"]) / (uuid + ".lock")
    # Use the existing canonical lock directory; no new run or log directory.
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinated job owns this GPU") from exc
        try:
            result = runner(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
            if result.stdout.strip():
                raise RuntimeError("Selected GPU already has a compute process; nothing was stopped")
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    args, config = options(argv)
    uuid = selected_uuid(config)
    if args.check:
        print(json.dumps({"status": "CHECK_PASSED", "gpu_uuid": uuid, "configuration": config,
                          "model_loaded": False, "run_directory_created": False}))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["LEROBOT_ASYNC_LOG_TO_FILE"] = "0"
    with gpu_lock(config, uuid):
        from lerobot.scripts.hotswitch_cli import serve
        print(json.dumps({"status": "LOADING", "gpu_uuid": uuid, "listen": f"{config['host']}:{config['port']}"}), flush=True)
        serve(SimpleNamespace(registry=config["registry"], host=config["host"], port=config["port"],
                              backend=config["backend"], dtype=config["dtype"], device="cuda", ready_file=None))


if __name__ == "__main__":
    main()
