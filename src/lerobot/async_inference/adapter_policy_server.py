"""PI05 Adapter entry for the original Evo-RL AsyncInference service.

Only model construction differs. RPCs, processors, action chunks and clients are
the upstream implementations. Weights are loaded on SendPolicyInstructions,
just as in the original server.
"""

import json
import os
import subprocess
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import draccus
import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.pi05.adapter_inference import check_adapter_files, load_adapter_policy
from lerobot.transport import services_pb2_grpc

FORMAL_ADAPTER = (
    "/nfs/baiyuntian/assets/model/straighten_cash_joint_adapter_formal_20261104_r4_node3_gpu1"
    "/postrun_r1/bundles/A"
)
AUTHORIZED_GPU_UUID = "GPU-4e74582b-8cca-a9c3-8d95-73f7babc3c2a"
GPU_LOCK = f"/nfs/lerobot/s101/hot_switch/20261103/locks/{AUTHORIZED_GPU_UUID}.lock"


@dataclass
class AdapterPolicyServerConfig(PolicyServerConfig):
    host: str = "10.61.5.1"
    port: int = 8100
    adapter_path: str = FORMAL_ADAPTER
    base_path: str | None = None
    gpu: int = 1
    gpu_uuid: str = AUTHORIZED_GPU_UUID
    check: bool = False


def select_gpu(config: AdapterPolicyServerConfig) -> None:
    """Select physical GPU before any CUDA context/model allocation."""
    if config.gpu != 1 or config.gpu_uuid != AUTHORIZED_GPU_UUID:
        raise ValueError("This deployment is authorized only for physical GPU1 and its recorded UUID")
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU selection must precede CUDA initialization")
    result = subprocess.run(
        ["nvidia-smi", f"--id={config.gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip() != config.gpu_uuid:
        raise ValueError("Physical GPU UUID differs from the authorized GPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_uuid


class AdapterPolicyServer(PolicyServer):
    def _preprocessor_overrides(self, policy_specs):
        overrides = super()._preprocessor_overrides(policy_specs)
        overrides["tokenizer_processor"] = {
            "tokenizer_name": str(Path(self.config.adapter_path).resolve() / "tokenizer")
        }
        return overrides

    def _load_policy(self, policy_specs):
        if policy_specs.policy_type != "pi05":
            raise ValueError("The Adapter entry accepts policy_type=pi05 only")
        if Path(policy_specs.pretrained_name_or_path).resolve() != Path(self.config.adapter_path).resolve():
            raise ValueError("Client pretrained_name_or_path must match server adapter_path")
        if policy_specs.device != "cuda:0":
            raise ValueError("Use policy_device=cuda:0 (the selected physical GPU1)")
        return load_adapter_policy(
            self.config.adapter_path, device=policy_specs.device, base_override=self.config.base_path
        )


@draccus.wrap()
def serve(config: AdapterPolicyServerConfig):
    select_gpu(config)
    audit = check_adapter_files(config.adapter_path, base_override=config.base_path)
    print(
        json.dumps(
            {
                "status": "CHECK_PASSED",
                "model_loaded": False,
                **{
                    key: audit[key]
                    for key in (
                        "adapter_path",
                        "base_path",
                        "target_count",
                        "tensor_count",
                        "manifest_sha256",
                    )
                },
            }
        ),
        flush=True,
    )
    if config.check:
        return

    # Coordinate with the existing GPU1 jobs; do not allocate a dummy reservation.
    import fcntl

    with open(GPU_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["nvidia-smi", f"--id={config.gpu_uuid}", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            raise RuntimeError("GPU1 has an existing compute process; it was left untouched")

        # Same service type, worker count and transport as upstream policy_server.
        policy_server = AdapterPolicyServer(config)
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            server = grpc.server(executor)
            services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
            if server.add_insecure_port(f"{config.host}:{config.port}") != config.port:
                raise RuntimeError("Failed to bind requested port")
            server.start()
            print(
                json.dumps(
                    {
                        "status": "LISTENING",
                        "address": f"{config.host}:{config.port}",
                        "model_loaded": False,
                        "load_trigger": "SendPolicyInstructions",
                    }
                ),
                flush=True,
            )
            try:
                server.wait_for_termination()
            except KeyboardInterrupt:
                pass
            finally:
                policy_server.stop()
                server.stop(grace=2).wait()


if __name__ == "__main__":
    serve()
