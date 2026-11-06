"""Read-only provenance collection. Never imports or constructs robot drivers."""

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


def collect_environment(repo=None, include_torch=False):
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    spec = importlib.util.find_spec("lerobot")
    report["lerobot_import"] = spec.origin if spec else None
    report["packages"] = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "grpcio", "protobuf"):
        try:
            dist = importlib.metadata.distribution(name)
            direct = dist.read_text("direct_url.json")
            report["packages"][name] = {"version": dist.version,
                "direct_url": json.loads(direct) if direct else None}
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if repo:
        provenance = Path(repo) / "source_provenance.json"
        if provenance.is_file():
            source = json.loads(provenance.read_text(encoding="utf-8"))
            report["source_provenance"] = {k: source[k] for k in ("repository", "head", "source_tree_sha256")}
            report["source_provenance"]["file"] = str(provenance)
        for key, args in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--short"]),
                          ("remotes", ["remote", "-v"])):
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
            report[key] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = str(error)
    if include_torch:
        import inspect

        import torch

        report["torch_runtime"] = {"cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        try:
            from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

            report["gemma_rmsnorm_forward"] = str(inspect.signature(GemmaRMSNorm.forward))
        except (ImportError, AttributeError) as error:
            report["gemma_rmsnorm_error"] = str(error)
    return report


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
