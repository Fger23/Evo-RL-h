"""Reproduce the real-weight value preflight without exposing a GPU or robot."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/hotswitch/straighten_cash_value.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from torch.utils.data import default_collate

    import lerobot
    from lerobot.configs.value_train import ValueTargetsConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config

    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    started = time.perf_counter()
    dataset = LeRobotDataset("straighten_cash_registered_r2", root=config["source_dataset_root"])
    model_config = Pistar06Config(device="cpu", dtype="float32", push_to_hub=False,
        vision_repo_id=config["vision_repo_id"], language_repo_id=config["language_repo_id"])
    policy = make_policy(model_config, ds_meta=dataset.meta)
    hook = policy.build_training_raw_batch_hook(dataset, ValueTargetsConfig())
    preprocessor, _ = make_pre_post_processors(model_config, dataset_stats=dataset.meta.stats)
    batch = hook(default_collate([dataset[0]]), 0)
    target = float(batch["observation.value_target"][0])
    batch = preprocessor(batch)
    policy.eval()
    with torch.no_grad():
        loss, _ = policy.forward(batch)
        value = policy.predict_value(batch)
    if not torch.isfinite(loss) or not torch.isfinite(value).all():
        raise ValueError("Real CPU value preflight produced nonfinite output")
    report = {"status": "REAL_CPU_VALUE_PREFLIGHT_PASSED", "elapsed_s": time.perf_counter() - started,
        "loss": float(loss), "predicted_value": float(value[0]), "real_target": target,
        "cameras": model_config.image_features, "source": str(Path(lerobot.__file__).resolve()), "gpu_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
