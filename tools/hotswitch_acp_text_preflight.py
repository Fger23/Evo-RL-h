"""CPU-only ACP prompt/tokenizer checks with real recorded observations.

Conditioning probes select positive/negative/dropout paths explicitly; they are
not stored dataset labels and do not count as ACP training or task evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/hotswitch/straighten_cash_value.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from torch.utils.data import default_collate

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import ACPConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.rl.acp_hook import build_acp_raw_batch_hook
    from lerobot.rl.acp_tags import build_acp_tagged_task

    torch.set_num_threads(4)
    dataset = LeRobotDataset("straighten_cash_registered_r2", root=settings["source_dataset_root"])
    config = PreTrainedConfig.from_pretrained(settings["base_path"])
    features = dataset_to_policy_features(dataset.meta.features)
    config.output_features = {"action": features["action"]}
    config.input_features = {k: v for k, v in features.items() if k != "action"}
    config.device = "cpu"
    config.tokenizer_name = settings["tokenizer_name"]
    processor, _ = make_pre_post_processors(config, pretrained_path=settings["base_path"],
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": config.tokenizer_name},
            "normalizer_processor": {"stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features}, "norm_map": config.normalization_mapping},
        })
    tokenizer = next(step for step in processor.steps if hasattr(step, "input_tokenizer"))
    rows = []
    indices = sorted({0, min(257589, len(dataset) - 1), min(257590, len(dataset) - 1), len(dataset) - 1})
    for index in indices:
        sample = dataset[index]
        if "Advantage:" in sample["task"]:
            raise ValueError("Recorded task already contains an ACP marker")
        for label, positive, dropout in [("positive", True, 0.0), ("negative", False, 0.0), ("dropout", True, 1.0)]:
            direct_text = sample["task"] if dropout else build_acp_tagged_task(sample["task"], positive)
            probe = default_collate([copy.deepcopy(sample)])
            probe["complementary_info.acp_indicator"] = torch.tensor([int(positive)], dtype=torch.int64)
            hook = build_acp_raw_batch_hook(ACPConfig(enable=True, indicator_dropout_prob=dropout), settings["seed"])
            training_batch = hook(probe, 0)
            if training_batch["task"] != [direct_text]:
                raise AssertionError("Training and deployment task text disagree")
            result = processor(training_batch)
            deployment_batch = default_collate([copy.deepcopy(sample)])
            deployment_batch["task"] = [direct_text]
            deployment_result = processor(deployment_batch)
            if not torch.equal(result["observation.language.tokens"], deployment_result["observation.language.tokens"]):
                raise AssertionError("Training/deployment token IDs differ for the same conditioning")
            full_prompt = result["task"][0]
            uncapped = tokenizer.input_tokenizer(full_prompt, truncation=False, padding=False)["input_ids"]
            attended = int(result["observation.language.attention_mask"].sum())
            decoded = tokenizer.input_tokenizer.batch_decode(result["observation.language.tokens"], skip_special_tokens=True)[0]
            expected_marker = None if dropout else f"Advantage: {'positive' if positive else 'negative'}"
            marker_count = decoded.count("Advantage:")
            if marker_count != (0 if dropout else 1) or (expected_marker and expected_marker not in decoded):
                raise AssertionError("ACP marker was duplicated, changed or truncated")
            if len(uncapped) > config.tokenizer_max_length or attended != len(uncapped):
                raise AssertionError("Actual task/state prompt was truncated")
            rows.append({"index": index, "conditioning_probe": label, "uncapped_tokens": len(uncapped),
                "attended_tokens": attended, "max_length": config.tokenizer_max_length,
                "marker_count": marker_count, "decoded_prompt": decoded})
    hook = build_acp_raw_batch_hook(ACPConfig(enable=True,
        indicator_dropout_prob=settings["acp_indicator_dropout_prob"]), settings["seed"])
    tasks = [dataset[0]["task"]] * 32
    probes = hook({"task": tasks.copy(), "complementary_info.acp_indicator": torch.tensor([i % 2 for i in range(32)], dtype=torch.int64)}, 0)
    tagged = sum("Advantage:" in text for text in probes["task"])
    report = {"status": "REAL_CPU_ACP_TEXT_PREFLIGHT_PASSED", "gpu_used": False,
        "scope": "Processor conditioning probes only; no dataset labels generated or training performed",
        "dataset_root": settings["source_dataset_root"], "tokenizer_name": config.tokenizer_name,
        "max_length": config.tokenizer_max_length, "rows": rows,
        "training_dropout_probe": {"configured_probability": settings["acp_indicator_dropout_prob"],
            "seed": settings["seed"], "samples": 32, "tagged": tagged, "untagged": 32 - tagged},
        "training_and_deployment_positive_tokens_match": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(json.dumps({"probes": len(rows), "max_uncapped_tokens": max(r["uncapped_tokens"] for r in rows)}))


if __name__ == "__main__":
    main()
