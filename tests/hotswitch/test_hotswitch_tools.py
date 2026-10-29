"""Operator validation and actual dataclass CLI parsing, entirely offline."""

import json

import draccus
import pytest

from lerobot.async_inference.hotswitch_tools import train_task, write_switch_request
from lerobot.async_inference.task_bundle import joint_target_pattern
from lerobot.configs.default import PeftConfig
from lerobot.scripts.lerobot_hotswitch import main


def test_alpha_and_dropout_cli_parse():
    config = draccus.parse(PeftConfig, args=["--r=32", "--lora_alpha=64", "--lora_dropout=0"])
    assert (config.r, config.lora_alpha, config.lora_dropout) == (32, 64, 0)
    assert PeftConfig().lora_alpha is None
    assert PeftConfig().lora_dropout is None


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"lora_alpha": 0}, {"lora_dropout": 1}, {"lora_dropout": -0.1}])
def test_invalid_adapter_arguments_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        PeftConfig(**kwargs)


def test_local_mailbox_has_only_task_and_operation(tmp_path):
    target = tmp_path / "switch.json"
    result = write_switch_request(target, "B", "same-operation")
    assert json.loads(target.read_text()) == result == {"task_id": "B", "operation_id": "same-operation"}
    with pytest.raises(ValueError):
        write_switch_request(target, "../untrusted/model")
    assert json.loads(target.read_text()) == result


def test_incomplete_training_plan_does_not_start_trainer(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"common_base": None}))
    with pytest.raises(ValueError, match="common_base"):
        train_task(plan, "A", execute=True)


def test_help_is_available_without_model_load(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for command in (
        "env_check",
        "audit_targets",
        "train_task",
        "export_task",
        "serve",
        "switch_task",
        "benchmark",
        "smoke",
    ):
        assert command in text


def test_launcher_rejects_mismatched_task_structure_and_upload_defaults(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"fixture content for fingerprint only")
    dataset = tmp_path / "dataset" / "meta"
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text('{"features": {}}')
    plan = {"common_base": str(base), "tasks": {}}
    configs = []
    for task in ("A", "B"):
        config = {
            "policy": {"type": "pi05", "pretrained_path": str(base), "push_to_hub": False},
            "peft": {"r": 32 if task == "A" else 16, "target_modules": joint_target_pattern()},
            "dataset": {"repo_id": "local/test", "root": str(dataset.parent)},
            "output_dir": str(tmp_path / task),
            "steps": 3,
        }
        config_path = tmp_path / f"{task}.json"
        config_path.write_text(json.dumps(config))
        configs.append((config_path, config))
        plan["tasks"][task] = {"config_path": config_path.name}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="same LoRA structure"):
        train_task(plan_path, "A")
    del configs[0][1]["policy"]["push_to_hub"]
    configs[0][0].write_text(json.dumps(configs[0][1]))
    with pytest.raises(ValueError, match="push_to_hub=false"):
        train_task(plan_path, "A")
