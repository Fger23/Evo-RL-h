"""CPU optimizer/checkpoint resume coverage; no model download or CUDA API calls."""

import ast
import copy
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from lerobot.utils.peft_audit import FrozenPeftBaseAudit, validate_audit_coverage


class ToyPeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.peft_config = {"default": "cpu-test-only"}
        self.base = nn.Linear(2, 2)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(2, 2))

    def get_base_model(self):
        return self


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model.lora_A.square().sum().backward()
    optimizer.step()


def save_cpu_checkpoint(model, optimizer, checkpoint, step):
    """Persist actual CPU tensors/state and the trainer's real step-file layout."""
    (checkpoint / "pretrained_model").mkdir(parents=True, exist_ok=True)
    (checkpoint / "training_state").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint / "pretrained_model/cpu_model.pt")
    torch.save(optimizer.state_dict(), checkpoint / "training_state/cpu_optimizer.pt")
    # Small artifacts stand in for real PEFT/tokenizer/processor files; the model
    # and optimizer tensors above are genuinely serialized and restored on CPU.
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "config.json",
        "train_config.json",
        "policy_preprocessor.json",
        "processor_state.safetensors",
        "tokenizer/tokenizer.json",
    ):
        path = checkpoint / "pretrained_model" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU_FIXTURE_ONLY")
    (checkpoint / "training_state/training_step.json").write_text(json.dumps({"step": step}))


def first_segment(root):
    model = ToyPeft()
    optimizer = torch.optim.SGD([model.lora_A], lr=0.1, momentum=0.9)
    audit = FrozenPeftBaseAudit(model, root, required_groups=("other",))
    for _ in range(2):
        update(model, optimizer)
    checkpoint = root / "checkpoints/000002"
    save_cpu_checkpoint(model, optimizer, checkpoint, 2)
    audit.checkpoint(model, 2, checkpoint)
    return model, optimizer, audit, checkpoint


class ResumeCoverageTests(unittest.TestCase):
    def test_saved_model_config_tokenizer_processor_or_optimizer_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _, audit, checkpoint = first_segment(root)
            audit.finish(model, 2)
            final = root / "peft_base_audit.json"
            validate_audit_coverage(final, 2, require_final=True)
            other = root / "checkpoints/copied-but-not-selected"
            shutil.copytree(checkpoint, other)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                validate_audit_coverage(final, 2, require_final=True, expected_checkpoint=other)
            with self.assertRaisesRegex(ValueError, "declared saved directory"):
                validate_audit_coverage(other / "training_state/peft_base_audit.json", 2)
            for relative in (
                "pretrained_model/adapter_model.safetensors",
                "pretrained_model/config.json",
                "pretrained_model/train_config.json",
                "pretrained_model/processor_state.safetensors",
                "pretrained_model/tokenizer/tokenizer.json",
                "training_state/cpu_optimizer.pt",
            ):
                with self.subTest(relative=relative):
                    path = checkpoint / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"ALTERED_AFTER_SAVE")
                    with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                        validate_audit_coverage(final, 2, require_final=True)
                    with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                        FrozenPeftBaseAudit(
                            model,
                            root,
                            required_groups=("other",),
                            start_step=2,
                            previous_report=checkpoint / "training_state/peft_base_audit.json",
                        )
                    path.write_bytes(original)
            missing = checkpoint / "pretrained_model/config.json"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                validate_audit_coverage(final, 2, require_final=True)

    def test_uncheckpointed_parameter_only_audit_cannot_be_formal_evidence(self):
        for strict in (False, True):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as directory:
                model = ToyPeft()
                optimizer = torch.optim.SGD([model.lora_A], lr=0.1)
                audit = FrozenPeftBaseAudit(
                    model, directory, required_groups=("other",), require_checkpoint_binding=strict
                )
                update(model, optimizer)
                if strict:
                    with self.assertRaisesRegex(ValueError, "actually saved"):
                        audit.finish(model, 1)
                else:
                    result = audit.finish(model, 1)
                    self.assertEqual(result["checkpoint_binding_status"], "NOT_CHECKPOINTED")
                    with self.assertRaisesRegex(ValueError, "not bound"):
                        validate_audit_coverage(result, 1, require_final=True)

    def test_interrupted_checkpoint_resume_then_two_more_updates_cover_the_whole_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, first_audit, checkpoint = first_segment(root)
            previous = checkpoint / "training_state/peft_base_audit.json"
            checkpoint_evidence = previous.read_bytes()
            self.assertEqual(json.loads(checkpoint_evidence)["status"], "CHECKPOINT_PASSED")
            # Simulate interruption here: no successful finish() for attempt one.
            model = ToyPeft()
            model.load_state_dict(torch.load(checkpoint / "pretrained_model/cpu_model.pt", weights_only=True))
            optimizer = torch.optim.SGD([model.lora_A], lr=0.1, momentum=0.9)
            optimizer.load_state_dict(
                torch.load(checkpoint / "training_state/cpu_optimizer.pt", weights_only=True)
            )
            start = json.loads((checkpoint / "training_state/training_step.json").read_text())["step"]
            resumed = FrozenPeftBaseAudit(
                model, root, required_groups=("other",), start_step=start, previous_report=previous
            )
            for _ in range(2):
                update(model, optimizer)
            final_checkpoint = root / "checkpoints/000004"
            save_cpu_checkpoint(model, optimizer, final_checkpoint, 4)
            resumed.checkpoint(model, 4, final_checkpoint)
            final = resumed.finish(model, 4)
            self.assertEqual(final["status"], "PASSED")
            self.assertEqual(final["start_step"], 2)
            self.assertEqual(final["updates_this_attempt"], 2)
            self.assertEqual(
                [(part["start_step"], part["end_step"]) for part in final["coverage"]], [(0, 2), (2, 4)]
            )
            self.assertEqual(final["before"]["sha256"], first_audit.before["sha256"])
            self.assertEqual(previous.read_bytes(), checkpoint_evidence)
            summary = validate_audit_coverage(
                root / "peft_base_audit.json", 4, required_groups=("other",), require_final=True
            )
            self.assertEqual(summary["start_step"], 0)
            self.assertEqual(summary["end_step"], 4)
            self.assertEqual(summary["segments"], 2)

    def test_resume_rejects_missing_started_gap_and_wrong_step_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _, _, checkpoint = first_segment(root)
            previous = checkpoint / "training_state/peft_base_audit.json"
            original = previous.read_bytes()
            with self.assertRaisesRegex(ValueError, "requires"):
                FrozenPeftBaseAudit(model, root, required_groups=("other",), start_step=2)
            for mutation in ("STARTED", "GAP", "WRONG_STEP", "FALSE_UPDATE"):
                with self.subTest(mutation=mutation):
                    record = json.loads(original)
                    if mutation == "STARTED":
                        record["status"] = "STARTED"
                    elif mutation == "GAP":
                        record["coverage"][0]["start_step"] = 1
                    elif mutation == "WRONG_STEP":
                        record["steps_completed"] = 3
                    else:
                        record["coverage"][0]["adapter_groups_updated"]["other"] = False
                    previous.write_text(json.dumps(record))
                    with self.assertRaises(ValueError):
                        FrozenPeftBaseAudit(
                            model, root, required_groups=("other",), start_step=2, previous_report=previous
                        )
            previous.write_bytes(original)

    def test_restored_parameter_or_referenced_history_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer, _, checkpoint = first_segment(root)
            previous = checkpoint / "training_state/peft_base_audit.json"
            altered = copy.deepcopy(model)
            with torch.no_grad():
                altered.lora_A.add_(1)
            with self.assertRaisesRegex(ValueError, "Loaded Base/adapter"):
                FrozenPeftBaseAudit(
                    altered, root, required_groups=("other",), start_step=2, previous_report=previous
                )
            resumed = FrozenPeftBaseAudit(
                model, root, required_groups=("other",), start_step=2, previous_report=previous
            )
            update(model, optimizer)
            final_checkpoint = root / "checkpoints/000003"
            save_cpu_checkpoint(model, optimizer, final_checkpoint, 3)
            resumed.checkpoint(model, 3, final_checkpoint)
            resumed.finish(model, 3)
            previous.write_bytes(previous.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "content hash changed"):
                validate_audit_coverage(root / "peft_base_audit.json", 3, require_final=True)

    def test_empty_resume_attempt_cannot_claim_all_prior_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _, _, checkpoint = first_segment(root)
            resumed = FrozenPeftBaseAudit(
                model,
                root,
                required_groups=("other",),
                start_step=2,
                previous_report=checkpoint / "training_state/peft_base_audit.json",
            )
            with self.assertRaisesRegex(AssertionError, "No new training updates"):
                resumed.finish(model, 2)
            report = json.loads((root / "peft_base_audit.json").read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertFalse(report["coverage_complete"])

    def test_frozen_and_lora_scalar_parameters_hash_without_dtype_view_error(self):
        model = ToyPeft()
        model.frozen_scalar = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        model.lora_scalar = nn.Parameter(torch.tensor(1.0))
        with tempfile.TemporaryDirectory() as directory:
            audit = FrozenPeftBaseAudit(model, directory, required_groups=("other",))
            with torch.no_grad():
                model.lora_scalar.add_(0.5)
            optimizer = torch.optim.SGD([model.lora_A, model.lora_scalar], lr=0.1)
            checkpoint = Path(directory) / "checkpoints/000001"
            save_cpu_checkpoint(model, optimizer, checkpoint, 1)
            audit.checkpoint(model, 1, checkpoint)
            result = audit.finish(model, 1)
            self.assertTrue(validate_audit_coverage(result, 1, require_final=True)["coverage_complete"])

    def test_actual_trainer_save_branch_publishes_last_only_after_passed_checkpoint_audit(self):
        source = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and any(
                isinstance(value, ast.Attribute) and value.attr == "save_checkpoint"
                for value in node.test.values
            )
        )
        block = compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec")
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = ToyPeft()
                optimizer = torch.optim.SGD([model.lora_A], lr=0.1)
                audit = FrozenPeftBaseAudit(model, root, required_groups=("other",))
                if changed:
                    update(model, optimizer)
                checkpoint = root / "checkpoints/000001"
                published = Mock()

                def save(**kwargs):
                    save_cpu_checkpoint(
                        kwargs["policy"], kwargs["optimizer"], kwargs["checkpoint_dir"], kwargs["step"]
                    )

                def publish(path, observer=published):
                    self.assertEqual(
                        json.loads((path / "training_state/peft_base_audit.json").read_text())["status"],
                        "CHECKPOINT_PASSED",
                    )
                    observer(path)

                namespace = {
                    "cfg": SimpleNamespace(save_checkpoint=True, output_dir=root, steps=1),
                    "step": 1,
                    "is_saving_step": True,
                    "is_main_process": True,
                    "logging": logging,
                    "get_step_checkpoint_dir": lambda *args, target=checkpoint: target,
                    "save_checkpoint": save,
                    "frozen_base_audit": audit,
                    "accelerator": SimpleNamespace(
                        unwrap_model=lambda value: value, wait_for_everyone=lambda: None
                    ),
                    "policy": model,
                    "optimizer": optimizer,
                    "lr_scheduler": None,
                    "preprocessor": None,
                    "postprocessor": None,
                    "update_last_checkpoint": publish,
                    "wandb_logger": None,
                }
                if changed:
                    exec(block, namespace)
                    published.assert_called_once_with(checkpoint)
                else:
                    with self.assertRaisesRegex(AssertionError, "Joint LoRA branches"):
                        exec(block, namespace)
                    published.assert_not_called()
                    self.assertFalse((checkpoint / "training_state/peft_base_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
