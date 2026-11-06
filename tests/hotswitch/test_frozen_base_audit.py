import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lerobot.utils.peft_audit import FrozenPeftBaseAudit


class ToyPeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.peft_config = {"default": "toy"}
        self.base = nn.Linear(2, 2)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(2, 2))

    def get_base_model(self):
        return self


class FrozenAuditTests(unittest.TestCase):
    def test_real_optimizer_update_adapter_only_preserves_base_hash(self):
        model = ToyPeft()
        with tempfile.TemporaryDirectory() as temp:
            audit = FrozenPeftBaseAudit(model, temp, required_groups=("other",))
            optimizer = torch.optim.SGD([model.lora_A], lr=.1)
            model.lora_A.square().sum().backward()
            optimizer.step()
            result = audit.finish(model, 1)
            self.assertEqual(result["status"], "PASSED")
            self.assertEqual(result["before"]["sha256"], result["after"]["sha256"])

    def test_base_value_change_is_persisted_as_failure(self):
        model = ToyPeft()
        with tempfile.TemporaryDirectory() as temp:
            audit = FrozenPeftBaseAudit(model, temp)
            with torch.no_grad():
                model.base.weight.add_(1)
            with self.assertRaisesRegex(AssertionError, "changed"):
                audit.finish(model, 2)
            report = json.loads((Path(temp) / "peft_base_audit.json").read_text())
            self.assertEqual(report["status"], "FAILED")

    def test_unfreezing_base_is_rejected_and_previous_attempt_is_preserved(self):
        model = ToyPeft()
        with tempfile.TemporaryDirectory() as temp:
            audit = FrozenPeftBaseAudit(model, temp)
            audit.finish(model, 1)
            previous = Path(temp) / f"peft_base_audit_{audit.audit_id}.json"
            prior_content = previous.read_bytes()
            second = FrozenPeftBaseAudit(model, temp)
            model.base.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "Unexpected trainable Base"):
                second.finish(model, 2)
            self.assertEqual(previous.read_bytes(), prior_content)

    def test_unchanged_adapter_or_nonfinite_update_is_not_reported_as_base_mutation(self):
        for nonfinite in (False, True):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as temp:
                model = ToyPeft()
                audit = FrozenPeftBaseAudit(model, temp, required_groups=("other",))
                if nonfinite:
                    with torch.no_grad():
                        model.lora_A.fill_(float("nan"))
                with self.assertRaisesRegex(AssertionError, "Joint LoRA branches"):
                    audit.finish(model, 2)
                report = json.loads((Path(temp) / "peft_base_audit.json").read_text())
                self.assertEqual(report["status"], "FAILED")
                self.assertTrue(report["frozen_base_unchanged"])

    def test_nonfinite_initial_adapter_is_rejected(self):
        model = ToyPeft()
        with torch.no_grad():
            model.lora_A.fill_(float("inf"))
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "NaN/Inf"):
            FrozenPeftBaseAudit(model, temp)
