import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from lerobot.scripts.lerobot_train import update_policy


class NanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0 + 1

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


class TinyPeft(nn.Module):
    def __init__(self, nan_loss):
        super().__init__()
        self.peft_config = {"default": "toy"}
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.nan_loss = nan_loss

    def forward(self, batch):
        loss = self.weight * float("nan") if self.nan_loss else NanGradient.apply(self.weight)
        return loss, {}


class FakeAccelerator:
    def autocast(self):
        return nullcontext()

    def unwrap_model(self, model, **kwargs):
        return model

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, maximum):
        return torch.nn.utils.clip_grad_norm_(parameters, maximum)


class FiniteGuardTests(unittest.TestCase):
    def test_nonfinite_loss_or_gradient_cannot_reach_optimizer_step(self):
        for nan_loss in (False, True):
            with self.subTest(nan_loss=nan_loss):
                model = TinyPeft(nan_loss)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                before = model.weight.detach().clone()
                with self.assertRaisesRegex(FloatingPointError, "Nonfinite PEFT"):
                    update_policy(SimpleNamespace(), model, {}, optimizer, 1.0, FakeAccelerator())
                self.assertTrue(torch.equal(before, model.weight))
                self.assertIsNone(model.weight.grad)
