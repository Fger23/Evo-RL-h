#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pretrained import PreTrainedPolicy


def _peft_trainable_params(policy, params):
    """Keep group hyperparameters while excluding every frozen Base tensor."""
    params = list(params)
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if not expected:
        raise ValueError("PEFT optimizer has no trainable parameters")
    if params and isinstance(params[0], dict):
        filtered = []
        actual = []
        for group in params:
            kept = [parameter for parameter in group["params"] if parameter.requires_grad]
            if kept:
                filtered.append({**group, "params": kept})
                actual.extend(kept)
    else:
        filtered = [parameter for parameter in params if parameter.requires_grad]
        actual = filtered
    identifiers = [id(parameter) for parameter in actual]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
        raise ValueError("PEFT optimizer parameters must cover each trainable adapter tensor exactly once")
    return filtered, expected


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    peft_training = getattr(cfg, "peft", None) is not None or bool(getattr(policy, "peft_config", None))
    if peft_training:
        params, expected = _peft_trainable_params(policy, params)
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    optimizer = cfg.optimizer.build(params)
    if peft_training:
        actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Built PEFT optimizer contains unexpected, missing, or duplicate parameters")
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
