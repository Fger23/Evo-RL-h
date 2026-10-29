"""CPU fakes exercise resident backend without downloading or connecting hardware."""

import unittest
from types import SimpleNamespace

import torch

from lerobot.async_inference.hotswitch_backend import ResidentBackend


class FakeProcessor:
    steps = []

    def __init__(self, offset=0):
        self.offset, self.resets = offset, 0

    def __call__(self, value):
        return value + self.offset if isinstance(value, torch.Tensor) else value

    def reset(self):
        self.resets += 1


class FakeModel:
    def __init__(self):
        self.config = None
        self.model = SimpleNamespace(config=None)
        self.task = None
        self.resets = 0

    def set_adapter(self, name):
        self.task = name

    def get_base_model(self):
        return self

    def eval(self):
        return self

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, prefix_steps=0, action_prefix=None):
        assert batch["task"] == self.task
        result = torch.full((1, 4, 1), 1.0 if self.task == "A" else 2.0)
        if prefix_steps:
            result[:, :prefix_steps] = action_prefix
        return result


class BackendTests(unittest.TestCase):
    def test_actual_warmup_alignment_normalization_and_a_b_a(self):
        # validate_task_pair is separately tested against on-disk bundles.
        from unittest.mock import patch

        bundles, processors = [], {}
        for name, offset in [("A", 10), ("B", 20)]:
            manifest = {
                "task_text": name,
                "action_schema": {"names": ["joint"]},
                "rtc": {"mode": "training_time", "prefix_steps": 2},
                "camera_mapping": {},
                "acp": {"enabled": False},
            }
            bundles.append(SimpleNamespace(task_id=name, manifest=manifest))
            processors[name] = (
                FakeProcessor(),
                FakeProcessor(offset),
                SimpleNamespace(chunk_size=4, rtc_training_max_delay=2),
            )
        with patch("lerobot.async_inference.hotswitch_backend.validate_task_pair"):
            backend = ResidentBackend(
                bundles, {"shared": FakeModel()}, processors, load_counts={"base": 1, "adapters": 2}
            )
        self.assertFalse(backend.ready("A"))
        observations = {t: {"observation.state": torch.zeros(1)} for t in ("A", "B")}
        backend.warmup(observations)
        self.assertTrue(all(backend.ready(t) for t in ("A", "B")))
        outputs = []
        for task in ("A", "B", "A"):
            backend.activate(task)
            raw, processed = backend.predict(task, observations[task])
            outputs.append(raw)
            self.assertTrue(torch.equal(processed - raw, torch.full_like(raw, 10 if task == "A" else 20)))
            p_raw, p_processed = backend.predict(task, observations[task], 2, raw[:2])
            self.assertTrue(torch.equal(p_raw[:2], raw[:2]))
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertEqual(backend.load_counts, {"base": 1, "adapters": 2})
        with self.assertRaisesRegex(ValueError, "training range"):
            backend.predict("A", observations["A"], 3, raw[:3])


if __name__ == "__main__":
    unittest.main()
