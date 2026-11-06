import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend


class Preprocessor:
    def __init__(self, decoded):
        self.steps = [SimpleNamespace(input_tokenizer=SimpleNamespace(batch_decode=lambda tokens: [decoded]))]

    def __call__(self, observation):
        return {"observation.language.tokens": []}


class AcpConditionTests(unittest.TestCase):
    def backend(self, decoded):
        backend = SharedAdapterBackend.__new__(SharedAdapterBackend)
        backend.active_task = "A"
        backend.device = "cpu"
        backend._dual = False
        backend.manifests = {"A": {"task_text": "straighten cash", "acp": {"enabled": True}}}
        predict = Mock(return_value=torch.ones(1, 4, 2))
        backend.models = {"shared": SimpleNamespace(predict_action_chunk=predict)}
        backend.processors = {"A": (Preprocessor(decoded), lambda action: action)}
        return backend, predict

    def test_missing_duplicated_or_contradictory_marker_never_reaches_inference(self):
        for decoded in ("Task only", "Advantage: positive Advantage: positive",
                        "Advantage: positive Advantage: negative"):
            with self.subTest(decoded=decoded):
                backend, predict = self.backend(decoded)
                with self.assertRaisesRegex(ValueError, "lost, duplicated, or contradicted"):
                    backend.infer("A", {})
                predict.assert_not_called()

    def test_single_positive_marker_reaches_policy_and_keeps_raw_processed_aligned(self):
        backend, predict = self.backend("Task: cash Advantage: positive, State: 128")
        raw, processed = backend.infer("A", {})
        predict.assert_called_once()
        self.assertEqual(tuple(raw.shape), (4, 2))
        self.assertTrue(torch.equal(raw, processed))
