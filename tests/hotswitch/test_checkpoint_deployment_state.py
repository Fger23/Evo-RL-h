import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lerobot.utils.train_utils import save_checkpoint


class MemoryTokenizer:
    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / "tokenizer.json").write_text(json.dumps({"vocab": "actual-in-memory-version"}))


class Config:
    type = "pi05"

    def save_pretrained(self, path):
        (path / "config.json").write_text('{"type":"pi05","use_peft":true}')


class RestoredPeft:
    peft_config = {"default": True}
    config = Config()

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text('{}')


class Preprocessor:
    steps = [SimpleNamespace(input_tokenizer=MemoryTokenizer())]

    def save_pretrained(self, path):
        (path / "policy_preprocessor.json").write_text('{}')


class CheckpointTests(unittest.TestCase):
    def test_restored_peft_without_cli_peft_preserves_config_and_actual_tokenizer(self):
        cfg = SimpleNamespace(peft=None, save_pretrained=lambda path: None)
        with tempfile.TemporaryDirectory() as temp, patch("lerobot.utils.train_utils.save_training_state"):
            save_checkpoint(Path(temp), 3, cfg, RestoredPeft(), None, preprocessor=Preprocessor())
            checkpoint = Path(temp) / "pretrained_model"
            self.assertEqual(json.loads((checkpoint / "config.json").read_text())["type"], "pi05")
            tokenizer = json.loads((checkpoint / "tokenizer/tokenizer.json").read_text())
            self.assertEqual(tokenizer["vocab"], "actual-in-memory-version")

    def test_missing_tokenizer_is_explicitly_rejected_for_pi05_peft(self):
        cfg = SimpleNamespace(peft=None, save_pretrained=lambda path: None)
        preprocessor = Preprocessor()
        preprocessor.steps = []
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("lerobot.utils.train_utils.save_training_state"),
            self.assertRaisesRegex(ValueError, "initialized tokenizer"),
        ):
            save_checkpoint(Path(temp), 3, cfg, RestoredPeft(), None, preprocessor=preprocessor)
