"""Tokenizer export never brings full-model weights or external file references."""

import json
import tempfile
import unittest
from pathlib import Path

from lerobot.async_inference.hotswitch.manifest import copy_tokenizer_assets


class TokenizerAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.destination = self.root / "minimal"
        self.config = self.source / "tokenizer_config.json"
        self.config.write_text('{}', encoding="utf-8")
        (self.source / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")

    def symlink(self, path, target, directory=False):
        try:
            path.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            self.skipTest(f"Host cannot create symlinks: {error}")

    def test_copy_preserves_asset_bytes_and_excludes_weights_indexes_and_model_metadata(self):
        retained = {"tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
                    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja", "chat_templates/tool.jinja"}
        ignored = {"model-00001-of-00003.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                   "generation_config.json", "config.json", "README.md", "custom_tokenizer.py"}
        for name in retained | ignored:
            path = self.source / name
            path.parent.mkdir(exist_ok=True)
            if name not in ("tokenizer_config.json", "tokenizer.json"):
                path.write_bytes(("unaltered-asset-" + name).encode())
        copied = copy_tokenizer_assets(self.source, self.destination)
        actual = {path.relative_to(self.destination).as_posix() for path in self.destination.rglob('*') if path.is_file()}
        self.assertEqual(set(copied), retained)
        self.assertEqual(actual, retained)
        for name in retained:
            self.assertEqual((self.source / name).read_bytes(), (self.destination / name).read_bytes())

    def test_config_and_supported_default_data_are_required(self):
        for missing in ("tokenizer_config.json", "tokenizer.json"):
            with self.subTest(missing=missing):
                source = self.root / missing
                source.mkdir()
                for name in ("tokenizer_config.json", "tokenizer.json"):
                    if name != missing:
                        (source / name).write_text('{}')
                with self.assertRaises(ValueError):
                    copy_tokenizer_assets(source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_versioned_assets_are_retained_but_do_not_replace_a_default(self):
        name = "tokenizer.0.13.3.json"
        (self.source / name).write_text('{"versioned": true}')
        self.config.write_text(json.dumps({"fast_tokenizer_files": [name]}))
        copy_tokenizer_assets(self.source, self.destination)
        self.assertEqual((self.source / name).read_bytes(), (self.destination / name).read_bytes())
        (self.source / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "no supported tokenizer data"):
            copy_tokenizer_assets(self.source, self.root / "version-only")

    def test_explicit_missing_absolute_parent_or_backslash_reference_is_rejected(self):
        for value in ("../tokenizer.json", "/outside/tokenizer.json", "C:\\outside\\tokenizer.json",
                      "tokenizer.99.json", "subdir/tokenizer.json"):
            with self.subTest(value=value):
                self.config.write_text(json.dumps({"tokenizer_file": value}))
                with self.assertRaisesRegex(ValueError, "asset reference"):
                    copy_tokenizer_assets(self.source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_custom_code_invalid_lists_and_empty_assets_are_rejected(self):
        for config in ({"auto_map": {"AutoTokenizer": "custom.Tokenizer"}},
                       {"fast_tokenizer_files": "tokenizer.json"}, {"fast_tokenizer_files": [None]},
                       {"chat_template_files": ["../template.jinja"]}):
            self.config.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                copy_tokenizer_assets(self.source, self.destination)
        self.config.write_text('{}')
        (self.source / "tokenizer.json").write_bytes(b'')
        with self.assertRaisesRegex(ValueError, "empty"):
            copy_tokenizer_assets(self.source, self.destination)

    def test_source_root_symlink_is_rejected(self):
        alias = self.root / "source-link"
        self.symlink(alias, self.source, directory=True)
        with self.assertRaisesRegex(ValueError, "directory symlink"):
            copy_tokenizer_assets(alias, self.destination)

    def test_selected_file_symlink_is_rejected(self):
        outside = self.root / "outside.json"
        outside.write_text('{}')
        (self.source / "tokenizer.json").unlink()
        self.symlink(self.source / "tokenizer.json", outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            copy_tokenizer_assets(self.source, self.destination)

    def test_chat_template_directory_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.symlink(self.source / "chat_templates", outside, directory=True)
        with self.assertRaisesRegex(ValueError, "directory symlink"):
            copy_tokenizer_assets(self.source, self.destination)

    def test_destination_cannot_overwrite_or_modify_source(self):
        with self.assertRaises(FileExistsError):
            copy_tokenizer_assets(self.source, self.source)
        with self.assertRaisesRegex(ValueError, "must not modify"):
            copy_tokenizer_assets(self.source, self.source / "nested-export")


if __name__ == "__main__":
    unittest.main()
