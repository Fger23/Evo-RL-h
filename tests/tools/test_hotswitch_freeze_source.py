"""CPU-only frozen-source proof checks; no training/GPU/model import occurs."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"freeze_review_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    tool = load_tool("hotswitch_freeze_source")
    source, run = tmp_path / "source", tmp_path / "run"
    files = {
        "pyproject.toml": '[project]\nname="cpu-frozen-fixture"\nversion="0.0.0"\n',
        "setup.py": "# CPU fixture\n",
        "MANIFEST.in": "recursive-include src *.py\n",
        "README.md": "# CPU source fixture\n",
        "LICENSE": "CPU fixture only\n",
        "source_provenance.json": json.dumps(
            {"repository": "https://github.com/Fger23/Evo-RL", "head": "a" * 40}
        ),
        # find_spec must inspect this without executing the package body.
        "src/lerobot/__init__.py": "raise RuntimeError('MODEL_PACKAGE_MUST_NOT_BE_IMPORTED')\n",
        "src/lerobot/scripts/lerobot_train.py": "# trainer CPU fixture\n",
        "src/lerobot/scripts/lerobot_train_trc.py": "# trc CPU fixture\n",
        "src/lerobot/scripts/lerobot_value_train.py": "# value CPU fixture\n",
        "src/lerobot/scripts/lerobot_value_infer.py": "# labels CPU fixture\n",
        "src/lerobot/policies/pi05/modeling_pi05.py": "# model CPU fixture\n",
        "src/lerobot/utils/peft_audit.py": "# audit CPU fixture\n",
        "src/lerobot/utils/train_utils.py": "# checkpoint CPU fixture\n",
        "src/lerobot/data/loader.py": "# Importable source, not operational data\n",
        "tools/data/resource.json": '{"needed_by_code":true}\n',
        "configs/hotswitch/run.json": '{"fixture":true}\n',
        "tests/example_test.py": "# included source test\n",
        "docs/guide.md": "# included guide\n",
        "requirements-test.txt": "# no packages needed\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for name in ("hotswitch_freeze_source.py", "hotswitch_evorl_pipeline.py", "hotswitch_export_run.py"):
        destination = source / "tools" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "tools" / name, destination)
    return tool, source, run


def rewrite_manifest(run, transform):
    path = run / "source_snapshot_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    transform(manifest)
    manifest["snapshot_tree_sha256"] = hashlib.sha256(
        json.dumps(manifest["snapshot_files_sha256"], sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_freeze_verify_records_actual_bytes_and_does_not_follow_mutable_source(fixture):
    tool, source, run = fixture
    manifest = tool.freeze_source(source, run)
    snapshot = run / "source_snapshot"
    assert Path(manifest["snapshot"]).resolve() == snapshot.resolve()
    assert (
        manifest["snapshot_tree_sha256"]
        == hashlib.sha256(json.dumps(manifest["snapshot_files_sha256"], sort_keys=True).encode()).hexdigest()
    )
    config = snapshot / "configs/hotswitch/run.json"
    summary = tool.verify_frozen_source(run, execution_root=snapshot, config_path=config, require_formal=True)
    assert summary["status"] == "VERIFIED"
    assert "src/lerobot/data/loader.py" in manifest["snapshot_files_sha256"]
    assert "tools/data/resource.json" in manifest["snapshot_files_sha256"]
    original = snapshot / "src/lerobot/scripts/lerobot_train.py"
    before = original.read_bytes()
    (source / "src/lerobot/scripts/lerobot_train.py").write_text("# Later mutable changes\n")
    assert original.read_bytes() == before
    tool.verify_frozen_source(run, execution_root=snapshot, config_path=config, require_formal=True)


def test_cache_data_outputs_are_excluded_and_generated_cache_is_harmless(fixture):
    tool, source, run = fixture
    for relative in (
        "src/lerobot/__pycache__/old.pyc",
        "data/private.parquet",
        "outputs/model.safetensors",
        "work/scratch.py",
        ".ruff_cache/config",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"NOT_EXECUTABLE_SOURCE")
    manifest = tool.freeze_source(source, run)
    names = manifest["snapshot_files_sha256"]
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any(name.startswith(("data/", "outputs/", "work/", ".git/")) for name in names)
    snapshot = run / "source_snapshot"
    generated = snapshot / "src/lerobot/__pycache__/generated.pyc"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"CACHE_NOT_PART_OF_SOURCE")
    tool.verify_frozen_source(run, execution_root=snapshot, require_formal=True)


@pytest.mark.parametrize(
    "change", ["modified", "missing", "extra", "nested_data", "nested_logs", "nested_work"]
)
def test_snapshot_file_mutation_deletion_or_addition_is_rejected(fixture, change):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    snapshot = run / "source_snapshot"
    target = snapshot / "src/lerobot/scripts/lerobot_train.py"
    if change == "modified":
        target.write_text("# Changed frozen trainer\n")
    elif change == "missing":
        target.unlink()
    else:
        relative = {
            "extra": "src/lerobot/new_behavior.py",
            "nested_data": "src/lerobot/data/extra.py",
            "nested_logs": "src/lerobot/logs/extra.py",
            "nested_work": "tools/work/extra.py",
        }[change]
        new_source = snapshot / relative
        new_source.parent.mkdir(parents=True, exist_ok=True)
        new_source.write_text("# New unrecorded code\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.verify_frozen_source(run, execution_root=snapshot, require_formal=True)


@pytest.mark.parametrize("invalid", ["empty", "tree", "parent", "absolute", "backslash"])
def test_forged_manifest_inventory_tree_or_path_is_rejected(fixture, invalid):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    outside = run.parent / "outside_test_source.py"
    outside.write_text("# Outside harmless test file\n")

    def mutate(manifest):
        names = manifest["snapshot_files_sha256"]
        if invalid == "empty":
            names.clear()
        elif invalid != "tree":
            path = {
                "parent": "../outside_test_source.py",
                "absolute": str(outside.resolve()),
                "backslash": "src\\lerobot\\unexpected.py",
            }[invalid]
            names[path] = hashlib.sha256(outside.read_bytes()).hexdigest()

    rewrite_manifest(run, mutate)
    if invalid == "tree":
        path = run / "source_snapshot_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["snapshot_tree_sha256"] = "not-a-content-hash"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        tool.verify_frozen_source(run, require_formal=True)


def test_freeze_never_overwrites_existing_evidence_or_recurses_into_source(fixture):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    manifest = run / "source_snapshot_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        tool.freeze_source(source, run)
    assert manifest.read_bytes() == before
    with pytest.raises(ValueError):
        tool.freeze_source(source, source / "nested-run")


def test_wrong_execution_root_and_mutable_external_config_are_rejected(fixture):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    with pytest.raises(ValueError):
        tool.verify_frozen_source(run, execution_root=source, require_formal=True)
    with pytest.raises(ValueError):
        tool.verify_frozen_source(
            run,
            execution_root=run / "source_snapshot",
            config_path=source / "configs/hotswitch/run.json",
            require_formal=True,
        )


def test_symlink_escape_in_source_is_rejected_before_copy(fixture):
    tool, source, run = fixture
    outside = run.parent / "outside_link_target.py"
    outside.write_text("# Harmless outside fixture\n")
    try:
        (source / "src/lerobot/escape.py").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Host cannot create test symlinks: {exc}")
    with pytest.raises(ValueError):
        tool.freeze_source(source, run)


@pytest.mark.parametrize("metadata", ["source_provenance", "snapshot_manifest"])
def test_metadata_symlink_flag_is_checked_before_reading_any_content(fixture, monkeypatch, metadata):
    tool, source, run = fixture
    if metadata == "snapshot_manifest":
        tool.freeze_source(source, run)
        target = run / "source_snapshot_manifest.json"
    else:
        target = source / "source_provenance.json"
    original_is_symlink, original_read_text = Path.is_symlink, Path.read_text
    reads = []

    def flagged(path):
        return path == target or original_is_symlink(path)

    def guarded_read(path, *args, **kwargs):
        if path == target:
            reads.append(path)
            raise AssertionError("Metadata content was read before rejecting the symlink")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", flagged)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(ValueError):
        if metadata == "snapshot_manifest":
            tool.verify_frozen_source(run, require_formal=True)
        else:
            tool.freeze_source(source, run)
    assert not reads


def test_package_gate_uses_real_fresh_process_spec_without_importing_model_package(fixture):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    snapshot = run / "source_snapshot"
    script = (
        "import importlib.util,json,pathlib; "
        f"s=importlib.util.spec_from_file_location('verify_freeze', {str(snapshot / 'tools/hotswitch_freeze_source.py')!r}); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        f"print(json.dumps(m.verify_frozen_source(pathlib.Path({str(run)!r}), "
        f"execution_root=pathlib.Path({str(snapshot)!r}),require_formal=True,check_package=True)))"
    )
    for package_root, success in ((snapshot / "src", True), (source / "src", False)):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(package_root)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=run.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if success:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["status"] == "VERIFIED"
        else:
            assert result.returncode != 0
        assert "MODEL_PACKAGE_MUST_NOT_BE_IMPORTED" not in result.stderr


def test_formal_source_failure_precedes_gpu_query_dataset_copy_and_fingerprinting(fixture, monkeypatch):
    tool, source, run = fixture
    tool.freeze_source(source, run)
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    pipeline = load_tool("hotswitch_evorl_pipeline")
    prepare = importlib.import_module("hotswitch_prepare_data")
    config = json.loads((ROOT / "configs/hotswitch/straighten_cash_value.json").read_text())
    dataset = run.parent / "dataset"
    dataset.mkdir()
    (dataset / "prepare_manifest.json").write_text('{"status":"validated"}')
    config["source_dataset_root"] = str(dataset)
    config_path = run.parent / "mutable_config.json"
    config_path.write_text(json.dumps(config))
    targets = run.parent / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "vlm": ["q"],
                "expert": ["q"],
                "projections": ["a", "b", "c", "d"],
                "target_modules": ["q", "a", "b", "c", "d"],
            }
        )
    )
    gpu = Mock(side_effect=AssertionError("GPU gate must not be reached"))
    copy = Mock(side_effect=AssertionError("Data copy must not be reached"))
    fingerprints = Mock(side_effect=AssertionError("Data fingerprint must not be reached"))
    monkeypatch.setattr(pipeline, "require_gates", lambda *args: None)
    monkeypatch.setattr(pipeline, "require_idle_gpu", gpu)
    monkeypatch.setattr(pipeline.shutil, "copytree", copy)
    monkeypatch.setattr(prepare, "fingerprints", fingerprints)
    # main() must reject its mutable execution/config roots before querying a
    # device or touching training data, even when mechanism gates are supplied.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--config",
            str(config_path),
            "--profile",
            "formal",
            "--run-root",
            str(run),
            "--targets",
            str(targets),
            "--execute",
            "--smoke-report",
            "unused",
            "--benchmark-report",
            "unused",
        ],
    )
    with pytest.raises(ValueError) as error:
        pipeline.main()
    assert any(word in str(error.value).lower() for word in ("snapshot", "source", "frozen"))
    gpu.assert_not_called()
    copy.assert_not_called()
    fingerprints.assert_not_called()
