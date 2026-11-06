"""Freeze reviewed source into a new run root and verify it without model imports.

Example: python tools/hotswitch_freeze_source.py freeze --source-root . --run-root /path/new_run
Training must subsequently execute new_run/source_snapshot/tools/... with that
snapshot's src on PYTHONPATH. No datasets, weights, Git internals or bytecode are copied.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

TREE_ROOTS = ("src", "tools", "configs", "tests", "docs")
PACKAGING_REQUIRED = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE")
ROOT_OPTIONAL = ("setup.cfg", "source_provenance.json", "audit.md", "requirements.in",
                 "requirements-ubuntu.txt", "requirements-macos.txt", "docs-requirements.txt")
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
ROOT_OPERATIONAL_DIRS = {"work", "weights", "data", "logs", "checkpoints", "outputs", "runs"}
NON_SOURCE_SUFFIXES = {".pyc", ".pyo", ".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                      ".bin", ".parquet", ".arrow", ".npy", ".npz", ".h5", ".hdf5", ".mp4", ".avi",
                      ".mov", ".mp3", ".wav", ".db", ".sqlite"}
CORE_REQUIRED = {"src/lerobot/__init__.py", "src/lerobot/scripts/lerobot_train.py",
                 "src/lerobot/policies/pi05/modeling_pi05.py", "tools/hotswitch_evorl_pipeline.py"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(files):
    # This is intentionally identical to the existing deployment snapshot schema.
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def ignored(relative, *, ignore_egg_info=True):
    return (relative.parts[0] in ROOT_OPERATIONAL_DIRS
            or any(part in IGNORED_DIRS or (ignore_egg_info and part.endswith(".egg-info"))
                   for part in relative.parts))


def checked_file(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Noncanonical or escaping snapshot path: {relative!r}")
    candidate = root / relative
    for part in [candidate, *candidate.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Snapshot symlink is not allowed: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Snapshot artifact escapes its root or is not a file: {relative}")
    return resolved


def source_files(root):
    result = {}
    for folder in TREE_ROOTS:
        directory = root / folder
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise ValueError(f"Source tree symlink is not allowed: {directory}")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if ignored(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlink is not allowed: {relative}")
            if not path.is_file() or path.suffix.lower() in NON_SOURCE_SUFFIXES:
                continue
            result[relative.as_posix()] = checked_file(root, relative.as_posix())
    for name in (*PACKAGING_REQUIRED, *ROOT_OPTIONAL):
        if (root / name).exists():
            result[name] = checked_file(root, name)
    return dict(sorted(result.items()))


def snapshot_inventory(root):
    """Also catch new files added after freezing; caches and logs are not source."""
    result = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        # Existing legacy snapshots may record egg-info. Rehash that recorded
        # metadata honestly, and catch newly added metadata as extra inventory.
        if ignored(relative, ignore_egg_info=False) or path.suffix.lower() in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError(f"Snapshot symlink is not allowed: {relative}")
        if path.is_file():
            result[relative.as_posix()] = checked_file(root, relative.as_posix())
    return dict(sorted(result.items()))


def source_identity(root):
    provided = {}
    provenance = root / "source_provenance.json"
    if provenance.exists() or provenance.is_symlink():
        provided = json.loads(checked_file(root, "source_provenance.json").read_text(encoding="utf-8"))
    if (root / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True).strip()
        method = "git_HEAD_and_worktree_bytes"
    else:
        head, status, method = provided.get("head"), None, "source_provenance.json_HEAD_and_actual_bytes"
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", head):
        raise ValueError("Freeze requires a Git HEAD or the deployed source_provenance.json HEAD")
    return {"repository": provided.get("repository", "https://github.com/Fger23/Evo-RL"),
            "head": head.lower(), "git_status": status, "head_source": method}


def freeze_source(source_root, run_root):
    source_root = Path(source_root).resolve(strict=True)
    run_root = Path(run_root).resolve()
    if run_root.is_relative_to(source_root):
        raise ValueError("Run root must be outside the source tree being copied")
    snapshot = run_root / "source_snapshot"
    manifest_path = run_root / "source_snapshot_manifest.json"
    if snapshot.exists() or manifest_path.exists() or (run_root.exists() and any(run_root.iterdir())):
        raise FileExistsError(f"Freeze requires a new empty run root: {run_root}")
    for name in PACKAGING_REQUIRED:
        if not (source_root / name).is_file():
            raise FileNotFoundError(f"Required repository packaging metadata is absent: {name}")
    for name in ("src", "tools", "configs"):
        if not (source_root / name).is_dir():
            raise FileNotFoundError(f"Required source directory is absent: {name}")
    identity = source_identity(source_root)
    inputs = source_files(source_root)
    if not CORE_REQUIRED.issubset(inputs):
        raise ValueError("Source tree is missing the actual trainer, model or pipeline")
    before = {name: file_hash(path) for name, path in inputs.items()}
    snapshot.mkdir(parents=True)
    for name, path in inputs.items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    copied = {name: file_hash(path) for name, path in snapshot_inventory(snapshot).items()}
    after = {name: file_hash(path) for name, path in source_files(source_root).items()}
    if copied != before or after != before:
        raise RuntimeError("Source changed while freezing; partial snapshot preserved for inspection")
    manifest = {"schema_version": 1, **identity, "snapshot": str(snapshot), "source_root": str(source_root),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "snapshot_files_sha256": copied, "snapshot_tree_sha256": tree_hash(copied),
                "hardware_connected": False, "weights_or_datasets_copied": False}
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2))
    verify_frozen_source(run_root)
    return manifest


def verify_frozen_source(run_root, *, execution_root=None, config_path=None, require_formal=False, check_package=False):
    run_root = Path(run_root).resolve(strict=True)
    manifest_path = checked_file(run_root, "source_snapshot_manifest.json")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = recorded.get("snapshot_files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen source snapshot file inventory must be nonempty")
    if recorded.get("snapshot_tree_sha256") != tree_hash(files):
        raise ValueError("Frozen source snapshot tree hash differs from its file inventory")
    lexical_snapshot = run_root / "source_snapshot"
    if lexical_snapshot.is_symlink():
        raise ValueError("source_snapshot must not be a symlink")
    snapshot = lexical_snapshot.resolve(strict=True)
    declared = recorded.get("snapshot")
    if not isinstance(declared, str) or not Path(declared).is_absolute():
        raise ValueError("Manifest must declare the actual absolute source_snapshot root")
    if not snapshot.is_relative_to(run_root) or Path(declared).resolve(strict=True) != snapshot:
        raise ValueError("Manifest snapshot must be this run root's actual source_snapshot")
    if execution_root is not None and Path(execution_root).resolve(strict=True) != snapshot:
        raise ValueError("Formal pipeline must execute from run_root/source_snapshot")
    for relative in files:
        checked_file(snapshot, relative)
    actual_paths = snapshot_inventory(snapshot)
    if set(actual_paths) != set(files):
        raise ValueError("Frozen snapshot inventory changed (missing or additional source files)")
    required = set(CORE_REQUIRED)
    if require_formal:
        required.update(PACKAGING_REQUIRED)
        required.add("src/lerobot/utils/peft_audit.py")
        if any(any(part.endswith(".egg-info") for part in PurePosixPath(name).parts) for name in files):
            raise ValueError("Formal snapshot must exclude editable egg-info metadata")
    if not required.issubset(files):
        raise ValueError("Frozen source snapshot is missing critical trainer/model/audit/packaging files")
    for relative, expected in files.items():
        if file_hash(actual_paths[relative]) != expected:
            raise ValueError(f"Frozen source snapshot differs from its actual start manifest: {relative}")
    if config_path is not None:
        config = Path(config_path).resolve(strict=True)
        if not config.is_relative_to(snapshot) or config.relative_to(snapshot).as_posix() not in files:
            raise ValueError("Formal config must be recorded copied content inside the frozen snapshot")
    origin = None
    if check_package:
        try:
            spec = importlib.util.find_spec("lerobot")
        except (ValueError, ModuleNotFoundError) as error:
            raise ValueError("Parent lerobot import origin cannot be resolved; fix PYTHONPATH") from error
        package = snapshot / "src/lerobot"
        if spec is None or not spec.origin or Path(spec.origin).resolve(strict=True) != package / "__init__.py":
            raise ValueError("Parent lerobot import origin must be frozen_snapshot/src; fix PYTHONPATH")
        if not spec.submodule_search_locations or any(Path(p).resolve(strict=True) != package for p in spec.submodule_search_locations):
            raise ValueError("Parent lerobot package search roots escape the frozen snapshot")
        origin = spec.origin
    return {"status": "VERIFIED", "manifest_sha256": file_hash(manifest_path),
            "snapshot_tree_sha256": tree_hash(files), "snapshot": str(snapshot),
            "repository": recorded.get("repository"), "head": recorded.get("head"),
            "files_verified": len(files), "parent_package_origin": origin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    freeze.add_argument("--run-root", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--execution-root", type=Path)
    verify.add_argument("--config-path", type=Path)
    verify.add_argument("--formal", action="store_true")
    verify.add_argument("--check-package", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_source(args.source_root, args.run_root)
        print(json.dumps({key: result[key] for key in ("snapshot", "snapshot_tree_sha256", "head")}))
    else:
        print(json.dumps(verify_frozen_source(args.run_root, execution_root=args.execution_root,
            config_path=args.config_path, require_formal=args.formal, check_package=args.check_package)))


if __name__ == "__main__":
    main()
