import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("export_run", ROOT / "tools/hotswitch_export_run.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def coverage_report(end, *, start=0, initial="initial", final="updated", audit_id="fixture"):
    groups = ("vlm", "expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    def snapshot(tag):
        return {"sha256": "a" * 64, "parameter_inventory": [["base", [1], "float32"]],
                "adapter_groups": {key: {"sha256": hashlib.sha256(f"{key}/{tag}".encode()).hexdigest(),
                    "finite": True, "names": [key]} for key in groups}}
    before, after = snapshot(initial), snapshot(final)
    segment = {"audit_id": audit_id, "start_step": start, "end_step": end, "updates": end - start,
               "base_sha256": before["sha256"], "frozen_base_unchanged": True,
               "required_groups": list(groups), "adapter_groups_updated": dict.fromkeys(groups, True),
               "adapter_values_finite": True,
               "before_adapter_groups_sha256": {key: value["sha256"] for key, value in before["adapter_groups"].items()},
               "after_adapter_groups_sha256": {key: value["sha256"] for key, value in after["adapter_groups"].items()}}
    return {"schema_version": 2, "status": "PASSED", "audit_id": audit_id, "coverage_complete": True,
            "start_step": start, "steps_completed": end, "updates_this_attempt": end - start,
            "required_groups": list(groups), "before": before, "after": after, "frozen_base_unchanged": True,
            "coverage": [segment], "previous_report": None, "adapter_values_finite": True,
            "adapter_groups_updated": dict.fromkeys(groups, True)}


def bind_saved_fixture(report, checkpoint, *, final=True):
    """Fixture bytes are audit inputs only, never presented as actual model weights."""
    from lerobot.utils.peft_audit import checkpoint_artifact_manifest
    (checkpoint / "pretrained_model").mkdir(parents=True)
    (checkpoint / "training_state").mkdir()
    (checkpoint / "pretrained_model/adapter_model.safetensors").write_bytes(b"dummy audit fixture")
    (checkpoint / "pretrained_model/config.json").write_text('{}')
    (checkpoint / "training_state/training_step.json").write_text(json.dumps({"step": report["steps_completed"]}))
    sidecar = copy.deepcopy(report)
    sidecar["status"] = "CHECKPOINT_PASSED"
    sidecar["checkpoint"] = checkpoint_artifact_manifest(checkpoint)
    sidecar_path = checkpoint / "training_state/peft_base_audit.json"
    sidecar_path.write_text(json.dumps(sidecar))
    if not final:
        return sidecar, sidecar_path
    full = dict(sidecar, status="PASSED", final_checkpoint_report={"path": str(sidecar_path),
        "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest()})
    return full, sidecar_path


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    run, source, base, tokenizer = (tmp_path / key for key in ["run", "source", "base", "tokenizer"])
    for path in [run, source, base, tokenizer]:
        path.mkdir()
    (source / "prepare_manifest.json").write_text('{"status":"validated"}')
    checkpoint = run / "adapter_A/checkpoints/last/pretrained_model"
    value_checkpoint = run / "value_train/checkpoints/last/pretrained_model"
    for path in [checkpoint, value_checkpoint]:
        path.mkdir(parents=True)
        (path.parent / "training_state").mkdir()
        (path.parent / "training_state/training_step.json").write_text('{"step":200}')
    labeled = run / "labeled_dataset"
    (labeled / "meta").mkdir(parents=True)
    (labeled / "data/chunk-000").mkdir(parents=True)
    image = "observation.images.front"
    info = {"total_frames": 2, "fps": 30, "features": {
        "action": {"names": ["joint.pos"], "dtype": "float32", "shape": [1]},
        image: {"dtype": "video", "shape": [2, 2, 3]},
    }}
    (labeled / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=["A real task"]).to_parquet(labeled / "meta/tasks.parquet")
    pd.DataFrame({"complementary_info.value": [-0.5, -0.1],
                  "complementary_info.advantage": [-0.1, 0.1],
                  "complementary_info.acp_indicator": [0, 1]}).to_parquet(labeled / "data/chunk-000/file-000.parquet")
    training = {"steps": 200, "dataset": {"root": str(labeled), "repo_id": "fixture"},
                "acp": {"enable": True, "indicator_field": "complementary_info.acp_indicator", "indicator_dropout_prob": 0.3}}
    (checkpoint / "train_config.json").write_text(json.dumps(training))
    (value_checkpoint / "value_train_config.json").write_text(json.dumps({"steps": 200, "dataset": {"root": str(source)}}))
    policy = {"rtc_training_max_delay": 37, "chunk_size": 50, "tokenizer_max_length": 200, "tokenizer_name": str(tokenizer),
              "output_features": {"action": {"shape": [1]}},
              "input_features": {"observation.state": {"shape": [1]}, image: {"shape": [3, 2, 2]}}}
    (checkpoint / "config.json").write_text(json.dumps(policy))
    settings = {"source_dataset_root": str(source), "base_path": str(base),
                "profiles": {"pilot": {"adapter_steps": 200, "value_steps": 200}}}
    stage = {"status": "passed", "exit_code": 0}
    stages = {key: dict(stage) for key in ["value-train", "value-infer", "adapter"]}
    stages["value-infer"]["argv"] = [f"--dataset.root={labeled}",
        f"--inference.checkpoint_path={value_checkpoint}", "--acp.enable=true"]
    (run / "pipeline_status.json").write_text(json.dumps({"profile": "pilot", "config": settings, "stages": stages}))
    observation = tmp_path / "observation.safetensors"
    save_file({"observation.state": np.zeros(1, np.float32), image: np.zeros((3, 2, 2), np.float32)}, observation)
    return SimpleNamespace(run_root=run, base_path=base, tokenizer_path=tokenizer,
        offline_observation=observation, output=tmp_path / "export", prefix_steps=6,
        action_unit="recorded_robot_units_unverified")


def test_pilot_metadata_retains_true_profile_labels_and_single_task(run_fixture):
    _, metadata = exporter.build_metadata(run_fixture)
    assert metadata["training_source"]["status"] == "pilot_completed"
    assert metadata["training_source"]["formal_training_completed"] is False
    assert metadata["training_source"]["second_task_trained"] is False
    assert metadata["training_source"]["frozen_base_audit"]["status"] == "NOT_ENABLED"
    assert metadata["training_source"]["tokenizer_provenance"]["status"] == "PILOT_EXTERNAL_TOKENIZER"
    assert metadata["training_source"]["shared_base_file_fingerprint"]["status"] == "NOT_CAPTURED"
    assert metadata["training_source"]["label_summary"] == {"frames": 2, "positive": 1, "negative": 1}
    assert metadata["acp"]["prompt_strategy"] == "positive"
    assert metadata["acp"]["beta"] == 1 and metadata["acp"]["cfg_enabled"] is False
    assert metadata["rtc"]["prefix_steps"] == 6


def test_success_flag_cannot_hide_incomplete_checkpoint(run_fixture):
    step = run_fixture.run_root / "adapter_A/checkpoints/last/training_state/training_step.json"
    step.write_text('{"step":199}')
    with pytest.raises(ValueError, match="differs from configured budget"):
        exporter.build_metadata(run_fixture)


def test_complete_checkpoint_cannot_hide_failed_pipeline_stage(run_fixture):
    path = run_fixture.run_root / "pipeline_status.json"
    status = json.loads(path.read_text())
    status["stages"]["adapter"]["exit_code"] = 1
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="not completed"):
        exporter.build_metadata(run_fixture)


def test_formal_requires_matching_full_duration_base_hash_evidence(tmp_path):
    report_path = tmp_path / "adapter_A/peft_base_audit.json"
    report_path.parent.mkdir()
    with pytest.raises(ValueError, match="requires"):
        exporter.frozen_base_evidence(tmp_path, "formal", 160000, {})
    training = {"verify_peft_base": True}
    with pytest.raises(FileNotFoundError):
        exporter.frozen_base_evidence(tmp_path, "formal", 160000, training)
    report, _ = bind_saved_fixture(coverage_report(160000), tmp_path / "adapter_A/checkpoints/last")
    report_path.write_text(json.dumps(report))
    evidence = exporter.frozen_base_evidence(tmp_path, "formal", 160000, training)
    assert evidence["status"] == "PASSED" and len(evidence["source_report_sha256"]) == 64
    for changes in [{"steps_completed": 200}, {"status": "STARTED"}, {"frozen_base_unchanged": False},
                    {"after": {"sha256": "b" * 64}}]:
        report_path.write_text(json.dumps(dict(report, **changes)))
        with pytest.raises(ValueError):
            exporter.frozen_base_evidence(tmp_path, "formal", 160000, training)


def test_formal_resume_requires_a_real_continuous_prior_checkpoint_chain(tmp_path):
    previous, previous_path = bind_saved_fixture(coverage_report(80000, final="middle", audit_id="previous"),
        tmp_path / "adapter_A/checkpoints/080000", final=False)
    final = coverage_report(160000, start=80000, initial="middle", final="final", audit_id="current")
    final["coverage"] = [*previous["coverage"], *final["coverage"]]
    final["previous_report"] = {"path": str(previous_path), "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest()}
    final, _ = bind_saved_fixture(final, tmp_path / "adapter_A/checkpoints/last")
    report_path = tmp_path / "adapter_A/peft_base_audit.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(final))
    evidence = exporter.frozen_base_evidence(tmp_path, "formal", 160000, {"verify_peft_base": True, "resume": True})
    assert evidence["coverage"]["segments"] == 2
    assert evidence["coverage"]["updates_this_attempt"] == 80000
    previous_path.write_text('{}')
    with pytest.raises(ValueError, match="hash changed"):
        exporter.frozen_base_evidence(tmp_path, "formal", 160000, {"verify_peft_base": True, "resume": True})


def test_export_detects_modification_of_the_recorded_source_snapshot(run_fixture):
    root = run_fixture.run_root / "source_snapshot"
    root.mkdir()
    files = {}
    for relative in ["src/lerobot/__init__.py", "src/lerobot/scripts/lerobot_train.py", "src/lerobot/policies/pi05/modeling_pi05.py",
                     "tools/hotswitch_evorl_pipeline.py"]:
        code = root / relative
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("# Original test fixture source\n")
        files[relative] = hashlib.sha256(code.read_bytes()).hexdigest()
    record = {"snapshot": str(root), "snapshot_tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
              "snapshot_files_sha256": files}
    (run_fixture.run_root / "source_snapshot_manifest.json").write_text(json.dumps(record))
    _, metadata = exporter.build_metadata(run_fixture)
    assert metadata["training_source"]["source_code"]["status"] == "VERIFIED"
    code.write_text("# Changed after launch\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        exporter.build_metadata(run_fixture)


def test_source_evidence_rejects_empty_inventory_and_unverified_tree(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    path = tmp_path / "source_snapshot_manifest.json"
    path.write_text(json.dumps({"snapshot_files_sha256": {}, "snapshot_tree_sha256": "anything"}))
    with pytest.raises(ValueError, match="nonempty"):
        exporter.source_code_evidence(tmp_path, "formal")
    path.write_text(json.dumps({"snapshot_files_sha256": {"x": "y"}, "snapshot_tree_sha256": "anything"}))
    with pytest.raises(ValueError, match="tree hash differs"):
        exporter.source_code_evidence(tmp_path, "formal")


def test_formal_uses_saved_actual_tokenizer_and_never_external_fallback(tmp_path):
    configured = tmp_path / "configured"
    with pytest.raises(ValueError, match="checkpoint/tokenizer"):
        exporter.tokenizer_evidence(tmp_path, "formal", configured)
    (tmp_path / "tokenizer").mkdir()
    evidence = exporter.tokenizer_evidence(tmp_path, "formal", configured)
    assert evidence["status"] == "CHECKPOINT_SAVED_TOKENIZER"
    assert evidence["saved_actual_source"] == str(tmp_path / "tokenizer")


def test_formal_base_fingerprint_binds_before_after_and_export_time_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    from hotswitch_evorl_pipeline import begin_shared_base_fingerprint, finish_shared_base_fingerprint

    base, run = tmp_path / "tiny_base", tmp_path / "run"
    base.mkdir()
    run.mkdir()
    (base / "config.json").write_text('{"type":"pi05","scope":"tiny CPU fixture"}')
    save_file({"fixture": np.zeros(1, dtype=np.float32)}, base / "model.safetensors")
    evidence = run / "shared_base_fingerprint.json"
    with pytest.raises(ValueError, match="captured"):
        exporter.shared_base_file_evidence(run, "formal", base)
    begin_shared_base_fingerprint(base, evidence)
    with pytest.raises(FileExistsError):
        begin_shared_base_fingerprint(base, evidence)
    completed = finish_shared_base_fingerprint(base, evidence)
    assert completed["status"] == "PASSED" and completed["before"] == completed["after"]
    assert exporter.shared_base_file_evidence(run, "formal", base)["status"] == "PASSED"
    save_file({"fixture": np.ones(1, dtype=np.float32)}, base / "model.safetensors")
    with pytest.raises(ValueError, match="changed after"):
        exporter.shared_base_file_evidence(run, "formal", base)


def test_formal_base_mutation_during_training_cannot_be_marked_passed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    from hotswitch_evorl_pipeline import begin_shared_base_fingerprint, finish_shared_base_fingerprint

    base = tmp_path / "tiny_base"
    base.mkdir()
    (base / "config.json").write_text('{"type":"pi05","scope":"tiny CPU fixture"}')
    save_file({"fixture": np.zeros(1, dtype=np.float32)}, base / "model.safetensors")
    evidence = tmp_path / "shared_base_fingerprint.json"
    begin_shared_base_fingerprint(base, evidence)
    (base / "config.json").write_text('{"type":"pi05","scope":"changed CPU fixture"}')
    with pytest.raises(ValueError, match="changed during"):
        finish_shared_base_fingerprint(base, evidence)
    assert json.loads(evidence.read_text())["status"] == "FAILED"
