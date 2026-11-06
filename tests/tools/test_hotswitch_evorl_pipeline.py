import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "evorl_pipeline", Path(__file__).resolve().parents[2] / "tools/hotswitch_evorl_pipeline.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_formal_workflow_cannot_run_with_only_cpu_or_simulation_smoke(tmp_path):
    smoke, benchmark = tmp_path / "smoke.json", tmp_path / "benchmark.json"
    smoke.write_text(json.dumps({"status": "real_cpu_passed", "frozen_base_unchanged": True}))
    benchmark.write_text(json.dumps({mode: {"status": "GPU_OFFLINE_PASSED", "samples": 200} for mode in ["shared", "dual"]}))
    with pytest.raises(ValueError, match="Real GPU"):
        pipeline.require_gates(smoke, benchmark)
    smoke.write_text(json.dumps({"status": "real_gpu_passed", "frozen_base_unchanged": True}))
    pipeline.require_gates(smoke, benchmark)
    for bad_status in ["SIMULATION_PASSED", "NOT_PASSED", "FAILED"]:
        benchmark.write_text(json.dumps({mode: {"status": bad_status, "samples": 200} for mode in ["shared", "dual"]}))
        with pytest.raises(ValueError, match="did not pass"):
            pipeline.require_gates(smoke, benchmark)
    benchmark.write_text(json.dumps({"shared": {"status": "SIMULATION_PASSED", "samples": 200}}))
    with pytest.raises(ValueError, match="both"):
        pipeline.require_gates(smoke, benchmark)


def test_run_directory_cannot_recursively_contain_source_copy(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        pipeline.validate_run_root(tmp_path / "source", tmp_path / "source/run")
    with pytest.raises(ValueError, match="outside"):
        pipeline.validate_run_root(tmp_path / "source", tmp_path / "source")
    pipeline.validate_run_root(tmp_path / "source", tmp_path / "run")


def test_every_row_requires_finite_real_value_advantage_and_binary_indicator(tmp_path):
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"total_frames":2}')
    path = tmp_path / "data/chunk-000/file-000.parquet"
    table = pd.DataFrame({"complementary_info.value": [-0.8, -0.1],
                          "complementary_info.advantage": [0.1, -0.1],
                          "complementary_info.acp_indicator": [1, 0]})
    table.to_parquet(path, index=False)
    assert pipeline.validate_labels(tmp_path) == {"frames": 2, "positive": 1, "negative": 1}
    table.loc[1, "complementary_info.advantage"] = float("nan")
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Nonfinite"):
        pipeline.validate_labels(tmp_path)


def test_pilot_budget_and_sanitized_label_destination_are_distinct_from_formal(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/hotswitch/straighten_cash_value.json").read_text())
    pilot = pipeline.commands(config, "pilot", tmp_path, ["q_proj"], "python")
    formal = pipeline.commands(config, "formal", tmp_path, ["q_proj"], "python")
    assert "--steps=200" in pilot["adapter"]
    assert "--steps=160000" in formal["adapter"]
    assert "--verify_peft_base=true" in formal["adapter"]
    assert not any(argument.startswith("--verify_peft_base") for argument in pilot["adapter"])
    assert f"--dataset.root={tmp_path / 'labeled_dataset'}" in pilot["value-infer"]
    assert f"--dataset.root={config['source_dataset_root']}" not in pilot["value-infer"]
    assert "--peft.full_training_modules=[]" in pilot["adapter"]
    assert "lerobot.scripts.lerobot_train_trc" in pilot["adapter"]


def test_repository_annotation_writer_updates_only_private_copy_and_feature_metadata(tmp_path):
    import numpy as np

    from lerobot.scripts.lerobot_value_infer import _write_columns_in_place

    original, labeled = tmp_path / "original", tmp_path / "labeled"
    (original / "data/chunk-000").mkdir(parents=True)
    (original / "meta").mkdir()
    (original / "meta/info.json").write_text('{"features":{},"total_frames":2}')
    path = Path("data/chunk-000/file-000.parquet")
    pd.DataFrame({"index": [0, 1], "episode_index": [0, 0]}).to_parquet(original / path, index=False)
    before = (original / path).read_bytes()
    shutil.copytree(original, labeled)
    columns = {"complementary_info.value": np.array([-0.8, -0.1], dtype=np.float32),
               "complementary_info.advantage": np.array([-0.2, 0.1], dtype=np.float32),
               "complementary_info.acp_indicator": np.array([0, 1], dtype=np.int64)}
    infos = {key: {"dtype": "int64" if key.endswith("indicator") else "float32", "shape": [1]}
             for key in columns}
    _write_columns_in_place(labeled, np.array([0, 1]), columns, infos)
    assert (original / path).read_bytes() == before
    assert json.loads((original / "meta/info.json").read_text())["features"] == {}
    assert set(json.loads((labeled / "meta/info.json").read_text())["features"]) == set(columns)
    assert pipeline.validate_labels(labeled) == {"frames": 2, "positive": 1, "negative": 1}
