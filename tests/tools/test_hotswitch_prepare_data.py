"""Preparation must preserve original labels and reject corrupt registered rows."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[2] / "tools/hotswitch_prepare_data.py"
)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


def source(root, orphan=False, duplicate=False):
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    camera = "observation.images.front"
    video = root / f"videos/{camera}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video; duration mocked")
    features = {
        "action": {"dtype": "float32", "shape": [1]},
        "observation.state": {"dtype": "float32", "shape": [1]},
        camera: {"dtype": "video", "shape": [2, 2, 3]},
    }
    info = {"features": features, "fps": 30, "robot_type": "fake", "codebase_version": "v3.0",
            "total_episodes": 1, "total_frames": 2,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root / "meta/info.json").write_text(json.dumps(info))
    stats = {k: {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0], "count": [2], "q01": [-0.9], "q99": [0.9]}
             for k in ["action", "observation.state"]}
    (root / "meta/stats.json").write_text(json.dumps(stats))
    pd.DataFrame({"task_index": [0]}, index=["test task"]).to_parquet(root / "meta/tasks.parquet")
    episode = dict(episode_index=0, length=2, episode_success="success", tasks=["test task"],
                   dataset_from_index=0, dataset_to_index=2,
                   **{"data/chunk_index": 0, "data/file_index": 0,
                      "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                      f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": 0,
                      f"videos/{camera}/from_timestamp": 0.0, f"videos/{camera}/to_timestamp": 2 / 30})
    pd.DataFrame([episode]).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    rows = [dict(index=i, episode_index=0, frame_index=i, timestamp=i / 30, task_index=0,
                 action=[float(i)], **{"observation.state": [float(i)]}) for i in range(2)]
    if orphan:
        rows.append(dict(rows[0], index=2, episode_index=1))
    if duplicate:
        rows[1]["frame_index"] = 0
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    return root


def test_orphan_removed_sources_unchanged_and_offsets_align(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "duration", lambda _: 2 / 30)
    first, second = source(tmp_path / "first", orphan=True), source(tmp_path / "second")
    before = {p: prepare_data.fingerprints(p) for p in [first, second]}
    result = prepare_data.prepare([first, second], tmp_path / "result", copy_videos=True)
    assert (result["total_episodes"], result["total_frames"], result["discarded_rows"]) == (2, 4, 1)
    assert result["status"] == "validated"
    assert result["tasks"] == ["test task"]
    assert all(prepare_data.fingerprints(p) == hashes for p, hashes in before.items())
    meta = pd.read_parquet(tmp_path / "result/meta/episodes/chunk-000/file-000.parquet")
    assert meta.episode_success.tolist() == ["success", "success"]
    assert meta.dataset_from_index.tolist() == [0, 2]
    assert meta["videos/observation.images.front/file_index"].tolist() == [0, 1]
    assert "complementary_info.acp_indicator" not in json.loads((tmp_path / "result/meta/info.json").read_text())["features"]


def test_corrupt_registered_episode_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "duration", lambda _: 2 / 30)
    broken = source(tmp_path / "broken", duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_data.prepare([broken], tmp_path / "result", copy_videos=True)
    assert not (tmp_path / "result/prepare_manifest.json").exists()


def test_existing_output_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        prepare_data.prepare([], tmp_path)
