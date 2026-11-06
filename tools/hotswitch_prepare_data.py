"""Build a new LeRobot v3 dataset from registered episodes only.

Never modifies a source. Videos are referenced by symlinks (or copied explicitly),
while every data/metadata parquet is independently written. An incomplete recording
can leave rows whose episode is absent from meta/episodes; those rows are audited
and excluded before indices are assigned. No success or ACP labels are invented.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprints(root: Path) -> dict[str, str]:
    files = sorted((root / "meta").rglob("*.json"))
    files += sorted((root / "meta").rglob("*.parquet"))
    files += sorted((root / "data").rglob("*.parquet"))
    return {str(p.relative_to(root)): digest(p) for p in files}


def duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        candidate = Path(sys.base_prefix) / "bin/ffprobe"
        if not candidate.is_file():
            raise FileNotFoundError("ffprobe is required to validate video references")
        ffprobe = str(candidate)
    return float(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True).strip())


def aggregate_stats(stats: list[dict]) -> dict:
    """Same weighted parallel variance/quantile rule as LeRobot aggregate_stats."""
    result = {}
    for feature in sorted(set().union(*(s.keys() for s in stats))):
        items = [s[feature] for s in stats if feature in s]
        means = np.stack([np.asarray(s["mean"]) for s in items])
        counts = np.stack([np.asarray(s["count"]) for s in items])
        total = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        mean = (means * counts).sum(axis=0) / total
        variance = ((np.stack([np.asarray(s["std"]) ** 2 for s in items])
                     + (means - mean) ** 2) * counts).sum(axis=0) / total
        merged = {
            "mean": mean, "std": np.sqrt(variance), "count": total,
            "min": np.min(np.stack([s["min"] for s in items]), axis=0),
            "max": np.max(np.stack([s["max"] for s in items]), axis=0),
        }
        for key in items[0]:
            if key.startswith("q") and key[1:].isdigit() and all(key in s for s in items):
                merged[key] = (np.stack([s[key] for s in items]) * counts).sum(axis=0) / total
        result[feature] = {k: np.asarray(v).tolist() for k, v in merged.items()}
    return result


def check_rows(data: pd.DataFrame, episodes: pd.DataFrame, fps: float) -> None:
    known = {int(x) for x in episodes["episode_index"]}
    if {int(x) for x in data["episode_index"]} != known:
        raise ValueError("Data and registered episode identities differ")
    if data["index"].duplicated().any():
        raise ValueError("Duplicate data index")
    if data.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError("Duplicate (episode_index, frame_index)")
    for _, row in episodes.iterrows():
        part = data[data.episode_index == row.episode_index].sort_values("frame_index")
        size = int(row.length)
        if len(part) != size or not np.array_equal(part.frame_index.to_numpy(), np.arange(size)):
            raise ValueError(f"Incomplete registered episode {row.episode_index}")
        if not np.allclose(part.timestamp.to_numpy(), np.arange(size) / fps, atol=1e-4):
            raise ValueError(f"Timestamp mismatch in episode {row.episode_index}")
        begin, end = int(row.dataset_from_index), int(row.dataset_to_index)
        if end - begin != size or not np.array_equal(part["index"].to_numpy(), np.arange(begin, end)):
            raise ValueError(f"Dataset bounds mismatch in episode {row.episode_index}")


def prepare(sources: list[Path], output: Path, copy_videos: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not sources:
        raise ValueError("No source datasets matched the explicit range")
    for source in sources:
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Output must be outside every source")
    output.mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    report = {"status": "preparing", "sources": [], "videos": [], "video_extra_duration": [], "discarded_rows": 0}
    all_meta, all_data, all_stats, tasks = [], [], [], []
    frame_offset = episode_offset = 0
    reference = None
    video_index = collections.Counter()
    for source_index, source in enumerate(sources):
        before = fingerprints(source)
        info = json.loads((source / "meta/info.json").read_text())
        signature = {k: info[k] for k in ["features", "fps", "robot_type", "codebase_version"]}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Incompatible metadata in {source}")
        episodes = pd.concat([pd.read_parquet(p) for p in sorted((source / "meta/episodes").rglob("*.parquet"))], ignore_index=True)
        episodes = episodes.sort_values("episode_index").reset_index(drop=True)
        if not np.array_equal(episodes.episode_index.to_numpy(), np.arange(len(episodes))):
            raise ValueError(f"Registered episode IDs must be contiguous in {source}")
        if len(episodes) != info["total_episodes"] or int(episodes.length.sum()) != info["total_frames"]:
            raise ValueError(f"Metadata counts disagree in {source}")
        if "episode_success" not in episodes or not episodes.episode_success.isin(["success", "failure"]).all():
            raise ValueError(f"Missing genuine success/failure labels in {source}")
        data = pd.concat([pd.read_parquet(p) for p in sorted((source / "data").rglob("*.parquet"))], ignore_index=True)
        orphan = data[~data.episode_index.isin(episodes.episode_index)]
        data = data[data.episode_index.isin(episodes.episode_index)].copy().sort_values("index").reset_index(drop=True)
        check_rows(data, episodes, info["fps"])
        source_tasks = pd.read_parquet(source / "meta/tasks.parquet")
        task_mapping = {}
        for text, row in source_tasks.iterrows():
            if text not in tasks:
                tasks.append(text)
            task_mapping[int(row.task_index)] = tasks.index(text)
        data["task_index"] = data.task_index.map(task_mapping)
        if data.task_index.isna().any():
            raise ValueError("Unknown source task index")
        data["index"] += frame_offset
        data["episode_index"] += episode_offset
        episodes["episode_index"] += episode_offset
        episodes["dataset_from_index"] += frame_offset
        episodes["dataset_to_index"] += frame_offset
        episodes["data/chunk_index"] = 0
        episodes["data/file_index"] = source_index
        episodes["meta/episodes/chunk_index"] = 0
        episodes["meta/episodes/file_index"] = 0
        for camera, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            chunk_key, file_key = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
            pairs = episodes[[chunk_key, file_key]].drop_duplicates()
            original_chunks = episodes[chunk_key].copy()
            original_files = episodes[file_key].copy()
            for src_chunk, src_file in pairs.itertuples(index=False, name=None):
                src = source / info["video_path"].format(video_key=camera, chunk_index=int(src_chunk), file_index=int(src_file))
                if not src.is_file():
                    raise FileNotFoundError(src)
                seconds = duration(src)
                mask = (original_chunks == src_chunk) & (original_files == src_file)
                bounds = episodes.loc[mask, [f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp"]]
                if (bounds.iloc[:, 0] < 0).any() or (bounds.iloc[:, 1] > seconds + 1 / info["fps"]).any():
                    raise ValueError(f"Video timestamp outside file duration: {src}")
                spans = bounds.iloc[:, 1].to_numpy() - bounds.iloc[:, 0].to_numpy()
                expected = episodes.loc[mask, "length"].to_numpy() / info["fps"]
                tolerance = 1 / info["fps"] + 1e-5
                if (spans < expected - tolerance).any():
                    raise ValueError(f"Video interval shorter than requested action timestamps: {src}")
                for row_index, extra in zip(episodes.index[mask], spans - expected, strict=True):
                    if extra > tolerance:
                        report["video_extra_duration"].append({"source": str(source), "camera": camera,
                            "episode_index": int(episodes.at[row_index, "episode_index"]), "extra_seconds": float(extra),
                            "action": "Preserved original video bounds and sampling origin; content alignment not inferred."})
                dst_file = video_index[camera]
                video_index[camera] += 1
                dst = output / info["video_path"].format(video_key=camera, chunk_index=0, file_index=dst_file)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if copy_videos:
                    shutil.copyfile(src, dst)
                else:
                    dst.symlink_to(src.resolve())
                episodes.loc[mask, chunk_key] = 0
                episodes.loc[mask, file_key] = dst_file
                report["videos"].append({"path": str(dst.relative_to(output)), "source": str(src), "bytes": src.stat().st_size, "duration_s": seconds})
        stats = json.loads((source / "meta/stats.json").read_text())
        for key in ["action", "observation.state"]:
            if int(stats[key]["count"][0]) != len(data):
                raise ValueError(f"Source normalization stats include wrong row count: {source}, {key}")
        all_stats.append(stats)
        data.to_parquet(output / f"data/chunk-000/file-{source_index:03d}.parquet", index=False)
        all_data.append(data[["index", "episode_index", "frame_index", "timestamp"]])
        all_meta.append(episodes)
        report["sources"].append({
            "path": str(source), "sha256": before, "episodes": len(episodes), "frames": len(data),
            "episode_offset": episode_offset, "frame_offset": frame_offset,
            "success_counts": {str(k): int(v) for k, v in episodes.episode_success.value_counts().items()},
            "orphan_rows_excluded": orphan[["index", "episode_index", "frame_index", "timestamp"]].to_dict("records"),
        })
        report["discarded_rows"] += len(orphan)
        if fingerprints(source) != before:
            raise RuntimeError(f"Source changed while preparing: {source}")
        (output / "prepare_manifest.json").write_text(json.dumps(report, indent=2))
        frame_offset += len(data)
        episode_offset += len(episodes)
        print(json.dumps({"source": source.name, "episodes": len(episodes), "frames": len(data), "excluded": len(orphan)}), flush=True)
    merged_meta = pd.concat(all_meta, ignore_index=True)
    merged_data = pd.concat(all_data, ignore_index=True)
    check_rows(merged_data, merged_meta, reference["fps"])
    if not np.array_equal(merged_data["index"].to_numpy(), np.arange(frame_offset)):
        raise ValueError("Global indices are not consecutive")
    merged_meta.to_parquet(output / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(tasks))}, index=tasks).to_parquet(output / "meta/tasks.parquet")
    final_info = dict(info)
    for key in ["total_steps"]:
        final_info.pop(key, None)
    final_info.update(total_episodes=episode_offset, total_frames=frame_offset, total_tasks=len(tasks), splits={"train": f"0:{episode_offset}"})
    (output / "meta/info.json").write_text(json.dumps(final_info, indent=2))
    (output / "meta/stats.json").write_text(json.dumps(aggregate_stats(all_stats), indent=2))
    report.update(status="validated", output=str(output), total_episodes=episode_offset, total_frames=frame_offset, tasks=tasks,
                  validation={"unique_episode_frame": True, "consecutive_global_index": True, "metadata_bounds": True, "video_paths_and_durations": True, "source_hashes_unchanged": True},
                  labels="Original episode_success preserved; no ACP/value/advantage labels generated.")
    (output / "prepare_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--first", default="straighten_cash_0928_2_nopolicy")
    parser.add_argument("--last", default="straighten_cash_1010_1_nopolicy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking unchanged source video files")
    args = parser.parse_args()
    sources = sorted(p for p in args.source_root.glob("straighten_cash_*_nopolicy") if args.first <= p.name <= args.last)
    if not sources or sources[0].name != args.first or sources[-1].name != args.last:
        parser.error("Both exact endpoints must exist; refusing an incomplete dataset range")
    result = prepare(sources, args.output, args.copy_videos)
    print(json.dumps({k: result[k] for k in ["status", "total_episodes", "total_frames", "discarded_rows", "tasks"]}))


if __name__ == "__main__":
    main()
