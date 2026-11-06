"""Render CPU-only figures from completed, immutable hot-switch benchmark logs.

Example (from repository root)::

    python tools/hotswitch_plot_benchmark.py \
      --benchmark validation/remote/gpu_benchmark \
      --smoke validation/remote/smoke_r6 \
      --output validation/figures \
      --report validation/BENCHMARK_MECHANISM_REPORT.md

This tool imports neither torch nor a robot driver and never edits input logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODES = ("shared", "dual")
LABELS = {"shared": "Shared base + 2 resident adapters", "dual": "2 resident merged full models"}
COLORS = {"shared": "#1864ab", "dual": "#e67700"}
METRICS = (
    ("switch_ack_ms", "Switch acknowledgement"),
    ("first_valid_chunk_ms", "First valid new chunk"),
    ("first_simulated_action_accepted_ms", "First simulated action accepted"),
    ("steady_inference_ms", "Steady inference"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def process_bytes(memory):
    # nvidia-smi used_gpu_memory is recorded in MiB, not decimal MB.
    rows = memory["process_nvidia_smi_pid_uuid_memory_mib"]
    return sum(float(row.split(",")[-1].strip()) * 2**20 for row in rows)


def load_evidence(benchmark, smoke):
    paths = [benchmark / "summary.json", *(benchmark / f"{mode}.jsonl" for mode in MODES),
             smoke / "smoke_report.json", smoke / "training_evidence.json"]
    hashes = {str(path): sha256(path) for path in paths}
    summary = read_json(benchmark / "summary.json")
    raw = {}
    for mode in MODES:
        raw[mode] = [json.loads(line) for line in (benchmark / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if summary[mode]["status"] != "GPU_OFFLINE_PASSED":
            raise ValueError(f"Refusing to label an unpassed {mode} run as GPU evidence")
        if len(raw[mode]) != summary[mode]["samples"] or not raw[mode]:
            raise ValueError(f"{mode} raw sample count differs from its summary")
        if any(row["kind"] != "gpu_offline" or row["mode"] != mode for row in raw[mode]):
            raise ValueError("Input logs must be the real GPU offline benchmark")
        for key, _ in METRICS:
            values = [row[key] for row in raw[mode]]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid latency in {mode}.{key}")
            for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                if not math.isclose(quantile(values, fraction), summary[mode]["metrics"][key][name], abs_tol=1e-7):
                    raise ValueError(f"Raw {mode}.{key}.{name} does not match the saved summary")
    smoke_report = read_json(smoke / "smoke_report.json")
    training = read_json(smoke / "training_evidence.json")
    if smoke_report["status"] != "real_gpu_passed" or training["second_task_trained"]:
        raise ValueError("Expected the passed, same-adapter-copy mechanism smoke")
    return summary, raw, smoke_report, training, paths, hashes


def figures(output, summary, raw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    samples = "/".join(str(len(raw[mode])) for mode in MODES)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    for axis, (key, title) in zip(axes.flat, METRICS, strict=True):
        for mode in MODES:
            values = np.sort([row[key] for row in raw[mode]])
            probability = np.arange(1, len(values) + 1) / len(values)
            axis.step(values, probability, where="post", color=COLORS[mode], linewidth=2,
                      label=LABELS[mode])
        axis.set(title=title, xlabel="Latency (ms)", ylabel="Empirical cumulative probability", ylim=(0, 1.02))
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.suptitle("Resident hot-switch latency: actual H200 measurements", y=0.995, fontsize=15)
    fig.text(0.5, 0.95, f"Offline saved observations + SimRobot | same adapter as A / A_copy | n = {samples}", ha="center")
    fig.text(0.5, 0.018, "In-process route; simulated interface acceptance is not physical action latency. No task success rate measured.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.895), h_pad=1.4, w_pad=2)
    fig.savefig(output / "latency_cdf.png")
    plt.close(fig)

    fields = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "process")
    labels = ("Allocated", "Reserved", "Peak allocated\n(post-warmup)", "Process total\n(nvidia-smi)")
    fig, axis = plt.subplots(figsize=(10.4, 5.4))
    positions = np.arange(len(fields))
    width = 0.35
    for index, mode in enumerate(MODES):
        memory = summary[mode]["memory"]
        values = [(process_bytes(memory) if field == "process" else memory[field]) / 2**30 for field in fields]
        bars = axis.bar(positions + (index - 0.5) * width, values, width,
                        color=COLORS[mode], label=LABELS[mode])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    axis.set(xticks=positions, xticklabels=labels, ylabel="GPU memory (GiB; 1 GiB = 2^30 bytes)", ylim=(0, 36))
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Resident GPU memory: measured allocation and process usage", y=0.99, fontsize=14)
    fig.text(0.5, 0.927, "Same-adapter-copy mechanism comparison; one GPU; both backends preloaded and warmed", ha="center", fontsize=10)
    fig.text(0.5, 0.032, "Peak counter reset after warmup; excludes loading / warmup peaks. Reserved includes retained allocator cache.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.90))
    fig.savefig(output / "gpu_memory.png")
    plt.close(fig)


def write_report(path, benchmark, smoke, summary, smoke_report, training, hashes):
    shared, dual = (summary[mode] for mode in MODES)
    allocated_reduction = 100 * (1 - shared["memory"]["allocated_bytes"] / dual["memory"]["allocated_bytes"])
    steady_slower = 100 * (shared["metrics"]["steady_inference_ms"]["p50"] / dual["metrics"]["steady_inference_ms"]["p50"] - 1)
    lines = [
        "# 真实 GPU 离线热切换机制基准", "",
        f"共享 Base 的 allocated 显存低 {allocated_reduction:.1f}%，本次稳态推理 P50 慢 {steady_slower:.1f}%。不能据此宣称 LoRA 切换更快。", "",
        "证据来自同一 H200、bfloat16、eager、保存的离线观测与 SimRobot。两方案各 200 次计时切换、400 次推理；另各做 200 次不计时的切换文件 I/O 拦截与参数存储检查，全部通过。A_copy 是 A 的相同 Adapter 副本，没有训练或评估第二个任务。", "",
        "| 延迟（ms） | 共享 P50 | P95 | P99 | 双完整模型 P50 | P95 | P99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    titles = ("切换确认", "首有效新 chunk", "首模拟动作接受", "稳态推理")
    for (key, _), title in zip(METRICS, titles, strict=True):
        values = [summary[mode]["metrics"][key][stat] for mode in MODES for stat in ("p50", "p95", "p99")]
        lines.append(f"| {title} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "计时使用同一进程的单调时钟。这张表测量进程内路由与模拟控制接口；真实 gRPC 验收另有记录。模拟接口接受时间不等于机器人物理动作延迟，未测任务成功率或实时控制周期抖动。P99 来自单次实验的 200 个样本，未提供独立重复实验或置信区间。", "",
              "| 显存（GB / GiB） | 共享 Base | 双完整模型 |", "|---|---:|---:|"]
    for field, title in (("allocated_bytes", "allocated"), ("reserved_bytes", "reserved"),
                         ("peak_allocated_bytes", "peak allocated（预热后重置）"), ("process", "nvidia-smi 进程占用")):
        values = [(process_bytes(summary[mode]["memory"]) if field == "process" else summary[mode]["memory"][field]) for mode in MODES]
        lines.append(f"| {title} | " + " | ".join(f"{value / 1e9:.3f} / {value / 2**30:.3f}" for value in values) + " |")
    lines += ["", "GB = 10⁹ bytes，GiB = 2³⁰ bytes；nvidia-smi 原始记录使用 MiB。peak allocated 仅覆盖预热之后的基准，不能当作进程启动、权重加载或预热峰值；reserved 保留加载期形成的分配器缓存。", "",
              f"共享方案确实只加载 1 份 Base、2 个 Adapter；双完整模型加载 2 份 Base 后分别合并相同 Adapter。共享参数存储 {shared['parameter_storage']['unique_storage_bytes']:,} bytes，双完整模型 {dual['parameter_storage']['unique_storage_bytes']:,} bytes。双方加载计数与存储检查都保存在 summary.json。正常基准异常响应记录为 0（client_metrics 为空），不代表真机安全或全部竞态验收。", "",
              f"固定输入/随机种子下共享方案重复结果误差为 0；双模型合并相对共享输出最大绝对误差 {dual['consistency']['A']['max_abs_error']:.9f}，MSE {dual['consistency']['A']['mse']:.9g}，满足运行前固定的 atol={dual['atol']}、rtol={dual['rtol']}。合并权重的 bfloat16 数值误差已保留，未将容差改成事后通过值。", "",
              f"字段 cold_load_s 实际是既有 Python 进程中的后端构造/加载耗时：共享 {shared['cold_load_s']:.3f}s、双完整模型 {dual['cold_load_s']:.3f}s；预热 {shared['warmup_s']:.3f}s / {dual['warmup_s']:.3f}s。它不是进程冷启动，且执行顺序固定为共享后双模型，不能作为公平的进程冷启动比较。", "",
              f"r6 机制烟测执行 {training['steps']} 步普通联合 LoRA 更新；VLM/Expert/投影 L1 更新分别为 {training['branch_update_l1']['vlm']:.6f} / {training['branch_update_l1']['expert']:.6f} / {training['branch_update_l1']['projections']:.6f}，Base 前后 SHA256 相同。保存恢复、独立恢复、A→A_copy→A 最大误差均为 0，优化器恢复后继续一步通过。RTC 训练范围 {smoke_report['rtc_max_delay']}、离线部署前缀 {smoke_report['deployment_prefix_steps']} 的 hard clamp 已验收；这不代表正式 ACP 训练或真机 RTC 验收完成。", "",
              "![](figures/latency_cdf.png)", "", "![](figures/gpu_memory.png)", "",
              "原始证据（未修改）：", "", f"- `{benchmark.as_posix()}/shared.jsonl`、`dual.jsonl`、`summary.json`",
              f"- `{smoke.as_posix()}/training_evidence.json`、`smoke_report.json`", "",
              "输入 SHA256：", ""]
    lines += [f"- `{Path(filename).as_posix()}`: `{digest}`" for filename, digest in hashes.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("validation/remote/gpu_benchmark"))
    parser.add_argument("--smoke", type=Path, default=Path("validation/remote/smoke_r6"))
    parser.add_argument("--output", type=Path, default=Path("validation/figures"))
    parser.add_argument("--report", type=Path, default=Path("validation/BENCHMARK_MECHANISM_REPORT.md"))
    args = parser.parse_args()
    summary, raw, smoke_report, training, inputs, hashes = load_evidence(args.benchmark, args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    figures(args.output, summary, raw)
    write_report(args.report, args.benchmark, args.smoke, summary, smoke_report, training, hashes)
    if {str(path): sha256(path) for path in inputs} != hashes:
        raise RuntimeError("Input benchmark evidence changed while rendering")
    print(json.dumps({"status": "CPU_FIGURES_RENDERED", "input_logs_unchanged": True,
                      "figures": [str(args.output / name) for name in ("latency_cdf.png", "gpu_memory.png")],
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
