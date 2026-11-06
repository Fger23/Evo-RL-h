# 单任务价值模型、真实 ACP 标签与联合 Adapter 工作流

本次只训练整理纸币任务 A。`A_copy` 是同一 Adapter 的相同权重副本，用于双驻留、共享 Base 与 A→A_copy→A 机制测试，不是第二个训练任务。

## 真实路径与环境

```bash
TASK_ROOT=/nfs/lerobot/s101/hot_switch/20261103
CODE_ROOT="$TASK_ROOT/validation/clean_gpu_source_r1/source_snapshot"
cd "$CODE_ROOT"
PY="$TASK_ROOT/env-clean/bin/python"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 WANDB_MODE=disabled
```

独立 venv 复用原环境包，但原 editable lerobot 指向另一个仓库，因此必须保留上述 `PYTHONPATH`。不要在原环境重新安装 extras。环境与导入来源证据见 `validation/remote/remote_audit.json`。
已完成的 pilot 200 步使用旧的独立 `env`；上述 `env-clean` 随后完成了真实 GPU smoke、
实际主 trainer 的 2 步训练/恢复与冻结 Base 审计。不要把这两次训练的范围或环境合并记账。

训练数据是已验证的新副本 `$TASK_ROOT/data/straighten_cash_registered_r2`：19 份来源、291 个已登记 episode、332523 帧。16 个未登记录制行已排除。原始数据及已有聚合没有写入。视频区间和未知动作单位的限制见 `validation/remote/DATA_AUDIT.md`。

## 执行顺序与预算

`configs/hotswitch/straighten_cash_value.json` 记录真实路径和两套预算：

| profile | value | ACP/TRC 联合 Adapter | 含义 |
|---|---:|---:|---|
| pilot | 200 steps，batch 8 | 200 steps，batch 1 | 短程机制和性能测试，不是正式收敛模型 |
| formal | 50000 steps，batch 8 | 160000 steps，batch 8 | 预算来源为已有整钞配置，未声称对 LoRA 最优 |

串行步骤是 value 训练 → 将干净数据复制到 run_root/labeled_dataset → 在该副本上完整 value infer 并生成真实 advantage/indicator → 校验每行标签 → 使用仓库 `lerobot_train_trc.py` 训练联合 Adapter。不会写回原始或已有聚合数据。pilot 也对完整新数据进行 value inference，不把局部或随机标签当成真实 ACP 数据。

`--execute` 前必须有实际 GPU 机制报告与两个方案各至少 200 次的 `GPU_OFFLINE_PASSED` 基准。运行真实 GPU RPC 和模拟机器人切换验收后，才启动本次训练。没有资源时入口明确失败；不会抢占已在 GPU 5 上运行的计算任务。脚本使用物理 UUID、稳定锁和启动前复查；这些是本工作流之间的协调，不能代替集群调度器。

实际已启动的 pilot 根为 `$TASK_ROOT/runs/pilot_r1`，使用其 `source_snapshot` 冻结源码。
不要重启或覆盖该运行。下面用一个新目录示范计划检查；再次启动前仍需确认 GPU 空闲：

```bash
"$PY" tools/hotswitch_evorl_pipeline.py --profile pilot \
  --run-root "$TASK_ROOT/runs/pilot_example_new" \
  --targets "$TASK_ROOT/validation/smoke_r6/targets.json"
```

完成上述阶段门后实际运行：

```bash
"$PY" -u tools/hotswitch_evorl_pipeline.py --profile pilot \
  --run-root "$TASK_ROOT/runs/pilot_example_new" \
  --targets "$TASK_ROOT/validation/smoke_r6/targets.json" \
  --smoke-report "$TASK_ROOT/validation/smoke_r6/smoke_report.json" \
  --benchmark-report "$TASK_ROOT/validation/gpu_benchmark/summary.json" --execute
```

正式预算使用新的 run_root，例如 `formal_r1` 和 `--profile formal`。不得把 pilot 的 200 steps 报成 formal 的 50000/160000 steps。脚本拒绝覆盖已有阶段日志和模型目录；中断后的真正继续训练需用原 trainer 的 resume 流程，而非重跑同一 pipeline 阶段。

正式入口要求先冻结源码，再从实际快照运行。以下展示新目录的冻结和命令检查；
执行训练需在最后命令中增加已有的 `--execute`，并满足 GPU、机制报告和基准阶段门：

```bash
FORMAL_RUN="$TASK_ROOT/runs/formal_example_new"
"$PY" tools/hotswitch_freeze_source.py freeze --source-root "$PWD" --run-root "$FORMAL_RUN"
FORMAL_SOURCE="$FORMAL_RUN/source_snapshot"
export PYTHONPATH="$FORMAL_SOURCE/src"
"$PY" "$FORMAL_SOURCE/tools/hotswitch_freeze_source.py" verify \
  --run-root "$FORMAL_RUN" --execution-root "$FORMAL_SOURCE" --formal --check-package
"$PY" "$FORMAL_SOURCE/tools/hotswitch_evorl_pipeline.py" --profile formal \
  --run-root "$FORMAL_RUN" --targets "$TASK_ROOT/validation/smoke_r6/targets.json" \
  --smoke-report "$TASK_ROOT/validation/smoke_r6/smoke_report.json" \
  --benchmark-report "$TASK_ROOT/validation/gpu_benchmark/summary.json"
```

`formal --execute` 在读取数据、查询 GPU 或创建训练输出之前核对完整 inventory、tree SHA、
每个实际文件字节、冻结配置位置，以及父进程 `lerobot` 的导入来源。丢失/新增/改写源码、
路径逃逸或错误 `PYTHONPATH` 均拒绝执行。快照不复制数据集、模型权重、Git 元数据或缓存。

formal profile 的 Adapter 命令明确启用 `--verify_peft_base=true`。共享 trainer 对全部冻结 Base 参数在训练开始/结束计算内容哈希；正式导出要求 `adapter_A/peft_base_audit.json` 的 PASSED、unchanged=true、实际完成步数和一致前后哈希，同时关联实际冻结源码 manifest。当前旧 pilot snapshot 没有启用这项完整训练时段审计，因此其导出应标为 NOT_ENABLED，不能把数步 smoke 的 Base 哈希证明冒充整个 pilot 的证明。

`pipeline_status.json` 原子记录每个子进程 PID、GPU UUID、argv、退出码、起止时间和持续时间。日志分别为 `value-train.log`、`value-infer.log`、`adapter.log`。长期运行时由外部 shell 保持 supervisor 存活，并检查这些真实状态和日志，不以启动成功当成训练完成。

## CPU 前置检查

以下入口只用离线数据，且内部显式隐藏 GPU：

```bash
"$PY" tools/hotswitch_value_cpu_preflight.py \
  --output "$TASK_ROOT/validation/value_cpu_check_new.json"
"$PY" tools/hotswitch_acp_text_preflight.py \
  --output "$TASK_ROOT/validation/acp_text_check_new.json"
```

已执行证据见 `validation/remote/value_cpu_preflight.json` 和 `acp_text_preflight_r2.json`。ACP 文本预检中，真实状态与任务文字经过完整 processor 后，positive/negative 标记各恰一次，dropout 无标记；训练和部署 token IDs 相同，最大 152 tokens 小于 200。条件探针没有写入数据，也不是训练标签。

## 完成后导出

```bash
"$PY" tools/hotswitch_export_run.py \
  --run-root "$TASK_ROOT/runs/pilot_r1" \
  --base-path /nfs/zxlong10/models/lerobot/pi05_base \
  --targets "$TASK_ROOT/validation/smoke_r6/targets.json" \
  --offline-observation "$TASK_ROOT/validation/smoke_r6/offline_observation.safetensors" \
  --tokenizer-path /nfs/zxlong10/models/paligemma-3b-pt-224 \
  --output "$TASK_ROOT/runs/pilot_r1/bundles_minimal_tokenizer_r3" --prefix-steps 6 \
  --action-unit recorded_robot_units_unverified
```

导出器读取 actual training_step 和真实 train_config，不凭目录名判断完成；缺阶段、退出失败、步数未达预算、标签不完整或 tokenizer/数据来源不符时拒绝。复用现有 `manifest.export_task`，包括 Base 内容指纹、processor、动作顺序、归一化、RTC/ACP 元数据和训练来源。A_copy 的 Adapter SHA256 必须与 A 一致。

导出状态为 `EXPORTED_NOT_GPU_VALIDATED`。新包仍需实际 GPU 加载、预热、保存恢复及切换/RTC 验收和新基准。`recorded_robot_units_unverified` 表示真实采集单位尚缺证据；不能据离线模拟或 loss 宣称真机安全或任务成功率。本文不包含连接、校准或驱动机械臂的操作。

精简导出使用 r3 的 `export_task` tokenizer 资产筛选器；从 r1 训练入口冻结快照执行旧导出器
会复制整个外部模型目录，因此导出时须将 `PYTHONPATH` 与工具路径切到
`$TASK_ROOT/validation/clean_gpu_source_r3/source_snapshot`。原 `bundles` 及其失败基准保持不变。
