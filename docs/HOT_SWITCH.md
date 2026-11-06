# π₀.₅ 联合 Adapter 离线性能与常驻切换

本次用户已决定先训练整理纸币一个 Adapter。`A_copy` 是同一 Adapter 的机制测试副本，
不代表第二个任务。真实训练预算、任务成功率、模拟机器人和真机结果分开记录。
所有训练、导出和基准输出目录须为新目录；示例路径若已经执行过，请换新后缀，
不要覆盖旧结果。已通过的机制测试路径为 `validation/smoke_r6`。

## 路径与环境

源码：`E:\ABYT\a0810\1103\hot`，基于 Fger23/Evo-RL `e9d277ae…`。
远程隔离根：`/nfs/lerobot/s101/hot_switch/20261103`。
pilot 200 步使用该目录的 `env/bin/python`；后续 GPU 验收和部署命令使用已通过验证的
`env-clean/bin/python`。二者只读复用原环境的大依赖，不替换自定义 Transformers。
训练入口 GPU QA 源码冻结在 `validation/clean_gpu_source_r1/source_snapshot`；
数值对照、精简 tokenizer 导出与真实 RPC 基准入口使用 `validation/clean_gpu_source_r3/source_snapshot`，
修复后的实际执行状态以最终报告为准。
当前工作区后续更新的文档与状态元数据不改动该执行快照。
`source_provenance.json` 记录未提交源码文件哈希。原环境 editable lerobot 指向另一个仓库，
每个命令都必须显式设置本次 `PYTHONPATH`，不能直接运行原来的 console script。

```bash
ssh -p 2222 baiyuntian@10.61.5.3
TASK_ROOT=/nfs/lerobot/s101/hot_switch/20261103
CODE_ROOT="$TASK_ROOT/validation/clean_gpu_source_r3/source_snapshot"
cd "$CODE_ROOT"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=GPU-73a48847-6cea-193b-12ab-0468514fcd3a
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="$TASK_ROOT/env-clean/bin/python"
"$PY" -m lerobot.scripts.hotswitch_cli env-check --include-torch --output "$TASK_ROOT/validation/environment_new.json"
```

先只读查询 `nvidia-smi -i 5` 及计算进程，若 GPU 已被其他工作占用就停止。
训练工作流会再次检查 GPU，并使用 UUID 绑定指定物理卡。

## 数据准备与真实标签

原数据范围为 19 份 `straighten_cash_0928_2_nopolicy` 至
`straighten_cash_1010_1_nopolicy`，291 已登记 episodes、332523 frames。
旧聚合混入 16 帧未登记 episode，造成重复 frame_index，不能沿用其 ACP 标签。
新副本 `data/straighten_cash_registered_r2` 排除未登记帧，保留真实 episode_success。
原 data/meta 文件前后哈希相同；视频只建立引用。manifest 记录视频长尾区间，
只验证请求时间在视频范围内，不声称已人工确认图像动作内容同步。

可重建到一个**新的**目录：

```bash
"$PY" tools/hotswitch_prepare_data.py \
  --source-root /nfs/lerobot/s101/datasets \
  --first straighten_cash_0928_2_nopolicy --last straighten_cash_1010_1_nopolicy \
  --output "$TASK_ROOT/data/straighten_cash_registered_new"
```

Value 推理会写回数据；工作流只向新建 `labeled_dataset` 写入，保留原始数据和干净副本。
不生成假的 success/advantage 标签。带真实标签的数据才可进入 ACP/TRC trainer。

## 阶段门与机制自检

本地纯离线测试：

```powershell
cd E:\ABYT\a0810\1103\hot
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests/hotswitch -v
python -m lerobot.scripts.hotswitch_cli benchmark --switches 200 --output validation/simulation_benchmark
```

GPU smoke 使用真实 π₀.₅、真实数据、真实反传，联合覆盖 PaliGemma 语言层、FM Expert、
action_in/out 和 time_mlp_in/out。rank 32 / alpha 64 / dropout 0 为实验起点。
SigLIP 默认不注入；扩展 targets 使用 `audit_targets --extended`，视觉需显式 `--include-vision`。

```bash
"$PY" -u -m lerobot.scripts.hotswitch_training smoke \
  --base-path /nfs/zxlong10/models/lerobot/pi05_base \
  --dataset-root "$TASK_ROOT/data/straighten_cash_registered_r2" \
  --repo-id straighten_cash_registered_r2 --output "$TASK_ROOT/validation/smoke_new" \
  --action-unit recorded_robot_units_unverified --steps 3 --rank 32 --alpha 64 \
  --rtc-max-delay 37 --prefix-steps 6 --device cuda \
  --tokenizer-path /nfs/zxlong10/models/paligemma-3b-pt-224
```

37 来自该任务实际旧 TRC 配置；6 是本次离线实验前缀长度，不是确认过的真机延迟。
数步 smoke 不证明任务已经学会 RTC，也不算 ACP 验收。报告检查各分支真实更新、
全 Base 不变、保存/共享恢复、独立恢复、继续训练和 A→A_copy→A 一致性。
容差事先由 `--atol/--rtol` 设置，不能事后掩盖失败。
前次失败和重试日志均保留，只有成功生成的 smoke_report 才能通过阶段门。

## Value → Advantage → 单任务 Adapter

`configs/hotswitch/straighten_cash_value.json` 区分 pilot 与 formal。
pilot 为 value 200 步和 Adapter 200 步，仅用于先测性能。
formal 预算来自原整钞配置：value 50000 步、Adapter 160000 步，不能把 pilot 写成正式完成。
新 formal 配置还启用 `verify_peft_base`：从实际加载的 step 开始审计，在每次保存检查点后、
更新 last 指针前写入 `training_state/peft_base_audit.json`。恢复训练会核对该检查点的模型状态、
Base/Adapter 摘要及此前从零开始的连续覆盖链；本次 start/end 与实际新增更新数分别记录。
只有覆盖完整预算且所有联合分支有限更新的最终 `PASSED` 才可用于正式导出。
每份检查点证明还封存实际保存的 Adapter、config、processor、tokenizer 与训练状态文件 SHA；
最终报告关联真实最终检查点 sidecar 的路径和哈希。恢复与正式导出都会重算这些字节，
并核对实际选择的检查点目录；替换模型文件、缺文件或改向其他 last 目标都会拒绝验收。
缺少历史、坏链或没有新增更新的恢复尝试会拒绝验收；旧 pilot 快照未启用此项，须标 `NOT_ENABLED`。

以下先打印真实命令；增加 `--execute` 后依次执行，任何失败停止后续阶段：

```bash
"$PY" tools/hotswitch_evorl_pipeline.py --profile pilot \
  --run-root "$TASK_ROOT/runs/pilot_r1" \
  --targets "$TASK_ROOT/validation/smoke_r6/targets.json"

"$PY" -u tools/hotswitch_evorl_pipeline.py --profile pilot \
  --run-root "$TASK_ROOT/runs/pilot_r1" \
  --targets "$TASK_ROOT/validation/smoke_r6/targets.json" \
  --smoke-report "$TASK_ROOT/validation/smoke_r6/smoke_report.json" \
  --benchmark-report "$TASK_ROOT/validation/gpu_benchmark/summary.json" --execute
```

正式命令用 `--profile formal` 和新的 run-root。可用 `--stage value-train`、
`value-infer`、`adapter` 执行单独阶段；已有产物默认拒绝覆盖。
Adapter 训练调用现有 `lerobot_train_trc.py`，保留 pi05、正训练延迟和真实 ACP 校验。
预加载 Base 不合并；恢复训练的 Adapter 保持可训练且不重复 PEFT 包装。
完整工作流、不可变源码快照及导出命令见 [价值模型与 ACP/TRC 操作记录](hotswitch_value_pipeline.md)。

## 任务包导出与服务

```bash
"$PY" -m lerobot.scripts.hotswitch_training export_task --help
"$PY" -m lerobot.scripts.hotswitch_training audit_targets --help
```

导出需要 checkpoint、实际 Base、metadata JSON、targets JSON、离线观测及 tokenizer。
任务包包含 Adapter、policy config、processor 和 tensor stats、tokenizer 内容、
Base SHA256、相机映射、动作名称/顺序/单位/频率、任务文本、RTC/ACP 与训练来源。
Base 内容、结构、tokenizer、动作 schema 不兼容时明确拒绝；归一化可不同但随任务原子切换。
tokenizer 导出只保留经过校验的词表、配置、特殊/新增词元与模板资产，拒绝模型分片、索引及自定义代码。
首次旧任务包曾误复制完整 tokenizer 来源模型目录，证据保留；精简包使用新的独立目录，
实际 15 组提示词的 token IDs、attention masks 与原来源完全相同，不修改训练权重。
注册表格式：

```json
{"base_path":"/nfs/zxlong10/models/lerobot/pi05_base",
 "tasks":{"A":"/absolute/bundle/A","A_copy":"/absolute/bundle/A_copy"}}
```

初始化阶段一次加载共享 Base，预加载所有 Adapter 和 processors，并用实际离线形状预热
无前缀和非零前缀。ready 后才开放 RPC；切换不读权重、不重新 from_pretrained、
不移动完整模型、不 merge。legacy 单模型服务仍保留原入口。

```bash
"$PY" -m lerobot.scripts.hotswitch_cli serve \
  --registry "$TASK_ROOT/runs/pilot_r1/bundles_minimal_tokenizer_r3/registry.json" --device cuda --port 0 \
  --ready-file "$TASK_ROOT/validation/server_ready_example_new.json"
```

另一个终端重复设置上述路径与环境，读取自动分配的 loopback 地址，再启动模拟机器人客户端：

```bash
ADDRESS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["address"])' \
  "$TASK_ROOT/validation/server_ready_example_new.json")
"$PY" -m lerobot.scripts.hotswitch_cli offline-client \
  --registry "$TASK_ROOT/runs/pilot_r1/bundles_minimal_tokenizer_r3/registry.json" --task-id A \
  --address "$ADDRESS" --steps 600 --control-hz 30 --switch-every 150 \
  --output "$TASK_ROOT/validation/offline_rpc_new.json"
```

手动切换时让客户端使用 `--switch-every 0`，再在第三个终端读取当前 session 并提交意图：

```bash
SESSION_ID=$("$PY" -m lerobot.scripts.hotswitch_cli status --address "$ADDRESS" | \
  "$PY" -c 'import json,sys; value=json.load(sys.stdin)["session_id"]; assert value; print(value)')
SWITCH_ID=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
"$PY" -m lerobot.scripts.hotswitch_cli switch-task --address "$ADDRESS" \
  --session-id "$SESSION_ID" --task-id A_copy --switch-id "$SWITCH_ID"
```

switch-task 只提交意图；控制线程在确认已发命令、进入保持后提交并清理未执行队列。
没有运行客户端时，意图停留 pending，不会假报已激活。
session/task/epoch/request 联合隔离观测、结果和异常。RPC 超时以同一 switch-id 确认，
结果未知继续保持。推理锁覆盖 VLM、完整 Flow Matching 和 postprocessor。

## RTC 和控制语义

普通异步队列、legacy 梯度 RTC、Training-Time clean prefix 是三个独立能力。
新增后端只支持普通异步及 training_time，legacy 通过原链路回归保留并明确拒绝混用。
新任务 bootstrap 使用 prefix_steps=0 / action_prefix=None；该任务有自己的 chunk 后恢复 refill。
raw 与 processed 配对维护，控制接口确认接受后才增加 accepted_steps。
队列保持 `(raw, processed)` 两元接口，并用 `queue_identity`、`pending_identity`、
`last_accepted_identity` 追踪同一响应的 session/task/epoch/request_id 和 observation_step。
`action_step`、`chunk_index` 从零计数，确认后的 `accepted_step` 是从一开始的累计接受计数；
切换会清除旧队列身份。录制 frame 的 task 文本随任务包更新，不含 ACP 标记；未显式指定 collector ID 时随活动 task_id 更新。
改写、裁剪、NaN、形状错误、超出前缀覆盖的响应均拒绝继续执行。
prefix 每个 FM 步 hard clamp，已知 token time=0，训练 loss 排除已知前缀。

第一版录制客户端采用同步 refill 和显式验证的保持间隔，不宣称无停顿异步 TRC。
录制 hot 路径、离线 RPC 客户端和未来硬件验收入口按控制频率限制相邻已确认命令的最小间隔；
该本地控制钟跨任务切换保留。推理超过一个周期后，下一条缓存动作仍须等待这个最小间隔，
不通过紧接着快速发送多条缓存动作追赶时钟。等待期间仍检查停止与错误状态。
resident 录制需注入实际控制模式的 safe_hold 与限位/连续性 validate_action 回调；
标准 CLI 未配置回调时拒绝启用。录制客户端还会拒绝缺失或带有 `unverified` 等未确认标记的动作单位；
即使提供保持与限位回调也不能绕过此检查。动作顺序、控制频率或单位校验失败会进入保持、
清除未执行队列并停止选取动作。离线 CLI 与 SimulatedRobot 允许未确认单位的机制测试，不能据此部署真机。
原 RobotClient 默认构造即连接行为保留兼容，测试必须注入 FakeRobot 并 connect_robot=False。

## 公平基准与结果解释

训练 200 步后的首次 bf16 dense 合并对照失败：最大 raw 误差 0.0560401，
超过预先固定的 atol=0.02、rtol=0.01；当时只完成 shared 200 条，dual 没有有效样本。
该失败不提高容差重写。修复对照使用下列两个独立实验，实际通过状态查看各运行目录。
`unmerged` 是两份各自拥有完整 Base 和 Adapter 的策略，必须明确区分于两个 dense 合并模型。
FP32 对照在模型构造前统一精度，不能与 bf16 的绝对延迟、显存混成同一组比较。

```bash
"$PY" -m lerobot.scripts.hotswitch_cli benchmark \
  --registry "$TASK_ROOT/runs/pilot_r1/bundles_minimal_tokenizer_r3/registry.json" --device cuda \
  --dual-mode unmerged --dtype bfloat16 --switches 200 --atol 0.00001 --rtol 0.00001 \
  --source-manifest "$CODE_ROOT/../source_snapshot_manifest.json" \
  --output "$TASK_ROOT/validation/bf16_independent_example_new"
"$PY" -m lerobot.scripts.hotswitch_cli benchmark \
  --registry "$TASK_ROOT/runs/pilot_r1/bundles_minimal_tokenizer_r3/registry.json" --device cuda \
  --dual-mode merged --dtype float32 --switches 200 --atol 0.0001 --rtol 0.0001 \
  --source-manifest "$CODE_ROOT/../source_snapshot_manifest.json" \
  --output "$TASK_ROOT/validation/fp32_merged_example_new"
```

真实 RPC 的 200 次切换计时入口为 `python -m lerobot.scripts.hotswitch_rpc_benchmark`。
它连接已经加载、预热的服务，使用一个模拟控制器和客户端单调时钟测量完整往返。
若服务已有会话，必须从 status 显式传入 `--previous-session-id`，不会自动抢占。
恢复成功的传输异常也计入异常数量，整体验收要求为零。它不能测量真机响应或任务成功率。

```bash
"$PY" -m lerobot.scripts.hotswitch_cli benchmark \
  --registry "$TASK_ROOT/validation/smoke_r6/registry.json" --device cuda \
  --switches 200 --atol 0.02 --rtol 0.01 --output "$TASK_ROOT/validation/gpu_benchmark"
```

共享两 Adapter 与两份独立合并模型分别加载、预热、运行，使用相同 core 与模拟控制接口。
两模型均常驻同一 GPU；内存不足就失败，不做 CPU 换入换出。
bf16 merge 的误差单独检验，保留 max_abs/MSE 和事先声明的容差。
JSONL 是原始每次记录，summary 含 P50/P95/P99、首次有效 chunk、
首个模拟控制接口接受动作、稳态推理、allocated/reserved/peak/device 显存、
存储实例与加载计数、异常计数。所有端到端计时使用同一进程 perf_counter。
新基准还在 `input_provenance.json` 保存实际 argv、注册表 SHA、各任务 manifest 与其已校验的
完整产物哈希，以及显式 `--source-manifest` 的 SHA。每个模式加载前后及运行后复查输入，
这些校验位于计时区间外。旧机制基准没有此新记录，保留原文件并单独标识其证据范围。
冷加载、预热单列；真实控制周期抖动和物理动作延迟未测。
`offline-client` 另行记录每次模拟接口确认的本地 `perf_counter` 时刻、相邻确认间隔、
相对指定周期的有符号/绝对抖动及分位数，包含 refill 和切换引起的停顿。
这些值是客户端模拟控制时序，不能作为机器人实际周期或物理动作响应。
逐行 `cycle_s` 只含本轮工作和最小间隔等待，不含最后的外层 sleep，不作为周期抖动。
RPC 运行异常会写入 `OFFLINE_RPC_FAILED`、已确认的部分记录与错误；已有输出文件拒绝覆盖。
报告的 `inference_requests` 逐请求保留身份、观测执行步、前缀长度、输入/返回的 raw 与
processed 前缀，以及响应到达时的实际接受步和已消耗前缀数，不包含图像。
真实 RPC 验收须运行到非零前缀 refill，并核对这些值；仅有 bootstrap 或预热不能证明远程前缀协议通过。
不以 loss、离线误差或模拟接受动作时刻替代任务成功率。

已执行的 3 步 Adapter 机制基准（同 GPU，各 200 次）如下；不是正式任务质量结果：

| 指标 P50 | 共享 Base + 两 Adapter | 两完整合并模型 |
|---|---:|---:|
| 切换确认 | 9.331 ms | 2.875 ms |
| 首有效 chunk | 187.649 ms | 148.249 ms |
| 首模拟接口接受动作 | 187.669 ms | 148.282 ms |
| 稳态推理 | 178.521 ms | 148.046 ms |
| allocated | 7.57 GB | 14.99 GB |
| reserved | 24.36 GB | 32.06 GB |

GB 为十进制字节；进程显存还包括 CUDA 上下文和缓存。共享方案省显存但实测更慢。
双模型合并最大绝对误差 0.0134583，符合运行前声明的 atol=0.02、rtol=0.01。
原始 JSONL 与 P95/P99 见 `validation/remote/gpu_benchmark`。
`cold_load_s` 是同一已运行进程内的后端构造，未测独立进程冷启动；
peak allocated 在预热后重置，不能当成加载过程峰值。

## 真机验收（未授权，未执行）

审阅实际采集配置，补齐每个动作单位、顺序、频率、限位及新动作连续性阈值；
为实际控制模式实现和独立验证安全保持回调。先在机械臂脱离控制的离线回放上
验证实际机器人 API 返回动作未被改写，再由用户授权连接、标定和动作执行。
随后分别 rollout ACP 联合 Adapter 与旧全参数模型，按相同任务条件统计成功率。
本次不连接机械臂、不校准、不发送动作、不更改固件。

实际验收入口为 `tools/hotswitch_robot_acceptance.py`。当前只运行过 `--help`、默认 dry-run
和完全模拟 driver 的 CPU 测试；没有运行 `--execute-hardware`。默认 dry-run 不导入 driver、
不创建机器人、不联系策略服务，输出 `NOT_EXECUTED` 与缺项清单。例如：

```bash
"$PY" tools/hotswitch_robot_acceptance.py --help
"$PY" tools/hotswitch_robot_acceptance.py \
  --registry "$TASK_ROOT/validation/smoke_r6/registry.json" \
  --operator-config configs/hotswitch/robot_acceptance.operator.example.json \
  --dry-run --output "$TASK_ROOT/validation/hardware_acceptance_dry_run_new.json"
```

示例 operator 配置故意不可用于硬件；不能把空字段或未确认单位补成猜测值。
未来得到真机授权后，操作者须提供真实名称/顺序/单位/频率，每轴 `min`、`max`、
`max_delta_per_step`，实际控制模式的保持验证证据文件与 `max_hold_s`，以及明确的 A→B→A 验收序列。
只有显式的 `--execute-hardware`、新报告路径、完整 operator 配置、已验证单位，以及实时服务 ready、
session 和 schema 检查全部通过后，脚本才允许导入可信本地 `module:factory`。
当前包的 `recorded_robot_units_unverified` 会在 driver 构造前阻止执行。

factory 接收 operator 配置中的 `driver` 字典，构造函数必须没有硬件副作用。返回的 driver 合约如下，
实际 SDK、串口和控制模式由未来经审核的本地驱动实现，本脚本不猜测这些参数：

| 方法 | 合约 |
| --- | --- |
| `connect()` / `close()` | 脚本分别调用一次；不得隐式重连；关闭行为需符合已验证的控制模式 |
| `observe_policy()` | 返回当前任务需要的最新 LeRobot `observation.*` 字典 |
| `safe_hold()` | 实现经验证、可覆盖推理等待的保持，并返回 `{"held":true,"action_schema":完整schema,"reference_action":实际保持目标向量}` |
| `validate_action(action)` | 根据真实驱动约束审查有序执行向量，明确返回 `True` 才通过 |
| `send_action(action)` | 发送有序向量并返回控制接口实际接受的向量；不得把请求值冒充接受值 |

脚本在控制线程切换，只向服务发送注册过的 task_id；动作经过真实限位和连续性检查，
raw、processed、实际接受值及身份写入报告。命令被改写、前缀超界、超出保持时长或任何异常均停止，
尝试经验证的保持并关闭连接。协议验收完成仍不代表任务成功率或物理动作延迟已测得。
