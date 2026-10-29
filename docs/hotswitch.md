# π₀.₅ 共享 Base 热切换操作手册

这是当前 Fger23/Evo-RL 工作区的增量实现。所有自动化入口使用离线观测和模拟控制接口。正式训练和模型导出需要先补齐真实路径；示例中的 null 表示尚未确认，不是默认数据或设备。

## 从离线自检开始

在仓库根目录用可用 Python 执行，以下为 PowerShell：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m lerobot.scripts.lerobot_hotswitch --help
python -m lerobot.scripts.lerobot_hotswitch env_check --output work/environment.json
python -m pytest --noconftest tests/async_inference/test_hotswitch_runtime.py tests/async_inference/test_hotswitch_rpc.py tests/async_inference/test_hotswitch_backend.py tests/async_inference/test_task_bundle.py tests/async_inference/test_clean_prefix_sampling.py tests/hotswitch tests/scripts/test_recording_remote_policy.py tests/async_inference/test_policy_server.py -q
python -m lerobot.scripts.lerobot_hotswitch benchmark --switches 200 --output work/benchmark-async
python -m lerobot.scripts.lerobot_hotswitch benchmark --switches 200 --prefix-steps 3 --output work/benchmark-trc
```

pytest 使用 --noconftest 避开仓库全局硬件 fixture 和无关数据下载依赖；指定的测试只使用离线输入和模拟机器人。没有运行 tests/test_control_robot.py、串口、校准或真实机器人测试。测试环境不等于 π₀.₅ 模型依赖环境。

每个 benchmark 都真实运行现有 PolicyServer + localhost gRPC + RemotePolicyActionClient + 模拟控制接口。默认比较 shared 与 dual 两种常驻模拟后端，各 200 次交替切换。使用相同控制安全状态机、连接建立和预热规则。JSONL 保存切换、请求身份、控制接收和服务器本地持续时间；summary.json 给出 P50/P95/P99。模拟策略数据不能用于估算真实显存或宣布 LoRA 更快，物理动作延迟与成功率为空。

如需手动控制持续运行的模拟客户端，三个终端分别执行：

```powershell
# 服务端：没有 Base/Adapter 文件，也没有硬件
python -m lerobot.scripts.lerobot_hotswitch serve --simulate --prefix-steps 3 --host 127.0.0.1 --port 8080
# 客户端：只向内存模拟控制接口确认动作
python -m lerobot.scripts.lerobot_hotswitch offline_client --server 127.0.0.1:8080 --task A --simulated-observations --control-file work/switch.json --duration 60 --output work/offline-client
# 操作终端：客户端下一控制边界切换；每次等上一次确认后再投递
python -m lerobot.scripts.lerobot_hotswitch switch_task --control-file work/switch.json --task B
python -m lerobot.scripts.lerobot_hotswitch switch_task --control-file work/switch.json --task A
```

offline_client 在真实注册表服务器上运行时，用 `--observations mapping.json` 替代 `--simulated-observations`，mapping 为 task_id 到实际离线 NPZ 的路径映射。客户端仍只构造内存模拟机器人。合成观测只允许发送给显式标记 simulation 的策略。动作接收、任务身份和切换统计保存到输出目录。

## 环境与真实模型机制验证

不要对原来可运行的 π₀.₅ 环境直接组合安装 `[pi,peft]`；pi extra 指向自定义 Transformers 分支，通用 peft extra 依赖可能覆盖它。先复制/新建环境，读取 direct_url.json 和实际 AdaRMSNorm cond 接口，保存具体已验证的依赖提交。当前主机版本未通过这一门槛，尚无可信的生产依赖锁。

取得一份共同 Base 和离线实测观测后：

```powershell
python -m lerobot.scripts.lerobot_hotswitch audit_targets --base $BASE --device cpu --output work/targets.json
python -m lerobot.scripts.lerobot_hotswitch audit_targets --base $BASE --extended --device cpu --output work/targets-extended.json
python -m lerobot.scripts.lerobot_hotswitch smoke --base $BASE --observations $LABELED_NPZ --device $DEVICE --steps 3 --rtc-max-delay $TRAIN_MAX_DELAY --output work/real-smoke
```

`$BASE/$DEVICE/$LABELED_NPZ` 必须由本机真实资源确定；没有暗含的 GPU 号、任务文本或服务器 IP。NPZ 用 `allow_pickle=False` 读取，键为 policy 输入特征（例如 observation.state、observation.images.<camera>）和 `action`；不要放 Python object 数组。观测是未批处理样本，processor 负责 batching；真实 action 标签必须匹配模型 chunk/action 维度。模型 task 文本由机制测试参数或任务包注入。

smoke 顺序训练两个临时联合 Adapter，各至少 3 步，检查语言、Expert 及每个动作/时间投影更新、基础参数未改变、保存恢复、恢复后继续训练和固定随机条件 A→B→A 误差。结果写入 smoke_result.json，临时 Adapter 仅证明机制，不是正式任务产物，也不证明 ACP 或任务能力。GPU 用于该命令前必须确认未被其他任务占用。

联合 Q/V 目标和扩展 Q/K/V/O+MLP 目标见 configs/hotswitch 和 examples/hotswitch。SigLIP 默认不适配，显式 `audit_targets --vision` 后方可启用对应 pattern。建议起点 r=32、alpha=64、dropout=0，未宣称最优。实际目标清单需来自已加载模型的 named_modules。梯度检查点重计算保留随机状态。

## 两个真实任务训练与导出

填写 `configs/hotswitch/train_plan.json` 和 train_A.json/train_B.json。两者 policy.pretrained_path 必须引用内容相同的 Base；不得指向两个不同的全参任务模型。模型、dataset.root、output_dir、mechanism_validation_report 使用绝对路径；config_path 相对于 train_plan.json。填写明确设备、预算 steps/batch_size 和实际 rtc_training_max_delay。复制真实 smoke_result.json 路径到机制验收字段，不能手填通过结果。正式 ACP/TRC 要求此前以正数 --rtc-max-delay 验证真实模型的前缀路径；CUDA 训练还要求真实 GPU 机制报告。

```powershell
python -m lerobot.scripts.lerobot_hotswitch train_task --plan configs/hotswitch/train_plan.json --task A
python -m lerobot.scripts.lerobot_hotswitch train_task --plan configs/hotswitch/train_plan.json --task A --execute
python -m lerobot.scripts.lerobot_hotswitch train_task --plan configs/hotswitch/train_plan.json --task B --execute
```

默认只验证并打印真实 trainer 命令。正式 ACP/TRC 使用 lerobot_train_trc.py 原校验入口，真实标签统计仍在共享 trainer 中校验。没有标签不能宣称完成 ACP。普通机制自检使用 smoke 或普通 trainer。训练恢复沿用已保存 train_config.json 的 `--resume=true` 流程，恢复 Adapter 可训练且不重复包装。所有参数保持本仓库的真实命名。

填 metadata_A/B.json：动作名称顺序单位频率、任务文本、摄像头映射、实际 tokenizer fingerprint、target_layers、RTC、ACP、训练来源和 lerobot_features。`tokenizer_fingerprint` 使用 hotswitch_backend.tokenizer_fingerprint(实际 tokenizer)；必须与保存的 tokenizer 配置一致。任务文本不含 Advantage 标记，唯一注入层在后端。ACP 首版支持 positive 单分支、CFG=false、beta=1，并校验 tokenizer 截断；非 ACP 不添加标记。

```powershell
python -m lerobot.scripts.lerobot_hotswitch export_task --adapter $CHECKPOINT_A --base $BASE --metadata configs/hotswitch/metadata_A.json --output configs/hotswitch/bundles/A
python -m lerobot.scripts.lerobot_hotswitch export_task --adapter $CHECKPOINT_B --base $BASE --metadata configs/hotswitch/metadata_B.json --output configs/hotswitch/bundles/B
```

导出复制 Adapter、policy config、processor JSON 和统计文件，生成含内容指纹的 manifest.json。任意一方被改动会拒绝。Base 指纹比较权重/配置内容，不比较目录名称。归一化差异随 processor 切换。任务组合不兼容时启动前失败。

## 启动、预热与运行中切换

填写 registry.json 的 base_path 和两份真实离线 observation NPZ，路径相对于 registry 所在目录。没有离线输入不会标记 ready。预热包括真实模型形状、无前缀以及任务配置的非零前缀；输出不会进入执行队列。

```powershell
python -m lerobot.scripts.lerobot_hotswitch serve --registry configs/hotswitch/registry.json --host 127.0.0.1 --port 8080
```

默认绑定 loopback，沿用可信 pickle RPC；不扩大网络暴露范围。旧单模型服务启动方式继续存在；注册表模式明确拒绝旧客户端路径 setup。状态和健康可通过 `ControlPolicy` 的 capabilities/status 操作查询。协议版本 1 包含 session、task、epoch、request 身份，proto 已使用真实生成器更新。

重新生成绑定使用仓库已有流程：`python -m grpc_tools.protoc -I src --python_out=src --grpc_python_out=src src/lerobot/transport/services.proto`，本次工具版本为 1.73.1，隔离安装后生成，没有手工改造生成文件。

客户端详细 API 见 hotswitch_client.md：构造 RemotePolicyRecordConfig 的 hotswitch_task_id 和 hotswitch_control_file，向现有 RemotePolicyActionClient 提供模拟 Robot 或已验证的真实 hold callback/限制。实时控制仍归 recording_loop 所有。控制线程可以 `request_task_switch('B')` 后由下一次 get_action 提交；也可显式 `commit_task_switch()`，之后 get_action 自动重新采样观测。每条动作发送前 validate，控制接口接受后 acknowledge。

运行中控制文件入口：

```powershell
python -m lerobot.scripts.lerobot_hotswitch switch_task --control-file work/switch.json --task B
python -m lerobot.scripts.lerobot_hotswitch switch_task --control-file work/switch.json --task A
```

这个入口仅投递 task_id+operation_id，运行客户端在控制边界读取；不会直接切服务端而留下旧机器人队列。单槽文件用于串行人工请求，连续写入可能合并为最后一次；需要逐次确认时使用客户端 API 和状态回复。相同 operation_id 重试幂等。recording_loop 在活跃录制 episode 内明确禁止任务切换，必须结束当前 episode 才能改任务。

## 真实模型离线对照与后续验收

```powershell
python -m lerobot.scripts.lerobot_hotswitch benchmark --registry configs/hotswitch/registry.json --device $DEVICE --mode both --switches 200 --output work/real-offline-benchmark
```

shared 只保留一份 Base、两个常驻 Adapter。dual 从同一 Base 和同一对 Adapter 生成独立常驻完整模型，各自合并仅发生在加载阶段，不改主方案 Base。两边预建连接、预热并复用相同客户端安全切换。先独立验证合并输出一致性，再将真实延迟对比作为工程结论；当前该项未执行。

记录冷启动、加载、预热、首次和稳定切换；端到端时间只使用客户端单调时钟，服务端只报告本地推理持续时间。显存采集包括 allocated/reserved/peak、实际参数唯一存储、Adapter 字节数和 nvidia-smi 可取得的设备/进程内存。WDDM 不可用项保持 null。200 样本的 P99 仍有较大不确定性。

真机验收另行授权：先审核控制模式、hold 实现、跳变/速度阈值、camera/action schema、发出命令后的返回语义，再执行离线复播和安全停止演练；通过后才允许实际 A→B→A rollout。检查旧 epoch 丢弃、接管/释放、episode 归属、raw/processed 对齐、TRC 延迟超界和硬件改写命令。没有 rollout 只报告机制与离线误差，不报告任务成功率。
