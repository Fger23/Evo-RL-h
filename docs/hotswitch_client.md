# 远程录制客户端热切换

`RemotePolicyActionClient` 保留原有单模型模式。设置
`RemotePolicyRecordConfig.hotswitch_task_id` 后改用版本 1 的 `ControlPolicy` / `Predict`
RPC；不再调用旧的整模型 `SendPolicyInstructions`。旧服务不支持新 RPC 时明确失败。

客户端构造函数只接收已有机器人对象，不调用 `robot.connect()`。
`stub=` 和 `channel=` 可注入模拟传输。通用 `RobotClient` 另提供
`robot=FakeRobot(), connect_robot=False, stub=...`，测试时必须显式使用这条无硬件路径。

## 控制线程调用顺序

```python
cfg = RemotePolicyRecordConfig(
    enable=True,
    hotswitch_task_id="A",
    hotswitch_simulation=True,  # 仅限模拟机器人
    actions_per_chunk=actual_chunk_size,
    chunk_size_threshold=0.5,
)
client = RemotePolicyActionClient(cfg, fake_robot, actual_fps, stub=fake_stub)
client.start()

# 可由其他线程排队，但只有控制线程会提交切换。
operation_id = client.request_task_switch("B")
action = client.get_action(fake_robot.get_observation(), task=None, timestep=0)
client.validate_action_before_send(action)
accepted = fake_robot.send_action(action)
client.acknowledge_action(accepted)
```

`get_action()` 在控制边界进入保持、使旧 generation 失效、等待已提交推理结束、
清理未执行队列、提交任务切换，并重新采集观测。也可先在控制线程调用
`commit_task_switch()` 以单独测量确认延迟，下一次 `get_action()` 仍会重新采集观测。
切换超时会查询状态、使用相同 operation ID 重试，再核对当前 session/task/epoch。
无法确认最终状态时保持停止，禁止继续旧任务动作。

每次动作必须先校验控制 processor 输出，再在控制 API 接受后确认。未经确认的动作
不会计入实际执行步数。processor 或硬件改写动作时，客户端清空前缀历史并保持停止；
已经发送的命令无法撤回，必须单独检查硬件控制 API 的返回语义。

配置 `hotswitch_control_file` 可从本地控制文件提交 `{ "task_id": "B",
"operation_id": "唯一字符串" }`。控制线程在下一次动作边界读取，重复 ID 不重复提交。
文件只含任务 ID，不通过网络加载任意模型路径。
录制模式在 episode 入口、第一帧之前读取并提交待处理请求。episode 期间出现的新
控制文件保持未消费，延至下一次录制入口处理；不会中途改变任务归属。

## Training-Time RTC 与录制归属

普通异步模式只有队列隔离。Training-Time RTC 任务要求服务明确声明对应能力，
任务包提供有效部署前缀长度与训练最大延迟。每个队列项同时保存归一化动作和执行动作，
新任务以零前缀 bootstrap，后续 refill 只使用该任务的 raw 队列。响应验证身份、
形状、有限性、前缀 hard clamp 和实际已接受命令数。推理期间已执行步数超出前缀覆盖时
丢弃响应并保持停止。后台旧异常与旧响应只计数，不能覆盖新任务状态。

Legacy 梯度引导 RTC 不接入此客户端热切换，配置时明确拒绝。

当前录制 dataset schema 没有统一逐帧策略身份字段，因此在一个录制 episode 内通过
客户端 API 请求任务变化会明确报错。可在同一连接的 episode 间切换，再开始下一 episode；实际任务
文本来自注册任务包。已有 `complementary_info.collector_policy_id` 字段时写入当前
task/session/epoch。无 dataset 的 teleop/reset 调用不标记为录制 episode。
人工接管、释放后的 reset 与原有数据写入流程保持不变。

## 真机前置条件

禁止把 `hotswitch_simulation=True` 用于真机。真机模式必须由已审查的集成代码传入
`hold_callback`，并为每一个动作名称配置正数 `hotswitch_max_jump` 和
`hotswitch_max_velocity`。回调必须适合实际控制模式；不能假定零位置或停止发包安全。
动作跳变和速度校验从最后一条确认命令开始，首次动作使用实际观测。未提供完整条件
时客户端拒绝启用硬件热切换。现有录制 CLI 没有自动选择通用保持动作；本次没有执行
任何真机连接、校准或命令。

离线回归：

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/hotswitch/test_recording_client.py tests/scripts/test_recording_remote_policy.py --confcutdir=tests/hotswitch -q
```

指标位于 `client._hotswitch.counters`：旧响应/异常丢弃、迟到 chunk、保持次数、
已接受命令数、bootstrap 请求数、队列耗尽次数与累计保持间隔。保持间隔从进入保持
到首条有效命令被控制 API 接受；持续未结束的保持不计入已完成间隔总和。
