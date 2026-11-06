# Evo-RL Hot Switch 推理操作指南

以下命令在服务器的 Bash 中执行。根目录统一为 `/home/baiyuntian/baiyuntian/Evo-RL-hot-switch`。开两个 SSH 终端，一个运行服务，一个发送推理请求。

## 1. 激活环境并绑定本仓库

```bash
ssh -p 2222 baiyuntian@10.61.5.1
cd /home/baiyuntian/baiyuntian/Evo-RL-hot-switch
source scripts/activate.sh
python -B - <<'PY'
import sys, lerobot
print(sys.executable)
print(lerobot.__file__)
PY
```

源码应显示 `/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/src/lerobot/__init__.py`。解释器复用 `/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/20261104_gpu1/runtime-clean/bin/python`。每个新终端都执行激活。节点路径配置在 `configs/hotswitch/node3.env`，IDE 的 SSH 解释器选择同一绝对路径，PYTHONPATH 指向当前仓库 `src/`；当前 venv 不需要重新 pip 安装。

CPU 客户端和 guardian 的 `CUDA_VISIBLE_DEVICES` 为空，模型子进程才使用选定 GPU。当前只授权物理 GPU1，UUID `GPU-4e74582b-8cca-a9c3-8d95-73f7babc3c2a`，模型内编号为 `cuda:0`。

## 2. 服务端选择 GPU 并启动

每次服务使用新的 SESSION；不要预先创建这个 SESSION 目录。

```bash
mkdir -p "$EVORL_INFER_ROOT"
SESSION="$EVORL_INFER_ROOT/service_$(date -u +%Y%m%dT%H%M%SZ)_$$"
echo "$SESSION"
bash scripts/hotswitch/server_start.sh --gpu 1 --port 8099 --backend shared --session "$SESSION" --check
```

`--check` 仅检查路径、源码和 GPU UUID，不加载模型。实际前台运行：

```bash
bash scripts/hotswitch/server_start.sh --gpu 1 --port 8099 --backend shared --session "$SESSION"
```

后台常驻运行使用下面这一条，前台和后台任选一种，不要重复启动：

```bash
nohup bash scripts/hotswitch/server_start.sh \
  --gpu 1 --port 8099 --backend shared --session "$SESSION" \
  > "${SESSION}.launcher.log" 2>&1 < /dev/null &
```

保存输出的 SESSION 完整路径，供第二个终端和停止命令使用。脚本持有现有共享 GPU 协调锁，并在选定卡已有计算进程时退出；不会停止其他进程。`--port 0` 可选择空闲端口，以 ready 文件的实际端口为准。`dual-unmerged` 是两个独立 Base 的对照模式，日常使用 `shared`。

注册表：`$EVORL_REGISTRY`，即正式 RUN 的 `postrun_r1/bundles/registry.json`。注册任务 A / A_copy；A_copy 权重与 A 相同。Base 仍从原 NFS 位置加载，Adapter 使用已完成的 A160000。

## 3. 客户端等待就绪与查询

第二个终端登录并激活，再把 SESSION 设置为第一个终端输出的真实路径：

```bash
cd /home/baiyuntian/baiyuntian/Evo-RL-hot-switch
source scripts/activate.sh
SESSION=/nfs/baiyuntian/assets/model/evorl_manual_inference/替换为实际服务目录
ADDRESS=$(python -B scripts/hotswitch/wait_ready.py --session "$SESSION" --timeout 900)
python -B -m lerobot.scripts.hotswitch_cli status --address "$ADDRESS" --timeout 30
```

## 4. 启动只返回预测的推理客户端

使用已有的真实离线观测，先查看默认 PLAN，再显式执行一次请求：

```bash
OBS="$EVORL_FORMAL_RUN/evidence/offline_observation.safetensors"
python -B examples/hotswitch/infer_client.py \
  --address "$ADDRESS" --task-id A --observation "$OBS" --output "$SESSION/prediction_A.json"
python -B examples/hotswitch/infer_client.py \
  --address "$ADDRESS" --task-id A --observation "$OBS" --output "$SESSION/prediction_A.json" --execute
```

默认 PLAN 不联网、不写结果；执行返回 50×12 的预测 chunk、原始/后处理动作、session/epoch/request 身份。客户端不加载 Base 或 Adapter，也不发送机器人控制指令。它是一次 bootstrap 推理示例，不是连续 RTC 控制循环。

服务会保留会话。再次运行新客户端时，用新输出文件并显式传 `--previous-session-id 上次结果中的session_id`，仅接管已结束的自己会话；否则重启自己的服务。不要自动接管其他客户端。

Python 调用接口保留在示例代码中：

```python
from infer_client import infer_once
result = infer_once(address, "A", observation)
print(result["processed_actions"])
```

激活脚本已把 `examples/hotswitch` 放入 PYTHONPATH。`observation` 包含三个 RGB 相机 tensor 和 12 维原始状态；图像键为 `observation.images.left_fixed`、`left_wrist`、`right_wrist`（每个完整键均以 `observation.images.` 开头），形状 `[3,480,640]`、float32 `[0,1]`，状态键 `observation.state`、形状 `[12]`。uint8 HWC 图像转 CHW 后除以 255。状态顺序和单位按真实数据集，不能自行猜测；不要重复加入 ACP 标签或归一化。

## 5. 连续离线推理 / SimRobot

在新的服务会话上运行：

```bash
python -B -m lerobot.scripts.hotswitch_cli offline-client \
  --registry "$EVORL_REGISTRY" --address "$ADDRESS" --task-id A \
  --steps 600 --control-hz 30 --switch-every 150 --timeout 60 \
  --output "$SESSION/offline_client.json"
```

该模式使用真实 RPC 和已保存观测，动作接受端是 SimRobot；A/A_copy 交替用于机制检查。`--switch-every 0` 固定 A。名义 30Hz 会受 refill hold 影响，不能据此声称稳定真机 30Hz 或任务成功率。

## 6. 日志、停止和远程端口

```bash
tail -f "$SESSION/server.stdout.log" "$SESSION/server.stderr.log"
bash scripts/hotswitch/server_stop.sh "$SESSION"
```

停止通过私有 token 的回环控制口，让 guardian 使用持有的 birth pidfd 清理自己的子进程并确认退出；不要使用按进程名批量 kill。

服务只监听回环。如需另一台机器访问，在该机器建立 SSH 隧道：

```bash
ssh -p 2222 -N -L 18099:127.0.0.1:8099 baiyuntian@10.61.5.1
```

客户端地址改为 `127.0.0.1:18099`；若服务选了端口 0，用 ready 文件的实际端口替换 8099。另一台客户端仍需兼容的协议代码和 torch/safetensors/grpc 环境，服务器的环境路径不能在不同机器直接激活。

当前入口仅进行推理/模拟。真机录制还需要经过验证的 safe_hold、动作限制和单位等回调；此指南没有连接或动作机器人。
