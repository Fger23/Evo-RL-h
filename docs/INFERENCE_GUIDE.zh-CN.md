# Evo-RL 服务端与双 SO101 客户端操作指南

两端共用同一个 `Evo-RL-hot-switch` 仓库、同一个 `hot-switch/unified-r1` 分支。本指南采用原 README 的写法：先激活本机环境、绑定当前源码，再直接用 `python -m` 启动，参数逐行展开。

## 1. 服务端：直接复制执行

先登录服务器：

```bash
ssh -p 2222 baiyuntian@10.61.5.1
```

登录后，在服务器终端完整复制下面这一段：

```bash
cd /home/baiyuntian/baiyuntian/Evo-RL-hot-switch

source "$PWD/20261104_gpu1/runtime-clean/bin/activate"
export PATH="$PWD/20261104_gpu1/runtime-clean/bin:$PWD/20261104_gpu1/runtime-base/bin:$PATH"
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export LEROBOT_ASYNC_LOG_TO_FILE=0 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -B -u -m lerobot.scripts.hotswitch_server \
  --gpu=1 \
  --host=10.61.5.1 \
  --port=8100 \
  --registry=/nfs/baiyuntian/assets/model/straighten_cash_joint_adapter_formal_20261104_r4_node3_gpu1/postrun_r1/bundles/registry.json \
  --backend=shared \
  --dtype=bfloat16
```

服务端加载正式 A160000 和共享 Base，在 `10.61.5.1:8100` 监听。看到包含 `ready` 和 `address` 的 JSON 后再启动客户端。进程以前台方式运行，日志直接写终端，tmux 可自行管理。

`--gpu=1` 指物理 GPU1，程序会核对授权 UUID；该卡在进程内显示为 `cuda:0`。`--port` 可改成你选定的空闲端口；客户端地址中的端口要同步修改。当前8099被其他用户占用，因此默认使用已验证的8100。

环境激活和 `export` 在同一终端完成后，再次启动服务只需执行上面的 `python -B -u -m ...` 命令及其参数。

## 2. 客户端：首次取得同一份代码

在连接双 SO101 的客户端机器执行，仓库放在 `~/code/Evo-RL-hot-switch`：

```bash
mkdir -p "$HOME/code"
git clone --branch hot-switch/unified-r1 \
  ssh://baiyuntian@10.61.5.1:2222/home/baiyuntian/baiyuntian/Evo-RL-hot-switch \
  "$HOME/code/Evo-RL-hot-switch"
```

目录已经存在时，更新源码执行：

```bash
cd "$HOME/code/Evo-RL-hot-switch"
git pull --ff-only origin hot-switch/unified-r1
```

这是首次安装或更新源码的操作。日常推理不会创建时间戳运行目录。

## 3. 客户端：直接复制执行

沿用原 `RTC_run.txt` 中的 `~/miniforge3` 和 `evo-rl` 环境。在客户端终端完整复制：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$HOME/code/Evo-RL-hot-switch"

export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export LEROBOT_ASYNC_LOG_TO_FILE=0 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -B -u -m lerobot.scripts.hotswitch_so101_client \
  --config="$PWD/configs/hotswitch/client_so101.json" \
  --remote_policy.server_address=10.61.5.1:8100 \
  --task-id=A \
  --steps=0
```

客户端使用本机 CPU，读取双 SO101 与三路相机，向服务端请求动作并控制机器人。`--steps=0` 表示持续运行到 Ctrl+C；例如 `--steps=300` 表示执行300条被确认的动作命令。

两端代码相同，各自使用本机已有 Python 环境。大模型权重只保存在服务端。客户端使用原来的串口、相机和校准文件。

## 4. 可直接修改的启动参数

| 服务端参数 | 当前值 / 含义 |
| --- | --- |
| `--gpu` | `1`，当前节点授权的物理 GPU |
| `--host` | `10.61.5.1`，本机监听地址 |
| `--port` | `8100`，可自选空闲端口 |
| `--registry` | 上面命令中的正式模型注册表路径 |
| `--backend` | `shared`，共享 Base 常驻 Adapter |
| `--dtype` | `bfloat16` |

| 客户端参数 | 当前值 / 含义 |
| --- | --- |
| `--config` | 本机 `configs/hotswitch/client_so101.json` |
| `--remote_policy.server_address` | `10.61.5.1:8100`，也可写作 `--address` |
| `--task-id` | `A` |
| `--steps` | `0`，持续运行 |
| `--previous-session-id` | 显式重连自己上一会话时填写 |

例如把服务端改为 `--port=8123`，客户端就对应使用 `--remote_policy.server_address=10.61.5.1:8123`。

这些参数对应当前共享 Base 热切换服务和双臂推理控制入口。原 README 中 `teleop`、`dataset` 等参数属于录制入口；机器人硬件配置在当前客户端的 JSON 中读取。

## 5. 当前双臂和相机配置

`configs/hotswitch/client_so101.json` 已填入原 `RTC_run.txt` 的配置：

| 配置项 | 当前值 |
| --- | --- |
| 类型 / ID | `bi_so_follower` / `bimanual_follower` |
| 左臂 / 右臂串口 | `/dev/ttyACM0` / `/dev/ttyACM1` |
| 左固定相机 | RealSense `260322276426` |
| 左腕相机 | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:2:1.0-video-index0` |
| 右腕相机 | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1:1.0-video-index0` |
| 图像 / 频率 | RGB 640×480 / 30fps |
| 单位 | `training_units="normalized"`，两臂 `use_degrees=false` |

你已确认采集时没有添加 `use_degrees=true`，因此身体关节使用校准行程刻度 −100～100，夹爪使用0～100。默认 `max_delta_per_step=2`；原模型及其历史 manifest 保持原样。

客户端沿用 `HF_LEROBOT_CALIBRATION/robots/so_follower`，需要已有 `bimanual_follower_left.json` 和 `bimanual_follower_right.json`。自定义过 `HF_HOME` 时仍遵循该设置。程序不会自动交换设备、自动校准或执行 motor setup。

## 6. 检查、停止与重连

在各自终端完成前面的环境激活和源码绑定后，可先检查配置：

```bash
# 服务端：只检查配置和 GPU UUID，不加载模型
python -B -m lerobot.scripts.hotswitch_server \
  --gpu=1 \
  --host=10.61.5.1 \
  --port=8100 \
  --check
```

```bash
# 客户端：检查配置、已有校准和串口路径，不连接机器人
python -B -m lerobot.scripts.hotswitch_so101_client \
  --remote_policy.server_address=10.61.5.1:8100 \
  --check
```

如果监听失败，在服务器检查端口：

```bash
ss -H -ltnpe 'sport = :8100'
```

客户端按 Ctrl+C 会尝试保持当前位置并关闭通信，保留力矩；服务端按 Ctrl+C 会停止服务并释放 GPU 锁。已有会话时，先结束自己原来的客户端，再将上次输出的 `session_id` 通过 `--previous-session-id` 显式传入，程序不会自动抢占会话。

服务使用明文 RPC，客户端需要能访问服务器私网地址，请在原可信内网使用。已完成 GPU1 模型加载和离线观测 RPC 验证，尚未进行双 SO101 实机运行验收。
