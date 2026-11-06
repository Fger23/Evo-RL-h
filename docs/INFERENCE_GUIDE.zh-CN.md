# Evo-RL：同一套代码，分别启动服务端和客户端

两端使用同一个 `Evo-RL-hot-switch` 仓库、同一个 `hot-switch/unified-r1` 分支。仓库里同时包含服务端和客户端，启动入口分别是 `scripts/server.sh` 和 `scripts/client.sh`。

服务端负责加载模型和 GPU 推理；客户端负责读取双 SO101 和三路相机、请求预测并执行动作。模型权重只需要保存在服务端。

## 1. 服务端完整启动命令

从本机登录服务器：

```bash
ssh -p 2222 baiyuntian@10.61.5.1
```

登录成功后，在服务器终端执行：

```bash
cd /home/baiyuntian/baiyuntian/Evo-RL-hot-switch
bash scripts/server.sh --gpu 1
```

脚本自动使用服务器已有的 Python 环境，绑定当前仓库的 `src/`，加载正式 A160000 和共享 Base，预热后监听 `10.61.5.1:8099`。看到包含 `ready` 和 `address` 的 JSON 后，再启动客户端。

在服务器任意目录，也可以直接用这一条命令启动：

```bash
bash /home/baiyuntian/baiyuntian/Evo-RL-hot-switch/scripts/server.sh --gpu 1
```

当前服务器只授权物理 GPU1，脚本会核对 GPU UUID 并使用现有共享锁。进程内的 `cuda:0` 对应这张物理 GPU1。若卡上已有计算进程，启动会报错退出。

如需手动激活服务器环境、在终端运行 Python，可执行：

```bash
cd /home/baiyuntian/baiyuntian/Evo-RL-hot-switch
source scripts/activate.sh
python server.py --gpu 1
```

以上两种方式选一种即可。使用 `bash scripts/server.sh` 时，环境选择和源码绑定由脚本完成。

## 2. 客户端首次取得同一份代码

下面的命令在连接双 SO101 的客户端机器执行，仅首次下载时运行。客户端仓库统一放在 `~/code/Evo-RL-hot-switch`：

```bash
mkdir -p "$HOME/code"
git clone --branch hot-switch/unified-r1 \
  ssh://baiyuntian@10.61.5.1:2222/home/baiyuntian/baiyuntian/Evo-RL-hot-switch \
  "$HOME/code/Evo-RL-hot-switch"
```

这会从服务器仓库取得同一份源码。客户端复用原有 Python 环境，权重仍保存在服务端。这里的目录创建属于首次安装；日常推理不会新建时间戳运行目录。

如果这个客户端目录已经存在，更新源码时执行：

```bash
cd "$HOME/code/Evo-RL-hot-switch"
git pull --ff-only origin hot-switch/unified-r1
```

## 3. 客户端完整启动命令

沿用原 `RTC_run.txt` 中的客户端环境：`~/miniforge3` 下的 `evo-rl`。在客户端新终端执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$HOME/code/Evo-RL-hot-switch"
bash scripts/client.sh
```

已经激活 `evo-rl` 并进入仓库根目录后，日常启动只需要：

```bash
bash scripts/client.sh
```

已激活该环境时，在客户端任意目录也可以执行：

```bash
bash "$HOME/code/Evo-RL-hot-switch/scripts/client.sh"
```

客户端脚本会使用本机已激活的环境，并把当前仓库的 `src/` 放在源码搜索路径首位。客户端默认连接 `10.61.5.1:8099`，使用任务 A，持续控制两台 SO101，直到按 Ctrl+C。默认模式就是实际机器人控制。

## 4. 两端日常命令对照

| 机器 | 在各自仓库根目录执行 |
| --- | --- |
| 服务端 | `bash scripts/server.sh --gpu 1` |
| 客户端，原 `evo-rl` 环境已激活 | `bash scripts/client.sh` |

两端运行的是同一份代码。服务端环境复用服务器已有 `20261104_gpu1/runtime-clean`，客户端环境复用原 SO101 采集环境。各自机器的 Python 环境路径可以不同。

tmux 由你自行创建和管理。程序以前台方式运行，日志输出到当前终端；启动不需要 `RUN_DIR`、`--session`、ready 文件或额外运行目录。

## 5. 当前配置与可选命令

服务端读取 `configs/hotswitch/server.json`；客户端读取 `configs/hotswitch/client_so101.json`。两份配置都在同一个仓库中。

服务端显式指定现有地址和端口：

```bash
bash scripts/server.sh --gpu 1 --host 10.61.5.1 --port 8099
```

客户端显式指定现有服务地址：

```bash
bash scripts/client.sh --address 10.61.5.1:8099
```

只检查配置，不启动模型或连接机器人：

```bash
# 在服务端仓库执行：检查服务配置和 GPU UUID
bash scripts/server.sh --check
```

```bash
# 在客户端仓库、原 evo-rl 环境中执行：检查配置、已有校准和串口路径
bash scripts/client.sh --check
```

客户端已使用你原 `RTC_run.txt` 的设备配置：

| 配置项 | 当前值 |
| --- | --- |
| 机器人类型 / ID | `bi_so_follower` / `bimanual_follower` |
| 左臂 / 右臂串口 | `/dev/ttyACM0` / `/dev/ttyACM1` |
| 左固定相机 | RealSense `260322276426` |
| 左腕相机 | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:2:1.0-video-index0` |
| 右腕相机 | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1:1.0-video-index0` |
| 图像 / 频率 | RGB 640×480 / 30fps |
| 关节单位 | `training_units="normalized"`，两臂 `use_degrees=false` |

你已确认采集时没有添加 `use_degrees=true`，因此身体关节沿用校准行程刻度 −100～100，夹爪沿用0～100，单位配置已经填好。

客户端沿用原校准目录 `HF_LEROBOT_CALIBRATION/robots/so_follower`，需要已有 `bimanual_follower_left.json` 和 `bimanual_follower_right.json`；自定义过 `HF_HOME` 时仍遵循该设置。串口与相机路径应对应原先的左右设备，程序不会自动交换设备、自动校准或执行 motor setup。

默认每步最大位置变化为2个控制单位，配置项为 `max_delta_per_step`。服务器接口使用明文 RPC，客户端需要能访问服务器私网地址；请在原可信内网使用。

## 6. 停止与重新连接

在客户端按 Ctrl+C，程序会尝试保持当前关节位置并关闭通信，保留力矩。在服务端按 Ctrl+C，服务停止并释放 GPU 锁。

如果客户端提示服务端已有会话，先结束自己原来的客户端，再用上次启动输出的 `session_id` 作为 `--previous-session-id` 参数重新连接。程序不会自动抢占其他客户端会话。

已完成 CPU 回归和 GPU1 实际模型加载、离线观测 RPC 推理验证；尚未进行双 SO101 实机运行验收。当前 refill 会同步推理并保持位置，实际控制包含推理等待时间。
