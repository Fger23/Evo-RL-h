# Evo-RL 服务端 / 双 SO101 客户端

日常只需要各一条命令。在各自机器的仓库根目录执行；tmux 由你自行创建和管理。启动不再创建时间戳运行目录，不需要 RUN_DIR、--session 或 ready 文件。

## 服务端

```bash
bash scripts/server.sh --gpu 1
```

它自动选择本机已有 Python 环境、绑定当前仓库源码，在前台加载正式 A160000 和共享 Base，预热后监听 8099。看到 ready JSON 后即可连接客户端。日志直接写终端；Ctrl+C 正常停止。授权 GPU1 的 UUID、模型注册表、监听地址和精度集中在 `configs/hotswitch/server.json`。

参数可直接覆盖，例如：

```bash
bash scripts/server.sh --gpu 1 --host 0.0.0.0 --port 8099
```

配置默认绑定当前节点的私网地址10.61.5.1（已在服务器网卡上核对），供另一台客户端连接；原 RPC 为明文、无网络认证的接口，只在可信内网使用。物理 GPU1 在进程内显示为 cuda:0。节点仍校验授权 UUID、持有原共享 GPU 锁；卡上已有计算进程会退出，不会停止其他进程。

只检查配置和 GPU UUID、不加载模型：

```bash
bash scripts/server.sh --check
```

普通已激活环境也可以直接用 `python server.py --gpu 1`。没有安装新环境或升级现有依赖。

## 客户端：两台 SO101 实际控制

`configs/hotswitch/client_so101.json` 已接入你指定分支 RTC_run.txt 的实际设备配置；在客户端核对这些设备身份仍对应原先左右臂和相机；训练单位已按你确认的原默认设置填好。以后固定这一条：

```bash
bash scripts/client.sh
```

也可在已激活的客户端环境中执行：

```bash
python client.py
```

默认模式就是双 SO101 真机控制，不是模拟器或仅打印预测。持续运行到 Ctrl+C；需要有限步数可加 `--steps 300`。模型权重只在服务端加载，客户端无需模型注册表或 NFS 权重目录。

## 一次性配置项来自原代码

直接复用原 `BiSOFollower/SOFollower`，不是重新实现机器人驱动。原 README 给出的机器人 ID 和校准文件约定已经放入配置：

- `robot.id`: `bimanual_follower`，来自原 RTC_run.txt。
- `calibration_dir=null` 时直接沿用原代码 HF_LEROBOT_CALIBRATION/robots/so_follower（通常为 `~/.cache/huggingface/lerobot/calibration/robots/so_follower`，兼容你原来的 HF_HOME），需要已有 `<id>_left.json` 和 `<id>_right.json`。
- 左右串口沿用原 RTC_run.txt 的 `/dev/ttyACM0` / `/dev/ttyACM1`；重启后仍需核对左右身份，建议使用稳定的 by-id 路径。
- 三路相机：左臂 `fixed`、左臂 `wrist`、右臂 `wrist`，对应模型 `left_fixed/left_wrist/right_wrist`，RGB640×480、30fps。
- 左臂6维在前，右臂6维在后，动作名称与当前12维模型相同。

直接基线 Evo-RL README 只有占位符，但你指定的 test-evorl/training-RTC 分支 RTC_run.txt 保存了实际配置，现已原样接入：左固定 RealSense 序列号260322276426，左腕 USB路径 pci-0000:c6:00.3-usb-0:2:1.0-video-index0，右腕 USB路径 pci-0000:c6:00.3-usb-0:1:1.0-video-index0。不会自动扫描或交换设备。

`address` 默认当前服务器私网地址 `10.61.5.1:8099`（没有沿用原脚本的旧服务器10.61.5.3、旧端口8090或GPU7）；客户端需能路由到该网卡。你也可以自行建立 SSH 隧道并把地址设为 `127.0.0.1:18099`。

2026-11-06 你确认采集时没有添加 `use_degrees=true`，因此沿用原驱动默认 `use_degrees=false`：身体关节使用校准行程刻度 −100～100，夹爪使用0～100。客户端已预置 `training_units="normalized"`，两臂均保持 `use_degrees=false`，无需再填写单位。用户确认来源保存在客户端配置的 `training_units_confirmation` 中；原模型 manifest 的历史 unverified 标记和模型文件保持原样。

`max_delta_per_step` 默认2个相应控制单位，拒绝突然跳变；可在 `limits` 为各个关节设置更窄的 min/max/max_delta_per_step。它不是碰撞检测或真机安全认证，需按你的原台架配置调整。

## 客户端环境与启动前检查

复用客户端原有采集环境即可。脚本优先使用 EVORL_PYTHON、已激活的 venv/conda 或仓库 .venv；不会去激活另一台服务器的绝对路径。需指定本机解释器时仍可一条命令：

```bash
EVORL_PYTHON=/你的客户端环境/bin/python bash scripts/client.sh
```

检查本地配置，不导入机器人 SDK、不联网、不连接硬件：

```bash
bash scripts/client.sh --check
```

真实启动先核对服务端模型指纹、任务、动作顺序、三路相机和频率，再创建驱动。缺失配置会在连接前报出具体项。不会启动自动校准、motor setup 或重写 PID；要求原校准与电机内配置一致，已为位置模式。连接时先写入当前位置目标再使能力矩，避免沿用旧目标。

## 停止、重连与推理控制

客户端 Ctrl+C：停止发新动作、尝试保持当前位置并关闭通信；保持力矩不会被自动关闭。服务端 Ctrl+C：停止 RPC，退出进程并释放 GPU 锁。

服务端已有客户端会话时，不会自动抢占。结束你自己的上一客户端后，可显式重连：

```bash
bash scripts/client.sh --previous-session-id 上次启动输出的session_id
```

客户端使用原 HotSwitchClientCore：确认实际送出的命令后才推进计数，RTC refill 保留原始/执行前缀校验。当前 refill 是同步推理并保持，不宣称无停顿30Hz。服务端 task A_copy 只是同权重机制副本，客户端默认只控制任务 A。

本次代码测试使用 CPU 假机器人和真实 RPC，不等于真机验收；服务器验证模型加载时也不会执行机器人客户端。
