# Evo-RL Hot Switch

基于 [Fger23/Evo-RL](https://github.com/Fger23/Evo-RL) 的改进代码仓库，提供 π05 联合 LoRA、共享 Base 的常驻 Adapter 热切换、gRPC 推理和双 SO101 客户端。

当前主分支为代码发布版：保留运行代码、训练与导出工具、配置、操作文档和测试源码，省略演示网站、媒体文件及生成的测试数据。服务器上的模型权重、训练数据、Python 环境和运行日志仍使用各自原有路径。

## 获取代码

```bash
git clone https://github.com/Fger23/Evo-RL-h.git Evo-RL-hot-switch
cd Evo-RL-hot-switch
```

服务端和客户端使用同一份代码。完整的环境激活、源码绑定和可复制命令见 [操作指南](./操作指南.md)。

## 启动入口

在指南所述环境中，服务端执行：

```bash
python -B -u -m lerobot.scripts.hotswitch_server \
  --gpu=1 \
  --host=10.61.5.1 \
  --port=8100
```

客户端执行：

```bash
python -B -u -m lerobot.scripts.hotswitch_so101_client \
  --remote_policy.server_address=10.61.5.1:8100 \
  --task-id=A \
  --steps=0
```

端口可自定义，两端保持一致。机器人、相机和原校准配置位于 `configs/hotswitch/client_so101.json`，已按用户确认采用 `normalized` 单位和 `use_degrees=false`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/lerobot/` | Evo-RL / LeRobot 运行代码 |
| `configs/hotswitch/` | 模型注册、训练和两端配置 |
| `tools/` | 数据准备、训练、导出与验收工具 |
| `scripts/` | 环境和启动辅助入口 |
| `tests/` | 测试源码；大体积输入数据未随代码发布 |
| `docs/` | 设计、来源和操作文档 |

CPU 回归和服务器 GPU1 模型加载、离线观测 RPC 已验证；双 SO101 实机运行尚未验收。演示媒体和测试数据不属于当前代码发布包，原项目资料可在上游仓库查阅。

保留原项目许可证与来源信息，见 [LICENSE](./LICENSE) 和 [RTC 来源说明](./docs/RTC_PROVENANCE.md)。
