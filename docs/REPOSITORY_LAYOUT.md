# 仓库结构与迁移

服务器仓库根目录：`/home/baiyuntian/baiyuntian/Evo-RL-hot-switch`。这是带 Git 历史的工作仓库，分支 `hot-switch/unified-r1`。

| 目录 | 内容 |
| --- | --- |
| `src/lerobot/` | 原 Evo-RL / LeRobot 主包；热切换位于 `async_inference/hotswitch/` |
| `tests/hotswitch/` | 热切换、联合 LoRA、冻结 Base、录制归属的 CPU 回归 |
| `tests/operations/` | 服务所有权 guardian 和真实回环 gRPC 的 CPU 测试 |
| `scripts/activate.sh` | 激活现有环境并绑定当前仓库 |
| `scripts/hotswitch/` | 启动、就绪检查、停止与 guardian |
| `examples/hotswitch/` | 推理客户端示例 |
| `configs/hotswitch/` | LoRA、数据、训练和节点部署配置 |
| `tools/` | 数据准备、训练流程、导出与验收工具 |
| `docs/` | 使用、设计与原项目文档 |
| `20261104_gpu1/` | Git 忽略的旧环境、冻结快照和历史操作代码，保留原路径 |

## 合并来源

1. 完整 Git 基线 `e9d277ae0650fb3fb7e991fcfa87399b20ce1a66`，包括原有 examples、website、docker、benchmarks 和根配置。
2. 正式训练批准源码，包含 Value 恢复时 CPU postprocessor caller 修正。
3. 单独审查并 CPU 验证的录制归属补丁：`recording_loop.py`、`recording_remote_policy.py`、`utils/recording_attribution.py`。
4. 新操作指南中的启动和推理工具，改为相对仓库根定位；guardian 字节保持原已验收版本。

190b2a2 整合版本的运行时 `src/` 仅统一 CRLF/LF 换行。后续单命令版本新增 hotswitch_server / hotswitch_so101_client，给原 CLI 增加监听地址和正常退出，并给异步日志增加终端输出开关；训练实现和 HotSwitchClientCore 未修改。原 626/627 冻结树仍原位保留。18 个基线缺失的数据集源码文件来自同一 Fger23/Evo-RL 历史提交 `4425ccd`，已在原训练源码中恢复。归档输入和源文件 SHA 见 `source_provenance.json`。

上游基线把 16 个完整 Markdown 文档误标为 Git 符号链接，Linux checkout 会失败。本仓库保留正文并把这些条目修正为普通文件。

## 单命令入口

服务端 `bash scripts/server.sh --gpu 1`；双 SO101 客户端 `bash scripts/client.sh`。设备配置来自用户指定 test-evorl/training-RTC 的 RTC_run.txt，见 docs/RTC_PROVENANCE.md。两个脚本不依赖 tmux、RUN_DIR 或每次启动目录；客户端不依赖服务端绝对环境路径或模型权重。旧的 scripts/hotswitch/server_start.sh 仅保留为按次留证入口。

## 环境与运行数据

环境没有重新安装或升级：`/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/20261104_gpu1/runtime-clean/bin/python`。`source scripts/activate.sh` 使用 `PYTHONPATH` 绑定当前根目录 `src/`。单独激活旧环境或使用旧指南仍会绑定旧冻结源；不要混用两套入口。

模型、数据、日志、checkpoint 和本次迁移验收证据全部在 `/nfs/baiyuntian/assets/model/`。没有复制大权重进 Git，也没有修改原模型、正式结果、冻结源和原验收报告。历史训练脚本留在原目录，不作为新仓库的默认服务入口。

## 回退

原目录没有搬动，原激活入口仍完整保留在 `/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/20261104_gpu1/inference_ops_guide_r1/activate_env.sh`，绑定正式 GPU 验收的旧 626 文件快照。在新的独立终端 source 该入口即可回到旧运行绑定；请先正常停止自己用新入口启动的服务。回退不修改模型和 checkpoint，不需重训。

## 验证范围

190b2a2 目录整合只做了 CPU 验证。后续单命令版本的 232 项 CPU 回归全部通过（179 core、38 tools、8 guardian、7 inference），其中新增 19 项覆盖双臂假硬件、真实回环 RPC、模型指纹拒绝和连接失败清理。新服务端命令已在授权 GPU1 真实加载 A160000，并用已有离线观测获得有限的 50×12 预测；随后正常退出，没有创建新的运行目录。没有重训、连接机器人或重跑 GPU 性能基准，不能把这次启动验证写成真机验收。
