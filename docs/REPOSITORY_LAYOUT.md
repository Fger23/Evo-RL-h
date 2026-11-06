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

所有运行时 `src/` 文件仅统一 CRLF/LF 换行；按 LF 规范化后与最新 627 文件冻结快照逐字节相同，运行逻辑没有再改动。18 个基线缺失的数据集源码文件来自同一 Fger23/Evo-RL 历史提交 `4425ccd`，已在原训练源码中恢复。归档输入和源文件 SHA 见 `source_provenance.json`。

## 环境与运行数据

环境没有重新安装或升级：`/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/20261104_gpu1/runtime-clean/bin/python`。`source scripts/activate.sh` 使用 `PYTHONPATH` 绑定当前根目录 `src/`。单独激活旧环境或使用旧指南仍会绑定旧冻结源；不要混用两套入口。

模型、数据、日志、checkpoint 和本次迁移验收证据全部在 `/nfs/baiyuntian/assets/model/`。没有复制大权重进 Git，也没有修改原模型、正式结果、冻结源和原验收报告。历史训练脚本留在原目录，不作为新仓库的默认服务入口。

## 回退

原目录没有搬动，原激活入口仍完整保留在 `/home/baiyuntian/baiyuntian/Evo-RL-hot-switch/20261104_gpu1/inference_ops_guide_r1/activate_env.sh`，绑定正式 GPU 验收的旧 626 文件快照。在新的独立终端 source 该入口即可回到旧运行绑定；请先正常停止自己用新入口启动的服务。回退不修改模型和 checkpoint，不需重训。

## 验证范围

本次目录整合检查 Git 基线继承、全部运行时源 SHA、环境绑定、CLI 帮助、Bash/嵌入 Python 语法、CPU 热切换与录制回归、真实回环 gRPC 客户端和 pidfd 清理。没有重跑正式训练、没有启动真实 GPU 模型或连接机器人。新仓库不声称经过新的 GPU 性能验收。
