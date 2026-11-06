# 相对原生 Evo-RL 的修改说明

基线是 [Fger23/Evo-RL@d7719ea](https://github.com/Fger23/Evo-RL/tree/d7719ea3320c04ffc047d32a8a1af2e80a9807ee)，直接从该提交开始构建。本版没有复制旧 hot-switch 的训练、RPC、会话、队列、客户端或命令封装实现。Git 历史保留旧版本，旧模型和环境仍位于原路径。

## 唯一修改的上游源码

`src/lerobot/async_inference/policy_server.py` 有三项小改动：

1. 把原有 `get_policy_class(...).from_pretrained(...)` 提取为 `_load_policy()`。默认行为原样保留，专用 Adapter 服务端通过子类覆盖它。
2. 把原有前处理器的 device/rename 覆盖字典提取为 `_preprocessor_overrides()`。默认字典不变；专用入口额外将 tokenizer 路径指向 bundle 中已核对 SHA 的副本。归一化参数、处理器顺序、tokenization 算法不变，磁盘上的保存配置不改。
3. 删除原来读取不存在的 `PI05Config.action_dim` 的一条调试输出。否则原生 PI05 推理会在这条日志处抛异常。动作计算和维度不改。

原 `Ready / SendPolicyInstructions / SendObservations / GetActions` 协议和服务类型、观测队列、时间戳、动作分块、后处理和控制循环保留。客户端源码、transport/protobuf、模型计算源码、训练源码、README、pyproject 均保持该上游提交的内容。

## 新增入口和加载器

| 新增文件 | 用途 |
| --- | --- |
| `src/lerobot/async_inference/adapter_policy_server.py` | 继承原 PolicyServer，只覆盖模型加载和 tokenizer 本地路径；沿用原 gRPC 服务。新增服务器 GPU/端口/Adapter 路径参数和只读检查。 |
| `src/lerobot/policies/pi05/adapter_inference.py` | 严格装载已有 Base + A160000 未合并 PEFT；不修改 PI05 网络或采样函数。 |
| `tests/async_inference/test_adapter_server.py` | 使用真实 loopback gRPC 与 CPU 假模型测试原协议兼容性。 |
| `tests/async_inference/test_adapter_loading.py` | CPU 权重恢复、PEFT、损坏拒绝和配置兼容测试。 |
| `操作指南.md`、本文件 | 具体启动命令及差异说明。 |

原生 PI05 `from_pretrained` 会捕获部分权重错误并返回未完整恢复的模型，因此专用加载器直接复用原生 key remapper 后执行严格 Base 权重恢复，失败立即抛出。随后按保存的 76 个精确目标挂载 LoRA，并逐张量核对全部 152 个 Adapter 权重。不会用默认 PEFT targets 替换训练时的 targets，也不合并成 dense 模型。

Adapter SHA256 固定为 `ae2253b81accf9d2f417f3d720137ddd5ca6f5c4809f576cdf779352f6574f22`。Base、处理器、tokenizer 使用导出 bundle 的 SHA 清单。12 维状态/动作、三路相机、chunk50、rank32、alpha64 均核验。

旧训练配置比原生配置多 `rtc_training_max_delay` 和 `tokenizer_name` 两个字段，专用加载器仅在内存中适配：前者对应原生的训练用 `action_conditioning/max_delay`；后者由保存的 tokenizer pipeline 和 bundle 本地路径承担。推理关闭训练用 gradient checkpointing 和 compile，保留 BF16。其他未知配置字段仍报错，不修改模型目录。

## 客户端保持原生

纯推理继续运行 `python -m lerobot.async_inference.robot_client`。带人工介入和数据集录制继续运行原 `lerobot-human-inloop-record --remote_policy.*`。两者都通过原 `RemotePolicyConfig` 发送服务器上的 Adapter 路径，客户端不需要 PEFT 或大模型副本。

原生默认客户端没有启用其另存的 RTC 客户端文件；本次没有接入旧 hot-switch 的 prefix、epoch 或双任务机制，也不继承它的验收指标和多客户端隔离保证。A 是唯一训练过的 Adapter。

## 发布和验证范围

按此前“只需要代码”的要求，发布时省略 `website/` 演示站点及媒体；测试源码保留，不上传权重、数据集、Python 环境或生成测试资源。

测试和文件检查不等于实机验收。此修改不执行机器人连接、校准或动作；服务端仅在原生客户端握手时加载模型。原生机器人连接、校准、力矩和停止行为均保留，请使用已有校准和真实设备参数。服务器沿用已有 PEFT `0.19.1` 环境，上游不同安装组合需另外验证。

本版在 Linux 上完成全部 75 项 async_inference CPU 测试（包含新增 29 项）及实际 A160000 完整 Base+PEFT 严格加载。用保存的离线观测调用原生 `_predict_action_chunk`，得到有限的 `50×12` 原始和后处理动作，整个验证没有初始化 CUDA。原生旧测试退出时有一条后台日志句柄关闭警告，保留在原始记录中；未修改上游测试或日志实现。此次没有启动 GPU 模型，也没有执行实机测试。
