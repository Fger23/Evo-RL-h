# RTC 来源核对

当前直接基线是 Fger23/Evo-RL 提交 e9d277ae0650fb3fb7e991fcfa87399b20ce1a66，不是直接 checkout Fger23/test-evorl 的 training-RTC 分支。

2026-11-06 核对该分支固定提交 98e434d4afbf39dc79060c89e8fdf3d1c3c7fbe7：configuration_pi05.py 的 Git blob 与本项目基线相同；_sample_training_rtc_prefix_mask、_build_flow_matching_inputs、_reduce_training_rtc_loss 三个训练核心函数与当前代码逐字相同。该分支额外包含不同的推理 prefix helper，不能声称整套推理实现逐字一致。

本项目在既有训练时 RTC 上增加联合 LoRA、共享 Base 热切换、会话/epoch/request 隔离和原始/执行前缀对齐。训练时 clean-prefix RTC、传统梯度 RTC、推理控制队列是独立路径；没有把三个概念混为同一实现。本次单命令入口没有替换这些训练核心函数。

来源：
- https://github.com/Fger23/Evo-RL/commit/e9d277ae0650fb3fb7e991fcfa87399b20ce1a66
- https://github.com/Fger23/test-evorl/blob/98e434d4afbf39dc79060c89e8fdf3d1c3c7fbe7/src/lerobot/policies/pi05/modeling_pi05.py

单命令客户端的具体双臂ID、串口和相机来自该固定提交的 RTC_run.txt；当前服务器已改为node3私网10.61.5.1、GPU1和正式A160000，未沿用旧GPU7或旧模型。该脚本未记录straighten_cash采集时的实际use_degrees，所以训练单位不会仅凭驱动默认值冒充已确认。
