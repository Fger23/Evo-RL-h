# π₀.₅ 联合 Adapter 的实测环境锁定

`configs/hotswitch/environment.lock.json` 记录实际版本、Transformers 提交及关键源码 SHA256、163 个已安装分发包、导入路径和验收状态。它是实测版本与源码锁，不是包含所有 wheel 哈希的完全独立离线镜像；从零重建整个依赖图尚未执行。

| 环境 | 状态 | packaging / setuptools | `pip check` |
|---|---|---|---|
| `/nfs/lerobot/s101/hot_switch/20261103/env` | r6 机制、200 次双方案 GPU 基准、600 步离线 RPC 通过；pilot 原环境 | 26.0 / 65.5.0 | 失败，两项原有约束冲突 |
| `/nfs/lerobot/s101/hot_switch/20261103/env-clean` | Linux CPU 110 个 unittest、38 个工具测试通过，无跳过；真实 GPU smoke、ACP/TRC 主入口、续训和无更新拒绝验收通过 | 25.0 / 80.9.0 | exit 0，`No broken requirements found.` |

原 `evo-rl` 和 `env` 均未修改。新 `env-clean` 由原 `/home/baiyuntian/miniforge3/envs/evo-rl/bin/python` 创建，使用 `venv --system-site-packages` 只读复用原来的大依赖，仅在自己的 site-packages 覆盖两项打包工具，并安装本仓库的独立 editable 元数据。它没有重新安装或升级 torch、PEFT、自定义 Transformers。下述 GPU 结果来自 `env-clean` 的独立实际执行，没有继承旧环境的 GPU 通过状态。

## 实际版本与来源

| 组件 | 锁定值 |
|---|---|
| Python | 3.10.12，Linux x86_64 |
| PyTorch / CUDA build | 2.7.1 / 12.6 |
| torchvision | 0.22.1 |
| Transformers | 4.53.3，`dcddb970176382c0fcf4521b0c0e6fc15894dfe0` |
| PEFT / Accelerate | 0.19.1 / 1.13.0 |
| grpcio / protobuf | 1.73.1 / 6.31.1 |
| safetensors / tokenizers | 0.7.0 / 0.21.4 |
| NumPy / SciPy | 2.2.6 / 1.14.1 |
| datasets / av / torchcodec | 4.1.1 / 15.1.0 / 0.5 |

安装记录 `direct_url.json` 指向 Hugging Face 的 `fix/lerobot_openpi` 分支及上述完整提交。四个实际安装文件已与[该提交的官方源码](https://github.com/huggingface/transformers/tree/dcddb970176382c0fcf4521b0c0e6fc15894dfe0/src/transformers/models)逐字节比对，全部相同；没有以提交号掩盖这四处的手工补丁。证据见 `validation/remote/transformers_source_comparison.json`。

| 实际安装文件 | SHA256 |
|---|---|
| `gemma/modeling_gemma.py` | `236196720ba5446704cc95a221aa43dd217e39afd72d92496f6307b1cd13a18a` |
| `paligemma/modeling_paligemma.py` | `b72b421de47b58662f9fc3f663100a333d31913d2d55a0a93a26fb364cf0657b` |
| `siglip/modeling_siglip.py` | `ef2e99500f263fdd78db9d4d9a255e29d844eb92f53c25ad640f8056700ef84b` |
| `siglip/check.py` | `93efc9ab01e707e4a68c00cb04620ef958091e11a8018d98588d3bba1cef960e` |

实际 `GemmaRMSNorm.forward` 接口为 `(self, x, cond=None)`，返回归一化结果与 gate。本仓库的 `_install_tokenwise_adarms` 只对模型实例补齐 per-token 条件广播，使训练的 `(B,T,D)` clean-prefix 条件也能通过缓存式 Expert 推理路径；没有改写已安装的 Transformers。仅检查版本号或 `PeftModel` 类型不能替代这个接口和真实前向的验证。

## extras 冲突与新增入口

实测 PEFT 0.19.1 的 `Requires-Dist` 对 Transformers 没有版本下界；本次冲突来自仓库原来的 `peft` extra 引入 `lerobot[transformers-dep]`，后者要求 `transformers>=4.57.1,<5.0.0`，排除了实测自定义分支 4.53.3。PEFT 的[正式发布配置](https://github.com/huggingface/peft/blob/v0.19.1/setup.py)与已安装元数据一致。

新增独立 `pi-peft` extra：

```toml
pi-peft = [
  "transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0",
  "peft==0.19.1",
  "scipy>=1.10.1,<1.15",
]
```

原 `pi` 和 `peft` extras 保持原值。不要将 `pi-peft` 与 `peft`、通用 `transformers-dep` 或引入这些约束的 `all` 一起用于同一个 π₀.₅ 环境。新增 extra 已通过本地 TOML/PEP 508 与实际安装元数据的约束核查，但尚未执行全新环境的完整 resolver 安装；证据见 `validation/remote/pi_peft_extra_metadata_check.json`。安装动作仅涉及两项本地工具和 editable 元数据；依赖一致性来自实际 `pip check` 与 CPU 导入，自定义 Transformers 保持实测内容。

`grpcio-tools` 不在运行环境中；生成后的绑定已纳入源码。需要重新生成协议时，单独构建环境使用 `grpcio-tools==1.73.1`，不要在运行环境升级 protobuf。

## 已执行的独立环境修复

以下为已执行命令，`env-clean` 和 `env_check_source` 已存在；重建应选另一个空目录，避免覆盖现有环境。

```bash
TASK_ROOT=/nfs/lerobot/s101/hot_switch/20261103
/home/baiyuntian/miniforge3/envs/evo-rl/bin/python -m venv \
  --system-site-packages "$TASK_ROOT/env-clean"
PY="$TASK_ROOT/env-clean/bin/python"
export PIP_REQUIRE_VIRTUALENV=true PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export TMPDIR="$TASK_ROOT/work/envclean_tmp"
"$PY" -m pip install --ignore-installed --no-deps --no-cache-dir \
  --disable-pip-version-check --index-url https://pypi.org/simple \
  packaging==25.0 setuptools==80.9.0
cd "$TASK_ROOT/env_check_source"
"$PY" -m pip install --ignore-installed --no-deps --no-build-isolation \
  --no-cache-dir --disable-pip-version-check -e "$TASK_ROOT/env_check_source"
"$PY" -m pip check
```

`env_check_source` 是单独复制的本仓库 `src`、`pyproject.toml`、`setup.py`、README、LICENSE 等构建文件；源归档 SHA256 为 `a237bc68e906e28f9688d84c3a1207d11ed2963f697c0e52550e410266a35764`。没有改动运行中的 source snapshot。

`venv` 的 ensurepip 原先自带 **本地** `setuptools-65.5.0.dist-info`。`--ignore-installed` 覆盖后，旧元数据与 80.9.0 同时存在，曾导致 `pip check` 与 `importlib.metadata` 选择不同版本。已将这个新 venv 内的旧目录移入任务目录留证，未删除或修改共享环境的任何文件：

```python
from pathlib import Path
import shutil
clean = Path("/nfs/lerobot/s101/hot_switch/20261103/env-clean").resolve(strict=True)
old = (clean / "lib/python3.10/site-packages/setuptools-65.5.0.dist-info").resolve(strict=True)
assert old.is_relative_to(clean)
destination = Path("/nfs/lerobot/s101/hot_switch/20261103/work/envclean_metadata_quarantine/setuptools-65.5.0.dist-info")
assert not destination.exists()
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(old), str(destination))
```

最终 setuptools 导入与元数据均为 80.9.0，`pip check` exit 0；torch、PEFT、自定义 Transformers、π₀.₅、TRC 入口、协议绑定与 `pkg_resources` 的 CPU 导入通过，20 个后端离线测试通过。`pkg_resources` 会发出弃用提示；本次没有把它当作依赖正确性的证据。原环境再次只读核查仍为 packaging 26.0 / setuptools 65.5.0。

## 固定运行源码，避免导入另一个仓库

原安装的 editable `lerobot` 指向 `/home/baiyuntian/training-rtc/Evo-RLT-rtc`。因此即使虚拟环境正确，也必须显式选择本次的冻结源码并用 `python -m`，不能直接调用旧的 console scripts。

pilot 实际冻结根为 `/nfs/lerobot/s101/hot_switch/20261103/runs/pilot_r1/source_snapshot`，602 个文件，tree SHA256 为 `305942b0bf859bd78403ca4ce89ac0b04439520681ef00d5d5ec8fed8e2ed319`；完整清单在同一 run 根的 `source_snapshot_manifest.json`。正式训练修改后应建立并记录新的 snapshot，而不是覆盖这个 pilot snapshot。

env-clean GPU QA 实际使用另一份不可变源码：`/nfs/lerobot/s101/hot_switch/20261103/validation/clean_gpu_source_r1/source_snapshot`，620 个文件。其 tree SHA256 为 `21c086d77146021d002140bf51cf4ca9cd0000f0a69ca717c19cbd9f2248a78a`，manifest SHA256 为 `7541b2fdc41bbf3f83bb013068abe736e35b43dd4b0a1b194814f06d2c151fdc`。运行时显式设置该 snapshot 的 `PYTHONPATH`；本次锁文件和文档的后续状态更新没有写入或改变远端 snapshot。

后端 CPU 子集复查命令（完整 Linux 110 + 38 个测试的实测日志见下文）：

```bash
TASK_ROOT=/nfs/lerobot/s101/hot_switch/20261103
SOURCE_SNAPSHOT="$TASK_ROOT/validation/clean_gpu_source_r1/source_snapshot"
PY="$TASK_ROOT/env-clean/bin/python"
export PYTHONPATH="$SOURCE_SNAPSHOT/src" CUDA_VISIBLE_DEVICES=""
"$PY" -m pip check
"$PY" -m unittest discover -s "$SOURCE_SNAPSHOT/tests/hotswitch" \
  -p 'test_backend*.py' -v
"$PY" - <<'PY'
import importlib.util, importlib.metadata as metadata
from pathlib import Path
origin = Path(importlib.util.find_spec("lerobot").origin).resolve()
snapshot = Path("/nfs/lerobot/s101/hot_switch/20261103/validation/clean_gpu_source_r1/source_snapshot").resolve()
assert origin.is_relative_to(snapshot)
assert metadata.version("packaging") == "25.0"
assert metadata.version("setuptools") == "80.9.0"
print(origin)
PY
```

snapshot 的 `src` 里可能包含 `.egg-info`，会影响 `importlib.metadata.distribution("lerobot")` 的扫描优先级。运行源码以实际 `find_spec`/模块 `__file__` 为准；审计新环境的 editable 安装来源时，应显式读取 `env-clean/lib/python3.10/site-packages/lerobot-0.4.4.dist-info/direct_url.json`。本次该文件确实指向独立 `env_check_source`。

## env-clean 实际 GPU QA

2026-11-03 19:33:59–19:41:52（Asia/Shanghai），队列的 `clean_qa` 阶段实际 exit 0，耗时 472.539 秒。运行使用上述不可变 QA snapshot 和 `/nfs/lerobot/s101/hot_switch/20261103/env-clean/bin/python`。Linux CPU 前置检查实际通过 110 个 unittest 和 38 个工具 pytest，无 Linux 跳过；日志为 `validation/remote/cpu_unittest.log`、`cpu_tools_pytest.log`，不与早先 20 个后端测试重复相加。

| 实际阶段 | 结果 |
|---|---|
| 联合 Adapter smoke，3 次更新 | 零初始化 LoRA 与 Base 最大差 0；训练后启用/禁用 Adapter 最大差 0.0212644；禁用后恢复 Base 最大差 0；76 个实际目标层全部调用 |
| 保存、共享/独立恢复、A→A_copy→A | 最大差均为 0；优化器恢复后实际再更新 1 步通过；训练延迟上限 37、预热 prefix=6 的 raw hard clamp 精确相等 |
| ACP/TRC 实际主训练入口，预算 2 | 实际完成 0→2；Base 哈希不变，六组 Adapter 均更新且有限，实际保存 tokenizer、优化器和 checkpoint 字节绑定通过验证 |
| 从 checkpoint 1 继续，仍为总预算 2 | 实际更新 1→2；连续证据为 0→1 + 1→2，实际 checkpoint/tokenizer/artifact/sidecar 验证通过 |
| 从已完成预算的 checkpoint 2 继续 | 按预期以 `No new training updates in this audit attempt` 拒绝，0 次更新，未发布新的 `last` checkpoint |

结构化汇总为 `validation/remote/clean_gpu_qa_summary.json`，实际远端原始结果在 `/nfs/lerobot/s101/hot_switch/20261103/validation/clean_gpu_qa_r1`，队列退出证据在 `validation/after_pilot_r1/queue_status.json` 的 `clean_qa` 阶段。smoke 的单步同步 GPU 墙钟为 0.6494、0.5645、0.4872 秒，包含解码、预处理、前后向、优化器和有限性检查，不包括模型加载与 checkpoint 哈希；它不是正式训练吞吐或真机控制时延。smoke 的 processed-prefix 一致性没有单独量化，此处的 prefix 精确相等只指预热时检查的 raw prefix。

这些结果确认 env-clean 的离线 GPU 机制、ACP/TRC 入口和保存续训链；正式长训练与真机验证仍为 **NOT_EXECUTED**，没有任务成功率结论。完整依赖图的全新 resolver 安装仍未执行。旧环境的原始依赖冲突及其历史 GPU 结果保留独立记录。

环境原始证据还包括 `validation/remote/environment_metadata.json`、`environment_clean_cpu.json`、`environment_original_after_clean.json`、`environment_clean_pip_check.txt`、`env_clean_install.log`。
