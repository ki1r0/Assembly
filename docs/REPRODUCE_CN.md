# 从源码复现：安装、接线测试、装配 rollout

这份流程对应 `launch.py` 的真实执行路径。先验证环境和工具连接，再尝试任务。接线测试通过、rollout 正常结束与原生装配成功是三个不同结果。

## 1. 准备独立依赖

需要 Linux、Python 3.11 或 3.12、Docker、NVIDIA 驱动与 NVIDIA Container Toolkit，以及 Isaac Sim 5.1 支持的 RTX GPU。本机实际验证使用 Python 3.11.17、单张 RTX A5000（GPU 2，24 GB 级别）和现有镜像 `nvcr.io/nvidia/isaac-lab:2.3.0`；物理计算使用 CPU，图像渲染使用所选 GPU。没有使用本地大模型或 learned VLA checkpoint。

在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python launch.py --help
nvidia-smi
docker info
# 首次使用时按 NVIDIA 的许可和容器访问要求取得镜像：
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
```

本机使用已存在的镜像，没有验证新账户从 NVIDIA registry 下载镜像的流程。宿主依赖与容器内 Isaac/PyTorch 分开；不要把宿主的 `.venv` 当成 Isaac Python 环境。

## 2. 提供相同资产

按 [ASSETS.md](ASSETS.md) 准备完整的 USD 目录和机器人引用文件。仅创建几个同名空文件不能运行任务；换成不同的 CAD 几何也不能直接沿用当前校准。仓库不包含这些资产，只有获得同一组资产的使用者才能复现当前场景。若只公开源码、未另行提供合法取得资产的渠道，这仍是复现的外部前提。

`ASSEMBLY_ASSET_ROOT` 指向这些资产的根目录。launcher 只读挂载它，不要求旧项目的 Python 代码、旧虚拟环境或历史 rollout。配置中的场景偏移和私有评估参数由 backend 使用，不是模型输入。

## 3. 配置视觉模型 API

```bash
cp .env.example .env
# 编辑 .env，填入自己的 API key、base URL、model ID 和资产根目录。
chmod 600 .env
set -a
source .env
set +a
```

`.env` 使用 Bash 环境变量语法；含空格的资产路径要加引号。launcher 不自动读取 `.env`，运行前必须 `source` 并导出变量，或通过自己的进程管理器注入。

`OPENAI_MODEL` 是提供商实际接受的模型标识，不含 RPent 的 `openai-chat:` 前缀。本次 SoCLaas 配置使用的标识是 `x-test-1`，响应返回的模型名为 `glm-5.3-flash`。其他部署应填写自身标识，不能假定 `glm-5.3-flash` 是所有服务的请求别名。`OPENAI_BASE_URL` 使用 SDK 的 API 根地址（例如 `https://provider.example/v1`），不要填写完整的 `/chat/completions` 路径。

所选 API 需要支持图像输入、工具调用和流式响应。本次只验证当前 GLM 提供商；“OpenAI 兼容”不保证所有提供商都接受相同的 reasoning 参数或具有相同视觉能力。密钥只进入宿主 planner 环境，不写入 `configuration.json`。源码目录只读挂载到容器时，launcher 将根目录 `.env` 覆盖挂载为 `/dev/null`，避免随源码挂载暴露密钥。不要把其他凭据文件放入源码或资产目录。

## 4. 先运行真实接线测试

将 `2` 换成你实际空闲的宿主 GPU 编号：

```bash
python launch.py --gpu 2 --budget-s 120 --run-dir runs/my-smoke
```

`--run-dir` 必须是新的 episode 目录。测试应依次完成 observe、读取空 memory 索引、沿 world X 移动 -5 mm、打开夹爪到 0.08 m、stop。它不尝试插入，`native_success=false` 是正常结果。

检查：

```bash
cat runs/my-smoke/episode/planner-summary.json
cat runs/my-smoke/episode/backend-summary.json
cat runs/my-smoke/checks/raster-live-comparison.json
```

预期 planner `error=null`，terminal source 为 `model_stop`，reason 为 `integration_test_complete`；backend 没有错误且视频帧数大于零；raster 比较为 true。实际位移和误差见 `episode/ipc-tools.jsonl`。首次冷启动可耗时一两分钟，120 秒决策预算在 backend 就绪、agent loop 开始后计时，不是整个容器启动的总时限；backend 启动另有 600 秒超时。

## 5. 完整装配 rollout

```bash
python launch.py --gpu 2 --rollout --budget-s 600 --run-dir runs/my-assembly
```

默认 `nominal-aligned` 场景，从未抓持的初始状态开始。保留固定主相机 960×720、真实手腕相机 224×224、seed 0、当前任务描述、几何控制器和独立原生评估。没有按模型的结果改变 prompt 或成功标准。

可选泛化场景通过 `--case bolt-position-shift`、`--case bolt-horizontal-pose`、`--case assembly-position-shift` 选择。先复现默认场景，再比较泛化；水平起始姿态的技能限制见 [LIMITATIONS.md](LIMITATIONS.md)。降低分辨率不是等价复现实验。

## 6. 找到结果与排查错误

| 文件 | 用途 |
| --- | --- |
| `configuration.json` | 本次场景和实际启动命令 |
| `private/planner.log` / `private/backend.log` | API/planner 和 Isaac 启动及执行错误 |
| `private/backend-failed.json` / `private/backend-exception.txt` | 如果存在，定位 backend 异常 |
| `episode/planner-summary.json` | 模型/预算/求助/停止终点及 planner 错误 |
| `episode/backend-summary.json` | 独立 `native_success`、物理步数、视频状态及 backend 错误 |
| `episode/rpent-activity/GLM_ACTIVITY_LOG_CN.md` | 可读活动日志，链接完整请求/响应 |
| `episode/rpent-activity/request_*/` | 实际 API 请求和服务返回，未补造未返回的推理 |
| `episode/ipc-tools.jsonl` / `episode/actions.jsonl` | 工具指令及测量反馈 |
| `episode/episode.mp4` | rollout 视频 |

只有 backend 的独立 `native_success=true` 才是任务成功。模型说“完成”、程序返回零或插入技能返回候选状态都不足以证明它。预算耗尽、模型求助/停止和运动急停应分别报告；当前 planner 在预算超时时返回非空 error，launcher 因此退出码为 1 并打印 traceback。若 terminal 为 `wallclock_budget` 且 backend error 为 null，应记录为任务超时，不能仅凭 launcher 的 `RPent planner failed` 判断 setup 损坏。错误时先看 summary 和两份日志，不要只重复启动。

headless 环境的 GLFW 警告不能单独作为失败依据：本机接线测试存在这些警告但正常出图、执行并录制视频。真正的 backend traceback、缺失 USD 引用、无图像、API 错误或空视频需要处理。

launcher 在结束时停止并删除自己的容器，日志已保存；它不停止其他实验的容器。`runs/`、`.venv/`、`.env` 不进入源码包。完整逐帧图像、深度、机器人状态和 API 历史记录会占用数 GB 磁盘，应在实验前预留空间。分享运行日志前仍要审查图像、对话和环境路径。

API 模型及远程服务可能变化，工具选择也具有随机性；一次实测验证安装和执行流程，不保证任何硬件/账号上逐条动作或任务结果完全一致。实际本次证据见 [SETUP_ROLLOUT_REPORT_CN.md](SETUP_ROLLOUT_REPORT_CN.md)。
