# Setup 与真实 rollout 验证报告

日期：2026-10-17（Asia/Singapore）。结论：**独立仓库的安装、模型调用、真实 Isaac 执行、记录与容器清理已验证；本次完整装配任务未达到原生成功。** 这不等于保证任意机器、账号或模型版本都产生相同动作，也不构成装配成功率估计。

## 实际执行范围

直接运行此仓库，没有复制另一套实验代码来跑。创建新的仓库内 `.venv`，安装 `requirements-dev.txt`；宿主各核心模块的实际导入路径都在此仓库中。模拟器只挂载当前仓库、当前 episode 输出和外部资产目录，没有挂载旧项目的 Python 代码。原有任务 prompt、几何/伺服技能、场景 seed 0、相机、私有评估和成功判据没有改变。没有因模型未完成或未求助而改 prompt。

环境：Python 3.11.17、RTX A5000（宿主 GPU 2，单 GPU）、镜像 `nvcr.io/nvidia/isaac-lab:2.3.0`、物理 CPU、固定主相机 960×720 与手腕相机 224×224。提供商请求别名 `x-test-1`，实际响应模型 `glm-5.3-flash`，无本地 learned VLA/checkpoint。GPU 显存一次采样约 2724 MiB；这不是峰值或最低配置声明。

| 组件 | 本机实测版本 |
| --- | --- |
| 宿主 `pydantic-ai-slim` | `2.53.0` |
| 宿主 `pydantic-ai-harness` | `0.53.0` |
| 宿主 `openai` | `3.31.0` |
| 宿主 `httpx2` | `2.13.1` |
| 宿主 `numpy` | `2.4.6` |
| 宿主 `scipy` | `1.17.1` |
| 宿主 `opencv-python-headless` | `5.0.0.93` |
| 容器 `python` | `3.11.13` |
| 容器 `torch` | `2.7.0+cu128` |
| 容器 `numpy` | `1.26.0` |
| 容器 `scipy` | `1.15.3` |
| 容器 `numba` | `0.59.1` |
| 容器 `imageio` | `2.37.0` |
| 容器 `imageio-ffmpeg` | `0.6.0` |

## 可复用命令与结果

完整步骤见 [REPRODUCE_CN.md](REPRODUCE_CN.md)。填好自己的 `.env` 后，在已激活的 `.venv` 中执行：

```bash
set -a
source .env
set +a
python launch.py --gpu 2 --budget-s 120 --run-dir runs/my-smoke
python launch.py --gpu 2 --rollout --budget-s 600 --run-dir runs/my-assembly
```

实际 episode 使用 `runs/setup-smoke-20261017` 和 `runs/setup-rollout-20261017`；这些名字已经使用，不能原地重跑。GPU 编号也应换成使用者自己的空闲设备。

| 结果 | 接线测试 | 完整装配 |
| --- | --- | --- |
| Backend 就绪前耗时 | 105.30 s | 105.08 s |
| 决策阶段时长 | 21.57 s | 600.01 s |
| API 请求数 | 5 | 12 |
| HTTP 返回 | 全部 200 | 全部 200 |
| Backend 物理步数 | 185 | 9709 |
| Backend error | null | null |
| 终点 | model_stop / integration_test_complete | wallclock_budget |
| Native success | false（未尝试任务，正常） | false |
| 视频帧数 | 37 | 1941 |
| 视频时长（12 FPS） | 3.08 s | 161.75 s |
| 输出体积 | 0.14 GB | 6.37 GB |
| Launcher 退出码 | 0 | 1（预算超时） |

两段视频均为 960×720；用 imageio/FFmpeg 实际解码了首帧、末帧并统计全片帧数，与 backend 的录制计数一致。两个 episode 的实时机器人深度与参考 raster 比较均通过，保留原精度检查。自有容器均已停止和删除，未停止其他实验容器。

完整装配在 600 秒决策预算处退出，planner 的 `error` 是 `planner timed out after 600s`。因此 launcher 退出码为 1 并输出 `RPent planner failed` traceback；这应按 summary 解释为本次任务超时，而不是依赖导入失败、API 认证失败或 backend 崩溃。启动约 105 秒不包含在这 600 秒决策预算中。

## GLM 实际活动

先 observe，从当前固定 RGB 图像选中约 (755,385) 的螺栓；几何技能返回 `grasp_lifted`，实际图像也显示螺栓随夹爪抬起。随后选择约 (546,352) 的孔进行插入，工具返回 `contact_stalled_after_entry` 和落座候选，但 native success 为 false。GLM 依次释放、settle、向上撤离 0.15 m、重新观察并放大图像。

GLM 随后改变了自己对“上右方/约两点钟孔”的判断，认为应使用约 (493,313) 的另一孔。它重新抓取螺栓；一次猜测的失效 feature_ref 被执行器拒绝，返回 `invalid_action`；模型从新观测取到有效引用后继续第二次插入。第二次插入执行到预算终点，被正常中断。本次没有模型求助或模型主动 stop，也没有自动执行人类帮助。

| 顺序 | Backend 指令 | 实际返回 | 耗时 |
| --- | --- | --- | --- |
| 1 | `grasp_visual` | `grasp_lifted` | 77.75 s |
| 2 | `insert_visual` | `contact_stalled_after_entry` | 122.32 s |
| 3 | `set_gripper` | `gripper_commanded` | 0.90 s |
| 4 | `settle` | `settled` | 2.60 s |
| 5 | `move_relative` | `reached` | 12.33 s |
| 6 | `observe` | `observed` | 0.00 s |
| 7 | `grasp_visual` | `grasp_lifted` | 52.49 s |
| 8 | `insert_visual` | `invalid_action` | 0.00 s |
| 9 | `insert_visual` | `interrupted` | 230.38 s |

这里描述模型的活动，不把它对目标孔的重新猜测当作事实。episode 结束后，私有评估记录过 `through_selected_cover_hole=true` 的 2100 个样本、`entered_selected_casing_opening=true` 的 1412 个样本，但 `seat_ready` 与 `task_success` 都为 0 个样本，`held_seat_certified=false`，时序评估最终 `failed=true`。因此不能把失败简单解释为“从未穿孔”，也不能从落座候选或视频观感宣布完整装配成功。单独的谓词计数不证明这些条件在同一时间成立。没有为本次 setup 验证修改评估逻辑。

## 发现并修复的 setup 问题

根目录 `.env` 按原指南创建后，会随整个源码目录的只读 bind mount 出现在容器中。虽然 actor 文件权限仍拒绝读凭据，且本次请求未包含该文件，但这不满足“凭据留在宿主”的隔离目标。

已修复 `launch.py`：根目录 `.env` 存在时，将其在容器内覆盖挂载为 `/dev/null`。发现问题后先将本次测试凭据临时移出挂载目录，运行结束后恢复宿主 `.env`（权限 0600）；不需要改变模型、控制器或当前 rollout 的任务输入。

验证分两层：

- 修改 launcher 的现有测试，检查挂载遮盖、API key 不出现在 Docker 参数/配置文件以及自有容器清理。修复后全部 **7 项测试通过**。
- 使用同一 Isaac 镜像实际启动无网络、无 GPU 的短容器，按同样的源码只读挂载和 `/dev/null` 遮盖方式，断言容器内 `.env` 内容为空，且容器环境不存在 `OPENAI_API_KEY`。实际退出码 0。

完整 rollout 的模型与执行代码没有因这项启动隔离修复而重跑；修改后的遮盖挂载独立完成了真实 Docker 验证。其他凭据文件应放在源码/资产目录之外，不应假设任意私有文件都会自动被遮盖。

指南还补全了：Bash 导出环境变量、模型 ID 不含 RPent 前缀、API 根 URL 格式、外部资产/引用文件要求、启动与决策预算区别、超时退出码解释、具体结果路径、磁盘开销和单次模型行为的不可确定性。

## 对“别人能够复现”的边界

本机验证覆盖全新宿主 Python 环境、真实模型请求、真实 GPU 仿真和执行；NVIDIA 镜像及 CAD/机器人 USD 已预先存在，没有在新账户、新操作系统或 Python 3.12 上重新验证。没有声称从公开仓库自动下载私有 CAD 即可运行。

使用者必须准备相同且合法取得的资产及完整引用、可用的视觉/工具调用 API 与 NVIDIA 容器环境。源代码本身不包含这些资产。模型/API 版本及随机决策可能变化；seed 0 固定仿真初始化，不固定远程模型逐条动作。可复现的实测结论是执行流程可运行、本次任务结果为 native false，而不是“保证装配成功”。

## 对应日志与视频

以下证据仅存于本机 ignored `runs/`，未放入源码 ZIP 或 GitHub 源码中。API 请求包含图像与对话，分享前仍需审查。

- [安装日志](../runs/setup-install.log)
- [机器可读验证汇总](../runs/setup-validation-evidence.json)
- [真实 Docker 凭据遮盖验证](../runs/setup-env-mount-check.json)
- [接线测试 GLM 活动日志](../runs/setup-smoke-20261017/episode/rpent-activity/GLM_ACTIVITY_LOG_CN.md)
- [接线测试 backend summary](../runs/setup-smoke-20261017/episode/backend-summary.json)
- [接线测试视频](../runs/setup-smoke-20261017/episode/episode.mp4)
- [完整 rollout GLM 活动日志](../runs/setup-rollout-20261017/episode/rpent-activity/GLM_ACTIVITY_LOG_CN.md)
- [完整 rollout planner summary](../runs/setup-rollout-20261017/episode/planner-summary.json)
- [完整 rollout backend summary](../runs/setup-rollout-20261017/episode/backend-summary.json)
- [完整 rollout 启动器日志](../runs/setup-rollout-20261017/private/launcher.log)
- [完整 rollout 视频](../runs/setup-rollout-20261017/episode/episode.mp4)
- [结束后私有评估谓词诊断](../runs/setup-rollout-20261017/private/posthoc-predicate-diagnostic.json)
