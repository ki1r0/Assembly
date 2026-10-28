# xArm 装配 pipeline 与 CAPE Peg Benchmark

本仓库新增 `pipeline/` 保存截至 **2026-10-28** 的 xArm7/G1 Isaac 适配；`benchmark/` 独立保存 peg insertion 的过程评价与实例运行。原仓库顶层 assembly-agent-core 源码保持不动，旧入口不代表本次 xArm 分支。

**这是仿真验证与真机迁移的代码包，不是已经在真机验收的驱动版本。** 本次按用户要求先上传，实际 xArm、RealSense 和 FT 接口及标定由新工作站按硬件配置接入。

| 路径 | 内容 |
|---|---|
| `pipeline/` | 当前 RPent 模型循环、工具、通用感知/技能、xArm IK/G1 几何、Isaac 后端 |
| `benchmark/` | 独立 CAPE peg 评价器、dev/test 初态、校准和 rollout runner |
| `scripts/` | 独立 Python 环境安装、官方 xArm 模型生成 |
| `storage-tools/` | 去重请求恢复与存储管理 |
| `results/` | 最佳物理证据、最新失败及本包验证记录，避免混淆 |

从 [安装与仿真复现](docs/SETUP_CN.md) 开始；真机工作见 [设备接入边界](docs/REAL_ROBOT_MIGRATION_CN.md)。[模块说明](docs/PIPELINE_CN.md)、[benchmark 复用](docs/BENCHMARK_EXTENSION_CN.md)、[当前限制](docs/CURRENT_LIMITATIONS_CN.md)、[结果索引](results/INDEX_CN.md)。

运行记录、密钥、视频、模型权重、CAD 和生成 USD 不进入 Git。公开机器人模型通过官方源码重新生成；装配资产需要合法取得后按安装文档放置。源码原有第三方声明见 `THIRD_PARTY_NOTICES.md` 和 `pipeline/THIRD_PARTY_NOTICES.md`；本整理不额外授予外部资产许可。
