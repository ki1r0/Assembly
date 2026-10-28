# 新工作站安装与仿真复现

环境：Linux、Python 3.11、Docker、与 NVIDIA Container Toolkit 兼容的 NVIDIA 驱动/GPU。仿真使用 `nvcr.io/nvidia/isaac-lab:2.3.0`（Isaac Sim 5.1），主机 Python 不安装 Isaac。接受 NVIDIA 对应许可后使用镜像。本次新目录已用独立 Python 环境安装依赖；真机接入另外见 `REAL_ROBOT_MIGRATION_CN.md`。

```bash
git clone https://github.com/ki1r0/Assembly.git
cd Assembly
bash scripts/setup_python.sh
cp pipeline/.env.example pipeline/.env
```

本地编辑 `.env`，提供 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、实际可用的多模态 `OPENAI_MODEL` 和 `ASSEMBLY_ASSET_ROOT`。API 必须支持图像与工具调用，不保证服务商模型别名一直不变。密钥不上传。

## 资产

装配 CAD 和 NVIDIA 机器人资产不随 GitHub 源码再分发。将合法取得的原资产目录放到 `ASSEMBLY_ASSET_ROOT`，保持 `parts/`、`CAD_edit/`、`vendor/` 以及 USD 的相对依赖。当前 CAD cache 生成器保留原任务集合，会读取其他 gearbox 部件；因此应提供原装配资产集合，不能只给一份 bolt 文件。缺失资产会明确失败。

必需目录和引用由以下文件列出：

- `pipeline/launch.py` 的 `required`：当前任务的直接 USD 依赖。
- `pipeline/tools/export_public_cad.py`：公开 CAD 缓存的全部 USD 列表与 Panda 夹指几何来源。
- `pipeline/tools/calibrate_part_mass.py`：Hub 质量计算的 STL 输入。

Panda 镜像相对位置：`vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`。即便用 xArm，目前公开 CAD cache 准备仍读取这份公开夹爪几何；没有隐瞒这个历史依赖。

## 生成 xArm 模型

仅做离线几何测试时，无需 Isaac/USD 转换：

```bash
PYTHON=pipeline/.venv/bin/python bash scripts/prepare_xarm_offline.sh
```

生成器不覆盖已有模型。已有输出时勿重复生成；在新工作站应重新生成绝对mesh路径。
完整仿真模型仍使用下述脚本，不要先单独生成URDF再直接重复调用完整脚本。

```bash
GPU=0 bash scripts/prepare_xarm_model.sh
```

脚本下载官方 `xArm-Developer/xarm_ros2` 的既有实验版本，展开 G1 URDF 并在 Isaac 中转换 USD。URDF 使用当前工作站绝对 mesh 路径，所以必须在新机器重新生成，不能移动旧生成目录后直接沿用。输出已有时拒绝覆盖，避免无意改掉已标定模型。实际夹爪未确认前，G1 只是仿真假设。

## 无模型场景 smoke

首次运行前，从合法取得的USD生成公开CAD采样缓存：

```bash
ASSEMBLY_ASSET_ROOT=/absolute/path/to/assets bash scripts/prepare_cad_cache.sh
```

公共GitHub CI自动准备官方机器人URDF/网格，执行`pytest tests -m 'not cad_assets'`。
它明确不覆盖需要未分发装配CAD的4项集成测试，不能代替全量验收。
在有合法资产的工作站，额外执行`cd pipeline && .venv/bin/python -m pytest tests -m cad_assets -q`；
默认`pytest tests -q`仍包含这些测试，缺失资产仍会失败，不会静默跳过。

```bash
set -a
source pipeline/.env
set +a
pipeline/.venv/bin/python pipeline/launch.py \
  --robot xarm7 --task peg --action-backend generic \
  --controller reference_ik_high_pd --case nominal-aligned \
  --peg-mvp --scene-smoke --xarm-physics-hz 240 --gpu 0 \
  --run-dir "$PWD/pipeline/runs/scene-smoke-01"
```

smoke 验证相机、深度、机器人和 evaluator 能启动，不代表装配成功。使用新目录记录每次运行，不复用旧 IPC。

## 模型 rollout

将上面 `--scene-smoke` 换成 `--rollout --budget-s 3600 --peg-sim-budget-s 600`。平躺使用 `--case bolt-horizontal-pose`。Hub 使用 `--task hub-cover --case nominal-aligned`，移除 `--peg-mvp` 与 peg 专用预算参数。保留所有保护；不把诊断加速参数自动推广到带载任务。

```bash
cd pipeline
.venv/bin/python -m pytest tests -q
cd ..
PYTHONPATH=benchmark pipeline/.venv/bin/python -m pytest benchmark/tests -q
PYTHONPATH=benchmark pipeline/.venv/bin/python -m assembly_bench.cli --help
```

运行资料在 `pipeline/runs/<run>/`：视频 `episode/episode.mp4`，动作 `episode/actions.jsonl`，模型活动 `episode/rpent-activity/`。请求使用共享 `_shared/requests.sqlite3` 去重，迁移运行记录时必须一并带走，不能只复制 request.ref.json。详细存储工具为 `storage-tools/manage.py`。
