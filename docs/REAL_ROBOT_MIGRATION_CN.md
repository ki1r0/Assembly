# 真机接入边界

本包保存的是 xArm7/G1 的 Isaac 仿真适配，不是已经在真机上验收的部署版本。按用户本次要求先上传代码，实际硬件配置在新 workstation 调整。没有执行过真实 xArm、RealSense、FT 闭环，也没有把仿真成功声明为真机成功。

## 可以复用的部分

- `pipeline/planner.py`：RPent 模型决策循环。
- `pipeline/generic_toolkit.py`：工具参数、动作反馈及请求记录。
- `pipeline/assembly_skills.py`、`cad_registration.py`、测量模块：公开 RGB-D/CAD 几何估计。
- `pipeline/xarm_gripper.py`、`arm_support_geometry.py`：G1/机械臂公开模型几何；必须与实际夹爪核对。
- `pipeline/m1_reach/`：运行控制和日志。这里的终止标记不能替代硬件急停。

## 必须替换的设备边界

| 当前代码 | 仿真行为 | 真机接入工作 |
|---|---|---|
| `launch.py` | 启动 Isaac 容器和 planner | 真机启动器需启动设备后端；当前启动器不会连接真实机器人 |
| `generic_backend.py` | Isaac articulation、模拟相机、关节反力、工具命令执行与私有评价 | 实现实际设备后端；不能把此文件中的 Isaac 调用直接用于硬件 |
| `robot_kinematics.py`、`xarm_robot_cfg.py` | USD 关节模型、仿真 PD/IK、仿真摩擦补偿 | 使用实际 xArm 控制接口与真实 TCP；仿真 PD/摩擦值不能当成驱动参数 |
| `config/public_camera.json` | 固定理想 RGB-D 相机标定 | 替换为 RealSense 实测内参、深度尺度及 base-from-camera 外参 |
| `xarm_frames.py` | 公开模型的坐标换算 | 核对实际末端、FT 安装和夹爪 TCP；世界单位 m、旋转 wxyz |
| 后端 wrench | 模拟关节反力转换 | FT 实际驱动、零偏、重力/工具补偿、传感器到 TCP 力矩变换 |
| 后端夹爪接触 | 仿真两指接触与压力 | 真实夹爪若没有两指压力读数，不可伪造相同反馈或沿用依赖它的力控策略 |

现有 `RobotToolkit.rpc()` 与 `IsaacToolkit.rpc()` 通过 `ipc/command_*.json` / `reply_*.json` 交换命令和观测。真机后端需实现真实可支持的工具，逐个保留单位、超时后实测位姿、部分执行、安全停止及错误语义；不支持的工具应明确不暴露，不能返回虚假完成。完整协议以这两个文件和 `generic_backend.py` 的命令分发为准。

## 在新工作站实施顺序

1. 按 `SETUP_CN.md` 安装并先复现仿真 smoke。
2. 确认实际机械臂/夹爪、RealSense、FT 型号和驱动；保存设备标定。不要复用仿真相机外参或默认 G1 假设。
3. 设备后端先验证只读关节、TCP、RGB-D、FT 的时间戳和单位，再验证受限小位移与停止。机器人硬件保护、急停和驱动错误必须直接终止动作。
4. 验证抓取、带载以及小范围接触，之后再连接模型进行装配。

独立 benchmark 的私有物体 pose 和接触标签只能在仿真评价中使用，不得成为真机 planner 输入。真机成功判定需要可获取的外部测量或人工审核，不能照搬 Isaac 私有真值。
