# 模块与来源

```
launch.py → Isaac 后端 + planner.py
planner.py → RPent ApiAgentLoop → generic_toolkit.py
工具参数 → IPC → generic_backend.py → xArm IK/关节/夹爪仿真
RGB-D/实测状态/动作反馈 → RPent → 下一次模型决策
私有过程测量 → 独立 CAPE evaluator → 完成/终止记录
```

| 模块 | 文件 |
|---|---|
| 模型循环与工具 | `pipeline/planner.py`, `generic_toolkit.py`, `toolkit.py`, `third_party/rpent/` |
| 通用感知/几何 | `cad_registration.py`, `aperture_measurement.py`, `rim_measurement.py`, `inhand_measurement.py` |
| 抓取/装配技能 | `assembly_skills.py`, `gripper_control.py` |
| xArm/G1 模型与坐标 | `xarm_robot_cfg.py`, `xarm_gripper.py`, `xarm_frames.py`, `arm_support_geometry.py` |
| 执行与物理 | `generic_backend.py`, `robot_kinematics.py`, `factory_cartesian.py`, `isaac_runtime/` |
| 日志与停止 | `m1_reach/`, `grasp_feedback.py` |
| 在线完成信号 | `peg_completion_feedback.py` |
| 私有过程评价 | `benchmark/assembly_bench/peg_mvp.py` |

这是当前 unified 实验代码的整理，不是把所有历史技能重新改写为原子动作。`generic_toolkit.py` 当前仍暴露 execute_grasp / assemble 宏；不能称其与此前细颗粒度实验工具完全相同。请求实际使用了什么工具应按对应 run 日志判断。

原仓库顶层文件保留为历史 assembly-agent-core，不覆盖。新增 `pipeline/` 是当前 xArm 分支；不要混用两个 launch.py。为保持算法行为，首次整理保留原扁平模块结构，以及因导入依赖仍需的其他任务代码。

来源：`panda-unified-assembly-peg+hub-20261018` 当前文件；benchmark 来源 `Human-AI-Collab/assembly_bench` 的 Oct25–27 peg MVP。改动仅为布局/路径、补齐存储工具、独立 peg CLI、runner 显式选择机器人、安装/模型生成文档与测试。没有改 prompt、物理参数、安全阈值或 evaluator 阈值。
