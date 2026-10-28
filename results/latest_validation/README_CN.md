# 本包验证（2026-10-28）

- 独立 Python 3.11 venv 全新安装 requirements，未借用旧实验 site-packages。
- `pipeline/tests`：342 passed，5 skipped，12.55 s；跳过包括需要旧完整项目与可选 USD 的集成测试，不代表这些被验收。
- `benchmark/tests`：23 passed，0.19 s，保留过程判据、未知状态和实例划分测试。
- 重新从官方 xArm URDF 生成新路径 USD；mesh 引用已检查指向新 package。
- 新目录 `--scene-smoke --robot xarm7 --peg-mvp`：通过。144 个物理步，RGB/depth 正常，backend error=null，未调用模型，未执行装配动作，不声称任务成功。
- 本次未重跑完整模型装配。历史任务结果见上层 INDEX，实机未验证。

场景测试使用原工作站合法资产根目录，但代码、URDF、USD、Python 环境和 evaluator 均来自新路径。资产不进入 GitHub，需在新机器按安装文档准备。
