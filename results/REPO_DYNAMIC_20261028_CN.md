# Assembly 仓库动态测试

## 资产与离线验证

- 正式仓库：ki1r0/Assembly，代码补丁8ffa58b；资产准备与测试分层434e6ce。
- GitHub Actions 40660159628 已完成success（Python3.11/3.12）。
- 公共CI实际生成官方xArm URDF及网格，不运行未分发CAD集成测试。
- 本机完整pipeline测试343通过、5跳过；CAD标记组4通过、2跳过；benchmark23通过。
- 资产来自已有合法本地安装；不是纯clone即可完成的完整仿真。

## 动态测试结果：未通过

从本仓库pipeline/launch.py及其独立环境启动xArm7/G1，480Hz，
reference_ik_high_pd，CAPE peg evaluator启用，无VLM、无负载，不使用旧实验后端。
保留原控制增益、运动阈值及保护。

- 六个正负10mm平移测试均reached。
- 首个0.08rad旋转返回motion_timeout：位置误差约0.538mm，角度误差约0.000907rad。
- 第二轴旋转reached；第三轴旋转在600秒墙钟预算处interrupted。
- 最终passed_unloaded=false，13536物理步，141视频帧，backend error=null。
- 未触发角速度紧急保护，不等于动态适配通过；夹爪阶段未执行，不声称已验证。
- 需继续排查旋转动作的收敛/停止条件与耗时，不能仅增大预算或放宽阈值称成功。

本机原始记录：pipeline/runs/repo-dynamic-20261028-01/。
视频：该目录episode/episode.mp4；最终结果checks/adapter-validation.json。
运行记录和视频不提交到Git，失败原始数据保留。
