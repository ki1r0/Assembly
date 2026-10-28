# 角速度相关补丁接入状态

2026-10-28：Assembly 是当前 CAPE 与 xArm pipeline 的主仓库。
正式代码在 pipeline/generic_backend.py，非 Human-AI-Collab 中的补丁附件。

夹爪物理步更新现在仅写夹爪 joint_ids，不重写机械臂目标。
回归测试覆盖 Panda、xArm 驱动关节及双臂索引。
该版本无 controlled-transfer 的 native 子步插值/速度前馈组合，
不照搬其参考重置分支，也不将 Panda 阻尼直接用于 xArm 或 OSC。
现有速度、关节限位、力保护与控制增益均保留。

GitHub Offline tests 在提交 15ee8b4 上失败，原因是未分发的外部资产：
xArm URDF、机器人网格和 public_cad_models 中的 CAD 缓存。
这是干净安装的资产准备/测试分层问题，不是角速度保护再次触发。
本补丁不通过跳过测试掩盖失败，也不上传许可未确认的资产；
CI 资产准备仍待修复，不能将定向测试通过报告为全套 CI 通过。

此接入不代表真机安全认证或 VLM 完整装配成功。
