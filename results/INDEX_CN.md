# 实验结果索引

本文件记录真实已观察结果，不把局部诊断或基础设施 smoke 当作完整任务成功。视频保留在原工作站，不将几十 GB 运行记录复制入 GitHub。

实验根目录：`/home/sunsiliang/robot-assembly-experiments/experiments/panda-unified-assembly-peg+hub-20261018/runs/xarm-speed-audit-20261027/`。

| Run | 结果 | 证据边界 |
|---|---|---|
| `upright-feedback-correction-01` | 直立插入、释放、撤退物理正例；约 12 mm 插入、径向误差约 0.208 mm、释放后稳定超过 7 s | 当时旧在线反馈未接好，模型最终求助；正式评分未校准 |
| `upright-cape-online-completion-02` | 135753 物理步，角速度安全停止 | 最新在线反馈修复后完整尝试，未完成装配 |
| `horizontal-complete-candidate-feedback-03` | 139824 物理步，模型停止 | 恢复多轮仍未完成 |
| `hub-local-friction-margin-grasp1-01` | 真实自由物体抬升约 57.6 mm | 无 VLM 的局部诊断，不是自主装配成功 |
| `hub-friction-margin-complete-03` | 109880 物理步，安全停止 | 完整自主尝试未成功 |

各目录视频：`episode/episode.mp4`；动作：`episode/actions.jsonl`。精选视频入口：`/home/sunsiliang/桌面/xArm仿真_精选视频_20261028/`（原工作站本地路径，新机器没有这些文件）。

`latest_validation/` 保存本次独立目录软件测试、场景 smoke 的小型 JSON 与说明，不含任务成功声明。
