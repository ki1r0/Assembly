# Peg evaluator 使用核对（2026-10-28）

## 找到的旧校准

`experiments/panda-grasp-belief-20261018/docs/EVALUATOR_VALIDATION_CN.md`记录两点钟孔独立bolt evaluator的物理正例：oracle-02-fixed-hold，4334物理步，native_success。该文同时明确限于旧Panda/Factory场景，不证明新机器人/新测量适配已标定。该文引用的本地runs路径本次不存在，不能把文档引用当成已重新核验原始录像/summary。

当前实验保留同一`IndependentBoltSeatingEvaluator`类和`config/bolt_insertion.yaml`。但其中evaluator_calibration仍记录cad_meters_per_source_unit=0.002，physical_positive=true，physical_negatives_rejected=true，status=historical_scale_mismatch_pending_revalidation；当前settings比例0.001、螺栓杆长22mm，不同于旧44mm尺度。旧校准未在当前尺度得到启用。

## 当前在线与离线不是同一判据

`generic_backend.py`读取bolt_insertion.yaml，调用peg_eval.observe(m)，随后native_success=status.task_success AND physical_calibration_matches_scale(...)。比例不符使校准有效性为false。

`--peg-mvp`同时创建`PegProcessEvaluator`，使用collider_geometry/collider_sweep分支且calibrated=False。它记录过程检查点和candidate_physical_success，但未将其输出回接在线native_success。这是判据/反馈接线未统一，不应只概括为“没有旧校准”。

直立upright-feedback-correction-01的实际最终旧trace：through_selected_cover_hole=true、深度约12mm、帽底间隙约2.5e-8m；但entered_selected_casing_opening=false、cap_support_contact=false、径向误差0.20774mm超过旧0.2mm，seat_ready=false、unscaled_evaluator_candidate_success=false，且比例校准不匹配。因此只把状态改成calibrated或比例改成0.001也不会自动解决，而且没有相应标定证据不能这样改。

## 新终态验证的实际范围

`runs/cape-peg-mvp-20261026/endpoint-acceptance-collider-v03-16/batch-status.json`：finished_requires_review，10次端点诊断，9次与预期标签匹配，1次backend_error；scope明确endpoint predicate calibration, not robot assembly or process validation。四次有效正例为true，偏孔/倒置负例为false；不能把环境失败记成错误判据，也不能将端点验证冒充完整过程标定。仍需完成物理复核与接线一致性。

## 结论

用了旧evaluator的代码，但没有将其历史校准当作当前0.001尺度的有效标定；新的collider过程判据已用于离线分析，却没有接管在线完成反馈。此前称“evaluator未校准”过于笼统。此次只定位/记录，没有改校准标志、阈值或覆盖原始结果。

## 用户澄清后的精确定位：25–27日 CAPE Peg MVP

用户指的是完整bench包装 `CAPE-Assembly / Peg MVP`，不是18日旧校准。入口与定义在Human-AI-Collab/assembly_bench/{peg_mvp.py,peg_mvp_runner.py}、configs/bench/tasks/peg_mvp.yaml；26日计划在本实验docs/ASSEMBLY_BENCH_PEG_MVP_PLAN_CN.md，27日复核报告在Human-AI-Collab/reports/CAPE_PEG_MVP_REVIEW_20261027_CN.md。

这批xArm确实挂载并使用了该PegProcessEvaluator：upright回合configuration包含private/peg-mvp-source.py到/opt/cape_peg/peg_mvp.py，criteria版本cape-peg-process-v0.3-collider-dev，离线final-score物理candidate为true。这条链不是完全没用新bench evaluator；未接完的是在线成功反馈，仍由旧peg_eval分支决定。

27日报告已指出10例独立终态验收中7例有效、3例backend_error，旧9/10汇总撤销。补测目录acceptance-infrastructure-retry-01的原始summary显示positive02及offset01均正常evaluator_calibration_complete；positive02已有重评为seat_predicate_stable=true，offset01已有原始过程metrics；offset02目录不存在。batch-status仍为running且未收齐，不可据此称验收完成。当前yaml仍pending，构造参数仍calibrated=False。此前首先引用18日校准没有回答用户所指版本，应以此定位为准。

## 2026-10-28 在线反馈修复

用户要求后已修复 --peg-mvp 分支：完成过程sample并调用PegProcessEvaluator后，在线native_success使用peg_completion_feedback.peg_process_completed的新物理过程结果，不再与旧scale-0.002 evaluator取AND。旧native单独写legacy_native_success；未启用peg-mvp的旧入口不变。

在线开发完成必须同时满足candidate_physical_success=true、全部五个goal predicates=true、有效初态、非特权且真实控制来源成立，评价状态为valid或uncalibrated。缺测/错误孔/未释放/未撤退/无稳定窗口/特权探针都不能结束成功。不改变任何几何阈值、稳定时间、安全保护或评分器calibrated参数。完成布尔只作为允许的环境终态反馈给模型，私有几何与接触身份不返回。

明确语义：native_success此时是“CAPE物理过程的在线开发完成”，不是“正式基准校准通过”。正式assembly_success仍由scorer按calibration状态返回，当前保持null、evaluation_status=uncalibrated。没有将未完成校准标为已完成；后续正式benchmark仍需独立校准收尾。

旧直立回合真实全量测量重放首次在线完成时间262.4125秒，formal grade仍null；结果online-completion-recorded-replay.json，不把该重放称为新的机器人成功。7项反馈测试验证无控制来源/初态无效/特权/缺测/未完成时拒绝成功，合并完整测试344 passed、3 skipped、12.70秒。

新物理自主回合runs/xarm-speed-audit-20261027/upright-cape-online-completion-02已在GPU0启动；user service xarm-upright-cape-completion-20261028，xarm7/nominal-aligned/reference_ik_high_pd/240Hz/默认1x速度，不提供旧成功坐标或固定动作。启动保存的后端源码已包含新完成反馈分支。当前仍运行，不预判native_success。
