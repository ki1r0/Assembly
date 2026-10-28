# 独立任务评价与扩展

`benchmark/assembly_bench/peg_mvp.py` 保持原 CAPE 过程评价器，`peg_mvp_runner.py` 保持实例驱动的逐次运行与汇总。`benchmark/data/instances.json` 保留原 dev/test 初态划分；`config/peg_mvp.yaml` 保存原任务定义。原定义为 Panda，本包不悄悄把它改成已校准 xArm benchmark。

本次仅将 CLI 缩为 peg 相关三个命令，移除旧 Factory CLI 的无关依赖；给 runner 增加显式 `--robot`，默认 xArm7 / reference_ik_high_pd。评价器算法和阈值未修改。指定 Panda 时可显式选择 isaac_osc。不要将不同 embodiment 的结果合并报告。

```bash
PYTHONPATH=benchmark pipeline/.venv/bin/python -m assembly_bench.cli evaluate-peg-mvp \
  --asset-root /absolute/path/to/assembly-assets \
  --robot xarm7 --controller reference_ik_high_pd \
  --split dev --limit 1 --gpu 0 --budget-s 3600 \
  --output "$PWD/pipeline/runs/peg-dev-01"

PYTHONPATH=benchmark pipeline/.venv/bin/python -m assembly_bench.cli rescore-peg \
  --run /absolute/path/to/run --output /absolute/path/to/new-score
```

`calibrate-peg-mvp` 是特权初态的端点正负例校准，不是 agent rollout；不能计入策略成功率。`assembly_success=null` / `uncalibrated` 不应计作已认证成功。基础设施错误应单列，不能算负例预测正确。

## 在线完成与离线分数

`pipeline/peg_completion_feedback.py` 仅当过程证据完整、初态有效、非特权真实流程且所有目标谓词满足时发出在线物理完成。旧判据保留在 `legacy_native_success`。正式过程校准尚未完成，因此 `native_success` 的开发完成信号并不等于正式 `assembly_success=true`。本次整理不改变这一语义。

## 扩展到其他任务

复用实例选择、独立进程、日志、未知/失败区分及汇总，新增任务自己的过程测量与判据；不要直接复用 peg 的毫米阈值。任务目标、公开几何和私有评价分开；agent 不得读取 private/ 中的 pose、接触身份或判据真值。新增任务先验证端点正负例，再验证完整过程、释放和稳定性，记录不确定状态。共享 runner 的 policy 边界目前是 launcher 命令，不意味着所有任务已经接入同一 scorer。
