# 本次整理修改清单

1. 原 unified 实验的 Python 模块、配置、运行依赖与测试复制到 `pipeline/`，排除 runs、diagnostics、缓存、密钥和虚拟环境。
2. `pipeline/launch.py` 默认 evaluator 路径由用户家目录改为本仓库 `benchmark/assembly_bench/peg_mvp.py`。
3. `pipeline/tools/run_*batch.py` 移除原 workstation 的资产默认路径，要求由环境提供；R1 资产也由调用方提供。
4. 复制请求存储工具到 `storage-tools/manage.py`，与原相对目录结构一致，避免运行后无法恢复 request 引用。
5. 独立保留 CAPE peg scorer、实例 runner。CLI 仅保留 peg 的三项功能，删除无关 Factory 导入；原 Wilson 区间函数移入 runner，算法不变。
6. runner 新增显式机器人参数，xArm 默认 reference_ik_high_pd，拒绝 xArm 使用 Panda 专属 isaac_osc；允许仅通过环境变量配置 API，不强制存在 .env。
7. 新增安装/机器人生成脚本与文档。机器人模型从官方源码重建，不提交含旧绝对路径的生成 URDF/USD。
8. benchmark 测试只调整实例文件路径；其余测试与算法保留。原顶层代码不改动。

未修改模型 system/task prompt、抓取候选算法、接触控制、物理参数、安全停止阈值、过程评价器或成功阈值。旧框架含有宏技能，本次未重新宣称其为纯原子动作系统。

CI 增加 pipeline 与独立 benchmark 测试，保留原顶层测试；未在 CI 下载大型资产或调用模型。
