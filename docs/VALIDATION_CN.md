# 源码整理与验证记录

日期：2026-10-17。此目录是独立 Git 源码仓库（main），未设置 remote、未提交、未推送；原实验目录未修改。

完成内容：提取最新 GLM/OpenAI 兼容 API + RPent agent/tools/memory + RGB-D 几何技能 + HRC Cartesian servo + Isaac 场景/原生评估；保留 RPent 通用 Env/VLA/RPC 接口；移除具体 learned-policy 推理服务、权重路径、模型抓取工具及归一化动作映射；独立保留真实手腕相机。API endpoint、model ID、GPU、资产路径和图像尺寸均可配置。

没有复制密钥、历史 memory、实验日志、图像视频或 USD/CAD/模型权重。保持已有接触去身份化、关节/力矩/原生运动急停和 actor 文件访问限制。Git 忽略本地环境及运行产物。

已验证：

- 全新 Python 3.11 虚拟环境安装 `requirements-dev.txt` 成功；`python -m pip check` 无依赖冲突。
- `python launch.py --help`、`python planner.py --help` 成功；实际 ApiAgentLoop 导入成功。
- `python -m pytest -q`：7 passed。覆盖通用 VLA HTTP RPC、public observation 字段筛选、终止端点不可覆写、可见局部孔壁圆拟合、私有文件及符号链接越界拒绝、真实 RPent agent loop 拒绝虚假 native success 后正常 stop、launcher 不向容器/配置文件传递 API key 并清理自有容器。
- 所有交付 Python 文件语法解析通过；模型专属代码、旧机器绝对路径及常见凭据文本模式扫描无命中；Git ignore 检查通过。

初始源码提取阶段未验证：当时没有调用远程 GLM API，也没有运行真实 GPU/Isaac 仿真或 peg insertion rollout。后续已执行真实验证，见 [setup / rollout 实测报告](SETUP_ROLLOUT_REPORT_CN.md)。HTTP RPC 与 agent loop 测试使用本地测试策略/FunctionModel，不是模型能力或装配成功证据。Python 3.12 尚未在本机执行，由附带 CI 配置覆盖。

已知限制：原有 horizontal-start 姿态恢复问题仍存在；先前四组最终实验原生成功均为 false。没有通过改成功判据来包装结果。历史 benchmark memory 不随源码发布，后续模型决策可能改变。

许可：RPent Apache-2.0 原文和版权头已保留；原 HRC/应用源码未提供许可证，需权利人确定后再公开分发。第三方 CAD/机器人资产外部配置，不随源码打包。
