# Assembly Agent Core：GLM + RPent + Isaac 装配框架

这是从当前实验 pipeline 提取的独立源码目录。默认链路是：GLM/OpenAI 兼容 API → RPent agent/tool loop → RGB-D 几何技能 → HRC Cartesian servo → Isaac Lab。保留真实手腕相机、接触反馈、求助/停止决策、原生评估和完整 API 日志。

已移除模型专属推理服务、checkpoint 配置、抓取工具和 LIBERO 归一化动作映射；RPent 通用 `BaseVLAClient.predict(obs, options)` 与 `BaseVLAFacade` / RPC 接口仍在。默认没有接入任何 learned VLA，未来需另接模型和明确动作适配。

## 安装与运行

完整的逐步流程、预期输出和排错方法见 [复现指南](docs/REPRODUCE_CN.md)，本次实测证据见 [setup / rollout 报告](docs/SETUP_ROLLOUT_REPORT_CN.md)。基础命令也见 [README.md](README.md)。宿主使用 Python 3.11/3.12，安装 `requirements-dev.txt`；准备 Docker、NVIDIA Container Toolkit、外部资产和 API 环境变量。复制 `.env.example` 为 `.env` 后自行填写，再执行：

```bash
set -a
source .env
set +a
python launch.py --gpu 0 --budget-s 120
python launch.py --gpu 0 --rollout --budget-s 600
```

第一条只运行接线测试，第二条运行完整装配 episode。默认固定主相机 960×720，手腕相机 224×224。交付源码包不包含密钥、权重、历史任务 memory 或 rollout 产物；USD/CAD 资产通过 `ASSEMBLY_ASSET_ROOT` 外部加载，不复制原机器项目。

## 需要先了解的边界

- 任务 prompt 保留当前装配任务的目标孔描述；没有借机更改任务、成功标准或控制器。
- 仿真器初始化与独立评估仍需要私有场景几何；这些信息不进入 actor 的 public observation。接触信号使用未按 Bolt 身份过滤的总力。
- 原生关节/力矩、速度及跟踪异常急停、数据验证和文件访问边界继续保留。
- 四组先前最终实验没有达到原生成功判据；不能把代码整理完成说成任务跑通。
- API 请求日志含图像及对话内容，分享前需审查；`.gitignore` 排除运行目录和环境文件。
- RPent 保留 Apache-2.0；继承的 HRC/应用源码未提供许可证，公开分发前还需确定许可。第三方 CAD/机器人资产未打包。

参见 [流程和信息边界](docs/ARCHITECTURE.md)、[VLA 接口](docs/VLA_INTERFACE.md)、[资产配置](docs/ASSETS.md)、[已知限制](docs/LIMITATIONS.md) 和 [来源/许可](THIRD_PARTY_NOTICES.md)。源码和运行结果已在本地验证；GitHub 目标仓库为 [ki1r0/Assembly](https://github.com/ki1r0/Assembly)，运行产物和本地凭据不纳入版本控制。
