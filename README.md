# ACP 复现：G1 动作与接触数据流水线

面向《Humanoid Mobility via Attention-based Contacts》的数据准备工程。将 OmniRetarget、LAFAN1、Mixamo、PHP 数据统一到 G1-23 球手模型，生成轨迹、场景、九部位几何接触标签、回放和离线质量报告。

**文档入口：[中文知识库](docs/README.md) · [快速开始](docs/01_使用指南/快速开始.md) · [完整手册](docs/01_使用指南/数据流水线手册.md)**

![流水线](docs/01_使用指南/图示/数据处理流程.svg)

```bash
# 在本项目根目录执行
.venv/bin/python scripts/acp_pipeline.py doctor
.venv/bin/python scripts/acp_pipeline.py list
.venv/bin/python scripts/acp_pipeline.py status mixamo
.venv/bin/python scripts/acp_pipeline.py catalog
```

本机资源配置见 `configs/machine.local.json`，跨机器配置从 [示例](configs/machine.example.json) 复制。两套环境分别负责 Holosoma/MuJoCo 重定向和 Drake 标注，统一入口自动选择。

结果按 [资产规范](docs/03_方法与规范/资产与版本管理规范.md) 分类：[数据集](outputs/数据集/)、[缓存](outputs/缓存/)、[实验归档](outputs/实验归档/)、[可视化](outputs/可视化/)、[管理索引](outputs/管理/)。原始数据仍存放在外部数据集目录。

已有数据处理与验证程序；训练模块尚未实现。处理完成、离线筛查通过、训练准入是不同状态，实际使用按每批次的质量报告与场景确认记录决定。

源码与命令查询：[仓库文件与架构查询](docs/03_方法与规范/仓库文件与架构查询.md)。阶段结束运行 `structure --stage "阶段说明"` 更新索引与记录。
