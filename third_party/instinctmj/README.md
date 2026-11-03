# InstinctMJ / SuperMimic 来源

本分支的 SuperMimic 方法实现参考本机 `/home/lyx/train/supermimic-extend/InstinctMJ`，基准提交 `7ca851e`。该目录有用户对 `mdp.py`、`perceptive_env_cfg.py` 的未提交修改；迁移时按实际文件复制到远端 `ACP_deps/InstinctMJ`，没有改写原目录，也没有修改远端其他项目。

参考模块：`src/instinct_mj/rl/supermimic/{runner,networks,kinematics,rotations,config}.py`。ACP 沿用真实采集、多步残差动力学拟合、冻结世界模型后的 BPTT、λ-return critic、AdamW 和 UTD 调度。`src/acp/training/supermimic/rotations.py` 保留原文件；其他模块适配 ACP 的 23DoF、球手、接触条件 actor 和高程图，不直接运行原仓库的 29DoF 深度图／参考姿态任务。

参考仓库：<https://github.com/project-instinct/InstinctMJ>。根目录许可证原文保留在 [LICENSE](LICENSE)，为 CC BY-NC 4.0。参考 SuperMimic 文件同时注明其基于 Chong Zhang / ETH Zurich 的 MIT 实现 [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)；原注释与署名保留。本项目不声称这些代码是论文作者的 ACP 官方实现。

完整实验接口、未验证项和引擎差异见 [SuperMimic迁移与4090部署](../../docs/01_使用指南/SuperMimic迁移与4090部署.md)。
