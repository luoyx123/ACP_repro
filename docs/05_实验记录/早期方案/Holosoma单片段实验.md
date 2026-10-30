# Holosoma 数据处理复现：第一条端到端样本

> 本页含已退役脚本的历史命令；当前操作请使用 [快速开始](../../01_使用指南/快速开始.md)。

本页保留首次单样例实验记录。最新的 145 条全量处理和额外自碰撞/脚滑验收见 [OMNI_TERRAIN_BATCH.md](OmniRetarget全身重拟合实验.md)；单样例的环境几何/限位通过不能替代完整离线筛选。

本轮已在用户的 `/home/lyx/train/holosoma` fork 实现 G1 23 DoF 目标适配，并从 OmniRetarget 的人体关键点/地形走通：

```text
Omni human_joints [604,53,3]，30 Hz + 配套 terrain URDF/mesh
  → 导入并绑定同一坐标系的地形
  → 人体关键点先插值到 50 Hz
  → Holosoma InteractionMeshRetargeter，目标 g1_23dof_rev_1_0
  → qpos [1006,30]，xyz + wxyz + 23 joints
  → 显式转换为 ACP 格式，计算速度/FK/碰撞几何
  → 九部位接触片段、未来 75 步 onset 目标、有效性掩码
  → MuJoCo 原生几何交叉检查、诊断报告、回放
```

这条样本是 Holosoma 重做人体→目标 23 DoF 的工程复现路径。论文 Table S1 对 Omni box climbing 的原路径是复用已重定向 G1 动作，再生成接触标注；两者不能混作同一数据处理方法。保留的 `g1_source.npz` 是原 G1 29 DoF qpos 对照，仍为 30 Hz，未当作已转换好的 23 DoF 动作使用。

## 实现位置

- ACP：`scripts/prepare_holosoma_g1.py` 根据本地 Unitree URDF/MJCF 生成专用资源，复制实际引用的 27 个 mesh，保存源/生成模型与网格哈希、23 关节名和语义 frame。
- Holosoma：`config_types/robot.py` 注册 `g1_23dof_rev_1_0`，`config_types/data_type.py` 新增 LAFAN/Mocap/SMPLH/SMPLX 及 Omni mappings；保留已有 G1 29 DoF、T1 与 Roban 行为。
- Holosoma：`prepare_omniretarget_terrain.py --output-fps 50` 在 IK 前插值人体关键点；`robot_retarget.py` 读取该 FPS，保存正确采样率，并按实际帧率调整脚锁速度阈值。
- ACP：`acp.data.holosoma` 检查 URDF/MJCF 关节顺序，转换 qpos，计算 50 Hz 参考状态、九部位几何接触和未来目标；掩码中明确区分 joint limits、环境穿透、速度诊断和未来窗口完整性。
- ACP：`src/acp/quality/native.py` 在求解用的原生场景上独立计算有符号距离及源 stance 对应的足底间隙。它没有积分动力学或推断接触力。
- ACP：`scripts/render_holosoma_reference.py` 导出运动学回放视频，绿色标记是几何标签的身体 landmark。

模型由官方 rev1.0 资源生成，腕部/腰部没有借用 ASAP 的另一种 23 DoF 集合。语义 frame 是本项目的工程定义，URDF/MJCF 同步生成；随机关节姿态下关键 frame 位置差异小于 1 µm。生成器还把官方 XML 中与 URDF 不同的关节 range/effort 同步到选定 URDF，逐项记录在 `models/g1_23dof_rev_1_0/provenance.json`。

## 复跑

从 ACP 项目根目录执行，Holosoma 依赖使用已有 `hsretargeting` 环境，ACP 标注使用项目 `.venv`：

```bash
.venv/bin/python scripts/run_holosoma_pipeline.py \
  --holosoma-root /home/lyx/train/holosoma \
  --holosoma-python /home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  --unitree-root /home/lyx/train/unitree_ros/robots/g1_description \
  --source /home/lyx/train/retarget/dataset/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_0.9.npz \
  --frames 604 --seed 0 --foot-support \
  --output outputs/实验归档/holosoma/holosoma_climb00_50hz_support_current_pose_seed0
```

脚本固定 NumPy sampling seed，依次保存模型、导入、重定向、标注、原生几何检查日志；manifest 保存命令、工作目录、源码/模型哈希及 Holosoma commit。临时导入数据在 Holosoma 的 `demo_data/acp_omniretarget_terrain` 中，与用户已有 demo 分开；动作结果在 ACP `outputs` 下。模型生成器不修改 Unitree 原资源。Holosoma 已有用户 README/Roban 场景改动保留。

本片段推荐启用 `--foot-support`，保持默认 1 mm XY 脚锁容差；去掉该参数并使用独立 `--output` 可复现无支撑代价的基线。支撑 soft cost 是本地 fork 的工程扩展，不能视为 ACP 论文披露配置；其 weight=200、gap tolerance=5 mm、距离查询范围=8 cm。只有源人体脚速度启发式判为 stance、且找到足下表面时才启用。

回放（把 `RUN` 换为具体输出目录）：

```bash
/home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  scripts/render_holosoma_reference.py \
  --scene /home/lyx/train/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/acp_omniretarget_terrain/climb_00_z_scale_0.9/g1_23dof_rev_1_0_w_multi_boxes_scaled_1.00_1.00_1.00.xml \
  --motion RUN/labeled/motion.npz --output RUN/reference.mp4
```

## 首次完整运行及边界

推荐输出 `outputs/实验归档/holosoma/holosoma_climb00_50hz_support_current_pose_seed0`：1006 帧，50 Hz；关节位置/速度限位诊断和环境穿透容差检查全部通过，最深 signed distance 约 -2.03 mm（阈值 -3 mm）。MuJoCo 与 Drake 原始距离接触 flags 一致。该几何结果不意味着动作已能稳定执行。

| 检查 | 无支撑代价基线 | 修正状态后启用支撑代价 |
| --- | --- | --- |
| 完整求解帧数 | 1006 | 1006 |
| 几何/关节位置/关节速度有效帧 | 100% | 100% |
| 源人体低速帧中左脚距表面 ≤1.5 cm | 22.5% | 96.3% |
| 源人体低速帧中右脚距表面 ≤1.5 cm | 18.5% | 92.4% |
| 左/右脚连续过滤后接触步数 | 198 / 166 | 848 / 839 |
| 左/右手连续过滤后接触步数 | 373 / 254 | 384 / 291 |

诊断中的 stance 只是源人体脚速度阈值启发式，并非支撑力真值。修正后左脚 gap p95 为 1.21 cm，右脚仍为 6.67 cm；短暂悬空、脚滑、接触语义仍需检查，不能据此宣布所有支撑质量问题已解决。

发现并修正了一个采样问题：先在 30 Hz 求解，再把机器人 qpos 插值到 50 Hz，会引入原求解帧中不存在的障碍物穿透；本片段出现 2 帧超过 3 mm，最深约 7.69 mm。正式脚本因此先插值人体点，再逐帧求解 50 Hz 机器人姿态。

基础 Holosoma 结果的部分站立阶段悬空，不应通过放大 contact 阈值掩盖。已有 foot-support 碰撞体选择原先只识别 Roban，本轮补充 G1 足底识别。首次 support 实验在约第 297 帧不可行，放宽到 3 mm 脚锁容差仍在约第 310 帧失败；两份独立日志保留。

随后定位到共享 `MjData` 状态错误：查询上一帧足底位置会修改状态，但支撑距离/雅可比没有恢复到当前 SQP 姿态。已在 `solve_single_iteration` 支撑查询前恢复当前 q 并调用 `mj_forward`。保持原 1 mm 容差重新求解完整片段成功，贴合结果如上表。新增回归测试在关闭时间成本时改变上一帧 Z，验证当前支撑解保持一致；去掉修复，该测试出现约 0.149 的姿态差异并失败。

ACP 测试 8 项、Holosoma retargeting 测试 12 项通过。结果目录包含 `manifest.json`、五阶段日志、`labeled/motion.npz`、`labeled/report.json`、`validation/mujoco_geometry_report.json`、`reference.mp4` 和两个回放截图。小型报告副本保存在 `docs/05_实验记录/数据快照/holosoma_support_*.json`。

接触标签当前使用工程距离阈值 1.5 cm、3 mm 穿透容差和最小连续 10 步。它们来自静态碰撞几何，不能称为 measured-force labels；Stage 2 的 hindsight 还需成功执行 rollout 和世界系竖直力 >1 N 事件。自碰撞、动力学稳定、执行器/力矩验证仍未完成，当前输出状态为 `provisional`，未标成可训练正式数据。

下一项验收剩余悬空/滑移片段和接触时序，再接入原始 LAFAN BVH。Mixamo 的 FBX 骨架导出和缺少 human joints 的 29→23 qpos 迁移保持独立适配步骤。
