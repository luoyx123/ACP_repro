# OmniRetarget robot-terrain 全量处理与评估

> 本页含已退役脚本的历史命令；当前操作请使用 [快速开始](../../01_使用指南/快速开始.md)。

更新：本文记录原 Holosoma → Unitree rev1.0 的历史路径。当前主线已改为 [ASAP 定义的 23 DoF 子集转换](../../02_数据集/OmniRetarget关节子集基线.md)，保留源根部、腿与腰；旧版整段筛选结果不能作为源数据有效性的结论。

2026-10-29。本轮范围仅为 `robot-terrain`，没有处理 `robot-object` 和 `robot-object-terrain`。评估对象是本项目转出的 G1 23 DoF 参考动作，源 29 DoF 数据另行保留。

145 条均已产出完整动作、九部位几何接触标签、未来接触目标、原生几何校验和逐条质量报告。共 **91,095 帧、50 Hz、约 30.32 分钟**。本轮保守筛选中 **0 条整段通过、145 条待修复**，不能整批直接作为 Stage 1 参考输入。

## 输入与处理路径

145 条来自 **29 组基础动作 × 5 个高度缩放**。112 条包含 `[T,53,3]` 人体点，33 条没有。训练／验证必须按基础动作分组，避免同一动作的缩放版本跨集合。

| 最终采用的路径 | 条数 |
| --- | ---: |
| 人体点 → Holosoma → 目标 23 DoF | 94 |
| 源 29 DoF FK → Holosoma → 目标 23 DoF | 32 |
| 人体输入失败后，改用源机器人 FK | 16 |
| 机器人 FK 恢复路径，去掉硬 XY 脚锁、保留支撑代价和碰撞约束 | 3 |
| 合计 | 145 |

33 条缺少人体点的轨迹全部完成迁移：32 条使用通常的 FK 路径，1 条使用去掉硬 XY 脚锁的恢复配置。恢复配置同样接受脚滑、穿透等检查，没有因求解完成而放宽质量门槛。

机器人间迁移使用数据集自带 `models/g1/g1_29dof_spherehand.urdf` 做源 FK，按 Holosoma mocap mapping 提取 15 个交互关键点。按关节名投影的 23 DoF 状态只用作 nominal；最终输出由 interaction-mesh 求解器重新优化。机器人 FK 点明确标记为 `source_robot_fk`，没有伪称恢复出了人体 mocap。当前路径没有额外的显式朝向任务，属于本项目的工程适配。

## 整体质量结论

下表是**轨迹条数**，同一条可能有多个问题；并非互斥分类。

| 检查 | 不通过条数 / 145 |
| --- | ---: |
| 原生碰撞模型自穿透超过 3 mm | 145 |
| 左脚接触期间水平速度 p95 超阈值 | 101 |
| 右脚接触期间水平速度 p95 超阈值 | 73 |
| 左脚源 stance 阶段近表面比例不足 | 69 |
| 右脚源 stance 阶段近表面比例不足 | 83 |
| 环境穿透超过 3 mm | 49 |
| 映射关键点误差超阈值 | 43 |
| 关节速度超过 URDF 限值 | 23 |
| 关节位置超限、帧数不完整 | 0 |

按帧加权，环境穿透无效比例约 **3.59%**；原生自碰撞标记约 **40.73%**，仅看相隔超过两个运动关节的身体对仍约 **33.23%**。环境最深穿透约 **86.7 mm**，出现在 `climb_21_z_scale_1.2`。因此当前问题不能只靠放宽几毫米阈值处理。

145 条的 MuJoCo／Drake 原始距离接触标记一致率均超过 **99.9%**，最低约 99.989%。这支持两套几何计算的对齐；它没有证明 IK 的非线性约束或动态执行可行。

自碰撞采用 MuJoCo 原生 collision filtering。报告额外列出远距离身体对及逐帧最深距离，以区分关节附近结构重叠；高频对包括左右手与同侧髋／大腿。相隔超过两个运动关节只是诊断分类，不能代替校准后的 allowed-collision matrix。源 URDF 也存在踝／小腿等结构碰撞重叠，后续应校准允许对，再加入求解约束。当前原生碰撞标记不应全部直接解释为实物碰撞真值。

脚滑使用语义足底 frame 的水平速度，排除接触段首尾一帧，属于筛选代理量；不是实测接触点的切向速度。源 stance 由脚速度 ≤0.3 m/s 的启发式获得，也不是支撑力标签。近表面距离查询封顶 10 cm。

## 筛选标准与阶段边界

标准在 [`configs/data_quality.json`](../../../configs/data_quality.json)，均为本项目工程阈值，并非论文披露的整段验收标准：

- 环境／自穿透容差 3 mm，整段不允许超限帧。
- 关节位置与有限差分速度不超 URDF 限值。
- 每只脚在源 stance 阶段距环境 ≤1.5 cm 的比例至少 90%。
- 接触段内足底 frame 水平速度 p95 ≤0.15 m/s。
- 映射关键点平均误差 ≤8 cm、总体 p95 ≤15 cm。
- 输出帧数完整，时间网格均匀，四元数有效，两引擎距离接触一致率至少 99.9%。

通过这些离线标准可形成 **Stage 1 参考候选**，并不要求先训练出 tracker。Stage 2 hindsight 数据则另需成功 tracker rollout 和实际接触力事件。当前候选清单为空，Stage 2 hindsight 数据也未生成。

## 50 Hz 的含义

论文 p.13 明确策略输出为 50 Hz、PD 为 200 Hz；Table S5 的接触时长和轨迹长度也按 50 Hz 控制步计数。论文没有要求原始动作文件必须存为 50 Hz。

本轮保留原始 30 Hz 文件，生成 50 Hz 查询版本：先插值人体点，或在目标时间网格上计算源机器人 FK，再逐帧求解目标姿态。没有对已解出的低频目标 qpos 做一次无约束插值后直接入库。

`--fps 30` 也已用独立 60 帧片段跑通：输出仍为 60 帧；1.5 秒未来窗口为 45 步，0.2 秒工程接触段过滤为 6 步。50 Hz 对应 75／10 步。插值不会增加源动作的信息量。

全量验收修正了两类尾帧取整问题：252 帧同频重采样少一帧，以及 523 帧、30 Hz 输入转换到 50 Hz 时少掉应保留的端点。前者复用 IK 结果补跑标签／校验，后者重新求解 `climb_01` 五个缩放版本，最终均为 871 帧。原始失败与修复前结果保留。

## 产物和复跑

本机结果目录：`outputs/实验归档/omni/omni_robot_terrain_all`。

- `artifact_index.json`：145 条最终采用产物的相对路径；恢复路径可能位于 `attempts` 中，按索引读取。
- `quality_summary.json`：全量汇总，包含全部输入分母。
- `quality_all_clips.csv` / `.json`：逐条指标和不通过原因。
- `QUALITY.md`：全量列表。
- `stage1_candidate_clips.txt`：本轮为空；`needs_repair_clips.txt` 包含 145 条。
- 各产物目录内的 `quality/diagnostics.npz`：逐帧自碰撞、映射误差和足底速度。
- `events_*.jsonl`、初次失败日志、`manifest.before_post_repair.json` 和 `attempts/pre_time_grid`：保存首次处理和修复记录。

小型报告副本见 [`docs/05_实验记录/数据快照/omni_terrain_quality_summary.json`](../数据快照/omni_terrain_quality_summary.json)、[`逐条 CSV`](../数据快照/omni_terrain_quality_all_clips.csv)、[`产物索引`](../数据快照/omni_terrain_artifact_index.json)。

从 ACP 项目根目录执行，使用已安装的 ACP `.venv` 和用户 Holosoma 环境：

```bash
.venv/bin/python scripts/batch_omni_terrain.py \
  --dataset-root /home/lyx/train/retarget/dataset/OmniRetarget_Dataset \
  --holosoma-root /home/lyx/train/holosoma \
  --holosoma-python /home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  --unitree-root /home/lyx/train/unitree_ros/robots/g1_description \
  --output outputs/实验归档/omni/omni_robot_terrain_all --workers 6 --fps 50

.venv/bin/python scripts/retry_omni_robot_fk.py \
  --run outputs/实验归档/omni/omni_robot_terrain_all \
  --dataset-root /home/lyx/train/retarget/dataset/OmniRetarget_Dataset \
  --holosoma-root /home/lyx/train/holosoma \
  --holosoma-python /home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  --workers 2

.venv/bin/python scripts/retry_omni_robot_fk.py \
  --run outputs/实验归档/omni/omni_robot_terrain_all \
  --dataset-root /home/lyx/train/retarget/dataset/OmniRetarget_Dataset \
  --holosoma-root /home/lyx/train/holosoma \
  --holosoma-python /home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  --workers 2 --relaxed

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  src/acp/quality/evaluate.py --run outputs/实验归档/omni/omni_robot_terrain_all --workers 4
```

CPU worker 的 BLAS／OpenMP 线程限制为 1，避免嵌套并行。支持断点续跑；`--clips` 与 `--force` 可重跑指定动作。新代码已修复时间网格问题；`resume_omni_postprocessing.py` 仅用于迁移旧产物的后处理失败。

ACP 测试 10 项通过；其 2 个 MuJoCo 测试在 ACP 环境跳过，在 Holosoma 环境单独运行通过。Holosoma retargeting 测试 12 项通过。

下一步先校准碰撞允许对、补自碰撞约束和非线性约束验收／步长回退，再调整支撑时序与脚锁；沿用这 145 条和同一版指标重跑。之后再扩展 LAFAN 29 DoF CSV 或原始 BVH。
