# 采用 ASAP 关节定义的 29 → 23 DoF 数据基线

> 统一入口与目录规范见 [快速开始](../01_使用指南/快速开始.md) 和 [完整手册](../01_使用指南/数据流水线手册.md)。本页保留各来源的详细适配与验证记录。

2026-10-29。当前数据主线改为：**已重定向的 G1-29 动作 → ASAP 定义的 23 个关节 → 接触标签与源数据逐帧对照**。全部 145 条 robot-terrain 已完成，50 Hz 共 91,095 帧。原始 30 Hz 的关节子集也单独保存，没有覆盖源数据或旧 Holosoma 输出。

## 为什么之前的结果会差

此前多数输入使用人体点重新拟合到另一款 23 DoF 模型；机器人 FK 路径也只用源关节值作为 nominal，通过 interaction mesh 重新优化整套姿态。这会再次改变根部、腿、腰与支撑关系，而不是只去掉 6 个自由度。

此外，两个“23 DoF”定义不同：

| 模型定义 | 腰部 | 每只手臂 | 固定／缺少的关节 |
| --- | --- | --- | --- |
| 此前 Unitree `g1_23dof_rev_1_0` | yaw | 肩 3 + 肘 1 + 腕 roll 1 | 腰 roll/pitch、双腕 pitch/yaw |
| ASAP `g1_29dof_anneal_23dof` | yaw/roll/pitch | 肩 3 + 肘 1 | 双腕 roll/pitch/yaw 共 6 个 |

本地 ASAP 配置 `humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml` 的 23 个关节包含腰部 3 DoF。对应 URDF 把 6 个腕关节直接定义为 `fixed`，保留零位 origin。仓库中没有找到独立的 29 DoF 轨迹逐步收缩到 23 DoF 的“退火”转换器。

ASAP 的 `scripts/data_process/fit_smpl_motion.py` 是 SMPL → 已约束机器人模型的拟合，包含关节范围裁剪和高斯平滑；末尾还按最低 mesh 高度整体调整根部 Z。这个整体落地处理不适合直接套用到已经和台阶对齐的 robot-terrain 动作。

本次采用 **ASAP 的关节定义**，从 OmniRetarget 自带 `g1_29dof_spherehand.urdf` 生成固定双腕的版本，保留源几何、质量和关节坐标定义。这样可把自由度减少造成的变化与更换模型造成的变化分开。本次直接关节子集是本项目面向已重定向数据的实现，不称为复现了 ASAP 的 SMPL 优化器。

## 已完成的处理与产物

`src/acp/robot/model.py` 生成 URDF／MJCF，依据本地 ASAP 文件验证 23 个关节的集合。生成模型的 FK 已用随机根姿态和关节角与原模型验证。MJCF 保存补充了双精度 body/joint 变换，避免默认 XML 导出的小数取整造成 FK 偏差。

`src/acp/robot/joints.py` 按关节名选取列，校验仅去掉六腕。根位置、根四元数、保留关节不优化、不裁剪、不平滑。50 Hz 版本先对源 29 DoF 使用相同时间网格插值，再选取 23 个关节。原始 30 Hz 子集直接选列，逐值不变。

`src/acp/pipeline/datasets/omni.py` 批量转换，并复用九部位几何接触标签、未来接触目标与双引擎校验。与新旧模型对应的手部／足部几何分组已适配，评估读取 manifest 中的 mapping。

结果目录为 `outputs/数据集/omni/asap23_baseline`：

- `assets/g1_asap23_spherehand.urdf`、`.xml`、`provenance.json`：关节定义及来源哈希；用于离线 FK／查看，尚未生成训练执行器配置。
- `inventory.json`、`artifact_index.json`：全部输入和最终产物索引。
- 每条 `source_rate_subset.npz`：源 30 Hz 的原样子集，canonical 顺序 `WXYZ,XYZ,23 joints`。
- 每条 `retargeted/*_original.npz`：50 Hz 查看／后处理接口，MuJoCo 顺序 `XYZ,WXYZ,23 joints`。
- 每条 `labeled/motion.npz`：规范化轨迹、九部位几何接触、未来接触目标等。
- 每条 `comparison/report.json`、`diagnostics.npz`：源 29 DoF、新子集、旧 IK 路径的配对指标。
- `paired_summary.json`、`paired_comparison.json`：全量配对汇总和逐条结果。
- `wrist_contact_repair_queue.json`：新增原生环境／自接触身体对或丢失源手部近表面状态的动作清单，按新增环境超限帧、丢失接触步数排序；不是物理碰撞真值。
- `quality_summary.json`：保留旧工程门槛的诊断结果，须结合下文解释。

源模型的半球手掌形状保留。固定腕部会改变它的位置和朝向，不等于手部所有几何接触都会保持。

## 全量比较结果

`src/acp/quality/compare.py` 将三种轨迹放在相同 50 Hz 时间戳比较，并另行校验源帧率导出是否逐值保持。

| 指标 | 结果 |
| --- | ---: |
| 处理动作 | 145 / 145 |
| 50 Hz 总帧数 | 91,095 |
| 原始帧率与 50 Hz 子集的根部、保留关节逐值校验 | 全部通过 |
| 双脚、躯干、骨盆、双肘最大 FK 位置误差 | 0 m |
| 上述部位最大旋转矩阵误差 | 0 |
| 新版本关节位置／速度超限动作 | 0 / 0 |
| 旧 IK 路径相对源动作的双脚位置平均误差 | 4.53 cm |
| 旧 IK 路径相对源动作的双脚位置最大误差 | 1.339 m |
| 旧 IK 路径根位置最大偏差 | 57.65 cm |
| 固定腕部后的手中心平均位移 | 0.876 cm |
| 固定腕部后的手中心最大位移 | 10.07 cm |
| 源手掌近表面状态保留率 | 56.30% |

双脚位置使用同名 `ankle_roll_link` 的世界位置比较；旧路径也包含模型几何定义差异，因此是相对源动作的总偏差。手掌近表面按主半球手掌 collision geometry 到环境的距离在 [−3 mm, 15 mm] 内定义，计数单位为“手 × 帧”，不是真实力接触。

`climb_21_z_scale_1.2` 的第 243/347 帧（零基编号 242）此前环境穿透约 86.7 mm，新版本与源 29 DoF 同为约 1.5 mm；整段的新／源环境距离最小值同为约 −8.9 mm。这说明此前的深脚部穿透在直接子集中没有重现。

**腕部固定仍引入了新的手部问题。** 在同一原生 contact 查询下，源环境超 3 mm 的帧数为 2,256，新版本为 11,913；其中 9,677 帧是源未超限、新版本超限。既有源超限与新版本超限重合 2,236 帧。全量最大环境深度分别约为源 49.2 mm、新版本 51.6 mm。不能把这条子集基线描述为所有接触问题已修好。

## 对旧筛选结果的修正解释

原始 G1-29 模型用同一原生碰撞定义查询时，91,095 帧也全部出现自接触超 3 mm，典型高频对为 ankle-roll／knee 等关节附近结构。腕部固定后的模型也有这些记录。因此“原生自碰撞检查每帧报警”不能直接推出源动作不可用，更不能把它全部归因于转换。

当前 `quality_summary.json` 保留旧门槛，会继续把 145 条都列为 `needs_repair`。这是一套尚未按源模型校准的整段零超限筛选；其自碰撞、低速 stance 和足底 frame 速度指标不应直接充当这批数据的最终验收结论。尤其本次腿、脚和根轨迹均未改变，足底速度代理量的源数据基线需要一起检查。

源模型已有的几何重叠、源动作的运动学残差，以及固定腕部新增的穿透／接触丢失应分别记录。`paired_*` 提供原生查询下的源／目标配对结果及新增身体对，后续物理碰撞允许对仍需按模型校准。固定腕部还会改变 MuJoCo 的 welded-body 碰撞过滤，因此 self pair 的变化也须结合过滤规则判断。

按“出现任意新增原生身体对或丢失任意源手部近表面步”生成的严格修复清单含 142 条，另 3 条在这些配对检查中未发现新增问题。新增环境身体对主要是左右半球手掌与箱体。这些是逐条定位条件，不是已校准的训练成功率。小型报告副本见 [配对汇总](../05_实验记录/数据快照/asap23_paired_summary.json) 和 [腕部接触修复清单](../05_实验记录/数据快照/asap23_wrist_contact_repair_queue.json)。

## 接下来的修复范围

1. 以本次子集作为固定基线，沿用源腿、腰和根轨迹。源已有的结构重叠用于校准碰撞允许对和验收定义；保留原始计数以便复查。
2. 优先处理腕部固定新增的手部接触变化：跟踪源接触面的接触位置／法向与手部碰撞几何，必要时只优化肩、肘；将脚、根和腰的保持条件纳入约束。仅拟合手中心不足以保持半球手掌的表面接触。
3. 对局部修正加入真实几何距离验收、步长回退、位置／速度约束和时序连续性。不可达的手姿态明确记录；23 DoF 无法保证精确复现任意 29 DoF 的腕部姿态。
4. 重新生成标签，与源及本次子集逐帧对照，之后再做全量回归。Stage 1 参考候选和 Stage 2 力接触 hindsight 分别验收。

本基线完成的是 **ASAP 关节定义的全量转换、接触标签、源／目标配对诊断和查看器适配**。后续已实现局部肩肘修正，并在三个完整动作上验证；详见 [局部接触修正](../03_方法与规范/肩肘接触局部修正.md)。全量修正及校准后的最终训练验收尚未完成。

## 复跑与查看

在 ACP 根目录执行，使用本机 Holosoma 环境：

```bash
PY=/home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/pipeline/datasets/omni.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/quality/evaluate.py \
  --run outputs/数据集/omni/asap23_baseline --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/quality/compare.py --workers 6

"$PY" scripts/view_omni_trajectory.py --clip climb_00_z_scale_0.9 --show-source
"$PY" scripts/view_omni_trajectory.py --clip climb_21_z_scale_1.2 \
  --show-source --frame 242 --paused
```

转换支持 `--clips NAME ...` 和断点续跑。`--fps 30 --output outputs/数据集/omni/asap23_baseline_30hz` 可保留 30 Hz 标签；与旧 50 Hz 路径的逐帧比较要求使用相同时间网格，当前 `compare_asap23.py` 的旧路径对照针对 50 Hz。

查看器默认指向新的 `outputs/数据集/omni/asap23_baseline`，HUD 额外显示源模型的原生距离与固定手腕造成的手中心位移。查看旧路径传 `--run outputs/实验归档/omni/omni_robot_terrain_all`。

视频：`outputs/可视化/历史回放/asap23_climb00_comparison.mp4`；问题帧图片：`outputs/可视化/历史回放/asap23_climb21_frame242.png`。这些均是运动学回放。

验证：ACP 测试 16 项通过、3 项本地 MuJoCo 集成测试在该环境跳过；新增实际源模型 FK 测试在 Holosoma 环境单独通过。全量 145 条均完成处理、配对检查及标签评估；新模型交互窗口、PNG 和 MP4 已验证。
