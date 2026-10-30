# LAFAN1：已重定向 G1-29 转为统一的 23 DoF

> 统一入口与目录规范见 [快速开始](../01_使用指南/快速开始.md) 和 [完整手册](../01_使用指南/数据流水线手册.md)。本页保留各来源的详细适配与验证记录。

输入为用户本地 `/home/lyx/train/dataset/LAFAN1_Retargeting_Dataset/g1`。采用与 OmniRetarget 相同的 ASAP 关节子集：**保留三轴腰部，六个腕关节固定为零，每臂保留肩部三轴和肘部**。继续使用当前圆形手掌模型，不通过人体点重新拟合整套轨迹。

## 数据与转换约定

数据集中 G1 部分共 **40 条 CSV、264,705 个原始帧、30 Hz**，属于 15 个动作名组。只处理 `g1/`，不处理 `h1/`、`h1_2/`，也不需要重新下载原始 BVH。输入为无表头的 36 列：`XYZ, QX QY QZ QW, 29 joints`。关节顺序依据仓库 README 和源 URDF 双重校验，使用显式关节名映射，不能直接截掉最后六列。

原始帧率导出逐值保留根位姿和选中的 23 个关节，共 30 列。另生成 **50 Hz、441,140 帧**的标准标签版本：根旋转 SLERP、根位置与关节线性插值，不延长原动作。50 Hz 为当前训练参考时钟，非输入数据必须满足的条件；`--fps 30` 可生成 30 Hz 标签版本。

不做根部整体抬高、重新落地、关节裁剪或全身平滑。源 CSV 不修改。原始四元数受 CSV 精度影响有约 1e-6 的单位长度误差，原始帧率子集保留这些值，SLERP 的 50 Hz 版本使用归一化旋转。

## 区分模型修订与自由度缩减

LAFAN 附带的 `robot_description/g1/g1_29dof_rev_1_0.urdf` 使用平手掌，与当前 Omni 来源的球手模型并不完全相同。关节名、运动轴及腿部运动学相同，但发现以下连接位置差异：

| 关节 | LAFAN 源 URDF | 统一球手 URDF |
| --- | --- | --- |
| `waist_roll_joint` 的 origin Z | 0.044 m | 0.035 m |
| `waist_pitch_joint` 的 origin Z | 0 m | 0.019 m |
| 双肩 pitch 的 origin Z | 0.24778 m | 0.23778 m |

两版 hip-roll 的 effort/velocity 参数也不同，全部差异和文件哈希记录在 `assets/lafan_provenance.json`。没有为了统一模型而改写这些 CSV 关节值。

每条输出提供两套比较：

1. **原始 LAFAN-29 → 统一球手 G1-29**：同样的根位姿与 29 个关节值，报告模型修订引起的 FK 差异；双脚 FK 单独断言一致。原始平手掌与半球手掌不作为同一个接触表面强行比较。
2. **统一球手 G1-29 → 统一球手 G1-23**：同一时间网格和几何，隔离冻结手腕造成的变化。逐值验证源帧率子集与 50 Hz 子集，比较双脚、躯干、骨盆、双肘的 FK，以及手部位移和接触变化。

因此查看器中的青色源骨架表示“CSV 关节值在统一球手 G1-29 上的参考”，HUD 有相应说明；它不是 LAFAN 原始平手掌机器人的完整网格。

## 全量检查结果

40 条均完成转换、标签、MuJoCo 复核和全帧配对检查，没有处理失败。目标 URDF 和 MJCF 的 SHA-256 与此前 Omni 基线分别相同。

| 指标 | 结果 |
| --- | ---: |
| 原始 30 Hz NPZ／CSV 根与保留关节逐值核对 | 40 / 40 通过 |
| 50 Hz 轨迹与源重采样后关节子集逐值核对 | 40 / 40 通过 |
| 腕部缩减的双脚／躯干／骨盆／双肘最大 FK 误差 | 0 m |
| LAFAN 原模型 → 统一模型的双脚最大 FK 误差 | 0 m |
| 模型修订引起的躯干最大 FK 差异 | 11.95 mm |
| 模型修订引起的肩／肘最大 FK 差异 | 6.46 mm |
| 23 DoF 关节位置／速度超限帧 | 0 / 0 |
| 冻结手腕后的手中心位移：均值／最大 | 18.41 / 45.12 mm |
| 统一 29 DoF 参考的手部近表面步数 | 5,534 |
| 固定手腕后仍保留的近表面步数 | 766，13.84% |
| 腕部缩减后全机器人环境距离从未超限变为超限的帧 | 892 |

近表面区间为 [−3 mm, 15 mm]，计数单位为“手 × 帧”，与 Omni 指标相同。892 是“源全机器人未超限、目标全机器人超限”的帧数；源已有其他部位超限时新增的手部身体对会另行记录，因此它不是所有新增手部穿透的总计数。

三个几何阶段的全机器人原生环境超 3 mm 帧数分别为：LAFAN 原模型 78,798；统一球手 29 DoF 116,641；统一球手 23 DoF 117,531。它们包含模型碰撞几何变化、源残差和手腕缩减影响，不能全部归因于 29→23，也不直接判定源动作的物理有效性。统一 29／23 的原生自接触计数均为 441,140 帧，仍存在已知关节附近结构重叠，需校准允许碰撞对。

严格配对复查清单包含 40 条，因为其条件是“任意新增原生环境／自接触身体对或丢失源手部近表面步”。走跑动作也可能有新增手／髋记录；不是只有手撑地动作需要检查。这个清单不是 40 条动作全部不可用的结论。机器可读摘要见 [lafan_asap23_paired_summary.json](../05_实验记录/数据快照/lafan_asap23_paired_summary.json)。

## 输出目录

默认 `outputs/数据集/lafan1/asap23_baseline`：

| 文件 | 内容 |
| --- | --- |
| `assets/g1_asap23_spherehand.urdf`、`.xml` | 与 Omni 主线一致的 23 DoF 模型 |
| `assets/original_robot.xml`、`original_scene.xml` | LAFAN 自带 29 DoF 模型及平地场景 |
| `assets/reference_robot.xml`、`reference_scene.xml` | 统一球手 29 DoF 参考及平地场景 |
| `assets/provenance.json`、`lafan_provenance.json` | 模型、数据版本、转换约定和差异 |
| `inventory.json`、`artifact_index.json` | 输入清单和查看索引 |
| `paired_summary.json`、`paired_comparison.json` | 全量转换和源／目标诊断 |
| `wrist_contact_repair_queue.json` | 冻结手腕新增的接触变化清单 |
| `quality_summary.json`、`QUALITY.md` | 保守离线筛选，非动作物理有效性的最终结论 |

每条 `clips/<动作名>/` 下：

- `source29.npz`：原始 30 Hz，转为 canonical `WXYZ,XYZ,29 joints`。
- `source_rate_subset.npz`：原始 30 Hz，`WXYZ,XYZ,23 joints`，含关节名。
- `source_rate_subset.csv`：原始 30 Hz，沿用输入根布局 `XYZ,XYZW,23 joints`，30 列、无表头。此文件已是 23 DoF，不能再用固定要求 36 列的 G1-29 CSV loader。
- `retargeted/<动作名>_original.npz`：标签帧率的 MuJoCo `XYZ,WXYZ,23 joints`，附源 FK 和 stance 诊断。
- `labeled/motion.npz`：规范轨迹、九部位几何接触、未来接触目标及有效掩码。
- `comparison/report.json`、`diagnostics.npz`：统一 29／23 的逐帧对照。
- `model_revision_report.json`、`model_revision_diagnostics.npz`：LAFAN 原模型与统一 29 DoF 的差异。
- `manifest.json`、`validation/`、`quality/`：处理状态、哈希、MuJoCo 复核和质量诊断。

## 复跑、查看和视频

在 ACP 项目根目录执行：

```bash
PY=/home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/pipeline/datasets/lafan1.py --workers 6
"$PY" scripts/summarize_lafan_asap23.py

"$PY" scripts/view_omni_trajectory.py --run outputs/数据集/lafan1/asap23_baseline \
  --clip walk1_subject1 --show-source

"$PY" scripts/view_omni_trajectory.py --run outputs/数据集/lafan1/asap23_baseline \
  --clip fallAndGetUp1_subject1 --show-source
```

转换支持 `--clips NAME ...`、`--force`、`--fps`、`--output`、`--dataset-root`、`--reference-urdf`。`--clips` 运行会把该输出目录的 inventory 设为所选动作；默认全量复跑会恢复 40 条清单。代码／输入／模型哈希及关键产物一致时跳过已完成片段；标签和筛选使用现有 ACP 工具。ACP `.venv` 负责 Drake 标签，Holosoma 环境负责 MuJoCo 模型、配对诊断及查看。

已导出 `outputs/可视化/历史回放/lafan_dance1_asap23.mp4`，为舞蹈前 10 秒运动学回放。查看器也支持暂停、逐帧、跳转诊断事件和 PNG 导出；操作见 [可视化说明](../01_使用指南/可视化与排错.md)。

局部修正工具也已适配本批的 canonical 源 NPZ 和平地 manifest。需要完整修正某条动作时使用：

```bash
RUN=outputs/数据集/lafan1/asap23_baseline_arm_repair
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/quality/repair.py \
  --baseline outputs/数据集/lafan1/asap23_baseline --output "$RUN" \
  --clips fallAndGetUp1_subject1 --passes 4 --smooth-weight 3 --workers 1
"$PY" src/acp/quality/finalize.py --baseline outputs/数据集/lafan1/asap23_baseline --run "$RUN"
"$PY" scripts/summarize_arm_repair.py --run "$RUN"
```

此入口本次仅做了 3 帧接口冒烟验证，没有把它列为完整动作修正结果；上面的全量指标均为未做局部肩肘修正的子集基线。

## 使用边界

本批完成的是**全量关节子集转换、接触标签和配对诊断**。冻结手腕对普通走跑的腿和根轨迹没有影响；倒地起身等手撑地动作仍可能改变手部近表面状态，修复清单单独保存。本批没有运行肩肘局部修正，也没有进行动力学验收。

地形设为平地 `z=0`，不从动作自动猜测台阶或障碍物。标签是 provisional 几何接触，不是 Stage 2 力接触 hindsight。原始 LAFAN 模型与统一模型的碰撞几何、结构重叠和原有地面残差需要与“腕部固定新增问题”分开解释。不能将保守质量筛选报警直接写成源动作不可用。

后续以修复清单为依据处理手撑地动作，沿用 [局部肩肘修正方法](../03_方法与规范/肩肘接触局部修正.md)，再校准碰撞允许对、接触 frame 和动态 tracker。划分训练／验证集时按原始动作组与受试者协议组织，避免同一长动作的邻近窗口跨集合泄漏。

验证：ACP 测试 20 项通过、5 项 MuJoCo 集成测试在该环境跳过；局部修正相关的 2 项 MuJoCo 集成测试在 Holosoma 环境单独通过。全部 40 条源帧率 CSV／NPZ 另行直接与原 CSV 逐值核对。LAFAN 交互窗口、PNG、251 帧 MP4 已验证。
