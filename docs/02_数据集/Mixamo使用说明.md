# Mixamo 原始 FBX → Holosoma → 圆形手 G1-23 → 接触标签

> 统一入口与目录规范见 [快速开始](../01_使用指南/快速开始.md) 和 [完整手册](../01_使用指南/数据流水线手册.md)。本页保留各来源的详细适配与验证记录。

2026-10-30 修复版。原始动作目录 `/home/lyx/train/dataset/Mixamo_dataset/animation` 共 2,453 条 FBX；22 点骨架已全部提取，520,661 源帧。character 和 animation_extra 中的静态参考姿态不作为动作输入。直接优化 23 DoF，不先拟合 G1-29。

## 当前入口

旧的 `outputs/实验归档/mixamo/mixamo_asap23` 批次已暂停，保留作错误适配基线。**新版全量输出是 `outputs/数据集/mixamo/20261030_asap23_v1`，全量仍在执行。** 进度读 `progress.json`，日志 `batch.log`，进程号 `batch.pid`。全量结束前不能把骨架提取完成等同于全部重定向完成。重点样例与本轮修复见 [MIXAMO_REPAIR.md](../05_实验记录/Mixamo坐下与爬梯修复.md)。

```bash
cd /home/lyx/train/ACP_repro
PY=/home/lyx/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python

# 查询当前批次；不会启动新的任务
.venv/bin/python scripts/acp_pipeline.py status mixamo
tail -n 10 outputs/数据集/mixamo/20261030_asap23_v1/batch.log

# 当前批次结束或停止后才能对同一目录断点恢复
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" src/acp/pipeline/datasets/mixamo.py \
  --output outputs/数据集/mixamo/20261030_asap23_v1 --workers 8 \
  --extracted-root outputs/缓存/mixamo/fbx_skeletons_v1 \
  --support-scenes configs/mixamo_test_scenes.json

# 坐下样例的运动学回放；青色为人体参考
"$PY" scripts/view_omni_trajectory.py --run outputs/数据集/mixamo/20261030_asap23_v1 \
  --clip 0f3625529b2f2b0f970819e47fb7de1a --show-source
```

调试用 `--clips HASH ... --output 单独目录`，因为指定片段会替换该目录 inventory。`--fps` 默认 50 Hz，可调整，不改变动作时长；源骨架按 FBX 的实际 fps 求值。

## 处理与修复

1. Blender 逐帧求值骨骼 head/tail，包含 armature 世界变换、FBX 单位及 pre/post rotation。结果是米、Z-up，原始文件不改写。
2. 用 rest 腿长比例缩放，源 −Y 前向转机器人 +X，只平移 XY 原点。**保留源世界 Z=0，不再按最低脚尖压低整段轨迹**，适用于坐姿、空中、倒立和爬梯。
3. 16 个有效映射点包括骨盆、躯干、髋膝踝脚尖、肩肘手。ToeBase 对应前方 0.14m 的 `*_ankle_roll_sphere_5_link`；`acp_*_foot` 仍是标签/诊断中的足底中心，二者不能混用。53 点容器中其他填零点不作监督。
4. 初始根部使用髋横轴和脊柱构成的完整三维姿态，可处理躺卧/倒立。初始化没有改写参考轨迹。
5. ground stance 要求 toe 速度 <0.10m/s、接近明确支撑面 4.5cm、持续至少 0.08s。相邻帧均处于 stance 才启用 1mm/frame XY 约束，落脚转换不沿用旧锚点。明确配置的横档顶面也可作为脚部支撑；不会将高处脚尖自动当作地面。
6. 手与躯干/髋/膝间启用 3mm 自碰撞避让。对配置了支撑物的动作，还按源动作阶段加入实际几何距离的手/骨盆支撑代价。细横档可能被线性化步长跨越，因此每步用实际几何作支撑误差检查并回退步长；不放宽最终几何门槛。
7. 只对不超过 1e-4rad 的求解数值限位残差投影到精确关节范围，记录最大修正量，然后对导出轨迹重新检查几何与速度。更大残差直接失败。

数值恢复依次尝试 Clarabel、同一锥规划上的 SCS 及高精度 SCS，约束残差必须 ≤1e-4。失败不补帧、不伪装为成功。Holosoma 原生初始化时的 trust-region fallback 沿用原实现；ACP 的备用后端不会另行删掉约束。

本轮在用户 Holosoma 中增加 `additional_pose_costs` 扩展点，默认返回空列表；实际 Mixamo 接触代价与步长回退实现在 ACP 的 `src/acp/pipeline/contact_solver.py`，其他调用默认行为不变。

## 动作名称与支撑场景

哈希文件名没有动作语义。独立下载了 [具名 Mixamo 参考集](https://huggingface.co/datasets/Linzhan/Mixamo-Animations-Characters) 中的 123 条相关动作，用归一化 32 点时序骨架、动作时长比对，得到 117 条名称候选，其中 5 条是爬梯候选。**这是轨迹相似度候选，不是原文件身份认证**；名称、分数、参考 FBX SHA-256、motion_id 与数据来源见 `configs/mixamo_semantics.json`，前三近邻见旧批次 `review/name_matches.json`。

批次首先处理 Sitting_6 和梯子候选，然后其他坐姿候选，最后其余动作。

`configs/mixamo_test_scenes.json` 为重点 6 条动作提供可编辑的测试椅面/横档。坐面按末段骨盆碰撞外形定位；横档按稳定脚部支撑阶段拟合规则间距，保留尺寸、姿态和拟合证据。生成命令：

```bash
"$PY" src/acp/pipeline/scene_fit.py \
  --run outputs/实验归档/mixamo/mixamo_asap23_priority_flat --output configs/mixamo_test_scenes.json
```

**这些是工程测试场景，不是 FBX 原场景或作者场景。** `review_status` 和 `approved_for_training` 明确阻止直接准入。还需检查梯子的扶手、横档深度、膝盖接触是否合理，并替换为目标训练场景。其他坐姿暂未建立椅子，报告 `missing_seat_geometry`，不会把平地无骨盆接触视为有效坐下标签。可用 `--support-scenes` 指定新的逐动作 box 配置；同一 URDF 几何传给 Drake 和 MuJoCo。

## 标签与验收

九部位几何 signed-distance 标签：距离 15mm、穿透容忍 3mm、最短接触 0.2s、未来窗口 1.5s。这些是当前工程约定，几何标签不等于 Stage 2 力接触 hindsight。

脚滑检查使用源 stance 与目标接触的内部帧，Mixamo 对应 toe 锚点速度；没有支撑阶段时不伪报脚滑。仍保存足底中心速度作原始诊断。坐下检查末段骨盆接触覆盖率；梯子检查足/手的预期支撑阶段与实际接触重合度，而非要求每条短动作每个部位都必须落地。

`configs/g1_spherehand_collision_exemptions.json` 只允许两组膝/踝外壳在踝 pitch 关节 3.5cm 内、深度不超过 4cm 的结构重叠，并绑定目标 URDF SHA-256。97 条旧样例、5,698 个接触用于范围校准。**原始自碰撞仍全部记录**，手碰腿、远离关节的膝踝碰撞、过深重叠均继续报错。这是工程碰撞规则，不是动力学安全认证。

每条动作保存 manifest、human_reference、retargeted、labeled、validation、quality 和求解日志。Drake 标签与原生 MuJoCo 逐帧交叉检查，最终还需 Stage 1 跟踪器 rollout；全量处理完成、几何通过和训练准入是不同状态。
