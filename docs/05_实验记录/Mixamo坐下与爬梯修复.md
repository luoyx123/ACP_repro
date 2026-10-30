# Mixamo 修复与坐下／爬梯回归

2026-10-30。旧批次已暂停；新版全量任务 `outputs/数据集/mixamo/20261030_asap23_v1` 已启动，按坐下和梯子候选优先排序。2,453 条输入的全量质量统计尚未完成，实时状态读该目录 `progress.json`。

## 已修复

- 脚尖对应前方 toe frame，保留源高度，完整三维初始根姿态。
- stance 仅使用明确支撑面，落脚转换不锁旧锚点；足滑指标不再混入非 stance、触地转换或足底中心旋转。
- 手与髋／腿／躯干避碰；导出时仅修正微小数值限位残差，重新验证实际轨迹。
- 坐面／横档使用同一份 URDF 进入 Drake 和 MuJoCo；按源阶段增加手／骨盆几何支撑代价。
- 查明单步爬梯右手接触丢失来自线性化步长跨过细横档：加入真实几何支撑误差检查和回退，右手从 0 接触帧恢复到 46 帧。回退不会接受进入横档内部的插值姿态。
- 建立目标 URDF 哈希绑定的踝关节外壳局部允许碰撞规则，保留 raw self 诊断，并继续拒绝其他身体对、远离关节或过深碰撞。
- 数值后端继续要求同一约束残差 ≤1e-4；高精度重试修复部分数值失败。评估失败与重定向失败分开记录，单条失败不取消其余片段。

## 重点动作实测

以下 6 条均已在**新版全量输出目录**完成：环境穿透超限、关节位置／速度超限、非允许自碰撞均为 **0 帧**。这些结论是离线几何检查结果，尚未进行动力学跟踪验证。

| 动作名称候选 | Clip | 50Hz 帧数 | 平均映射点误差 | 左／右手接触帧 | 骨盆接触帧 |
| --- | --- | ---: | ---: | ---: | ---: |
| Sitting_6 | `0f3625529b2f2b0f970819e47fb7de1a` | 222 | 1.81 cm | 0 / 49 | 94 |
| Climbing_Ladder | `27b793aeb4838dbb29025352815be1a0` | 39 | 2.45 cm | 20 / 12 | 0 |
| Climbing_3 | `6d6857f80acfb5ef3111f77e86d4eab9` | 149 | 3.41 cm | 93 / 102 | 0 |
| Climbing_6 | `8c56f6ba2d819e69ecf6764f4ec6542e` | 78 | 2.87 cm | 52 / 46 | 0 |
| Start_Climbing_Ladder | `bfe5d1961033c37b16af565b85dc1dad` | 102 | 2.75 cm | 46 / 14 | 0 |
| Climbing_6 | `de853577df76be590347f5759a8dccde` | 78 | 2.87 cm | 52 / 46 | 0 |

坐下：末段 20% 轨迹的骨盆接触覆盖率为 **100%**。梯子：源预期支撑阶段的手／足接触覆盖率均为 **100%（有该阶段的部位）**；某部位没有持续至少 0.2s 的源支撑阶段时显示空值，不硬造该类标签。详细逐动作数值见 [JSON 快照](数据快照/mixamo_repair_priority.json)。

## 可视化

已导出完整短片，青色为原人体骨架，绿色／黄色为接触点与支撑面见证点。视频诊断保留 raw self 数值，但只将超出局部允许规则的自碰撞标红。

- [坐下视频](../../outputs/实验归档/mixamo/mixamo_asap23_priority_support_v7/review/sit_down.mp4)
- [坐下末段截图](../../outputs/实验归档/mixamo/mixamo_asap23_priority_support_v7/review/sit_down_terminal.png)
- [梯子：单步视频](../../outputs/实验归档/mixamo/mixamo_asap23_priority_support_v7/review/climb_one_step.mp4)
- [梯子：两步视频](../../outputs/实验归档/mixamo/mixamo_asap23_priority_support_v7/review/climb_two_steps.mp4)

这些专用回归目录保留完整产物；全量结果统一输出到 `outputs/数据集/mixamo/20261030_asap23_v1`。交互回放和断点恢复命令见 [流程文档](../02_数据集/Mixamo使用说明.md)。

## 场景与剩余工作

当前椅面和横档是可编辑的**工程测试场景**，不是原始 FBX 或论文作者的环境。6 条重点轨迹目前只剩 `unconfirmed_support_scene` 准入原因，配置保留 `approved_for_training=false`。尚未建立场景的其他坐姿会明确报告 `missing_seat_geometry`；其余轨迹仍在全量重跑和质量检查。

需要结合实际训练环境检查横档间距／深度、扶手、动作名称匹配与意外膝盖接触。部分单步／进入梯子的样例出现膝盖与横档接触，原始几何标签如实保存，不能仅凭“接触覆盖率高”判定接触语义正确。场景确认后仍需 Stage 1 tracker 动力学 rollout；Stage 2 力接触 hindsight 单独生成。

名称来自 123 条具名参考的轨迹相似度匹配：117 条候选中包括 5 条梯子候选；身份与场景均没有冒充为原作者披露内容。完整来源、匹配分数、motion_id 与参考文件 SHA-256 在 `configs/mixamo_semantics.json`。

## 验证

- ACP：24 项通过，7 项因环境依赖跳过。
- Holosoma 环境下的 ACP 求解测试：2 项通过，包含同约束求解器恢复、真实手部间隙下降、跨横档步长回退及穿透中间态拒绝。
- Holosoma 23 DoF 回归：4 项通过，包含 URDF／MJCF frame 一致性与前帧足部查询后的当前姿态支撑计算。
- 6 条重点动作均经过 Drake 标签与原生 MuJoCo 几何交叉检查。
