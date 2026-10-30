# Mixamo 旧批次抽查（2026-10-30）

这是修复前的历史快照；旧任务已暂停，最新修复与新批次见 [MIXAMO_REPAIR.md](../Mixamo坐下与爬梯修复.md)。

全量任务仍在运行，本报告固定检查已完成的 448 条、79,048 帧，不是全量最终验收。

- 环境几何超限：155 帧，约 0.196%；关节位置超限 9 帧、速度超限 12 帧。
- 任一脚滑代理指标报警：326/448 条；接触足底点速度 P95 的跨动作中位数左右分别约 0.185/0.206 m/s。
- 原生自接触每帧报警包含已知结构重叠；跨超过两个活动关节的身体对超限占 8.31%，该分类仍不是校准后的物理碰撞矩阵。
- 人体映射点平均位置误差的跨动作中位数 4.19 cm、P95 为 5.41 cm，最差 48.69 cm；这些比较也包含人体与机器人比例、frame 语义差异。
- 当前旧门槛下没有训练候选；118 条仅有 raw self-collision 原因，不能因为结构重叠直接判定这些动作坏。

## 已定位的适配问题

1. `src/acp/pipeline/datasets/mixamo.py` 将人体 ToeBase 对应到 `acp_*_foot`，该 frame 在踝下方；现有 `*_ankle_roll_sphere_5_link` 才在前方 0.14 m。源踝／脚尖间的前后关系没有正确对应到目标，需先修正并做隔离回归。这是当前适配器的问题，不能归咎于 Mixamo 动作。
2. 整段最低脚尖落地假设不适合倒立／空中姿态。已完成样本中 3 条出现处理后的源骨盆 Z<0；最差样例 `2e13a31de3fb474c2e720669f5bc311b` 的落地偏移为 −1.159 m，骨盆约 −0.24 m，目标机器人明显失真。需要按支撑语义选择脚、手、躯干等地面估计，或标为场景待确认，不能继续盲用脚尖最低点。
3. 当前没有硬 XY 足底锁定，soft support 不保证无滑动。完成上述坐标和 frame 修复后，才适合评价是否要启用源 stance 对应的足底保持。
4. `quality_summary.json` 仍是早期两个样例的报告，完整批次结束后才由批处理覆盖。实时状态读 `progress.json`，本次已完成集合的汇总读 `review/snapshot.json`，不要把旧 summary 当全量结果。

## 可复查产物

- 较好长动作：`outputs/实验归档/mixamo/mixamo_asap23/review/good_long.mp4`，选自完整 2,228 帧动作的中间 8 秒。它仅有 raw self-collision 原因，不等于动力学已通过。
- 中等位置误差样例：`outputs/实验归档/mixamo/mixamo_asap23/review/median.png`；可见目标脚部悬空。
- 最差样例：`outputs/实验归档/mixamo/mixamo_asap23/review/worst.png`；可见倒地扭曲，与错误落地假设一致。
- 完成集合详细指标：`outputs/实验归档/mixamo/mixamo_asap23/review/completed_quality.json`。
- 固定快照：`docs/05_实验记录/数据快照/mixamo_review_snapshot.json`。

本轮只检查和记录，未改变正在运行批次的配置或输出。当前输出保留为调试基线，尚未判定可直接用于训练。
