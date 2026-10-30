# PHP 跑酷数据接入

PHP 本地已有通过 OmniRetarget 得到的 G1-29 训练动作，可以采用与 Omni/LAFAN 相同的按名称投影方法转换到统一球手 G1-23，再对配套箱体地形生成接触标签。无需重新从人体数据拟合。

## 来源和规模

机器路径从 `configs/machine.local.json` 读取：

- 数据根：`php_dataset`，本机 `/home/lyx/train/retarget/dataset/php`。
- 项目根：`php_root`，本机 `/home/lyx/train/php_parkour`。
- 实际源模型：`motion_matching/motion_matching/assets/robots/unitree_description/mjcf/g1.xml`。
- 同仓 URDF：`.../urdf/g1/main_mesh_collision_halfspherehand.urdf`，不拿它替代已验证的 MJCF 来解释包内 FK。

| 子集 | 动作数 | 采样率 | 配套地形 |
| --- | ---: | ---: | --- |
| `php-motions-low-climb-76-v1` | 40 | 50 Hz | 每条 10 个箱体变体 |
| `php-motions-locomotion-v1` | 100 | 50 Hz | 每条 10 个箱体变体 |
| 合计 | 140 | 103,856 源帧 | 全变体共 1,400 组动作／场景配对 |

箱体来自 `*_terrain.npy`，不是把 robot 身体位置当作地形拟合。源文件名中的参数不用于推测箱体位置，读取实际数组。

## 输入与模型核验

`joint_pos` 是 `[T,36]`，布局为 **XYZ、WXYZ、29 个关节**，顺序由包内 `joint_names` 确定；不是 Omni 原始的 WXYZ、XYZ。包内另含速度、身体世界位置／四元数和命令数据。本阶段只把运动学轨迹、关节名称和配套地形用于离线转换，不伪造力接触。

适配器用实际 PHP MJCF 在首、中、尾帧计算公共身体位置，对照 `body_pos_w`；误差超过 1e-5 m 会报错，结果写 `source_format_validation.json`。这是输入布局和源模型检查，未代替整段动力学／几何检查。

目标统一到与 Omni/LAFAN/Mixamo 相同的球手 G1-23。源 PHP 机器人与统一球手模型的限位／碰撞几何可能不同，因此保存源模型和统一 29DoF 参考两份证据。`comparison/report.json` 只隔离统一参考 29→23 的变化，不代表 PHP 原始模型与目标模型所有几何等价。

## 地形变体

数组形状 **[箱体数, 变体数, 10]**，每个条目为 `XYZ + WXYZ + 完整 XYZ 边长`。变体 0 只是供应的数据变体，未确认是无扰动的原始场景。

默认对全部 140 条动作各取变体 0，先做基线筛查；`--all-variants` 显式扩展到 1,400 组。动作本身在不同变体间复用，不会因地形变化自动重新拟合姿态，出现穿透／悬空应作为真实失败项保留。训练划分按同一原动作分组，十个变体不得泄漏到验证集。

## 使用

```bash
# 清点，不运行逐帧标注；不要在主批次活动时重写它的清单
.venv/bin/python scripts/acp_pipeline.py run php --audit-only \
  --output outputs/实验归档/php/input_audit

# 两条短样例：攀爬与行走
.venv/bin/python scripts/acp_pipeline.py run php --workers 2 \
  --clips php_climb_14ae61dde74c_v00 php_locomotion_b204a8ca16dc_v00 \
  --output outputs/实验归档/php/my_pilot

# 主批次 140 条，每条变体 0；活动批次会拒绝重复启动
.venv/bin/python scripts/acp_pipeline.py run php --workers 4 --background
.venv/bin/python scripts/acp_pipeline.py status php
.venv/bin/python scripts/acp_pipeline.py evaluate php --workers 4
```

每条保存原速率 23 关节子集、目标场景、Drake 接触、MuJoCo 验证和成对报告。`source_rate_subset_exact` 与 `resampled_subset_exact` 验证投影数值保持；脚、躯干和肘等保留链 FK 的误差应接近数值精度，手部变化独立统计。

已完成的两条 pilot 跑通全部处理阶段，但当前严格全片门槛均标 `needs_repair`，包括穿透、限位、自碰撞等；全量报告用于区分新增／继承问题。最终筛查数量查看主批次 `QUALITY.md`，不可仅从输入文件完整推断全部可直接训练。

回放入口和输出位置见 [快速开始](../01_使用指南/快速开始.md)。实现为 `src/acp/data/adapters/php.py` 与 `src/acp/pipeline/datasets/php.py`。

## 2026-10-30 全量基线记录

140 条均已尝试，135 条完成转换、标注、MuJoCo 验证和成对检查；5 条原始几何查询出现长时间停滞，保留在失败分母。已对脚部碰撞体／地面距离查询复现，不将其误记为关节名称转换失败。统一入口采用逐片段标注超时，异常条目可单独重试。135 条成功处理片段在当前严格全片筛查下均为 `needs_repair`；这与处理故障是两个不同队列。

抽查回放：[PHP 攀爬源／目标叠加截图](../../outputs/可视化/php/adapter_pilot.png)。后续应优先解决几何查询数值问题，再按原始／统一模型差异、地形变体和新增／继承问题推进质量修正，不默认修改源动作。
