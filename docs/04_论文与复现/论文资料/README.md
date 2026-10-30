# 论文阅读材料与来源

本目录同时收录两篇论文。原始文件都在 `assets/papaer/`（原始目录名 `papaer` 按现有素材保留）。

## ACP：Humanoid Mobility via Attention-based Contacts

原始文件：[2610.31608v1.pdf](../../../assets/papaer/2610.31608v1.pdf)。

- [论文精读笔记](NOTES.md)：边读边写的逐节笔记，目前覆盖 Abstract 与 Introduction（p.1–5）。
- [MarkItDown 提取 Markdown](2610.31608v1.md)：便于搜索和阅读，数学公式、双栏顺序、表格可能失真。
- [pdftotext -layout 提取](2610.31608v1.layout.txt)：用于补充核对，不替代原始 PDF。
- [来源记录](provenance.json)：包含原始 PDF 的 SHA-256、转换版本和视觉检查页码。
- 阅读时还将 PDF p.4、13、19、22、23、24 渲染为图像对照 Fig.2、公式及 Tables 3/S3–S6。

## AME-2：Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding

原始文件：[Zhang 等 - 2026 - ...pdf](../../../assets/papaer/Zhang%20等%20-%202026%20-%20Agile%20and%20Generalized%20Legged%20Locomotion%20via%20Attention-Based%20Neural%20Map%20Encoding.pdf)，
arXiv:2601.08485v1（2026-01-13，20 页）。

- [AME-2 精读笔记](NOTES_AME2.md)：按"元信息 / 问题与假设 / 方法论 / 工程 / 实验 / 批判借鉴"模板整理，
  方法论章节重排并重写公式；含与 ACP 的接口对照表与可迁移清单。
- [逐节精读](AME2_CLOSE_READING.md)：按章节顺序增长，当前含 Introduction 与 Related Works（逐段总结、引用对照、要点）。
- [MarkItDown 提取 Markdown](AME-2.md) / [pdftotext -layout 提取](AME-2.layout.txt)：同样以 layout 做核对依据。
- [来源记录](AME-2.provenance.json)。
- 官方代码：[leggedrobotics/ame2_minimal](https://github.com/leggedrobotics/ame2_minimal)（G1 + Isaac Lab 2.3.2；
  建图模型与 G1 学生训练按 README 于"下一篇论文"发布时才放出）。

## 转换方式

转换于 2026-10-29 / 2026-10-30，本地模式、未启用插件/云服务。MarkItDown 实际安装版本 **0.1.9**；
使用 `markitdown[pdf]`（`uv run --with "markitdown[pdf]==0.1.9"`），不是技能示例中的 0.1.8。

```sh
uv run scripts/extract_paper.py           # ACP，脚本里固定了文件路径
pdftotext -layout assets/papaer/2610.31608v1.pdf docs/04_论文与复现/论文资料/2610.31608v1.layout.txt
# AME-2：同样的 MarkItDown 调用，输入换成 assets/papaer/Zhang 等 - 2026 - ....pdf
pdftotext -layout "assets/papaer/Zhang 等 - 2026 - Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding.pdf" \
  docs/04_论文与复现/论文资料/AME-2.layout.txt
```

核查的官方入口：

- ACP：[作者项目页](https://leggedrobotics.github.io/ACP/)、[作者方法页](https://leggedrobotics.github.io/ACP/method.html)、
  [arXiv v1 HTML](https://arxiv.org/html/2610.31608v1)、[MGMT 前置工作项目页](https://wholebodylocomotion.github.io/)
- AME-2：项目页 `sites.google.com/leggedrobotics.com/ame-2`、[arXiv 2601.08485](https://arxiv.org/abs/2601.08485)、
  [代码 ame2_minimal](https://github.com/leggedrobotics/ame2_minimal)

所有详细方法与数值整理以**本地 PDF v1**为依据；在线网页用于资源核查，不能悄悄替换论文版本。
PDF 文字提取质量问题尤其包括 `N(0,0.03²)` 被串成 `0.032`、公式上下标丢失、奖励表灰色禁用区丢失；
AME-2 的 `σ̂_t² < 0.2²` 在 layout 文本里会显示成 `0.22`、`t_mask` 的上下标与分式也会串行。
