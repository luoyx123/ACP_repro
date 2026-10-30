# 仓库协作约定

## 阶段完成维护

每完成一个涉及代码、目录、命令入口或接口变化的阶段，必须在最终交付前执行 `.venv/bin/python scripts/acp_pipeline.py structure --stage "阶段说明"` 更新中文架构索引与阶段记录，并执行 `.venv/bin/python scripts/acp_pipeline.py check --manifests`。新增模块时同步维护 `src/acp/management/structure.py` 的中文用途说明。日常功能统一从 `scripts/acp_pipeline.py` 提供入口。

## CodeGraph

存在 `.codegraph/` 的仓库，理解或定位代码时优先使用 `codegraph explore`、`codegraph node` 或相应 MCP 工具，再用 rg 和文件读取。没有索引时跳过，不自动创建索引。目录重构后同步已有索引。
