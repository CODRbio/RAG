# 备份的 Prompt（当前未使用）

此目录存放已从主流程中移除、但保留备份的 prompt 文件。详见根目录 `docs/prompts_usage.md`。

| 文件 | 原用途 | 说明 |
|------|--------|------|
| `optimizer_auto_route.txt` | `SmartQueryOptimizer.optimize(..., auto_route=True)` | 当前统一用 `get_routing_plan()` 做代价感知路由，无人调用 `optimize(auto_route=True)`，故备份。 |
