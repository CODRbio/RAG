# 备份的 Prompt（当前未使用）

此目录存放已从主流程中移除、但保留备份的 prompt 文件。详见根目录 `docs/prompts_usage.md`。

## DEPRECATED: SmartQueryOptimizer + Tavily LLM 扩展（Chat/Research 1+1+1 统一后移除）

| 文件 | 原用途 | 说明 |
|------|--------|------|
| `optimizer_auto_route.txt` | `SmartQueryOptimizer.optimize(..., auto_route=True)` | 当前统一用 `get_routing_plan()` 做代价感知路由，无人调用 `optimize(auto_route=True)`，故备份。 |
| `optimizer_normal.txt` | SmartQueryOptimizer 多引擎查询生成 | 已废弃；改用 1+1+1 结构化查询（chat_generate_queries / generate_queries）。 |
| `optimizer_system.txt` | 同上 | 同上。 |
| `optimizer_routing_plan.txt` | get_routing_plan() 代价感知路由 | 已废弃。 |
| `optimizer_routing_plan_system.txt` | 同上 | 同上。 |
| `web_search_optimize.txt` | Tavily LLM 查询扩展 | 已废弃；expand 始终为 False。 |
| `web_search_optimize_system.txt` | 同上 | 同上。 |

对应代码备份：`src/retrieval/backup/smart_query_optimizer.py`、`query_optimizer.py`。
