# Prompt 使用说明（搜索词优化 / Chat / Deep Research / Validation）

## 1. 搜索词优化用到的提示词

检索统一走 `unified_web_search`，是否用「智能优化」由 `use_query_optimizer` 与 `smart_optimizer.enabled` 决定。

### 1.1 Smart Query Optimizer（主路径）

| 场景 | 条件 | 使用的 Prompt |
|------|------|----------------|
| **Auto 路由**（前端选「自动」或 `providers=None`） | `is_auto_route and use_smart` | `optimizer_routing_plan.txt` + `optimizer_routing_plan_system.txt` → `get_routing_plan()` 生成 primary/fallback 与每引擎查询 |
| **手动选引擎**（前端指定具体引擎） | `use_smart and not is_auto_route` | `optimizer_normal.txt` + `optimizer_system.txt` → `optimize(auto_route=False)` 为每个已选引擎生成查询 |

代码位置：`src/retrieval/smart_query_optimizer.py`（`get_routing_plan` / `optimize`）、`src/retrieval/unified_web_search.py`（`is_auto_route` 分支）。

### 1.2 Tavily 单查询扩展（备用路径）

当 **仅用 Tavily** 且 **未走 Smart 多查询**（单查询 + 开启扩展）时：

- `web_search_optimize.txt` + `web_search_optimize_system.txt`

代码位置：`src/retrieval/web_search.py`（`_generate_queries_sync`，受 `enable_query_expansion` 控制）。

---

## 2. 按场景汇总

### Chat

- **检索 / 搜索词优化**：同上 1.1（auto → routing_plan；手动引擎 → optimizer_normal + optimizer_system）。若走 Tavily 单查询扩展则用 1.2。
- **路由与回复**：`chat_route_system.txt`、`chat_route_classify.txt`、`chat_rag_system.txt`、`chat_direct_system.txt`、`chat_agent_hint.txt`、`chat_agent_autonomous_hint.txt`、`chat_agent_evidence_scarce_hint.txt`。
- **发起 Deep Research 澄清**：`chat_deep_research_clarify.txt`、`chat_deep_research_system.txt`。

### Deep Research Round 1（首轮调研）

- **检索 / 搜索词优化**：与 Chat 相同（`use_query_optimizer=True`）→ auto 用 routing_plan，手动用 optimizer_normal + optimizer_system。
- **规划与提纲**：`scope_research.txt`、`plan_outline.txt`。
- **首轮查询生成**：`generate_queries.txt`（Round 1 宽泛查询）；bilingual hints：`bilingual_hint_academic.txt`、`bilingual_hint_discovery.txt`、`bilingual_hint_gap_queries.txt`。
- **查询修复**：`repair_queries_scholar_google.txt`、`repair_queries_generic.txt`。

### Deep Research Round 2+（缺口填补轮）

- **检索**：`use_query_optimizer=False`，不再走 Smart Optimizer；查询由下面 LLM 生成。
- **缺口/精炼查询**：`generate_refined_queries.txt`、`generate_gap_queries.txt`。
- **写作与润色**：`generate_claims.txt`、`write_section.txt`、`translate_content.txt`、`generate_abstract.txt`、`limitations_section.txt`、`open_gaps_agenda.txt`、`coherence_refine.txt`、`coherence_refine_window.txt`。
- **轨迹压缩**：`trajectory_compress.txt`、`trajectory_compress_system.txt`。

### Validation（充分性 / 覆盖检查）

- **快速覆盖检查**（Micro-CoT）：`quick_coverage_check.txt`。
- **充分性评估**：`evaluate_sufficiency.txt`。

---

## 3. 已清理 / 备份的 Prompt

以下文件已移至 `src/prompts/backup/`，当前代码不再引用：

| 文件 | 说明 |
|------|------|
| `optimizer_auto_route.txt` | 原用于 `optimize(..., auto_route=True)`。当前统一使用「auto 用 get_routing_plan，手动用 optimize(auto_route=False)」，没有任何调用传 `auto_route=True`，故已备份；代码中 `auto_route=True` 分支改为复用 `optimizer_normal.txt`。 |

如需恢复「代价感知路由」的旧版单步 LLM 行为，可从 `backup/` 取回并恢复 `smart_query_optimizer` 中对应分支。
