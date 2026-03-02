# Deep Research 搜索流程与优化逻辑调研

本文档对 Research 的搜索流程、各轮检索方式做全方位梳理。**SmartQueryOptimizer / use_query_optimizer 已废弃**（见下文「废弃说明」）；Chat 与 Research 统一采用 **1+1+1 结构化查询**（recall + precision + discovery）。

---

## 零、废弃说明（SmartQueryOptimizer）

- **use_query_optimizer** 与 **query_optimizer_max_queries** 已从全代码库移除；不再有「查询优化器」开关。
- **SmartQueryOptimizer**、**query_optimizer** 及相关 optimizer prompts 已移入 **backup** 并标注 deprecated：
  - `src/retrieval/backup/smart_query_optimizer.py`、`query_optimizer.py`
  - `src/prompts/backup/` 下 `optimizer_*.txt`、`web_search_optimize*.txt`
- **Tavily LLM 查询扩展**（expand）始终为 False；`web_search.py` 中 `_generate_queries_sync` / `_generate_queries_async` 已标注 deprecated。
- **Chat** 与 **Research plan** 均改用 **1+1+1**：由 LLM 生成 1 recall + 1 precision + 1 discovery，经 `web_queries_per_provider` 单次检索并融合。

---

## 一、整体流程概览

```
Scoping → Plan(背景检索+生成大纲) → [Research → Evaluate]×N轮(每章节) → Write → Verify → ...
```

- **Plan**：一次「初步背景调查」检索，用于生成大纲。
- **Research**：按**章节**轮流、每章节最多 **max_section_research_rounds**（默认 3）轮；每轮 = 生成 query 包 → 分层检索(tiered search) → 收尾。
- **Evaluate**：评估证据充分度、产出 gaps；若有 gap 会触发 **evaluate_supplement** 补充检索。
- **Write/Verify**：优先用 section pool 做 rerank，不足时才落底检索。

---

## 二、「第一轮全面搜索」实际是什么？

**结论：没有单独的「先搜 4 个优化关键词」的一步；有两块不同的「首轮」逻辑。**

### 2.1 Plan 阶段的「背景检索」（1+1+1 结构化）

- **位置**：`plan_node`（`agent.py`）。
- **输入**：`topic`（用户题目）+ `preliminary_knowledge`（Sonar/clarify 阶段已有）。
- **流程**：与 Chat Round 2 一致：
  1. 调用 **generate_structured_queries_1plus1plus1**(topic, preliminary_knowledge, client) 得到 1 recall + 1 precision + 1 discovery；
  2. 构建 **web_queries_per_provider**：ncbi/semantic/scholar/google → [recall, precision]，tavily → [discovery]；
  3. `svc.search(query=recall_q, mode=hybrid, top_k=plan_top_k, filters={..., web_queries_per_provider})`。
- **用途**：给 outline 生成提供「初步知识」上下文。与 Chat 统一为 1+1+1，不再使用已废弃的 SmartQueryOptimizer。

### 2.2 Research 每轮的「第一轮」（Round 1）— 才是 recall + precision + discovery

- **位置**：`research_node` → `_generate_section_queries` → `_execute_tiered_search`。
- **每章节**的 **Round 1**（`section.research_rounds == 1`）时：
  - `gap_only_mode = False`，会生成：
    - **Recall**：`recall_queries_per_section` 条（Lite=2，Comprehensive=4），prompt 要求「宽网、同义词、多表述」；
    - **Precision**：`precision_queries_per_section` 条（Lite=2，Comprehensive=4），「方法/时间/对象约束」；
    - **Discovery**：1 条（自然语言问题，给 Tavily 等）；
    - 若已有 **gaps**（例如从上一轮 evaluate 带过来），还会加 **gap 相关** query（引擎感知或 fallback 关键词）。
  - 因此 **Lite 下是 2+2+1 + gaps，Comprehensive 下是 4+4+1 + gaps**，即「第一轮」是 **recall + precision + discovery 一起**，并不是「先单独做 4 个优化关键词再分解」。
- **Round 2+**：`gap_only_mode = True`，只生成 **gap 相关** query（+ 可能保留的 recall/precision），不再做「全面」的 recall/precision 铺网。

所以：**你说的「第一轮全面搜索」在实现里 = 每个章节的 Research Round 1 用「recall + precision + discovery（+ gap）」多类 query 做分层检索**；「4 个」在 Lite 下是 2 recall + 2 precision，Comprehensive 下是 4+4。

---

## 三、分解问题 3 轮：每轮怎么搜？

- **3 轮**由配置 **max_section_research_rounds**（默认 3）决定：每个章节最多 3 轮「research → evaluate」循环。
- **每轮**流程：
  1. **生成 query 包**：`_generate_section_queries(state, section, max_queries=...)`  
     - Round 1：recall + precision + discovery + gap（若有）；  
     - Round 2+：仅 gap 相关（引擎感知 + fallback）。
  2. **分层检索**：`_execute_tiered_search(..., max_tier=...)`，用上面 query 包按 tier 执行：
     - **Tier 1**（仅 biomedical）：NCBI，keyword 类 query（gap/recall/precision）。
     - **Tier 2**：Semantic Scholar（keyword）+ Tavily（discovery/NL）。
     - **Tier 3**：Scholar + Google（含 gap_scholar/gap_google + 本轮的 refined 查询）。
  3. **Tier 上限**由 **round** 和 **depth preset** 决定：
     - **Lite**：Round 1 用 `round1_max_tier=2`（只到 T2）；中间轮 `gapfill_max_tier=2`；最后一轮 `last_round_max_tier=3`（可到 T3）。
     - **Comprehensive**：每轮都可到 T3（round1/gapfill/last_round 均为 3）。
  4. 每 tier 结束后若 `section.gaps` 非空，会做 **Micro-CoT 覆盖检查**（`_quick_coverage_check`），通过则提前结束，不再跑后面的 tier。

| 轮次 | Query 来源 | Tier 上限 (Lite) | Tier 上限 (Comprehensive) |
|------|------------|------------------|----------------------------|
| Round 1 | recall + precision + discovery + gap | 2 | 3 |
| Round 2..N-1 | 主要 gap | 2 | 3 |
| Round N（最后一轮） | 主要 gap | 3 | 3 |

---

## 四、Chat 三轮检索（与 Research 一致思路）

- **Round 1 — 背景**：有 Perplexity 且 sonar_strength ≠ off 时调用 Sonar API，得到 `preliminary_knowledge_block` + `sonar_chunks`；否则调用**本地 LLM**（`chat_local_cognition.txt`）做无搜索的认知生成，仅得到 `preliminary_knowledge_block`。
- **Round 2 — 1+1+1**：仅当 `do_retrieval` 且 `effective_search_mode in ("hybrid", "web")`。用 `chat_generate_queries.txt` + Round 1 背景生成 1 recall + 1 precision + 1 discovery，构建 `web_queries_per_provider`，**单次** `retrieval.search`；Round 1（sonar_chunks）与 Round 2 结果经 **fuse_pools_with_gap_protection** 合并。
- **Round 3 — Gap**：证据不足时 `_generate_chat_gap_queries` 生成最多 **3** 条 gap 查询，每条一次 `retrieval.search`，结果再 fuse。

Research 内**不再使用** use_query_optimizer；Plan 背景检索已改为与 Chat 一致的 1+1+1（见 2.1）。

---

## 五、重叠与潜在冲突（标出）

### 5.1 统一 1+1+1（Chat 与 Research Plan）

- **Chat Round 2** 与 **Research plan_node** 均使用 **generate_structured_queries_1plus1plus1**（`src/retrieval/structured_queries.py`）+ **chat_generate_queries.txt**，产出 1 recall + 1 precision + 1 discovery，再经 `web_queries_per_provider` 单次检索。思路一致，无重叠冲突。
- **Research 每轮**（research_node）仍用 `_generate_section_queries`（recall/precision/discovery/gap）与 tiered search，与 Plan 的 1+1+1 阶段分离。

### 5.3 Write/Verify 的规则 query vs Research 的 LLM query

- **Write/Verify** 落底检索用 `_build_write_queries(topic, section_title[, extra_kw_suffix])`：**规则抽取**（英文关键词 + 固定 NL 问句），无 LLM、无优化器。
- **Research** 用 LLM 的 recall/precision/discovery/gap。

潜在冲突：若用户期望「写作阶段也用优化器或与 research 一致的 LLM 生成 query」，当前实现不一致（write 固定规则、且关闭优化器）。可考虑：write 落底时是否允许使用优化器或复用 research 的 query 风格。

### 5.4 Gap 补充：evaluate_supplement vs 下一轮 Research

- **evaluate_supplement**（evaluate_node 内）：用 `topic + section + gap` 拼成一条 query，直接 `svc.search`，无优化器、无 LLM 生成；结果标为 `pool_source="eval_supplement"`（走 gap 池融合）。
- **下一轮 Research**：gap 由 `_generate_engine_gap_queries` 生成**引擎定制** query（LLM），或 fallback 关键词。

重叠：都是「针对 gap 再搜一次」。区别是 evaluate_supplement 是**即时、简单拼接**补一点；下一轮 research 是**整轮、引擎感知**的 gap 查询。逻辑略重复，但层次不同，可保留；若希望统一，可考虑只保留一种（例如只做 evaluate_supplement 或只做下一轮 gap 轮）。

### 5.5 Sonar 强度与 UI

- **Sonar 前置**已从「开关」改为**强度选择器**（off / sonar / sonar-pro / sonar-reasoning-pro），默认 sonar-reasoning-pro；Chat Round 1 据此决定是否调用 Sonar 或本地 LLM 认知。

---

## 六、关键代码位置速查

| 逻辑 | 文件 | 位置/函数 |
|------|------|-----------|
| Plan 背景检索（1+1+1） | agent.py | plan_node：generate_structured_queries_1plus1plus1 + web_queries_per_provider + svc.search |
| 共享 1+1+1 生成 | src/retrieval/structured_queries.py | generate_structured_queries_1plus1plus1，web_queries_per_provider_from_1plus1plus1 |
| Chat Round 1（Sonar/本地认知） | routes_chat.py | step 4.9 Sonar；fallback 本地 LLM chat_local_cognition.txt |
| Chat Round 2（1+1+1） | routes_chat.py | _generate_chat_structured_queries，_chat_web_queries_from_1plus1plus1，filters.web_queries_per_provider |
| Chat Round 3（gap） | routes_chat.py | _generate_chat_gap_queries，[:3]，fuse_pools_with_gap_protection |
| Research 每轮 query 生成 | agent.py | _generate_section_queries（recall/precision/discovery/gap） |
| Research 每轮 tiered 检索 | agent.py | _execute_tiered_search |
| Evaluate 补充检索 | agent.py | evaluate_node 内 should_supplement，pool_source=eval_supplement |
| Write/Verify 落底 query | agent.py | _build_write_queries，web_queries_per_provider |
| 已废弃（备份） | src/retrieval/backup/，src/prompts/backup/ | smart_query_optimizer，query_optimizer，optimizer_*.txt |

---

## 七、小结表

| 问题 | 结论 |
|------|------|
| 查询优化器（SmartQueryOptimizer）还用吗？ | **已废弃**。全库移除 use_query_optimizer；相关代码与 prompts 移入 backup 并标注 deprecated。 |
| Chat 检索流程？ | **三轮**：Round 1 = Sonar 或本地 LLM 认知；Round 2 = 1+1+1 单次 search + fuse；Round 3 = gap 最多 3 条，通用 fuse。 |
| Plan 背景检索怎么做？ | **1+1+1**：topic + preliminary_knowledge → generate_structured_queries_1plus1plus1 → web_queries_per_provider → 单次 search。与 Chat Round 2 一致。 |
| Research 每轮怎么搜？ | 每轮：_generate_section_queries → _execute_tiered_search(T1→T2→T3)；Round 1 多类 query，Round 2+ 以 gap 为主；tier 上限由 preset 与是否首/末轮决定。 |
| 重叠/冲突 | 见第五节：Chat 与 Plan 已统一 1+1+1；Write 规则 query vs Research LLM；evaluate_supplement vs 下一轮 gap。 |
