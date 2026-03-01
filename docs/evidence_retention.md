# 证据保留机制（Evidence Retention）

本文档描述 Chat 和 Deep Research 两条路径下，从检索到最终 LLM 上下文的证据保留完整流程，包括参数含义、每个阶段的保留数量及关键设计决策。

---

## 通用语义：step_top_k 与 write_top_k

为保证 Chat 与 Deep Research 的参数语义一致，本项目统一约定：

| 参数 | 通用含义 | Chat | Deep Research |
|------|----------|------|----------------|
| **step_top_k** | **每轮检索的保留上限**：单次检索调用（local+web 合并后全局重排）输出条数不超过此值。 | 唯一一次主检索的保留上限 | 每一轮 research 检索、以及 evaluate 补充检索的单次输出上限 |
| **write_top_k** | **单个产出单元进 LLM 的证据上限**：「产出单元」= 一次完整回答或一个大纲章节。 | 一次问答 = 一单元；控制进入 LLM 上下文的证据数（EvidenceSynthesizer max_chunks） | 一个大纲章节 = 一单元；写作前从 section pool 重排后保留的证据数（gap 保护在此生效） |

- `step_top_k`：多轮场景下**每一轮**都按此阈值截断；单步场景（Chat）即主检索的阈值。None 时继承 `local_top_k`。
- `write_top_k`：Chat 与 DR 均使用。Chat 为一次问答的最终证据上限（None 时等于 step_top_k）；DR 为每个 section 写作时的证据上限（None 时由 preset 从 step_top_k 推导，如 `step_top_k × 1.5` 并受 cap 限制）。
- **DR 最终全文整合**（synthesize_node）：不再做证据截断，仅拼接各章节成文。

---

## 一、核心参数速查

| 参数 | 作用域 | 含义 |
|------|--------|------|
| `local_top_k` | Chat + DR | 传给 `RetrievalService.search(top_k=...)` 的检索预算；控制本地向量召回上限 |
| `step_top_k` | Chat + DR | **每次检索调用/每一轮**合并重排后的保留上限（`result_limit`）；多轮时每轮均适用；None 时继承 `local_top_k` |
| `write_top_k` | Chat + DR | **单个产出单元**进 LLM 的证据上限：Chat 一次问答；DR 每个大纲章节写作。None 时 Chat 等于 step_top_k；DR 由 `_compute_effective_write_k` 推导（≈ `step_top_k × 1.5`，受 preset cap 限制） |
| `actual_recall` | 内部 | `max(80, result_limit × 4)`；传给本地检索的召回放大系数，确保重排有足够候选 |
| `local_recall_k` | 内部 | `min(actual_recall, result_limit × 2)`；hybrid 模式下本地内部重排后的输出上限（为全局融合提供更多候选） |

---

## 二、Chat 路径

### 2.1 完整数据流

```
用户消息
  └─→ routes_chat.py：上下文分析 → 查询路由 → 查询构建
        └─→ RetrievalService.search(mode, top_k=local_top_k, filters={step_top_k, ...})
              ├─[local] HybridRetriever.retrieve(top_k=actual_recall, step_top_k=local_recall_k)
              │    ├─ Stage 1: Milvus dense+sparse 各召回 actual_recall 条
              │    ├─ Stage 2: 加权 RRF 融合（dense=0.6, sparse=0.4）
              │    ├─ Stage 3: BGE-M3 cross-encoder 重排 → local_recall_k 条
              │    └─ dedup_and_diversify（per_doc_cap=3）
              │
              ├─[web] unified_web_searcher.search_sync(...)   ← 并行执行
              │    ├─ Tavily / Scholar / Semantic Scholar / NCBI
              │    ├─ 智能查询优化（optimizer 展开多组查询）
              │    ├─ 内部 de-dup 后合并（~259 条原始结果）
              │    └─ 可选全文抓取（LLM 预判 ~11 条需全文）
              │
              ├─[超时控制] 软等待（soft-wait）
              │    ├─ 本地：硬超时 timeout_s（默认 60 s）
              │    └─ Web：软超时 min(timeout_s × 5, 300 s)（避免 Scholar ~145 s 被丢弃）
              │
              ├─ cross_source_dedup（过滤 web 中已在本地库的文献）
              │
              └─→ [ 无证据不足 ] 直接使用 main；[ 证据不足 ] 生成 gap query、补搜后：
                  fuse_pools_with_gap_protection(
                      main_candidates = local_hits + web_hits（或主检索 pack.chunks 转 hit）,
                      gap_candidates  = 补搜得到的 hit 列表（可为 []）,
                      top_k           = result_limit ← step_top_k 或 local_top_k,
                      gap_min_keep    = ceil(step_k × chat_gap_ratio),
                      gap_ratio       = chat_gap_ratio,
                      rank_pool_multiplier = chat_rank_pool_multiplier
                  )
                    ├─ 全局单次重排（main + gap 统一评分）
                    ├─ 初始取 top_k 切片（不做二次分数加权）
                    ├─ gap quota：至少保留 gap_min_keep 条 gap（不足时先从 ranked tail，再从未入榜 gap 回填）
                    └─ 返回 top result_limit 条 EvidenceChunk
  └─→ EvidenceSynthesizer(max_chunks=write_k) → context_str → LLM
       write_k = write_top_k 或 step_top_k 或 len(pack.chunks)；仅前 write_k 条进上下文，pack.chunks 全量参与引文解析
```

### 2.2 各阶段数量示例

以日志示例（`local_top_k=45, step_top_k=50, providers=tavily+scholar+semantic+ncbi`）为例：

| 阶段 | 数量 |
|------|------|
| `actual_recall` | `max(80, 50×4) = 200` |
| `local_recall_k` | `min(200, 50×2) = 100` |
| 本地 BGE 重排输出 | ≤ 100 条 |
| Web 原始结果 | ~314 条（4 provider × 多查询） |
| Web 内部去重后 | ~259 条 |
| cross_source_dedup 过滤 | 去掉已在本地库的文献 |
| 全局 fuse 输入 | 本地 ≤100 + web 剩余 |
| 检索输出（`result_limit`） | `step_top_k = 50` 条 |
| **进 LLM 上下文（`write_k`）** | **`write_top_k` 或 step_top_k，如 50 条** |

### 2.3 Chat 的 gap 补搜与融合（证据不足时）

当**证据不足**（evidence_scarce）且需要 RAG 时，Chat 会：

1. 用 LLM 生成至多 3 条 **gap query**（`_generate_chat_gap_queries`，prompt：`chat_gap_queries.txt`）。
2. 对每条 gap query 做一次补搜（`retrieval.search`，top_k = max(5, step_k//2)），结果放入 `gap_candidates`。
3. 若 `gap_candidates` 非空，将主检索的 `pack.chunks` 转为 main_candidates，与 gap_candidates 一起调用 **fuse_pools_with_gap_protection**，并显式传入 Chat 比例与放大倍率（默认 `chat_gap_ratio=0.2`、`chat_rank_pool_multiplier=3.0`）。
4. 融合后的 `new_chunks` 替换原 `pack.chunks`，再走 EvidenceSynthesizer 与 LLM。

- **gap 补搜失败**：某条 gap query 检索异常时打 **WARNING**（`[chat] gap supplement search failed for ...`），其余照常融合。
- **无 gap 或未触发证据不足**：不生成 gap query，主检索结果不经过 gap 池，fuse 时 gap_candidates=[]，无配额约束。

### 2.4 重排器模式（Chat）

Chat 路径固定使用 `bge_only`（速度优先；cascade/colbert 为 DR 写作阶段保留）。如用户未通过 `filters.reranker_mode` 传入，则自动设为 `bge_only`。

### 2.5 Chat 是否有多步骤？主检索 vs Agent 二次召回

**结论：Chat 的“步骤”只有一步主检索；Agent 工具会带来按需的二次（或多次）召回，但和 DR 的“多步骤”不是同一概念。**

| 对比项 | Chat | Deep Research |
|--------|------|----------------|
| **步骤含义** | 无“多步骤”流程：**仅一次**主检索（`do_retrieval` 时调用 `retrieval.search()`），得到 `step_top_k` 条证据后直接进 LLM 上下文 | 有固定多步骤：research 轮次 → evaluate（含 gap 补充）→ generate_claims → write，每步都可能涉及检索或 pool 重排 |
| **主检索** | 一次 `search(mode=hybrid, top_k=local_top_k)`，内部 local+web 并行 → fuse_pools → 截断到 `step_top_k` | 每轮 research 调用一次 `search()`，结果写入 section pool；evaluate 时可能再按 gap 补搜；write 时从 pool 重排取 `write_top_k` |
| **二次/多次召回** | **Agent 工具**：当 `agent_mode=assist` 时，LLM 可主动调用 `search_local`、`search_web`、`search_scholar`、`search_ncbi` 等。每次调用都会再执行一次 `svc.search(mode="local" 或 "web", top_k=10)`，属于**按需、模型决定**的额外检索 | 无“Agent 工具召回”；所有检索都是流程预设的（研究轮、评估补充、写作 fallback） |
| **证据合并** | 主检索得到 `pack.chunks`（如 50 条）。Agent 工具返回的 chunk 通过 `_collect_chunks` 收集，react 结束后 `drain_agent_chunks()` 得到 `agent_extra_chunks`，**按 chunk_id 去重追加**到 `pack.chunks`，故最终引文池 = 主检索 + Agent 追加（可能 50+ 条） | 所有证据先进入 section pool，写作前统一 `_rerank_section_pool_chunks(write_top_k)`，再经 fuse_pools（含 gap 保护）得到最终写作用证据 |

因此：

- **Chat 里没有“step”的多个步骤**：从检索服务视角只有**一步**主检索，`step_top_k` 表示的就是这一步的最终保留数。
- **Agent 里确实有二次（或 N 次）召回**：工具 `search_local` / `search_web` 等会再次调用 `RetrievalService.search()`，每次用工具传入的 `query` 和 `top_k`（默认 10，返回给模型时最多 15 条），结果与主检索证据**合并、去重**后一起参与引文解析。这类召回**不经过** fuse_pools（因为每次是单 mode 的 local 或 web），也不受 `step_top_k` 约束，仅受工具参数和 `_collect_chunks` 上限影响。

---

## 三、Deep Research 路径

Deep Research 分四个节点，每个节点都会操作 **per-section evidence pool**（按章节隔离的累积证据池）。

### 3.1 Section Evidence Pool

```python
state["section_evidence_pool"][section_title] = List[EvidenceChunk]
```

- 每次检索结果以 `pool_source` 标签追加到对应章节的池中（不去旧保新）。
- `pool_source` 决定写作时的 gap 保护级别：

| `pool_source` | 写入时机 | 类型 |
|---------------|----------|------|
| `"research_round"` | 研究轮次（research_node） | 主池（main） |
| `"eval_supplement"` | 评估节点发现 gap 后补充搜索 | **gap 池（受保护）** |
| `"write_stage"` | write_node 额外检索（pool 为空时） | 主池（main） |

### 3.2 完整数据流

```
Deep Research 任务
  │
  ├─[研究节点 research_node] × N 轮
  │    └─ RetrievalService.search(mode, top_k=step_top_k, ...)
  │         └─ 同 Chat hybrid 流程（全局融合后 step_top_k 条）
  │    └─ _accumulate_section_pool(pool_source="research_round")
  │
  ├─[评估节点 evaluate_node]
  │    ├─ 从 section pool 取出所有块
  │    ├─ _rerank_section_pool_chunks(top_k=eval_top_k≈20) → LLM 评估覆盖度
  │    └─ 若覆盖度不足 AND 存在 gaps：
  │         └─ 按 gap 逐条搜索（每 gap top_k=min(eval_top_k,10)）
  │              └─ _accumulate_section_pool(pool_source="eval_supplement")  ← gap 保护标签
  │
  ├─[生成主张节点 generate_claims_node]
  │    └─ _rerank_section_pool_chunks(top_k=write_top_k, ...)
  │         └─ 同 write_node 证据选择（见下）
  │
  └─[写作节点 write_node]
       ├─ write_top_k = _compute_effective_write_k(preset, filters)
       │     = max(preset_write_k, ui_write_top_k 或 step_top_k×1.5)，受 cap 限制
       │
       └─ _rerank_section_pool_chunks(pool_chunks, top_k=write_top_k)
            ├─ 按 pool_source 分池：
            │    ├─ main_candidates：pool_source ∉ {"eval_supplement"}
            │    └─ gap_candidates： pool_source == "eval_supplement"
            │
            └─→ fuse_pools_with_gap_protection(
                    main_candidates, gap_candidates, top_k=write_top_k,
                    gap_ratio=research_gap_ratio,
                    rank_pool_multiplier=research_rank_pool_multiplier
                )
                  ├─ 全局单次重排（main + gap 统一评分）
                  ├─ 初始取 top_k 切片（不做二次分数加权）
                  ├─ gap quota：至少保留 ceil(write_top_k × research_gap_ratio) 条 gap 块
                  ├─ 若不足：先从 ranked tail 回填，再从未入榜 gap pool 回填
                  └─ 返回 write_top_k 条，传入 LLM 写作
```

### 3.3 write_top_k 计算逻辑

```python
def _compute_effective_write_k(preset, filters):
    preset_write_k = preset.get("search_top_k_write", 12)   # 深度预设基线
    write_k_cap    = preset.get("search_top_k_write_max", 60)  # 上限

    if filters.get("write_top_k") and int(filters["write_top_k"]) > 0:
        return min(max(preset_write_k, int(filters["write_top_k"])), write_k_cap)

    step_top_k = int(filters.get("step_top_k") or 0)
    if step_top_k > 0:
        return min(max(preset_write_k, int(step_top_k * 1.5)), write_k_cap)

    return preset_write_k  # 使用 preset 默认值
```

| `depth` | `preset_write_k` | `write_k_cap` |
|---------|-----------------|---------------|
| `lite` | 8 | 30 |
| `comprehensive` | 12 | 60 |

### 3.4 各阶段数量示例

以 `step_top_k=50, write_top_k=None, depth=comprehensive` 为例：

| 阶段 | 数量 |
|------|------|
| 研究轮次每轮检索 | step_top_k = 50 条 |
| section pool 累计（N 轮后） | 50 × N 条（含重复） |
| evaluate 重排取证 | eval_top_k ≈ 20 条 |
| gap 补充（每 gap） | min(20, 10) = 10 条 |
| eval_supplement 入池 | ≤ gaps数 × 10 条 |
| write_top_k 推导 | max(12, 50×1.5)=75 → cap=60 → **60 条** |
| **写作最终证据** | **60 条（gap 中至少 15 条受保护）** |

---

## 四、fuse_pools_with_gap_protection 算法

两条路径的最终截断都经过此函数（`src/retrieval/service.py`）。

### 4.1 算法步骤

```
输入：main_candidates (N_m 条) + gap_candidates (N_g 条) + top_k
      可选：gap_min_keep（显式指定时覆盖比例推算）

Step 1  标记：每个候选打 _pool_tag = "main" | "gap"
Step 2  全局重排：_rerank_candidates(all_cands, top_k = rank_pool_k)
        rank_pool_k = min(max(ceil(top_k×multiplier), top_k + N_g), total)
        → 单次 cross-encoder 评分，消除跨源分布差异
Step 3  初始切片：取 reranked[:top_k]
Step 4  quota 保护：
        gap_min_keep = 调用方传入 或 默认 ceil(top_k × gap_ratio)
        Chat 默认 ratio=0.2，Research 默认 ratio=0.25（均可配置）
        若 top_k slice 中 gap 数 < gap_min_keep：
          先从 reranked 的 tail 取 gap 补齐；仍不足再从未入榜 gap pool 补齐
          强制替换 slice 中最低位 main，保持总数 = top_k
        effective_min_keep = min(理论配额, n_gap, top_k)，即 gap 池不足时用实际数量
Step 5  剥除 _pool_tag，返回 top_k 条

诊断输出（diag["pool_fusion"]）：
  main_in, gap_in, total_reranked, rank_pool_k, rank_pool_multiplier,
  gap_deficit_before_fill, gap_backfill_ranked, gap_backfill_unranked,
  gap_min_keep, gap_in_output, output_count
```

### 4.2 gap_min_keep 来源

| 路径 | gap_min_keep |
|------|----------------|
| **Chat** | `ceil(step_k × chat_gap_ratio)`（默认 0.2，可配置） |
| **Deep Research** | `ceil(top_k × research_gap_ratio)`（默认 0.25，可配置） |

### 4.3 两种 WARNING（不足配额时的日志）

- **情况 A（gap 池数量不足理论配额）**：当按比例算出的 `desired_min_keep > n_gap` 时，effective 配额降为 `n_gap`，并打 WARNING：`[fuse_pools] gap pool too small: desired_quota=... but only ... gap candidates available — using ... as effective quota`。属正常：允许不补到理论配额，但需留痕。
- **情况 B（回填后仍不足）**：执行完 Step 4 后，若输出中 gap 条数仍小于 effective_min_keep（极少见），打 WARNING：`[fuse_pools] gap quota not met after backfill: wanted=... in_output=...`。用于异常监控。

---

## 五、超时与可靠性

### 软等待（hybrid 模式）

```
t=0     本地检索 ────────────────── t≈8s 完成（无内部重排时）
                                    t≈15s 完成（BGE 重排 50 候选）
t=0     Web 搜索  ────────────────────────────────── t≈145s 完成（Scholar browser）
         ↑                                           ↑
     并行启动                              软等待覆盖（≤300s）

硬超时（60s）：本地检索超出此时间则放弃，记录 diag["local_timeout"]
软超时（300s）：Web 超出此时间则放弃，记录 diag["web_timeout"]
正常完成后：若 web_elapsed > timeout_s，记录 diag["soft_wait_ms"] = 实际等待毫秒数
```

### 日志解读

检索完成日志（`[chat] ⑤ 检索完成`）关键字段：

```
mode=hybrid | top_k=45 | step_top_k=50 | reranker_mode=bge_only
| chunks=50                    ← 最终证据数（= result_limit = step_top_k）
| sources=dense,web            ← 两源均贡献
| fusion(main=120,gap=0,out=50) ← pool fusion 统计
| soft_wait_ms=85000           ← web 比 timeout_s 多等了 85s（正常）
| 耗时=155000ms
```

---

## 六、相关代码位置

| 功能 | 文件 | 函数/常量 |
|------|------|-----------|
| 全局融合 + gap 保护 | `src/retrieval/service.py` | `fuse_pools_with_gap_protection()`（一次全局重排 + 确定性配额回填） |
| Chat 主检索（一步） | `src/retrieval/service.py` | `RetrievalService.search()` hybrid 分支 |
| **Chat gap 补搜与融合** | `src/api/routes_chat.py` | §5.6：证据不足时 `_generate_chat_gap_queries` → 补搜 → `fuse_pools_with_gap_protection(..., gap_ratio=chat_gap_ratio, rank_pool_multiplier=chat_rank_pool_multiplier)`；补搜失败打 WARNING |
| **Chat Agent 二次召回** | `src/llm/tools.py` | `_handle_search_local()`, `_handle_search_web()` → `svc.search(mode="local"\|"web", top_k=10)`；结果经 `_collect_chunks` → `drain_agent_chunks()` 合并到 `pack.chunks` |
| Agent 证据合并到主 pack | `src/api/routes_chat.py` | ⑧a：`agent_extra_chunks` 按 chunk_id 去重后 `pack.chunks.append(...)` |
| 本地向量 + 图 + 重排 | `src/retrieval/hybrid_retriever.py` | `HybridRetriever.retrieve()`, `_rerank_candidates()` |
| web 统一搜索 | `src/retrieval/unified_web_search.py` | `unified_web_searcher.search_sync()` |
| 跨源去重 | `src/retrieval/dedup.py` | `cross_source_dedup()` |
| DR section pool 累积 | `src/collaboration/research/agent.py` | `_accumulate_section_pool()` |
| DR pool 重排（gap 保护） | `src/collaboration/research/agent.py` | `_rerank_section_pool_chunks()`, `_DR_GAP_POOL_SOURCES` |
| DR write_top_k 计算 | `src/collaboration/research/agent.py` | `_compute_effective_write_k()` |
| gap pool 标识常量 | `src/collaboration/research/agent.py` | `_DR_GAP_POOL_SOURCES = frozenset({"eval_supplement"})` |

---

## 七、设计约束（勿随意修改）

1. **截断只发生一次**：`fuse_pools_with_gap_protection` 返回后，唯一的 `all_chunks[:result_limit]` 在 `search()` 末尾作为安全兜底（对 hybrid 是幂等的）。不得在 hybrid 分支内部提前截断。

2. **local_recall_k ≠ result_limit**：hybrid 模式中本地检索必须返回 `result_limit × 2` 条候选供全局重排使用，而不是等于 `result_limit`，否则全局重排的候选池会过小。

3. **eval_supplement 必须进 gap pool**：`_DR_GAP_POOL_SOURCES` 必须包含 `"eval_supplement"`，否则 gap 评估阶段补充的证据在写作时会与普通证据平等竞争，可能全部被挤出 top-k。

4. **soft-wait 不能改回 with-block**：`ex.shutdown(wait=False)` 之后继续 `fw.result(timeout=web_wait_s)` 是软等待的实现方式；若改回 `with ThreadPoolExecutor ... as ex:` 会强制等待所有线程完成，失去超时控制。
