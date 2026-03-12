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
- `write_top_k`：Chat 与 DR 均使用。Chat 为一次问答的最终证据上限（None 时等于 step_top_k）；DR 为每个 section 写作时的证据上限（None 时由 preset 从 step_top_k 推导，如 `step_top_k × 1.5`；已受 `search_top_k_write_max` 截断）。
- **DR 最终全文整合**（synthesize_node）：不再做证据截断，仅拼接各章节成文。

---

## 一、核心参数速查

| 参数 | 作用域 | 含义 |
|------|--------|------|
| `local_top_k` | Chat + DR | 传给 `RetrievalService.search(top_k=...)` 的检索预算；控制本地向量召回上限 |
| `step_top_k` | Chat + DR | **每次检索调用/每一轮**合并重排后的保留上限（`result_limit`）；多轮时每轮均适用；None 时继承 `local_top_k` |
| `write_top_k` | Chat + DR | **单个产出单元**进 LLM 的证据上限：Chat 一次问答；DR 每个大纲章节写作。None 时 Chat 等于 step_top_k；DR 由 `_compute_effective_write_k` 推导（≈ `step_top_k × 1.5`；受 `search_top_k_write_max` 截断） |
| `actual_recall` | 内部 | 当前实现与 contract 一致：**等于 `result_limit`**；hybrid 本地分支用此值参与配置（与 `local_recall_k` 区分见下） |
| `local_recall_k` | 内部 | `min(actual_recall, max(result_limit × 2, 20))`；hybrid 模式下传给本地 `retriever.retrieve(top_k=local_recall_k)` 的输出上限，为全局融合提供候选；当 `result_limit ≥ 10` 时等于 `result_limit` |

当前前端默认建议：

- Chat / Hybrid 默认 `local_top_k = 45`
- 保持 `step_top_k = 10`、`write_top_k = 15`
- 如果出现“本地库明明有内容，但最终混合结果里本地占比很低”，优先先提高 `local_top_k`，而不是先改 hybrid 融合逻辑

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
              │    ├─ Stage 3: 产出源内有序 local candidate pool（不做 Chat 最终裁决）
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
              └─→ [ 无证据不足 ] 使用 main pool 融合结果；[ 证据不足 ] 生成 gap query、补搜后再做跨池融合
  │
  ├─→ [Agent 可选] agent_mode=assist/autonomous 时，LLM 可调用工具补搜
  │    └─ search_local / search_web 以 pool_only 返回源内 candidate pool → agent_extra_chunks = drain_agent_chunks()
  │
  │
  │   ★ §5¾ 单次统一 BGE rerank（系统提示组装前，无论是否触发 gap）：
  │     _fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=chat_gap_candidates_hits, agent=[])
  │     → pack.chunks 更新为 BGE rerank 后的 write_top_k 条（含 gap 配额保护）
  │
  └─→ [Agent 追加融合，仅当 agent_extra_chunks 非空]
      _fuse_chat_main_gap_agent_candidates(
          main_candidates  = pack.chunks（已含 gap，无需再传 gap 候选）,
          gap_candidates   = []（禁止再传 gap，否则双重计数）,
          agent_candidates = agent_extra_chunks（可为 []）,
          top_k            = write_top_k 或 step_top_k,
          gap_ratio        = chat_gap_ratio（默认 0.2）,
          agent_ratio      = chat_agent_ratio（默认 0.1）,
          rank_pool_multiplier = chat_rank_pool_multiplier（默认 3.0）
      )
        ├─ 全局单次重排（main + gap + agent 统一评分）
        ├─ 初始取 top_k 切片（不做二次分数加权）
        ├─ gap quota：至少保留 ceil(top_k × 0.2) 条 gap
        ├─ agent quota：至少保留 ceil(top_k × 0.1) 条 agent
        └─ 返回最终 EvidenceChunk（Chat 不再对 fused pool 追加 score threshold 过滤）
  └─→ EvidenceSynthesizer(max_chunks=write_k) → context_str → LLM
       write_k = write_top_k 或 step_top_k 或 len(pack.chunks)；仅前 write_k 条进上下文，pack.chunks 全量参与引文解析
```

### 2.2 各阶段数量示例

以日志示例（`local_top_k=45, step_top_k=50, providers=tavily+scholar+semantic+ncbi`）为例：

| 阶段 | 数量 |
|------|------|
| `actual_recall` | `result_limit = 50`（当前实现无放大） |
| `local_recall_k` | `min(50, max(100, 20)) = 50` |
| 本地检索输出（rerank=False 时） | ≤ 50 条 |
| Web 原始结果 | ~314 条（4 provider × 多查询） |
| Web 内部去重后 | ~259 条 |
| cross_source_dedup 过滤 | 去掉已在本地库的文献 |
| 全局 fuse 输入 | 本地 ≤50 + web 剩余 |
| 检索输出（`result_limit`） | `step_top_k = 50` 条 |
| **进 LLM 上下文（`write_k`）** | **`write_top_k` 或 step_top_k，如 50 条** |

### 2.3 Chat 的 gap 补搜与融合（证据不足时）

当**证据不足**（evidence_scarce）且需要 RAG 时，Chat 会：

1. 用 LLM 生成至多 3 条 **gap query**（`_generate_chat_gap_queries`，prompt：`chat_gap_queries.txt`）。
2. 对每条 gap query 做并行补搜（`retrieval.search`，`pool_only=True`，top_k = max(10, step_k)），结果放入 `chat_gap_candidates_hits`（raw hits，**无内部 rerank**）。
3. gap 候选**暂存不融合**：`pack.chunks` 仍保持原始 RRF 池，不做任何中间 BGE rerank。

步骤 2-3 结束后，**无论是否触发 gap**，统一进入第 §5¾ 步（系统提示组装前）：

4. **单次统一 BGE rerank**（Phase `chat_pre_agent_fusion`）：调用 `_fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=chat_gap_candidates_hits, agent=[])`，执行全流程唯一一次联合 BGE rerank（`fuse_pools_with_gap_protection`），输出 `write_top_k` 条，同时完成：
   - 跨池 chunk_id 去重（gap 优先于 main）
   - gap 配额保护（`chat_gap_ratio=0.2`）
5. 融合结果写入 `pack.chunks`，再走 EvidenceSynthesizer → `context_str`（供系统提示和 LLM 使用）。

若后续 Agent 触发且 `agent_extra_chunks` 非空：

6. **Agent 追加融合（第二次 BGE rerank）**：Agent 在 §5¾ 的 `context_str` 基础上调用 LLM，LLM 通过工具检索发现**在 §5¾ 之后才产生的新 chunks**（`agent_extra_chunks`）。这些 chunks 在第一次 BGE rerank 时不存在，无法事先纳入排序，因此需要第二次 `_fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=[], agent=agent_extra_chunks)`。**两次 rerank 池组成不同，不重复：** 第一次无 agent chunks，第二次有；第二次以 §5¾ 输出的 `pack.chunks` 为 main，gap 传 `[]`（gap 已在 main 内，不得重传，否则双重计数）。agent 受 `chat_agent_ratio=0.1` 保护。若 `agent_extra_chunks` 为空（agent 未触发或工具无新 chunk），则第二次 rerank 跳过，整个路径仅 1 次 BGE rerank。

- **gap 补搜失败**：某条 gap query 检索异常时打 **WARNING**（`[chat] gap supplement search failed for ...`），其余照常融合。
- **无 gap 或未触发证据不足**：`chat_gap_candidates_hits=[]`，§5¾ 融合退化为对 main pool 单独做 BGE rerank。

### 2.4 重排器模式（Chat）

Chat 路径固定使用 `bge_only`（速度优先；cascade/colbert 为 DR 写作阶段保留）。如用户未通过 `filters.reranker_mode` 传入，则自动设为 `bge_only`。

### 2.5 Chat 是否有多步骤？主检索 vs Agent 二次召回

**结论：Chat 的“步骤”只有一步主检索；Agent 工具会带来按需的二次（或多次）召回，但和 DR 的“多步骤”不是同一概念。**

| 对比项 | Chat | Deep Research |
|--------|------|----------------|
| **步骤含义** | 无“多步骤”流程：**仅一次**主检索（`do_retrieval` 时调用 `retrieval.search()`），得到 `step_top_k` 条证据后直接进 LLM 上下文 | 有固定多步骤：research 轮次 → evaluate（含 gap 补充）→ generate_claims → write，每步都可能涉及检索或 pool 重排 |
| **主检索** | 一次 `search(mode=hybrid, top_k=local_top_k)`，内部 local+web 并行 → fuse_pools → 截断到 `step_top_k` | 每轮 research 调用一次 `search()`，结果写入 section pool；evaluate 时可能再按 gap 补搜；write 时从 pool 重排取 `write_top_k` |
| **二次/多次召回** | **Agent 工具**：当 `agent_mode=assist` 时，LLM 可主动调用 `search_local`、`search_web`、`search_scholar`、`search_ncbi` 等。每次调用都会再执行一次 `svc.search(mode="local" 或 "web", top_k=10)`，属于**按需、模型决定**的额外检索 | `research_node()` 内部 agent 工具补搜（`agent_supplement`）；`evaluate_node()` 补缺检索（`eval_supplement`） |
| **证据合并** | 主检索得到 `pack.chunks`。Agent 工具返回的 chunk 通过 `_collect_chunks` 收集，react 结束后 `drain_agent_chunks()` 得到 `agent_extra_chunks`。最终通过 `_fuse_chat_main_gap_agent_candidates()` 将 main + gap + agent 三池融合，agent 池受 `chat_agent_ratio=0.1` 软保护 | 所有证据先进入 section pool（按 pool_source 标记），写作前统一 `_rerank_section_pool_chunks(write_top_k)`，再经三池 fuse_pools（含 gap 0.2 + agent 0.25 保护）得到最终写作用证据 |

因此：

- **Chat 里没有“step”的多个步骤**：从检索服务视角只有**一步**主检索，`step_top_k` 表示的就是这一步的最终保留数。
- **Agent 里确实有二次（或 N 次）召回**：工具 `search_local` / `search_web` 等会再次调用 `RetrievalService.search()`，每次用工具传入的 `query` 和 `top_k`（默认 10）。给模型展示的 tool 返回仍可做短上下文裁剪，但入池的 `agent_extra_chunks` 必须保留完整候选池，不得再额外截断。agent 全部完成后，`agent_extra_chunks` 会作为 agent pool 参与最终三池融合（`_fuse_chat_main_gap_agent_candidates`），受 `chat_agent_ratio` 软保护。

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
| `"research_round"` | 研究轮次（research_node）主检索 | 主池（main） |
| `"eval_supplement"` | 评估节点发现 gap 后补充搜索 | **gap 池（受保护）** |
| `"agent_supplement"` | research_node 内部 agent 工具补搜 | **agent 池（受保护）** |
| `"write_stage"` | write_node 因章节池证据不足或异常触发的兜底补充检索（严禁将已用过的写作/验证证据回灌） | 主池（main） |
| `"revise_supplement"` | review_revise_agent_supplement 获取的定向补证结果（必须入池，保证 synthesize 阶段引文可溯源） | agent 池（受保护） |

### 3.2 完整数据流

```
Deep Research 任务
  │
  ├─[研究节点 research_node] × N 轮
  │    ├─ RetrievalService.search(mode, top_k=step_top_k, ...)
  │    │    └─ 同 Chat hybrid 流程（全局融合后 step_top_k 条）
  │    ├─ _accumulate_section_pool(pool_source="research_round")
  │    └─ agent 工具补搜（research_node 内部一步，非独立 LangGraph 节点）
  │         └─ _accumulate_section_pool(pool_source="agent_supplement")  ← agent 保护标签
  │
  ├─[评估节点 evaluate_node]
  │    ├─ ★ BGE rerank #1（phase=evaluate_pool_rerank）：
  │    │    _rerank_section_pool_chunks(query=topic+section, top_k=len(pool_chunks))
  │    │    → 对全量章节池做一次 BGE rerank，供覆盖度评估（reranker_mode 固定 bge_only）
  │    │    （全池 top_k 使 LLM 获得完整排序视角；不做小窗硬截断，过长时仅压缩/摘要）
  │    └─ 若覆盖度不足 AND 存在 gaps：
  │         └─ 最多取前 3 个 gaps，每 gap 单独检索一次
  │              top_k = step_top_k；未传时回退到 preset search_top_k_eval
  │              └─ _accumulate_section_pool(pool_source="eval_supplement")  ← gap 保护标签
  │
  ├─[生成主张节点 generate_claims_node]（comprehensive 默认执行，lite 跳过）
  │    └─ ★ BGE rerank #2（phase=generate_claims_pool_rerank）：
  │         _rerank_section_pool_chunks(query=topic+section, top_k=write_top_k)
  │         → 同 write_node 三池融合（见下），用于主张提取
  │
  ├─[写作节点 write_node]
  │    ├─ write_top_k = _compute_effective_write_k(preset, filters)
  │    │     = max(preset_write_k, ui_write_top_k 或 step_top_k×1.5)
  │    ├─ verification_k = max(15, ceil(write_top_k × 0.25))（目标态动态值）
  │    │
  │    └─ ★ BGE rerank #3（phase=write_pool_rerank，同时产出 write_chunks 和 verify_chunks）：
  │         _rerank_section_pool_chunks(query=topic+section, top_k=write_top_k,
  │                                     secondary_top_k=verification_k)
  │         ├─ 按 pool_source 分三池：
  │         │    ├─ main_candidates：pool_source ∈ {"research_round", "write_stage"}
  │         │    ├─ gap_candidates： pool_source == "eval_supplement"
  │         │    └─ agent_candidates：pool_source ∈ {"agent_supplement", "revise_supplement"}
  │         └─→ fuse_pools_with_gap_protection(top_k=write_top_k, secondary_top_k=verification_k,
  │                 gap_ratio=0.2, agent_ratio=0.25, multiplier=3.0)
  │               ├─ 全局单次 BGE rerank（main+gap+agent 统一评分，rerank_k = write_top_k × 3）
  │               ├─ _apply_quota_selection(write_top_k) → write_chunks（含 gap/agent 配额保护）
  │               └─ _apply_quota_selection(verification_k) → verify_chunks（复用已排序 pool，
  │                    无需重跑 BGE；gap_min=ceil(verification_k×0.2)，独立配额窗口）
  │
  ├─[验证节点 verify_node]
  │    ├─ light（unsupported ratio ≤ verify_light_threshold）：轻量告警，继续完成
  │    ├─ medium：记录 gaps，不回到 research
  │    └─ severe（unsupported ratio > verify_severe_threshold）：
  │         章节回到 research；若超 max_verify_rewrite_cycles 则 cap，继续完成
  │
  ├─[审核节点 review_gate_node]（skip_draft_review=false 时进入）
  │    ├─ revise：
  │    │    ├─ [返修补证 review_revise_agent_supplement]
  │    │    │    └─ 仅跑 1 轮定向 agent GAP 补证
  │    │    │    └─ review_revise_supplement_k = max(1, ceil((step_top_k or search_top_k_eval) * 0.5))
  │    │    │    └─ 新证据必须追加写入 Section Evidence Pool（pool_source="revise_supplement"）
  │    │    ├─ [整合重写 review_revise_integrate]
  │    │    │    └─ 消费旧章节文本 + 新补证 + 作者补充观点 + review 问题 → 生成新版章节
  │    │    └─ 回到 review_gate_node 由用户再次确认
  │    └─ approve / skip：继续
  │
  └─[最终综合 synthesize_node]
       ├─ 可选 final_agent_supplement（工具检索继承请求级 step_top_k）
       ├─ abstract / limitations / future directions / open gaps agenda 生成
       ├─ 全文 coherence refine + 最终 citation resolve
       └─ 不走章节级 write_top_k 截断
```

### 3.3 write_top_k 计算逻辑

```python
def _compute_effective_write_k(preset, filters):
    preset_write_k = int(preset.get("search_top_k_write", 12))
    ui_write_k     = int(filters.get("write_top_k") or 0)
    ui_step_k      = int(filters.get("step_top_k") or 0)

    if ui_write_k > 0:
        result = max(preset_write_k, ui_write_k)
    elif ui_step_k > 0:
        result = max(preset_write_k, int(ui_step_k * 1.5))
    else:
        result = preset_write_k

    # 注意：search_top_k_write_max 当前已接线，参与裁剪
    cap = int(preset.get("search_top_k_write_max", 0))
    if cap > 0:
        result = min(result, max(cap, preset_write_k))
    return result
```

| `depth` | `preset_write_k`（search_top_k_write） | `search_top_k_write_max`（已接线） |
|---------|----------------------------------------|----------------------------------------|
| `lite` | 10 | 40（参与裁剪） |
| `comprehensive` | 12 | 60（参与裁剪） |

### 3.4 各阶段数量示例

以 `step_top_k=50, write_top_k=None, depth=comprehensive` 为例：

| 阶段 | 数量 |
|------|------|
| 研究轮次每轮检索 | step_top_k = 50 条 |
| section pool 累计（N 轮后） | 50 × N 条（含重复）+ agent_supplement |
| evaluate 取证 | 优先使用章节池已累积证据；不做小窗硬截断 |
| eval_supplement（每 gap） | step_top_k = 50 条；最多 3 个 gaps |
| eval_supplement 入池 | ≤ 3 × 50 = 150 条（依赖后续 fuse 收敛） |
| write_top_k 推导 | max(12, 50×1.5) = 75（受 search_top_k_write_max = 60 cap，实际 = 60） |
| verification_k | max(15, ceil(60 × 0.25)) = 15（目标态动态值） |
| **写作最终证据** | **60 条（gap 中至少 12 条受保护，agent 中至少 15 条受保护）** |

---

## 四、fuse_pools_with_gap_protection 算法

两条路径的最终截断都经过此函数（`src/retrieval/service.py`）。

### 4.1 算法步骤

```
输入：main_candidates (N_m 条) + gap_candidates (N_g 条) + agent_candidates (N_a 条，可选) + top_k
      可选：gap_min_keep / agent_min_keep（显式指定时覆盖比例推算）

Step 1  标记：每个候选打 _pool_tag = "main" | "gap" | "agent"
Step 2  全局重排：_rerank_candidates(all_cands, top_k = rank_pool_k)
        rank_pool_k = min(max(ceil(top_k×multiplier), top_k + N_g + N_a), N_total)
        → 单次 cross-encoder 评分，消除跨源分布差异
Step 3  初始切片：取 reranked[:top_k]
Step 4  quota 保护：
        gap_min_keep = 调用方传入 或 默认 ceil(top_k × gap_ratio)
        Chat 默认 gap_ratio=0.2，Research 默认 gap_ratio=0.2（均可配置）
        agent_min_keep = 有 agent 池时 ceil(top_k × agent_ratio)
        Chat 默认 agent_ratio=0.1，Research 默认 agent_ratio=0.25
        若 top_k slice 中 gap 数 < gap_min_keep：先从 reranked 的 tail 取 gap 补齐；仍不足再从未入榜 gap pool 补齐；替换 slice 中最低位 main，保持总数 = top_k。agent 同理。
        effective_min_keep = min(理论配额, n_gap, top_k)，即池不足时用实际数量
Step 5  剥除 _pool_tag，返回 top_k 条

诊断输出（diag["pool_fusion"]）：
  main_in, gap_in, total_reranked, rank_pool_k, rank_pool_multiplier,
  gap_deficit_before_fill, gap_backfill_ranked, gap_backfill_unranked,
  gap_min_keep, gap_in_output, output_count
```

### 4.2 gap_min_keep / agent_min_keep 来源

| 路径 | gap_min_keep | agent_min_keep |
|------|----------------|-----------------|
| **Chat** | `ceil(step_k × chat_gap_ratio)`（默认 0.2，可配置） | 有 agent 追加时 `ceil(top_k × chat_agent_ratio)`（默认 0.1）；见 `_fuse_chat_main_gap_agent_candidates` |
| **Deep Research** | `ceil(top_k × research_gap_ratio)`（默认 **0.2**，可配置） | 章节 fuse 时 `ceil(top_k × research_agent_ratio)`（默认 **0.25**）；仅 `agent_supplement` 入 agent 池 |

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
| **Chat gap 补搜（暂存）** | `src/api/routes_chat.py` | 证据不足时 `_generate_chat_gap_queries` → 并行补搜（`pool_only=True`）→ 结果暂存 `chat_gap_candidates_hits`；不做中间 BGE rerank |
| **Chat §5¾ 统一 BGE fusion** | `src/api/routes_chat.py` | 系统提示组装前：`_fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=chat_gap_candidates_hits, agent=[])` → 唯一一次 BGE rerank，含跨池去重与 gap 配额保护 |
| **Chat Agent 追加融合** | `src/api/routes_chat.py` | `agent_extra_chunks` 非空时：`_fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=[], agent=agent_extra_chunks)` → 第二次 BGE rerank，含 agent 配额保护 |
| **Chat Agent 二次召回** | `src/llm/tools.py` | `_handle_search_local()`, `_handle_search_web()` → `svc.search(mode="local"\|"web", top_k=10, filters={pool_only=True})`；结果经 `_collect_chunks` → `drain_agent_chunks()` 合并到 `pack.chunks`；`_handle_search_scholar()` 通过 `_search_and_close()` 协程确保 aiohttp session 在 finally 块中关闭 |
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

1. **截断只发生一次**：`fuse_pools_with_gap_protection` 返回后，唯一的 `all_chunks[:result_limit]` 在 `search()` 末尾作为安全兜底（对 hybrid 是幂等的）。`pool_only=True` 时必须绕过这一步，返回未截断原始候选池。

2. **Chat 的 fused pool 不做绝对阈值二次过滤**：Chat 的 main+gap / main+gap+agent 融合已经完成统一排序与 quota 回填，不得再对 fused 结果追加 `fused_pool_score_threshold` 之类的绝对阈值过滤，否则会直接破坏 gap / agent 保护。

3. **local_recall_k 公式**：hybrid 模式中 `local_recall_k = min(actual_recall, max(result_limit × 2, 20))`，且当前 `actual_recall = result_limit`。本地分支用 `top_k=local_recall_k` 调用 retriever，为全局融合提供候选；不得改为仅用 `result_limit` 或与 contract 不符的公式。

4. **eval_supplement 必须进 gap pool**：`_DR_GAP_POOL_SOURCES` 必须包含 `"eval_supplement"`，否则 gap 评估阶段补充的证据在写作时会与普通证据平等竞争，可能全部被挤出 top-k。

5. **agent_supplement 必须进 agent pool**：`research_node()` 内部 agent 工具补搜的结果以 `pool_source="agent_supplement"` 入池，章节 fuse 时作为独立 agent pool 参与三池融合，受 `research_agent_ratio=0.25` 保护。若将其混入 main pool，agent 补搜证据在候选量大时可能被全部挤出。

6. **进入章节池的 Research 检索统一使用 pool_only**：凡是检索结果还会进入 Section Evidence Pool，并在后续 `_rerank_section_pool_chunks()` 统一重排的路径（如 `research_round`、`eval_supplement`、`review_revise_supplement`），必须设置 `pool_only=True`，避免在入池前提前 rerank / 截断。

7. **revise_supplement 必须入池**：`review_revise_agent_supplement` 获取的新证据必须追加写入当前章节的 Section Evidence Pool（`pool_source="revise_supplement"`），以保证最终全文合成阶段引文可精准溯源。

8. **write_stage 严禁回灌**：`write_stage` 仅限 `write_node()` 因章节池不足或异常触发的兜底补充检索结果。严禁将已用过的写作/验证证据回灌进章节池，否则会导致重排权重污染和数据冗余。

9. **gap_supplement 上下文溢出防护**：用户手动提交的 `gap_supplement` 虽绕过 `fuse_pools`，但在拼接进入 LLM 写作上下文前，必须动态扣减其对应的 `write_top_k` 额度或执行严格的 Token 截断防范机制，严防上下文溢出。

10. **soft-wait 不能改回 with-block**：`ex.shutdown(wait=False)` 之后继续 `fw.result(timeout=web_wait_s)` 是软等待的实现方式；若改回 `with ThreadPoolExecutor ... as ex:` 会强制等待所有线程完成，失去超时控制。

11. **BGE rerank 次数约束（Chat vs Deep Research）**：两条路径的 rerank 次数设计不同，均有充分理由。

    **Chat**（最多 2 次）：
    - **非 agent / agent 无新 chunk**：仅 §5¾ `chat_pre_agent_fusion` 一次（main+gap）。
    - **agent 路径且 `agent_extra_chunks` 非空**：§5¾ 一次（main+gap） + ⑧b 追加融合一次（pack.chunks+agent 新 chunk）。第二次原因：agent 工具调用在 §5¾ 之后发生，产生的新 chunks 在第一次 rerank 时不存在，必须事后补排。两次池组成不同，不冗余。
    - **禁止**：local main pool 单独 rerank、gap 补搜后立即做中间 fusion。

    **Deep Research**（每章节 2–3 次 BGE rerank + 1 次子集截取）：
    - **#1 evaluate_pool_rerank**（evaluate_node）：`top_k=len(pool_chunks)`，全池排序供覆盖度评估，query = topic+section。
    - **#2 generate_claims_pool_rerank**（generate_claims_node，comprehensive 才执行）：`top_k=write_top_k`，主张提取（lite 跳过，共 2 次）。
    - **#3 write_pool_rerank**（write_node）：`top_k=write_top_k`，写作正文，query = topic+section。
    - **#4 write_verify_pool_rerank**（write_node）：`top_k=verification_k`，引文验证，query 同 #3（原"verification"方向词已去除，不影响排名）。必须独立走 `fuse_pools_with_gap_protection(top_k=verification_k)` 以保证 gap/agent 配额按 verification_k 比例生效，不可截取 write_chunks。
    - Research 多次 rerank 是合理的：各节点目的不同（评估 vs 主张 vs 写作 vs 验证），top_k 不同导致 gap/agent 配额窗口不同，章节池在节点间持续累积，后续节点在更大的池上重新排序。

12. **跨池去重（gap/agent 优先于 main）**：`fuse_pools_with_gap_protection` 在合并候选池前按 chunk_id 去重，顺序：gap → agent → main（main 中与 gap/agent 重复的 chunk_id 被剔除）。此步骤发生在全局 BGE rerank 之前，确保每个 chunk 只参与一次排名。

13. **search_scholar 工具必须关闭 aiohttp session**：`_handle_search_scholar` 每次实例化 `SemanticScholarSearcher`，其内部 `_ensure_session()` 创建 `aiohttp.ClientSession`。必须在 search 完成后（无论成功或异常）通过 `try/finally` 调用 `await ss.close()`，否则 agent 结束后 event loop 关闭时触发 `Unclosed client session` 与 `RuntimeError: Event loop is closed`。实现模式：将 `ss.search(...)` 与 `ss.close()` 封装在同一个协程 `_search_and_close()` 中，再通过 `asyncio.run()` 在子线程执行。
