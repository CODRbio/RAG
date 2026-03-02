# Rerank 机制说明（改版后）

本文档描述在「Remove Local Rerank Duplication」「Graph Top-K UI and Pool-Only」「Chat gap pool-only + Local fuse_pools 统一」以及「Remove Dead Rerank Block from retrieve_hybrid」改版后的 rerank 调用链与设计不变量。

## 设计不变量

1. **Hybrid 模式**：local 与 graph 只产出**有序候选池**（RRF 序 / 图融合序），**不做** rerank、不做截断；**唯一**的 rerank 发生在 `fuse_pools_with_gap_protection` 内，对 local + web（及可选 gap）做一次全局排序并截断。
2. **Local-only 模式**：`retrieve_hybrid` 无 rerank 参数，始终返回完整候选池；service 内调 `fuse_pools_with_gap_protection(local_pool, [], top_k)` 做**一次**全局 rerank 再截断——与 hybrid 模式统一。
3. **Web-only 模式**：在 service 内对 web 结果做**一次** rerank（`_rerank_candidates` 或 `_embedding_rerank`）后转 chunk。
4. **Chat gap 补搜**：gap 子查询走 `pool_only=True`，service 跳过 fuse_pools，只返回原始池子；最终 routes_chat 将 main + 所有 gap 做**唯一一次** `fuse_pools_with_gap_protection`。

---

## 调用链总览

| 场景 | 入口 | Rerank 发生位置 | 次数 |
|------|------|------------------|------|
| **Chat — hybrid** | routes_chat → retrieval.search(mode=hybrid) | service: fuse_pools_with_gap_protection(local_main + web_main) | 1 |
| **Chat — hybrid + gap 补搜** | 主检索同上；每个 gap 调 retrieval.search(pool_only=True)；最后 routes_chat 合并 | ① 主检索：service 内 1 次 fuse_pools<br>② 每个 gap 查询：**不 rerank**（pool_only 直接返回原始池）<br>③ 主+gap 合并：routes_chat 内 1 次 fuse_pools | 1 + 1 = 2 |
| **Chat — local-only** | retrieval.search(mode=local) | service: fuse_pools_with_gap_protection(local_pool) | 1 |
| **Chat — local-only + gap 补搜** | 主检索同上；每个 gap 调 retrieval.search(pool_only=True)；最后 routes_chat 合并 | ① 主检索：service 内 1 次 fuse_pools<br>② 每个 gap 查询：**不 rerank**<br>③ 主+gap 合并：routes_chat 内 1 次 fuse_pools | 1 + 1 = 2 |
| **Chat — web-only** | retrieval.search(mode=web) | service: web 分支内 _rerank_candidates 或 _embedding_rerank | 1 |
| **Deep Research** | agent 多次 svc.search(mode=hybrid) | 每次 search：service 内 fuse_pools 一次 | 每次 1 |
| **DR section 写作** | _rerank_section_pool_chunks | fuse_pools_with_gap_protection(main_candidates, gap_candidates) | 1 |

---

## 关键代码位置

### 1. 唯一实现：`_rerank_candidates`

- **文件**：`src/retrieval/hybrid_retriever.py`
- **作用**：对候选列表做 BGE/ColBERT 重排，支持 bge_only | colbert_only | cascade；大候选集先经 funnel（_embedding_pre_filter）再进 cross-encoder。
- **输出**：按 rerank 分数排序的列表，调用方再按需 `[:top_k]`。

### 2. 全局融合与唯一 rerank

- **函数**：`fuse_pools_with_gap_protection`（`src/retrieval/service.py`）
- **输入**：main_candidates、gap_candidates（可选）、top_k、reranker_mode、skip_rerank 等。
- **逻辑**：
  - 将 main/gap 打标（_pool_tag）后合并为 all_cands；
  - **一次**全局 rerank：`skip_rerank` 时用 `_embedding_rerank`，否则 `_rerank_candidates`，得到 reranked；
  - 取 top_slice = reranked[:target_top_k]，再做 gap 配额补齐（gap_min_keep）；
  - 去掉内部 tag 后返回。
- **不变量**：对一次 search 调用，只做**一次** rerank。

### 3. Local / Hybrid 分支均不在 retriever 内 rerank

- **文件**：`src/retrieval/service.py`
  - hybrid 分支：`local_config.rerank = False`
  - local 分支：`config.rerank = False`，retriever 返回完整池后，service 调 fuse_pools
- **文件**：`src/retrieval/hybrid_retriever.py`
  - `retrieve_hybrid` 无内部 rerank，始终返回完整 fused_hits 池，不截断不排序（由上层 fuse_pools 统一 rerank）
  - `retrieve_vector(rerank=False)` → 返回完整 RRF candidates，不截断
  - `retrieve_with_graph` → 返回完整融合池，不排序不截断；PPR 数量由 graph_top_k 控制

### 4. pool_only 模式（Chat gap 子查询）

- **触发**：`filters["pool_only"] = True`
- **行为**：service.search() 中，无论 hybrid 还是 local 分支，跳过 fuse_pools，直接返回原始候选池（带 `_source_type` 标签）
- **调用方**：routes_chat `_search_one_gap()` 设置 pool_only，收集所有 gap 子查询的原始池子，最终与 main 一起做唯一一次 fuse_pools

### 5. Web-only 的单次 rerank

- **文件**：`src/retrieval/service.py`，mode=="web" 分支
- 取回 web_hits 后：若 skip_rerank 则 `_embedding_rerank`，否则 `_rerank_candidates`；失败时回退到 embedding 或按分排序截断。只做一次。

### 6. Chat gap 补搜（routes_chat 最终合并）

- 主检索：一次 retrieval.search(hybrid/local) → service 内一次 fuse_pools。
- 每个 gap query：retrieval.search(pool_only=True) → 跳过 rerank，返回原始池。
- 最后：routes_chat 将 main_candidates 与所有 gap_candidates 交给 fuse_pools_with_gap_protection 做**唯一一次**最终合并与 rerank。
- 总 rerank 次数 = 主检索 1 次 + 最终合并 1 次 = 2 次。

### 7. Deep Research section pool

- **文件**：`src/collaboration/research/agent.py`，`_rerank_section_pool_chunks`
- 将 section 的 main_candidates 与 gap_candidates 交给 fuse_pools_with_gap_protection；若未启用 fusion 或失败，则回退为 `_rerank_candidates(all_candidates, ...)`。每个 section 一次 rerank。

---

## skip_rerank 行为

- **filters["skip_rerank"] == True** 时：
  - fuse_pools 内用 `_embedding_rerank` 代替 `_rerank_candidates`（轻量 bi-encoder 排序）。
  - local/hybrid 分支：retriever 始终 `rerank=False`，不受 skip_rerank 影响；fuse_pools 内的 skip_rerank 会使用 embedding 排序。
  - web 分支：用 `_embedding_rerank` 代替 cross-encoder。
  - chat gap pool_only：子查询不做任何 rerank，skip_rerank 仅影响最终 routes_chat 的 fuse_pools。
