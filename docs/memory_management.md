# 内存与记忆管理维护文档 (Memory Management & Maintenance)

本文件详细说明了系统中的三层记忆架构（Session Memory, Working Memory, Persistent Store）的深度实现逻辑、存储结构、以及它们在 RAG 场景下的利用机制。

---

## 1. 记忆架构总览 (Architectural Overview)

系统采用分层记忆架构，旨在平衡 Token 消耗、上下文质量与跨会话的连贯性：

| 记忆层级 | 存储对象 | 作用域 | 跨 Session | 核心驱动机制 |
| :--- | :--- | :--- | :--- | :--- |
| **Session Memory** | 活跃对话 Buffer | 单次 Chat | ❌ 隔离 | **Token 容量驱动 (MAX: 40k chars)** |
| **Rolling Summary** | 长程历史压缩 | 单次 Chat | ❌ 隔离 | **无重叠 (Zero-Overlap) 滑动归纳** |
| **Evidence Pool** | 检索材料 (Chunks) | 单次 Chat | ❌ 隔离 | **命中提权 (Hit Boosting) + 置顶机制** |
| **Working Memory** | 画布项目核心事实 | 关联 Canvas | ✅ 共享 | **结构化 Fact 提取 + 冲突解决 (Newer Wins)** |
| **Persistent Store** | 用户偏好/习惯 | 全局用户 | ✅ 共享 | 结构化 Profile 注入 |

---

## 2. 会话级记忆：主流化 Summary Buffer (Session Memory)

系统放弃了机械的“固定轮次”管理，采用了业界领先的 **Summary Buffer** 模式。

### 2.1 Token 驱动的滑动驱逐 (Token-based Eviction)
- **容量上限**: `MAX_BUFFER_CHARS = 40,000` (约 10k Tokens)。
- **驱逐逻辑**: 采用 FIFO (先进先出) 策略。当 Buffer 溢出时，最旧的轮次会被移出 Buffer 并触发 `Rolling Summary` 更新。
- **零重叠上下文 (Zero-Overlap)**: 系统严格区分“已归纳”与“未归纳”内容。发给 LLM 的上下文永远是 `[Rolling Summary] + [Active Buffer]`。两者边界清晰，彻底消除了冗余 Token 浪费。

### 2.2 Assistant 回答“瘦身” (Response Slimming)
为防止 AI 冗长的思考过程或 RAG 证据块撑爆 Buffer，系统在记忆持久化前执行以下清洗：
- **拦截器**: 自动剥离 `assistant` 回答中的 `<think>`、`<search_results>` 和 `<evidence>` 标签内容。
- **目的**: 确保对话记忆中只保留 AI 的**最终结论**，大幅提升 Buffer 的有效信息密度。

### 2.3 Rolling Summary 质量保障
- **硬上限**: `MAX_SUMMARY_CHARS = 12,000`。
- **再压缩 (Re-compression)**: 每次合并新驱逐的内容时，LLM 会被要求对整体摘要进行重新提炼，剔除过时的探索路径，保持长程记忆的高清晰度。

---

## 3. 证据池管理：RAG 专项优化 (Evidence Pool)

文献 Chunk 被视作一种“动态外挂记忆”，其生命周期受以下算法控制：

### 3.1 命中提权 (Hit Boosting)
- **机制**: 当 AI 的 `tool_calls` 或回答实际引用了某个缓存的 Chunk 时，系统会增加其 `hit_count`。
- **淘汰豁免**: 高命中的 Chunk 在缓存中拥有更高权重。即便它在时间线上较旧，也会被优先保留，不容易被新话题冲掉。默认保留最近 10 轮检索结果。

### 3.2 置顶记忆 (Pinned Context)
- **功能**: 支持将“基石文献”的关键 Chunk 置顶入系统 Prompt。
- **熔断保护**: 置顶内容设有 `MAX_PINNED_CHARS = 30,000` 的硬上限，防止用户过度置顶导致主对话窗口崩溃。

---

## 4. 跨对话记忆：结构化 Fact 演进 (Working Memory)

通过 Canvas 桥接不同 Session 的关键在于“传递什么”以及“如何处理冲突”。

### 4.1 结构化事实提取 (Fact-based Extraction)
Working Memory 不再是一段散文摘要，而是带时间戳的 **Facts 列表**：
- **Fact 示例**: `[2026-03-12 10:00] 确定研究重点为冷泉生态系统的能量流动机制。`

### 4.2 冲突解决 (Conflict Resolution - Newer Wins)
- **机制**: 每次更新 Working Memory 时，系统会将“旧 Facts”与“新进展”同时发给 LLM。
- **指令**: 显式要求 LLM 进行版本演进。如果新旧信息矛盾（如用户修改了研究计划），AI 会以带最新时间戳的信息为准，将旧信息标记为 `[Revised]` 或直接覆盖。

---

## 5. 可观测性与调试 (Observability)

系统提供了“黑盒”透视能力，方便开发者监控记忆系统的运行状态。

### 5.1 调试接口 (Debug API)
- **端点**: `GET /api/debug/memory/{session_id}`
- **返回信息**:
    - **Buffer 剖面**: 当前活跃轮次、字符占用百分比。
    - **Summary 版本**: 当前压缩后的历史摘要内容。
    - **Cache 权重**: 缓存中每个 Chunk 的 `hit_count` 和 `is_pinned` 状态。

---

## 6. 核心参数参考表 (Config Table)

| 参数名 | 默认值 | 所在模块 |
| :--- | :--- | :--- |
| `MAX_BUFFER_CHARS` | 40,000 | `session_memory.py` |
| `MAX_SUMMARY_CHARS` | 12,000 | `session_memory.py` |
| `MAX_PINNED_CHARS` | 30,000 | `session_memory.py` |
| `MAX_EVIDENCE_TURNS`| 10 | `session_memory.py` |
| `FINAL_INTEGRATION_LIMIT`| 800,000 | `context_limits.py` |

---

## 7. 改进建议 (Future Roadmap)

1. **多级向量索引**: 将过期的 Session 存入向量库，支持“还记得上个月我聊过的那个话题吗？”这种超长跨度搜索。
2. **记忆重要度评分**: 自动识别对话中的关键结论并“加固”到 Working Memory 中，防止被滚动摘要稀释。
3. **分场景 Buffer**: 为 `Deep Research` 任务设定更大的专属 Buffer 空间。
