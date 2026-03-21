# 前端信息架构与工作流重构方案 (v2.0)

> 2026-03-15 实施状态：
> - 已切换为路由化工作区壳层，主入口为 `/chat`、`/ingest`、`/scholar`、`/papers/:paperUid`、`/analysis`、`/workspace/graph`
> - Scholar 已改为“检索与纳管区”，主动作收口为 `Open in Paper Workspace` 与 `Add to Analysis Pool`
> - `Compare / Discovery Workspace` 与 `Graph Workspace` 已落地；旧 `/compare` 作为兼容入口保留，旧 `/graph` 跳转到新图谱工作区
> - 分析池采用全局 persistent pool，基于 `paper_uid` 跨 Scholar / Paper Workspace / Graph Workspace 共享

## 1. 核心设计理念

当前前端虽然成功接入了强大的后端能力（LLM 总结、发现、图谱提取等），但信息架构仍停留在“能力堆砌”阶段，存在“未选对象先见操作”、“功能埋在侧栏深处”的问题。

本次重构的根本原则是：**“先选对象，再选动作，最后看结果。”**

我们将彻底抛弃在单一列表页（Scholar）无限叠加侧栏的方案，转而面向**使用场景**构建四大独立的前端工作区（Workspace）。

---

## 2. 四大核心工作区设计

### 2.1 Scholar 页 (检索与纳管区)
**定位**：系统的“入口”与“分发中心”，专注“找文献、筛文献、入库、绑定”。
- **对象**：全局文献库 / 外部检索引擎。
- **UI 呈现**：纯粹的列表视图。
- **支持的动作**：
  - 基础操作：搜索、下载、保存到库 (Collection/Library)。
  - 轻量用户态：加标签 (Tags)、收藏 (Favorite)、预览笔记。
  - **流转动作 (核心)**：
    - 点击单篇文献 ➔ `Open in Paper Workspace`
    - 勾选多篇文献 ➔ `Add to Analysis Pool` 进而跳转 Compare / Discovery Workspace。

### 2.2 Paper Workspace (单篇文献精读工作台)
**定位**：文献精读的核心阵地，替代原有的局部弹窗或右侧滑出面板。
- **对象**：锁定到单一 `paper_uid`。
- **路由建议**：`/papers/:paperUid`
- **页面布局（经典三栏）**：
  - **左栏 (Context & Nav)**：当前 Collection / Library 下的论文列表，或最近打开的历史记录，方便快速切换。
  - **中栏 (Content)**：全屏级的 PDF 阅读区（支持缩放、高亮、划词）。
  - **右栏 (AI Assistant & Overlay)**：明确的 Tab 面板：
    - **Summary**：生成并展示当前文献的结构化摘要。
    - **Ask**：针对当前文献的独立问答对话框。
    - **Notes**：当前文献的个人笔记区。
    - **Annotations**：针对特定图片/段落的 Anchored Annotation，支持在 PDF 上双向定位。
    - **Media / Figures**：提取并解析出的图片、表格和视觉元数据。
- **交互约束**：右上角或标题栏明确锁定状态 `Current target: [Title / paper_uid]`，所有 AI 动作（如 QA）直接携带该上下文，无需二次选择。

### 2.3 Compare / Discovery Workspace (多文献分析工作台)
**定位**：解决多实体聚合分析的问题，不与单篇文献阅读抢占屏幕空间。
- **对象**：一组 papers 数组 (`papers[]`) 或一组图节点种子。
- **入口**：由 Scholar 页多选触发，或由 Paper Workspace 侧栏“加入比较池”触发。
- **页面布局**：
  - **顶部 (对象池)**：Selected Papers 的 Chip 列表（支持快速移除/增补）。
  - **左侧 (分析动作选区)**：
    - *Compare*: Compare Matrix, Narrative Summary.
    - *Discovery*: Missing Core (引文补全), Forward Tracking (被引追踪), Experts (核心学者), Institutions (核心机构).
  - **右侧/主体 (结果输出区)**：
    - 根据左侧所选动作，渲染横向对比表格、Markdown 总结文本或推荐文献卡片列表。

### 2.4 Graph Workspace (图谱探索工作台)
**定位**：彻底改变“先给空图再搜节点”的逻辑，强制“范围先行”。
- **对象**：Graph Scope + Seed Nodes。
- **路由建议**：`/workspace/graph`
- **页面布局**：
  - **Top Bar (表单区 - 必填)**：
    - 图类型：`Entity` / `Citation` / `Author` / `Institution`
    - 范围：`Global` / `Collection` / `Library`
    - 种子：关联的 `paper_uid`, `author_id` 或 `institution_id`（支持模糊搜索带入）。
    - *只有填完范围和种子，才激活“查询图谱”按钮。*
  - **Main Area (双栏输出)**：
    - **左/中栏**：ForceGraph 可视化图表（保留原有交互体验）。
    - **右栏 (Graph Details)**：
      - AI 生成的图摘要 (Graph Summary)。
      - 核心指标 (PageRank 最高的节点、桥接点)。
      - 快照信息 (Snapshot Version & Status)。
      - 点击图上节点时的详情侧板。

---

## 3. 推荐的实施演进路线 (Roadmap)

为了平稳过渡并最快兑现业务价值，建议分步实施：

### 🏁 Phase 1: 打造极致的 Paper Workspace
*这是用户最迫切的需求，也是最符合真实科研工作流的一环。*
1. 新建 `/papers/:paperUid` 路由。
2. 移植现有的 PDF 渲染组件作为主视觉区。
3. 将现有的 Summary、Ask、Annotations 接口接入右侧 Tab 栏。
4. 确保在精读页面内，AI 的上下文被严格限制为当前论文，状态（笔记、标签）直接落库。

### 🏁 Phase 2: 重构 Graph Workspace 的“范围先行”逻辑
*解决当前图检索难以预期、用户认知模糊的问题。*
1. 隔离 Entity 图（HippoRAG）与 Typed Graph（Academic）的前端入口。
2. 开发统一的**图谱查询条件表单栏**。
3. 补齐图谱摘要（Summary）和关键指标（Metrics）的右侧常驻面板。

### 🏁 Phase 3: 抽离 Compare & Discovery 工作区
*为 Scholar 列表减负。*
1. 在 Scholar 列表增加“加入分析池”功能。
2. 建立独立路由集中处理多对象 API 调用。
3. 实现不同 Discovery 模式返回结果的一致性卡片渲染。

### 🏁 Phase 4: Scholar 的彻底瘦身
*收尾工作。*
1. 从 Scholar 的操作栏彻底移除“Summarize”, “Ask”, “Compare” 等深度阅读按钮。
2. 仅保留“添加到 Collection/Library”、“添加标签”和清晰的“在工作台中打开”跳转按钮。

---

## 4. 结论与下一步

这版设计将彻底改变系统的性质：从一个**“带有各种 AI 接口的文献列表”**，进化为一个**“基于对象的结构化科研工作台”**。

如果你对以上架构方案认可，我们接下来的编码动作将停止在旧的侧边栏上打补丁，而是**直接拉出 Phase 1 (Paper Workspace) 的骨架代码**，实现第一个路由页面及三栏布局。
