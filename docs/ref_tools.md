# 文献搜索与学术助手增强方案

**版本**：V4.0  
**日期**：2026-03-13  
**定位**：在现有 RAG 项目基础上，增强 Scholar / 文献发现 / 学术助手能力，而不是平行再造一套新系统。

---

## 一、文档目标

本文档用于把原有“文献搜索模块”升级为一个可落地、可复用、适配远程服务器部署的学术助手方案。核心目标不是单点做“找文”，而是打通：

1. 文献搜索
2. 文献入库
3. 文献精读
4. 文献发现
5. 图谱辅助
6. 对话式学术助手

同时必须遵守两个总原则：

1. **优先复用现有项目能力**，避免在 `ref_tools` 旁边再长出第二套检索、下载、任务、数据库、SSE 体系。
2. **所有核心能力以远程服务为中心设计**，因为本工具运行在远程服务器，前端只是调用方，不应承载状态真相。

---

## 二、核心架构原则

### 2.1 Remote-First，而不是 Browser-First

这里的“本地”指的是**服务器侧本地资源**，不是用户浏览器本地。

- 向量库、关系库、Redis、缓存、Playwright、下载器都部署在远程服务器
- 前端只通过 HTTP / SSE 调用
- 长任务必须支持 `task_id`、断点续传、心跳和终态事件
- 不能依赖前端内存状态保存任务上下文
- 不能把关键流程写成“只有单进程内存对象存在时才可用”

### 2.2 Reuse-First，而不是模块平行复制

`ref_tools` 不应重新实现以下能力：

- LLM 调用：复用 `src/llm/llm_manager.py`
- 混合检索：复用 `src/retrieval/service.py`
- 多源学术搜索：复用 `src/retrieval/unified_web_search.py`、`src/retrieval/semantic_scholar.py`
- 全文抓取：复用 `src/retrieval/web_content_fetcher.py`
- 下载器：复用 `src/retrieval/downloader/paper_downloader_refactored.py`
- Scholar 工作流与 SSE：复用 `src/api/routes_scholar.py`、`src/tasks/dispatcher.py`、`src/tasks/redis_queue.py`
- 共享数据库：复用 `src/db/engine.py`、`src/db/models.py`
- 文献元数据与入库记录：复用 `src/indexing/paper_metadata_store.py`、`src/indexing/paper_store.py`

### 2.3 图服务全局通用，而不是 ref_tools 私有

项目里已经存在多种“图”概念：

- LangGraph 工作流图：`src/collaboration/workflow/graph.py`
- HippoRAG 实体图：`src/graph/hippo_rag.py`
- 未来要补的文献引用图 / 作者合作图 / 机构图

这三类图不能混为一谈。后续应抽象出**统一图服务层**，由它管理：

- 图数据来源
- 图快照构建
- 图缓存与失效
- 子图查询
- 图摘要压缩（Graph Prompting）

`ref_tools` 只是这个统一图服务的一个业务使用方。

### 2.4 SQL 管理全局通用，而不是每个模块再开新库

当前项目已经有共享 SQL 引擎与统一模型层：

- `src/db/engine.py`
- `src/db/models.py`

因此：

- **业务主数据**必须进入共享关系库
- **不要**为 `ref_tools` 单独维护新的主 SQLite/JSON 库
- SQLite / DiskCache 可以继续用于短 TTL 的外部 API 缓存，但不应成为业务真相源

---

## 三、现有项目能力映射

下表定义 `ref_tools` 应复用的现有模块边界：

| 能力 | 现有模块 | 在学术助手中的角色 |
| --- | --- | --- |
| LLM 统一调度 | `src/llm/llm_manager.py` | 总结、问答、Graph Prompting、query rewrite |
| 本地/混合检索 | `src/retrieval/service.py` | 基于向量库和 Web 的证据召回主入口 |
| 学术搜索聚合 | `src/retrieval/unified_web_search.py` | 聚合 Scholar / Semantic Scholar / NCBI / Web |
| Semantic Scholar 接口 | `src/retrieval/semantic_scholar.py` | 学术搜索、引用补充、前沿论文发现 |
| 网页全文抓取 | `src/retrieval/web_content_fetcher.py` | 搜索结果补全文、证据增强 |
| 论文下载 | `src/retrieval/downloader/paper_downloader_refactored.py` | PDF 获取与下载策略执行 |
| Scholar API 与任务流 | `src/api/routes_scholar.py` | 搜索、建库、下载、推荐、SSE 流 |
| 长任务队列 | `src/tasks/dispatcher.py` `src/tasks/redis_queue.py` | 远程任务执行、事件流、断点续传 |
| 共享 SQL | `src/db/engine.py` `src/db/models.py` | 所有业务主数据统一落库 |
| 文献入库记录 | `src/indexing/paper_store.py` | collection 内 paper 真相源 |
| 文献元数据索引 | `src/indexing/paper_metadata_store.py` | DOI/title/authors/year 统一索引 |
| 实体图能力 | `src/graph/hippo_rag.py` `src/api/routes_graph.py` | 图检索与图展示基础 |
| 引文管理 | `src/collaboration/citation/manager.py` | 引文格式化、证据绑定、输出引用 |
| collection 与 library 绑定 | `CollectionLibraryBinding` | 将“文献搜索候选库”和“知识库 collection”打通 |

---

## 四、运行形态与远程部署约束

这是本方案必须明确写死的工程前提。

### 4.1 调用链路

标准调用形态应为：

```text
Frontend / Agent
    -> FastAPI routes
    -> Retrieval / Scholar / Graph / LLM services
    -> PostgreSQL / Redis / Vector DB / Parsed Data / Downloader
    -> SSE 持续回传进度与增量结果
```

### 4.2 必须满足的远程执行要求

1. 搜索、下载、推荐、图扩展都应由服务端执行。
2. 浏览器自动化、验证码处理、代理访问只在服务端存在。
3. 前端只消费最终结果、增量事件和任务状态。
4. 所有长流程都必须兼容已有 SSE 协议：
   - `task_id`
   - `after_id` / `Last-Event-ID`
   - `heartbeat`
   - `done / error / cancelled`
5. 任何“图对象”“缓存对象”都不能只存在于单 worker 内存中作为唯一真相。

### 4.3 图对象在远程多实例环境下的额外要求

文献引用图、作者图、机构图如果采用 NetworkX，有一个重要边界：

- **NetworkX 只能作为计算缓存，不应作为唯一持久层**

因此推荐策略是：

1. 原始事实数据落 SQL / parsed_data / metadata store
2. NetworkX 子图按需构建或基于快照加载
3. 可选地把图快照序列化到磁盘或对象存储
4. worker 重启后应能从持久层恢复

这点对远程服务器部署非常关键。

---

## 五、推荐的总体架构

### 5.1 分层结构

`ref_tools` 建议作为“学术助手业务层”，位于现有基础设施之上：

```text
API 层
  - 复用 /scholar
  - 增补 /ref 或 /academic-assistant 业务接口

业务服务层
  - ReferenceSearchService
  - ReferenceAssistantService
  - GlobalGraphService
  - LibraryStateService

基础服务层
  - RetrievalService
  - LLMManager
  - PaperDownloader
  - PaperStore / PaperMetadataStore
  - Redis Task Queue / SSE

存储层
  - PostgreSQL / SQLModel
  - Redis
  - Vector DB
  - parsed_data / PDF 文件目录
```

### 5.2 不建议的做法

以下方案不建议继续：

1. 单独新建一个 `ref_tools.db`
2. 在文档方案里把 SQLite 当作业务主库
3. 在 `ref_tools` 内复制一份 Scholar 搜索器、下载器、SSE 管道
4. 把引用图直接做成前端私有状态
5. 用单进程内存图承担跨请求、跨任务的唯一真相

---

## 六、全局通用能力设计

### 6.1 全局 SQL 管理层

当前共享 SQL 已经存在，后续 `ref_tools` 只应在此之上扩展，不再自建新库。

#### 现有可直接复用的表/实体

- `Paper`
- `PaperMetadata`
- `ScholarLibrary`
- `ScholarLibraryPaper`
- `CollectionLibraryBinding`
- 各类 task / job / canvas / citation 相关表

#### 设计要求

1. 文献候选、下载状态、入库映射必须走共享 SQL。
2. collection 与 scholar library 的关系继续复用 `CollectionLibraryBinding`。
3. DOI、标题归一化、元数据补全统一复用 `paper_metadata_store` 与 `dedup.normalize_doi/normalize_title`。
4. 新增表时优先做“通用资源态”设计，而不是只服务单一页面。

#### 推荐新增方向

如果后续要补“精读/收藏/标签/笔记”，建议走**通用资源状态模型**，而不是只给 `ScholarLibraryPaper` 加业务字段。推荐抽象：

- `resource_user_state`
  - `user_id`
  - `resource_type` (`paper`, `library_paper`, `canvas_doc`, ...)
  - `resource_id`
  - `read_status`
  - `favorite`
  - `archived`
- `resource_tag`
  - `user_id`
  - `resource_type`
  - `resource_id`
  - `tag`
- `resource_note`
  - `user_id`
  - `resource_type`
  - `resource_id`
  - `note_md`

这样可以作为项目全局通用能力复用，不会把文献助手做成孤岛。

### 6.2 全局图服务层

建议新增统一图服务，例如：

- `src/services/graph_service.py`
- 或 `src/graph/service.py`

其职责应统一包含：

1. **图类型隔离**
   - 实体图
   - 文献引用图
   - 作者合作图
   - 机构关联图
2. **图构建**
   - 从 `parsed_data` / `PaperMetadata` / `ScholarLibraryPaper` / 外部 API 构建
3. **图查询**
   - n-hop 邻居
   - 子图裁剪
   - Top-K 中心节点
4. **图压缩**
   - 将图结构转换为适合 LLM 的摘要文本
5. **图缓存**
   - snapshot 版本号
   - 热缓存
   - 失效重建

#### 对 `ref_tools` 的直接收益

`ref_tools` 后续做：

- Missing Core 推荐
- Forward Tracking
- 作者合作发现
- 机构聚类
- 研究脉络解释

都能直接调用统一图服务，而不需要在模块内部重复维护图逻辑。

---

## 七、学术助手功能设计

### 7.1 搜索入口

搜索入口优先复用现有 `Scholar API` 与 `RetrievalService`。

#### 推荐分工

1. **候选搜索**
   - 复用 `routes_scholar.py`
   - 来源包括 `google_scholar`、`semantic_relevance`、`semantic_bulk`、`ncbi`、`annas_archive`
2. **混合证据检索**
   - 复用 `RetrievalService.search(mode="local" | "web" | "hybrid")`
3. **全文增强**
   - 对非 PDF 网页命中，复用 `WebContentFetcher`
4. **去重与归一化**
   - 复用 `src/retrieval/dedup.py`
   - 统一 DOI / URL / arXiv / title 规范

### 7.2 搜索结果入库与候选库管理

现有 `ScholarLibrary` / `ScholarLibraryPaper` 已经非常接近“文献候选池”设计，应继续沿用。

推荐闭环：

```text
搜索结果
  -> 加入 ScholarLibrary
  -> 去重 / DOI 补全 / venue 规范化
  -> 下载 PDF
  -> ingest 到 collection
  -> 通过 CollectionLibraryBinding 保持库与知识库联动
```

这意味着 `ref_tools` 不需要重新设计一套“文献列表库”。

### 7.3 下载与入库

下载与入库必须继续走现有 scholar 长任务链路：

- 下载：`paper_downloader_refactored.py`
- 任务分发：`src/tasks/dispatcher.py`
- 状态流与事件流：`routes_scholar.py` + Redis stream

原因：

1. 远程服务器上浏览器下载是高失败率链路，现有模块已经有较多容错
2. SSE、任务取消、心跳、批量下载这些基础设施已存在
3. 下载后自动 ingest 的闭环也已打通

### 7.4 精读与问答

学术助手的精读能力应完全站在现有解析和检索体系上实现。

#### 复用路径

1. `parsed_data` 作为首选数据源
2. `RetrievalService` 召回相关 chunks
3. `LLMManager` 负责统一模型调用
4. `citation.manager` 负责回答中的引文绑定

#### 典型能力

- 结构化总结
- 定制问题回答
- 方法/结论/局限抽取
- 多文献对比
- 思维导图生成（继续坚持 markmap，不引入 Mermaid 作为主路径）

### 7.5 文献发现与学术发现

这是后续增强的核心差异点。

#### 发现链路建议

1. 先从本地 collection / library 中找到语义种子文献
2. 结合本地引用事实、元数据事实构建子图
3. 对缺失节点再去外部 OpenAlex / Semantic Scholar 查询补全
4. 将补全结果以异步任务和 SSE 增量回传

#### 关键说明

原方案里提到 OpenAlex，这个方向可以保留，但接入方式要符合当前项目风格：

- OpenAlex 作为**远程外部元数据补充源**
- 不作为主业务数据库
- 请求结果优先入缓存层，再落共享 SQL 中的标准化结果
- 调用必须在服务端，不能让前端直连

---

## 八、GraphRAG 在本项目中的正确落点

原方案中的 GraphRAG 思路可以保留，但要收敛为“与现有项目兼容的版本”。

### 8.1 当前可直接复用的图能力

- `HippoRAG` 已经具备实体图与图扩展检索基础
- `routes_graph.py` 已有图查询接口雏形

### 8.2 学术助手需要新增的图能力

在不推翻现有结构的前提下，优先补以下图：

1. **文献引用图**
   - 节点：paper
   - 边：cites / cited_by
2. **作者合作图**
   - 节点：author
   - 边：co_author
3. **机构图**
   - 节点：institution
   - 边：affiliated_with / collaborates_with

### 8.3 Graph Prompting 的后端位置

Graph Prompting 不应在前端做，也不应把整个图 JSON 直接喂给模型。

推荐后端流程：

```text
图查询结果
  -> 子图裁剪
  -> 指标提取（degree / pagerank / bridge）
  -> 结构压缩为 Markdown / bullet facts
  -> 注入 LLM prompt
```

### 8.4 当前阶段的现实建议

第一阶段不要追求“大而全学术图谱平台”，优先做：

1. collection 内部引用图
2. library 与 collection 的联动图
3. 基于 DOI/title 的外部引用补全

先把“有用且稳”的子图能力跑通。

---

## 九、缓存、限流与外部 API 治理

### 9.1 缓存策略

缓存要分层，不要混用：

1. **业务主数据**
   - 共享 SQL
2. **短期 API 结果缓存**
   - Redis / DiskCache / TTLCache
3. **图快照缓存**
   - 文件快照或对象存储快照
4. **全文抓取缓存**
   - 复用 `web_content_fetcher` 既有缓存机制

### 9.2 外部 API 调用要求

对 OpenAlex / Semantic Scholar / Crossref 等外部源，统一要求：

1. 服务端调用
2. 带超时
3. 带缓存
4. 带批量化
5. 带限流保护
6. 配额不足时自动降级到本地能力

### 9.3 工程红线

1. 不允许前端直连外部学术 API 作为主链路
2. 不允许逐条 DOI 串行调用可批处理接口
3. 不允许把外部 API 返回 JSON 直接当最终展示模型
4. 不允许模块内私自保存第二份 DOI 真相表

---

## 十、推荐实施路径

### Phase 1：先做“复用式整合”

目标：不新增大块基础设施，只打通现有能力。

- 以 `ScholarLibrary` 作为候选文献池
- 以 `CollectionLibraryBinding` 打通文献库和 collection
- 统一使用 `paper_metadata_store` 做 DOI/title 真相索引
- 文档中所有搜索、下载、入库、推荐链路明确指向现有模块

### Phase 2：补“全局图服务”

目标：把图能力从分散模块中抽到统一层。

- 抽象 `GlobalGraphService`
- 接入 citation graph / author graph / institution graph
- 支持图快照与子图查询
- 给 LLM 输出统一图摘要接口

### Phase 3：补“学术助手能力层”

目标：把搜索升级为可对话、可发现、可精读。

- 单篇精读总结
- 定向问答
- 多篇对比
- Missing Core / Forward Tracking
- 学者与机构发现

### Phase 4：补“通用资源状态层”

目标：把收藏、标签、笔记做成全局能力。

- `resource_user_state`
- `resource_tag`
- `resource_note`

这样后续 Scholar、Canvas、Project、Paper 都能统一复用。

---

## 十一、最终结论

`ref_tools` 的正确方向不是“再做一个独立文献系统”，而是：

1. 以前端调用远程服务的方式，挂接到现有 FastAPI + SSE + Redis + SQL + Retrieval 体系上
2. 以 `ScholarLibrary + CollectionLibraryBinding + PaperStore + PaperMetadataStore` 组成文献主链路
3. 以共享 SQL 作为唯一业务主数据层
4. 以统一图服务承载实体图、引用图、作者图、机构图
5. 以 `LLMManager + RetrievalService + Graph Prompting` 提供学术助手体验

换句话说，后续实现应遵循：

**搜索复用现有 Scholar，检索复用现有 Retrieval，状态复用现有 SQL，长任务复用现有 SSE，图能力抽为全局服务。**

这才是和当前项目最一致、也最适合远程服务器部署的演进路径。
