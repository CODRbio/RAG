# 文献搜索与学术助手增强方案

**版本**：V4.1
**日期**：2026-03-14
**定位**：在现有 RAG 项目基础上，增强 Scholar / 文献发现 / 学术助手能力，而不是平行再造一套新系统。

---

## 零、全局唯一论文标识符规范（paper_uid）

> **此章节为所有后续 Phase 计划的前置约束，任何新计划必须遵守。**

### 0.1 为什么需要 paper_uid

Phase 1 要求以 `paper_metadata_store` 作为 DOI/title 真相索引，以 `ScholarLibrary` 作为候选文献池，以 `CollectionLibraryBinding` 打通库与 collection。打通这些模块的前提是：**每篇论文在所有模块间必须有同一个稳定、唯一的标识符**。

当前痛点：没有 DOI 的论文缺乏此标识符，各模块各自为政生成不一致的 ID，导致跨模块 join 失败。

### 0.2 paper_uid 规则（优先级链）

`paper_uid` 是**论文作为学术作品的跨系统稳定标识符**，由以下优先级链确定性生成：

| 优先级 | 条件 | 格式 | 示例 |
| --- | --- | --- | --- |
| 1 | `normalized_doi` 非空 | `doi:{normalized_doi}` | `doi:10.1038/s41586-023-06345-3` |
| 2 | URL 或 title 中含 arXiv ID | `arxiv:{yymm.nnnnn}`（去版本号后缀） | `arxiv:2306.01234` |
| 3 | PMID 已知 | `pmid:{pmid}` | `pmid:36721000` |
| 4 | 以上均无 | `sha:{sha256(norm_title + "\|" + first_author_last_lower + "\|" + str(year or ""))[:16]}` | `sha:a3f2b1c4d5e6f789` |

**不变性保证：**
- 同一论文任何时候调用结果相同（确定性，无随机）
- 一旦 DOI 被后续补全，必须重新计算 `paper_uid` 并更新所有引用（sha → doi 升级）
- `normalized_doi` 始终通过 `dedup.normalize_doi()` 归一化，保证全局一致

### 0.3 概念边界

`paper_uid` 不替代以下标识符，这些标识符保留现状：

| 标识符 | 作用域 | 来源 | 是否被 paper_uid 替代 |
| --- | --- | --- | --- |
| `paper_id` | 单个 collection 内文件 | 文件名 stem（`pdf_path.stem`） | ❌ 不替代 |
| `cite_key` | 单次对话/引文列表 | `CiteKeyGenerator`（用户可见） | ❌ 不替代 |
| chunk fingerprint | chunk-level 内容去重 | `MD5(text)`，`dedup._fingerprint()` | ❌ 不替代 |

### 0.4 唯一权威实现

`paper_uid` 的计算**只能通过一个函数**：

```python
from src.retrieval.dedup import compute_paper_uid
```

### 0.5 工程红线

1. **任何模块不得私自实现论文级 ID 生成逻辑**，必须统一调用 `compute_paper_uid()`
2. **任何模块不得绕过 `compute_paper_uid()` 构造跨模块论文引用键**
3. **不允许模块内保存第二份论文去重表**（违反 ref_tools 第九节第 4 条工程红线）
4. 后续所有新增模块、新增 API、新增服务，在需要标识一篇论文时，必须调用此函数

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
  - `last_opened_at`（可后续补充）
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

需要特别明确：`resource_user_state` 是**用户态覆盖层**，不是业务主状态真相源。

- `ScholarLibraryPaper`、`Paper`、未来的 `graph_snapshots` 等领域表负责主状态
- `resource_user_state` 只承载：
  - `read_status`
  - `favorite`
  - `archived`
  - `last_opened_at`（若后续补）
- `resource_tag` / `resource_note` 跟随资源硬删除而清理，归档时默认保留
- 不允许在 `ScholarLibraryPaper`、`Paper`、Graph 事实/快照表里重复加 `favorite` / `tag` / `note` 这类用户态字段

#### 通用资源生命周期约束

`ref_tools` 后续新增的文献资源、图资源、用户态资源，都必须继承项目统一资源治理规则，参考：

- `docs/resource_management.md`

这里必须先写死几个约束，避免后续每个模块各自定义生命周期：

1. **归档不等于删除**
   - `archived` 表示“用户保留，但默认不参与常规列表与推荐”
   - 归档不触发文件、图事实、图快照、binding、note/tag 清理
   - 只有硬删除才触发联动资源清理
2. **归档资源默认不参加启动时 age cleanup**
   - 语义对齐现有 Canvas 规则
   - 后续 `paper` / `library_paper` / `graph_report` 等资源若支持归档，均沿用此语义
3. **TTL 只用于衍生资源，不用于业务主表**
   - Redis task state / task events
   - 临时 collection
   - checkpoint
   - 可重建的图快照缓存
   - 不包括 `Paper`、`ScholarLibraryPaper`、`PaperMetadata`、图事实表
4. **删除必须做联动失效**
   - 删除 `Paper` / collection / library / binding 时，相关图边、scope 快照、查询缓存都必须失效或重算
   - 不允许删除源资源后残留 scope 级图快照继续对外服务

#### 学术资源状态机

为了让 Scholar / Collection / Graph / 后续资源态层能对齐，文献资源推荐统一采用以下业务状态流转：

```text
discovered
  -> saved
  -> downloaded
  -> ingested
  -> graph_ready
  -> archived
  -> deleted
```

各状态语义如下：

- `discovered`
  - 仅搜索结果、临时候选或推荐结果
  - 尚未进入永久 `ScholarLibraryPaper`
- `saved`
  - 已进入 `ScholarLibraryPaper`
  - 但本地 PDF 还未就绪
- `downloaded`
  - PDF 已落到 library 目录
  - 尚未进入 `Paper` / collection
- `ingested`
  - 已进入 `Paper` / collection
  - 可以参与本地检索、绑定、后续图扩展
- `graph_ready`
  - 已被图事实层吸收
  - 可参与 citation / author / institution graph 查询
- `archived`
  - 用户显式归档
  - 业务上默认隐藏，但不删除 PDF、图事实、图快照、tag、note
- `deleted`
  - 硬删除
  - 触发文件、绑定、图快照、查询缓存等联动清理

该状态机是**概念层统一约束**，不要求当前阶段马上在单表里存一个 `status` 字段；当前实现可继续由多个领域表共同体现：

- `ScholarLibraryPaper` 体现 `saved`
- library 目录中的 PDF 体现 `downloaded`
- `Paper` 体现 `ingested`
- future `graph_facts` / `graph_snapshots` 体现 `graph_ready`

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

#### 生命周期要求

`GlobalGraphService` 不只是查询接口，还必须承担图资源生命周期治理：

1. **图事实**
   - 属于持久业务资源
   - 落共享 SQL
   - 不做 TTL 自动清理
2. **图快照**
   - 属于可重建衍生资源
   - 可以按 dirty/version 回收
   - 但不允许作为唯一真相
3. **快照状态机**
   - `building -> ready -> stale -> rebuilding -> ready`
   - 失败时记录 `error`，但不应立即删除旧的 `ready` 快照
4. **失效触发条件**
   - library ingest
   - library 删除 / 单篇删除
   - collection drop
   - `PaperMetadata` 补全升级
   - `paper_uid` 升级或重算
   - `CollectionLibraryBinding` 解除或改绑
5. **失效策略**
   - 先将相关 scope 标记为 `stale`
   - 由懒重建或后台任务重建
   - 删除源资源时，必须保证旧 scope 快照不再继续命中

#### 删除与归档约束

对 citation / author / institution 图，还需要明确以下边界：

1. 删除 `ScholarLibraryPaper` 时，不应直接删除 `PaperMetadata`
2. 删除 `Paper` / collection 时，相关图边与局部图快照必须失效
3. 删除 `ScholarLibrary` 或 `CollectionLibraryBinding` 时，对应 scope 的 graph snapshot 与 graph facts 需要按 scope 清理或重算
4. 归档 `paper` / `library_paper` 时，默认不删除图事实；查询结果是否隐藏由用户态过滤决定

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

其中 `LLMManager` 的缓存要单独看待：

- provider 原生 prompt/context cache 与应用层 response cache 不是一回事
- 统一行为、配置边界与 usage 口径以 `docs/configuration.md` 为准
- rollout / 监控 / 回滚以 `docs/operations_and_troubleshooting.md` 为准

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
- 明确图快照生命周期、失效策略与删除联动

### Phase 3：补“学术助手能力层”

目标：把搜索升级为可对话、可发现、可精读。

- 单篇精读总结
- 定向问答
- 多篇对比
- Missing Core / Forward Tracking
- 学者与机构发现
- 图片解析回填与图像证据利用
- 论文标注统一入 SQL，并按 paper-linked 规则增量写入向量库

#### Phase 3 存储补充

Phase 3 不再把图片解析和标注停留在“解析细节”层，新增的存储结构与边界固定为：

1. 新增 `resource_annotations` 作为唯一标注真相表，字段固定包含：
   - `id`
   - `user_id`
   - `resource_type`
   - `resource_id`
   - `paper_uid`
   - `target_kind`
   - `target_locator_json`
   - `target_text`
   - `directive`
   - `status`
   - `created_at`
   - `updated_at`
2. 图片解析结果仍写回 `enriched.json`，不新增 `paper_media` 之类第二真相表：
   - `figure_data`
   - `interpretation`
   - `ocr_text`
3. Milvus 不新开 collection，继续复用当前 paper collection，只扩展 `content_type`：
   - `image_caption`
   - `image_analysis`
   - `annotation`
4. 动态字段统一补齐：
   - `paper_uid`
   - `figure_id`
   - `image_path`
   - `bbox`
   - `page`
   - `resource_type`
   - `annotation_id`
5. 标注向量化边界写死：
   - 仅 `paper-linked` annotation 进入 Milvus
   - 纯全局 Canvas 批注只保留 SQL，不向量化

#### Phase 3 服务与接口补充

本阶段统一由 `ReferenceAssistantService` 编排，外部入口统一收口到 `/academic-assistant`：

- `POST /academic-assistant/papers/summary`
- `POST /academic-assistant/papers/qa`
- `POST /academic-assistant/papers/compare`
- `POST /academic-assistant/discovery/{mode}/start`
- `POST /academic-assistant/media-analysis/start`
- `POST /academic-assistant/annotations`
- `GET /academic-assistant/annotations`
- `GET /academic-assistant/task/{task_id}`
- `GET /academic-assistant/task/{task_id}/stream`

#### Phase 3 媒体解析执行规则

1. 图片解析默认不作为所有 ingest 的必跑步骤
2. 主路径是手动触发的后台任务或工具调用
3. 成功后同步完成两件事：
   - 回写 `enriched.json`
   - 增量 upsert Milvus 中的 `image_caption` / `image_analysis` / `annotation`
4. 图片证据可作为 assistant 的正式 evidence，允许返回 `page / bbox / figure_id` anchors
5. annotation 参与检索与 rerank，但不作为 bibliographic citation 输出

### Phase 4：补“通用资源状态层”

目标：把收藏、标签、笔记做成全局能力。

- `resource_user_state`
- `resource_tag`
- `resource_note`
- 统一承载 `archived` / `favorite` / `read_status` 等用户态覆盖字段

这样后续 Scholar、Canvas、Project、Paper、Graph Report 都能统一复用。

#### Phase 4 落地约束

本阶段实际实现时，固定采用以下策略：

1. 新增三张共享 SQL 表：
   - `resource_user_states`
   - `resource_tags`
   - `resource_notes`
2. `project` 只作为 API 语义别名，对外允许传入，但落库统一规范成 `canvas`
3. `paper` 的用户态按逻辑论文 `paper_uid` 承载，而不是按 collection 内某个 `Paper.id` 承载
4. `resource_annotations` 不接入 `resource_user_state`，避免与它已有的 `status` 冲突
   - annotation 只复用 `tag` / `note`
   - annotation 的主状态继续由 `resource_annotations.status` 负责
5. 现有 `Canvas.archived` 只保留兼容用途
   - 迁移时回填到 `resource_user_states(resource_type='canvas')`
   - `/projects/{canvas_id}/archive` / `/unarchive` 内部改走统一资源态服务，并保持对旧字段双写
6. 删除联动继续走应用层 cascade
   - 删除 `canvas` / `paper` / `scholar_library_paper` / `resource_annotation` 时，必须同步清理对应 `user_state` / `tag` / `note`

需要注意实施顺序：

- Phase 2 就要先把资源生命周期约束写死，尤其是 graph snapshot 的状态机、失效策略和删除联动
- Phase 4 才正式落 `resource_user_state` / `resource_tag` / `resource_note` 表
- 也就是说，**Phase 4 是建模落地阶段，不是生命周期首次定义阶段**

### Phase 5：前端对接与产品化落地

目标：把前四阶段的后端能力收口到现有前端主入口，形成连续可用的 Scholar 工作流，而不是再长出新的孤立页面。

- 在 `ScholarPage` 内接入单篇精读、定向问答、图片解析、annotation、resource state/tag/note
- 在现有 `Graph` 标签页内升级 typed graph，而不是新开独立图工作台
- 在 `TaskCenter` 中纳入 `academic_assistant` 长任务
- 保持 `Canvas` / `Chat` 主编辑流不被 Phase 5 第一轮打散

#### Phase 5 前端入口约束

第一轮前端对接固定遵循以下边界：

1. 主入口是 `ScholarPage`
   - 搜索结果卡片和 library 列表直接挂接 `favorite` / `archived` / `tag` / `note`
   - 单篇论文在右侧 assistant 面板内完成：
     - 单篇精读总结
     - 定向问答
     - media analysis 触发与状态展示
     - annotation 查看与编辑
   - 多篇论文在同一 assistant 面板内完成：
     - 多篇对比
     - Missing Core
     - Forward Tracking
     - 学者发现
     - 机构发现
2. typed graph 继续复用现有 `GraphExplorer`
   - `entity` 保持现有 HippoRAG 实体图体验
   - `citation` / `author` / `institution` 通过图类型切换进入 typed graph
   - 第一轮只要求子图查询、scope 选择、种子输入、摘要面板，不强上复杂分析 UI
3. `TaskCenter` 必须识别 `academic_assistant`
   - 包括 `media-analysis`
   - 包括 discovery 系列任务
4. 不新建独立 assistant workspace
5. 不提前做 Graph Report 前端
6. 不把 `resource_notes` 和 `resource_annotations` 混为同一类 UI
   - `resource_note` 是资源级笔记
   - `resource_annotation` 是锚点级批注

#### Phase 5 API 消费约束

前端对接时，统一通过已有后端入口消费，不允许在前端层自行拼装第二套业务链路：

1. 学术助手统一走 `/academic-assistant`
   - `papers/summary`
   - `papers/qa`
   - `papers/compare`
   - `discovery/{mode}/start`
   - `media-analysis/start`
   - `annotations`
   - `task/{task_id}`
   - `task/{task_id}/stream`
2. 通用资源态统一走 `/resources`
   - `state`
   - `tags`
   - `notes`
3. typed graph 统一走 `/graph/{graph_type}/...`
   - `stats`
   - `subgraph`
   - `summary`
   - `snapshots`
   - `snapshots/rebuild`

#### Phase 5 前端状态层约束

前端状态管理不允许把 scholar、assistant、resource overlay 全塞回一个 store。推荐并实际采用以下分层：

1. `useScholarStore`
   - 继续只负责 scholar 搜索、下载、文献库、批量操作
   - 不承载 summary / QA / note/tag 结果真相
2. `useAcademicAssistantStore`
   - 承接 summary / QA / compare / discovery / media-analysis / annotation 相关结果与任务状态
3. `useResourceStore`
   - 承接 `resource_user_state` / `resource_tag` / `resource_note`
   - 提供按 `resource_type + resource_id` 的统一读取入口

#### Phase 5 产品化红线

1. 不允许在 `ScholarPage` 内直接绕过 store 到处散写 `fetch('/academic-assistant/...')`
2. 不允许 typed graph 与 entity graph 在前端维护两套完全独立的图容器与状态语义
3. 不允许把 `project` 当作真正新的资源类型落到前端持久化键上
   - `project` 只是 `canvas` 的语义别名
4. 不允许前端直接调用外部学术 API、图补全 API、图片解析 API
5. 不允许把长任务结果只保存在页面局部状态中
   - 必须能通过 `task_id` 恢复
6. 不允许因为接入新能力而破坏原有 `Scholar` 搜索、下载、推荐、建库主链
7. 不允许因为接入 typed graph 而回归现有 `entity` 图体验

### 10.1 实现红线补充

除前文工程红线外，`ref_tools` 相关实现还必须遵守以下约束：

1. 不允许在 `ScholarLibraryPaper`、`Paper`、Graph 事实/快照表里重复加 `favorite` / `tag` / `note` 这类用户态字段
2. 不允许图快照成为唯一真相源
3. 不允许 archived 资源被启动清理误删
4. 不允许删除资源后残留 scope 级快照继续对外服务
5. 不允许为图片解析再单独新增第二套 media 真相表
6. 不允许 paper-linked annotation 只存在于向量库而不落 SQL
7. 不允许 annotation 被当成正式文献 citation 直接输出

---

### 10.2 Discovery Mode 命名规范

**后端 canonical 枚举**（唯一真相源：`src/services/reference_assistant_service.py::ReferenceAssistantService.discover()`）：

| 枚举值 | 说明 |
| --- | --- |
| `missing_core` | Missing Core 引用发现（citation 子图） |
| `forward_tracking` | Forward Tracking 被引发现（OpenAlex cited_by API） |
| `experts` | 学者发现（author 子图） |
| `institutions` | 机构发现（institution 子图） |

**前端 TS 类型现状**（`frontend/src/types/index.ts::DiscoveryMode`）：

```typescript
export type DiscoveryMode = 'missing-core' | 'forward-tracking' | 'experts' | 'institutions';
```

前端使用 kebab-case，与后端 `discover()` 内部 `if mode == "missing_core"` 逐一比较的写法不一致。`/discovery/{mode}/start` 的 `mode` 路径参数会原样转发给 `service.discover()`，导致 `missing-core` / `forward-tracking` 在当前版本无法命中。

**已实现的规范化层**：

- 路由层（`start_discovery`）在转发给服务前执行 `mode = mode.replace("-", "_")`，兼容前端 kebab 写法。
- `ReferenceAssistantService.discover()` 入口再次规范化（`strip/lower/replace("-","_")`），提供工具链与直接调用的双重保护。
- `discover_academic_resources` 工具（`src/llm/tools.py`）同样执行入口规范化，LLM 发起的工具调用与 HTTP 路径行为一致。
- `discover()` 返回的 `mode` 字段始终为 canonical underscore 值（`missing_core/forward_tracking/experts/institutions`），而非底层图类型。

---

### 10.3 Graph API 路由边界

当前路由层分两段，消费方必须明确区分：

#### entity 图（HippoRAG 专属路由）

| 路由 | 说明 |
| --- | --- |
| `GET /graph/stats` | HippoRAG 全图统计 |
| `GET /graph/entities` | 实体列表，支持 type/q 过滤 |
| `GET /graph/neighbors/{entity_name}` | n-hop 邻居子图 |
| `GET /graph/chunk/{chunk_id}` | chunk 详情（含 Milvus + parsed 回退） |
| `GET /graph/pdf/{paper_id}` | PDF 文件流，用于前端高亮溯源 |

上述路由直接调用 `HippoRAG`，**不经过 `GlobalGraphService`**，也没有 scope / snapshot 语义。

#### typed graph（GlobalGraphService 路由）

| 路由 | 适用 graph_type | 说明 |
| --- | --- | --- |
| `GET /graph/{graph_type}/stats` | citation / author / institution（entity 也可但只返回 HippoRAG 文件 mtime） | 快照元信息 + 事实统计 |
| `POST /graph/{graph_type}/subgraph` | citation / author / institution | 子图查询，懒触发快照重建 |
| `POST /graph/{graph_type}/summary` | citation / author / institution | 图摘要，供 LLM Prompt 使用 |
| `GET /graph/{graph_type}/snapshots` | citation / author / institution | 快照版本列表 |
| `POST /graph/{graph_type}/snapshots/rebuild` | citation / author / institution | 强制重建快照 |

**关键约束**：

1. `VALID_GRAPH_TYPES = {"entity", "citation", "author", "institution"}`——后端语法上接受 `entity`，但 `entity` 走的是 `EntityGraphAdapter`，`ensure_snapshot` 只返回 HippoRAG 文件 mtime，不做事实重建，`rebuild` 接口对 `entity` 无实质意义。
5. `query_subgraph` 中若显式传入 `snapshot_version` 但该版本不存在或未处于 ready 状态，服务端抛出 `ValueError`，路由层返回 HTTP 400；不传版本时仍回退到最新 ready 快照。
2. **前端已明确约束**：`getTypedGraphStats / queryTypedGraphSubgraph / summarizeTypedGraph / listTypedGraphSnapshots / rebuildTypedGraphSnapshot` 均使用 `Exclude<GraphType, 'entity'>` 类型签名，禁止前端通过 typed graph API 消费 entity 图。
3. `GraphSnapshotItem` 快照字段为 `built_from_revision`（与后端 `_snapshot_row_to_dict()` 一致），旧的 `source_revision` 保留为 `@deprecated` 兼容别名。
3. 前端消费 entity 图只走三条专属路由：`/graph/stats`、`/graph/entities`、`/graph/neighbors/{entity_name}`，不使用 `/graph/entity/...` 形式。
4. typed graph 的所有 POST 端点需要登录（`Depends(get_current_user_id)`），entity 图的 GET 端点 `/graph/chunk/{chunk_id}` 和 `/graph/pdf/{paper_id}` 接受可选登录（`get_optional_user_id`）。

---

### 10.4 Resource Overlay Capability 矩阵

`/resources` 接口的 `resource_type × capability` 支持关系（来源：`src/services/resource_state_service.py`）：

| resource_type | `user_state`（GET/PATCH `/resources/state`） | `tag`（GET/POST/DELETE `/resources/tags`） | `note`（GET/POST/PATCH/DELETE `/resources/notes`） |
| --- | --- | --- | --- |
| `canvas` | ✅ | ✅ | ✅ |
| `project`（`canvas` 的语义别名，落库统一为 `canvas`） | ✅ | ✅ | ✅ |
| `paper` | ✅ | ✅ | ✅ |
| `scholar_library_paper` | ✅ | ✅ | ✅ |
| `resource_annotation` | ❌ 不支持（自身已有 `status` 字段） | ✅ | ✅ |

**resource_id 语义约定**：

| resource_type | resource_id 使用的字段 | 类型 |
| --- | --- | --- |
| `canvas` / `project` | `Canvas.id`（UUID 字符串） | `str` |
| `paper` | `Paper.paper_uid`（跨 collection 逻辑身份） | `str` |
| `scholar_library_paper` | `ScholarLibraryPaper.id`（整数的字符串表示） | `str（"123"）` |
| `resource_annotation` | `ResourceAnnotation.id`（整数的字符串表示） | `str（"456"）` |

注意：`paper` 的 `resource_id` 不是 collection 内的 `paper_id`，而是全局唯一的 `paper_uid`，必须通过 `compute_paper_uid()` 生成。

**删除联动约束**：

删除 `canvas` / `paper` / `scholar_library_paper` / `resource_annotation` 时，应用层必须同时：
1. 调用 `get_resource_state_service().delete_resource_overlays()` 清理 user_state / tag / note
2. 调用 `delete_resource_annotations_for_resource()` 清理 `ResourceAnnotation`

`delete_project` 已实现以上两步联动。不允许依赖数据库级联来完成用户态数据清理。

`read_status` 合法值：`unread` / `reading` / `read`（传入其他值后端返回 400）。

---

### 10.5 Academic-Assistant 任务 SSE 契约

**SSE 端点**：`GET /academic-assistant/task/{task_id}/stream`

**重连机制**（与现有 scholar/chat SSE 协议一致）：

- 请求头 `Last-Event-ID`，或 query param `after_id`（优先取请求头）
- 初始值传 `-`，表示从第一条事件开始消费
- 每次事件回传格式为标准 SSE（`id: / event: / data:`），消费方应在断线重连时把已收到的最后一个 `id` 传回

**事件类型**：

| event | data 含义 | 是否终态 |
| --- | --- | --- |
| `progress` | `{stage: string, task_type: string, elapsed_s?: number}` | 否 |
| `heartbeat` | `{task_type: string, elapsed_s: number}` | 否 |
| `done` | 任务结果 payload（与 `GET /academic-assistant/task/{task_id}` 的 `payload.result` 一致） | ✅ 终态 |
| `error` | `{message: string}` | ✅ 终态 |
| `cancelled` | `{status: "cancelled"}` | ✅ 终态 |
| `timeout` | `{status: "timeout"}` | ✅ 终态 |

心跳间隔：5 秒（`_ASSISTANT_SSE_HEARTBEAT_INTERVAL = 5`）。

**SSE 终态兜底规则**：当任务已处于终态但事件队列中无 canonical 终态事件时，服务端合成以下事件：`completed` → `event: done`（data 为 `payload.result`），`error/cancelled/timeout` → 同名事件。前端 `terminalEvents` 列表额外包含 `completed` 作为过渡兜底，避免重连循环。

**任务种类与用户隔离校验**：

- stream 端点和 status 端点均校验 `task.kind == TaskKind.academic_assistant`，其他 kind 返回 400。
- 两个端点同时校验 `task.user_id == current_user_id`，跨用户访问返回 403（不可绕过）。
- `TaskKind` 枚举：`chat / dr / scholar / academic_assistant`（来源：`src/tasks/task_state.py`）。

**任务状态机**（`TaskStatus`）：

```text
submitted → running → completed
                    → error
                    → cancelled
                    → timeout
```

`AcademicAssistantTaskState.status` 前端类型包含全部 5 个终态值：`submitted / running / completed / error / cancelled / timeout`。

收到终态事件时，store 从持久化任务 ID 列表中移除；若 SSE 流意外断开，store 会先调用 `loadTask(task_id)` 恢复真实后端状态，仅在后端返回终态时才移除持久化 ID（避免网络抖动误删仍在运行的任务）。

---

## 十一、最终结论

`ref_tools` 的正确方向不是“再做一个独立文献系统”，而是：

1. 以前端调用远程服务的方式，挂接到现有 FastAPI + SSE + Redis + SQL + Retrieval 体系上
2. 以 `ScholarLibrary + CollectionLibraryBinding + PaperStore + PaperMetadataStore` 组成文献主链路
3. 以共享 SQL 作为唯一业务主数据层
4. 以统一图服务承载实体图、引用图、作者图、机构图
5. 以 `LLMManager + RetrievalService + Graph Prompting` 提供学术助手体验
6. 以前端现有 `ScholarPage + GraphExplorer + TaskCenter` 作为产品化入口，而不是平行再造第二套页面体系

换句话说，后续实现应遵循：

**搜索复用现有 Scholar，检索复用现有 Retrieval，状态复用现有 SQL，长任务复用现有 SSE，图能力抽为全局服务，前端继续以现有主入口承载产品化能力。**

这才是和当前项目最一致、也最适合远程服务器部署的演进路径。
