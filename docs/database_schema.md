# 数据库架构文档

> 本文档描述项目的数据库引擎、连接配置、表结构、各子系统的使用方式及 Schema 管理机制。

---

## 目录

1. [数据库引擎](#1-数据库引擎)
2. [连接配置与优先级](#2-连接配置与优先级)
3. [连接池与引擎参数](#3-连接池与引擎参数)
4. [完整表清单](#4-完整表清单)
5. [各子系统的数据库使用方式](#5-各子系统的数据库使用方式)
6. [Schema 管理机制](#6-schema-管理机制)

---

## 1. 数据库引擎

**PostgreSQL**（唯一支持的引擎）

- 当前地址：`127.0.0.1:5433`（端口 5433，非默认 5432）
- 数据库名：`rag`
- 驱动：`psycopg`（psycopg3）
- ORM：SQLModel（SQLAlchemy 封装）

---

## 2. 连接配置与优先级

配置解析逻辑：`src/db/engine.py:_resolve_db_url()`，优先级从高到低：

```
1. 环境变量 RAG_DATABASE_URL
       ↓ 不存在则
2. config/rag_config.local.json → database.url      ← 当前生效
       ↓ 不存在或无此字段则
3. config/rag_config.json → database.url            ← 模板占位符
       ↓ 均未配置则
4. 抛出 RuntimeError，启动失败并提示配置方法
```

**URL 自动标准化规则（`_normalize_db_url()`）：**

| 输入格式 | 自动转换为 |
|----------|-----------|
| `postgres://...` | `postgresql+psycopg://...` |
| `postgresql://...` | `postgresql+psycopg://...` |
| `postgresql+psycopg://...`（已显式指定驱动） | 不变 |

**配置方法：**

编辑 `config/rag_config.local.json`，设置 `database.url`：

```json
{
  "database": {
    "url": "postgresql+psycopg://用户名:密码@host:端口/数据库名"
  }
}
```

修改后重启应用生效。

---

## 3. 连接池与引擎参数

核心代码：`src/db/engine.py:get_engine()`

引擎为**单例**（全局 `_engine` 变量，首次调用时创建）。

| 参数 | 值 | 说明 |
|------|----|------|
| 连接池 | `QueuePool`（SQLAlchemy 默认） | 支持并发连接 |
| `pool_pre_ping` | `True` | 每次借用连接前检活，防止 stale 连接 |
| `echo` | `False` | 不打印 SQL（可在 config 中开启调试） |

**Session 工厂：**

```python
# FastAPI 依赖注入（src/db/engine.py:get_session）
def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session
```

各 Store 模块也直接调用 `get_engine()` 在内部管理 Session。

---

## 4. 完整表清单

ORM 定义文件：`src/db/models.py`

### 4.1 画布系统（Canvas）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `canvases` | `id` (str UUID) | `user_id`, `topic`, `abstract`, `stage`, `preliminary_knowledge`, `archived` | 研究/写作画布主记录 |
| `outline_sections` | `id` (str UUID) | `canvas_id`, `level`, `order`, `title`, `content` | 大纲结构节点 |
| `draft_blocks` | `id` (str UUID) | `canvas_id`, `section_id`, `content`, `word_count` | 草稿内容块 |
| `canvas_versions` | `id` (str UUID) | `canvas_id`, `version_num`, `snapshot_json` | 版本快照 |
| `canvas_citations` | `id` (str UUID) | `canvas_id`, `paper_id`, `cite_key` | 引用关系 |

关联关系：canvas → 其余4表均为 `cascade="all, delete-orphan"`

### 4.2 会话与对话（Session / Chat）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `sessions` | `session_id` (str) | `canvas_id`, `user_id`, `created_at`, `updated_at` | 对话会话 |
| `turns` | (`session_id`, `turn_index`) 复合 | `role`, `content`, `intent`, `evidence_pack_id`, `citations_json` | 单轮对话，含引用信息 |

关联关系：session → turns `cascade="all, delete-orphan"`

### 4.3 用户与权限

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `user_profiles` | `user_id` (str) | `username`, `password_hash`, `role`, `is_active`, `preferences_json` | 用户账号 |
| `user_projects` | `id` | `user_id`, `canvas_id`, `updated_at` | 用户-画布关联 |
| `revoked_tokens` | `token_hash` (str SHA-256) | `expires_at` | JWT 吊销黑名单 |

### 4.4 工作记忆（Working Memory）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `working_memory` | `id` | `canvas_id`, `summary`, `metadata_json`, `updated_at` | 画布级状态缓存 |

### 4.5 论文与导入（Papers / Ingest）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `papers` | (`collection`, `paper_id`) 复合 | `user_id`, `filename`, `file_path`, `file_size`, `chunk_count`, `library_id` | 已导入论文元数据 |
| `ingest_jobs` | `job_id` (str) | `user_id`, `collection`, `status`, `total_files`, `done_files`, `created_at` | 导入任务主记录 |
| `ingest_job_events` | `id` | `job_id`, `event_type`, `data_json`, `created_at` | 导入事件流（供 SSE） |
| `ingest_checkpoints` | (`job_id`, `paper_id`, `stage`) 复合 | `status`, `error_msg`, `updated_at` | 每文件每阶段断点续传状态 |

关联关系：ingest_job → events/checkpoints `cascade="all, delete-orphan"`

### 4.6 深度研究（Deep Research）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `deep_research_jobs` | `job_id` (str) | `user_id`, `canvas_id`, `session_id`, `status`, `started_at`, `depth` | 深度研究任务主记录 |
| `deep_research_job_events` | `id` | `job_id`, `event_type`, `data_json` | 任务事件流 |
| `deep_research_section_reviews` | `id` | `job_id`, `section_id`, `decision`, `feedback` | 章节审核门控记录 |
| `deep_research_resume_queue` | `id` | `job_id`, `state_json` | 可续跑任务队列 |
| `deep_research_gap_supplements` | `id` | `job_id`, `section_id`, `gap_desc`, `material_json` | 知识空缺与补充材料 |
| `deep_research_insights` | `id` | `job_id`, `section_id`, `content` | 研究洞见记录 |
| `deep_research_checkpoints` | (`job_id`, `phase`) 复合 | `state_json`, `updated_at` | LangGraph 状态断点（可恢复） |

关联关系：deep_research_job → 其余6表均为 `cascade="all, delete-orphan"`

### 4.7 文献库（Scholar Library）

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `scholar_libraries` | `library_id` (str UUID) | `user_id`, `name`, `folder_path`, `created_at` | 用户文献库（命名收藏） |
| `scholar_library_papers` | `record_id` (str UUID) | `library_id`, `paper_id`, `title`, `authors`, `venue`, `normalized_journal_name`, `download_status` | 文献库内论文记录 |
| `collection_library_bindings` | `id` | `user_id`, `collection_name`, `library_id` | 向量库-文献库绑定（1:1） |

关联关系：scholar_library → scholar_library_papers `cascade="all, delete-orphan"`

### 4.8 元数据缓存

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `paper_metadata` | `paper_id` (str) | `doi`, `title`, `authors`, `year`, `source`, `created_at` | 论文元数据（DOI/标题双索引） |
| `crossref_cache` | `normalized_title` (str) | `result_json`, `created_at` | Crossref 按标题查询缓存 |
| `crossref_cache_by_doi` | `normalized_doi` (str) | `result_json`, `created_at` | Crossref 按 DOI 查询缓存 |

### 4.9 全局图与学术助手

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `graph_facts` | `id` | `user_id`, `scope_type`, `scope_key`, `graph_type`, `src_node_id`, `relation_type`, `dst_node_id` | GlobalGraphService 的持久事实边 |
| `graph_snapshots` | `id` | `user_id`, `scope_type`, `scope_key`, `graph_type`, `snapshot_version`, `status`, `storage_path` | 图快照元数据；快照 payload 落文件 |
| `resource_annotations` | `id` | `user_id`, `resource_type`, `resource_id`, `paper_uid`, `target_kind`, `target_locator_json`, `target_text`, `directive`, `status` | Phase 3 通用标注真相表 |
| `resource_user_states` | `id` | `user_id`, `resource_type`, `resource_id`, `favorite`, `archived`, `read_status`, `last_opened_at` | Phase 4 用户态覆盖真相；唯一键是 `(user_id, resource_type, resource_id)` |
| `resource_tags` | `id` | `user_id`, `resource_type`, `resource_id`, `tag`, `normalized_tag` | Phase 4 自由标签；按规范化 tag 去重 |
| `resource_notes` | `id` | `user_id`, `resource_type`, `resource_id`, `note_md`, `updated_at` | Phase 4 资源级 Markdown 笔记，可多条 |

### 4.10 影响因子索引

| 表名 | 主键 | 主要字段 | 说明 |
|------|------|----------|------|
| `impact_factor_journals` | `id` | `journal_name`, `normalized_name`, `impact_factor`, `quartile` | 期刊影响因子数据 |
| `impact_factor_index_meta` | `key` (str) | `value` | 数据源版本/哈希（变更检测） |

---

## 5. 各子系统的数据库使用方式

### 5.1 共用 Session 模式

所有 Store 类通过两种方式获取 DB 会话：

```python
# 方式 A：FastAPI 路由依赖注入
@router.get("/...")
def endpoint(session: Session = Depends(get_session)):
    ...

# 方式 B：Store 内部自管理（更常见）
with Session(get_engine()) as session:
    row = session.get(Model, pk)
    session.add(row)
    session.commit()
```

### 5.2 各 Store 对应的表和文件

| 子系统 | Store 文件 | 操作的表 |
|--------|-----------|---------|
| 画布 | `src/collaboration/canvas/canvas_store.py` | canvases, outline_sections, draft_blocks, canvas_versions, canvas_citations |
| 会话 | `src/collaboration/memory/session_memory.py` | sessions, turns |
| 工作记忆 | `src/collaboration/memory/working_memory.py` | working_memory |
| 用户/项目 | `src/collaboration/memory/persistent_store.py` | user_profiles, user_projects |
| 认证 | `src/auth/session.py` | revoked_tokens |
| 论文 | `src/indexing/paper_store.py` | papers |
| 导入任务 | `src/indexing/ingest_job_store.py` | ingest_jobs, ingest_job_events, ingest_checkpoints |
| 深度研究 | `src/collaboration/research/job_store.py` | deep_research_* 全部7张表 |
| 文献库绑定 | `src/services/collection_library_binding_service.py` | scholar_libraries, scholar_library_papers, collection_library_bindings |
| 论文元数据 | `src/indexing/paper_metadata_store.py` | paper_metadata, crossref_cache, crossref_cache_by_doi |
| 全局图服务 | `src/services/global_graph_service.py` | graph_facts, graph_snapshots |
| 学术助手标注 | `src/indexing/assistant_artifact_store.py` | resource_annotations |
| 通用资源态 | `src/services/resource_state_service.py` | resource_user_states, resource_tags, resource_notes |

### 5.3 认证设计（JWT + 最小 DB 负载）

JWT Token 本身无状态，DB 只存**吊销记录**：

- 登出/吊销时：向 `revoked_tokens` 插入 token SHA-256 哈希 + 过期时间
- 验证时：查 `revoked_tokens` 表，命中则拒绝
- 容错设计：DB 不可用时 token 视为**未吊销**（fail-open，不影响正常请求）
- 清理：应用启动时按 `expires_at` 删除过期记录

### 5.4 元数据缓存（双索引设计）

`paper_metadata_store.py` 使用**单例 + 线程锁**：

```
查询流程：
  title → crossref_cache（normalized_title 主键，O(1)）
  doi   → crossref_cache_by_doi（normalized_doi 主键，O(1)）
  命中  → 返回缓存 JSON
  未命中 → 调用 Crossref API → 写入两张缓存表
```

### 5.5 断点续传（Ingest Checkpoints）

`ingest_checkpoints` 表以 `(job_id, paper_id, stage)` 为复合主键，支持：
- 每文件、每阶段（parse/chunk/embed/index）独立状态
- Upsert 语义：重复提交幂等
- 启动时清理：完成/报错的 Job 对应 Checkpoint 立即删除；孤立 Checkpoint > 7d 则清理

### 5.6 Milvus 统一 paper collection 扩展

Phase 3 没有新开 media / annotation 专用 collection，而是在现有 paper collection v2 上扩展 `content_type`：

- `text`
- `table`
- `image_caption`
- `image_analysis`
- `annotation`

对应动态字段补充：

- `paper_uid`
- `figure_id`
- `image_path`
- `bbox`
- `page`
- `resource_type`
- `annotation_id`

边界说明：

- `enriched.json` 仍是图片解析真相源
- Milvus 中的 `image_analysis` / `annotation` 只是向量影子，用于检索和 rerank
- `resource_user_states / resource_tags / resource_notes` 全部只落 PostgreSQL，不进入 Milvus

### 5.7 通用资源态规范化

Phase 4 对外允许多种资源语义，但落库统一做 canonicalization：

- `project`：API 别名，落库统一写成 `canvas`
- `canvas`：`resource_id = canvas_id`
- `paper`：`resource_id = paper_uid`
- `scholar_library_paper`：`resource_id = str(row.id)`
- `resource_annotation`：只允许 `tag/note`，不进入 `resource_user_states`

---

## 6. Schema 管理机制

### 6.1 启动时 Schema 初始化

位于 `src/api/server.py` lifespan 中：

```
init_db()
  └─ SQLModel.metadata.create_all(engine)  建表（幂等，跳过已有表）
  └─ _ensure_schema_updates()              追加新列（幂等）
```

### 6.2 Alembic 迁移管理

- 迁移脚本目录：`alembic/versions/`（12+ 个版本）
- 配置文件：`alembic.ini`、`alembic/env.py`
- URL 在运行时由 `src/db/engine._resolve_db_url()` 动态解析，`alembic.ini` 中的 `sqlalchemy.url` 仅为占位符

**常用命令：**

```bash
# 升级到最新版本
alembic upgrade head

# 查看当前版本
alembic current

# 查看迁移历史
alembic history --verbose

# 回滚一步
alembic downgrade -1
```

### 6.3 `_ensure_schema_updates()`（幂等列追加）

位于 `src/db/engine.py`，在 `init_db()` 末尾自动运行。
通过 `ALTER TABLE ADD COLUMN` 补充字段，列已存在时静默忽略：

| 表 | 列 | 补充原因 |
|----|----|----|
| `deep_research_jobs` | `started_at` | 任务耗时统计 |
| `canvases` | `preliminary_knowledge` | 预研知识字段 |
| `scholar_libraries` | `folder_path` | 磁盘路径追踪 |
| `scholar_library_papers` | `venue` | 期刊/会议名称 |
| `scholar_library_papers` | `normalized_journal_name` | 影响因子匹配用 |

### 6.4 关于 JSON 列

多处字段（`turns.citations_json`、`canvas.keywords`、`dr_job.state_json` 等）以 `TEXT` 存储 JSON 序列化内容，未使用 PostgreSQL 专属的 `JSONB` 类型。
如未来需要对 JSON 内容做 DB 层过滤查询，可按需迁移为 `JSONB` 列。

---

*核心文件：`src/db/engine.py`、`src/db/models.py`、`alembic/`*
*更新日期：2026-03-12*
