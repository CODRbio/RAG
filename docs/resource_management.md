# 资源管理文档

> 本文档描述项目所有资源的生命周期与回收策略。

---

## 目录

1. [资源总览](#1-资源总览)
2. [资源依赖关系图](#2-资源依赖关系图)
3. [各资源生命周期详述](#3-各资源生命周期详述)
4. [清理机制汇总](#4-清理机制汇总)
5. [配置参考](#5-配置参考)

---

## 1. 资源总览

| 类型 | 实例 | 位置 |
|------|------|------|
| 关系型数据库 | PostgreSQL，21 张表 | `rag_config.local.json` → `database.url` |
| 向量数据库 | Milvus Collections | Milvus 服务 |
| 文件系统 | PDF、解析中间产物 | `data/users/{user_id}/` |
| 内存 / 队列 | Redis 任务状态、会话内存 | Redis |

---

## 2. 资源依赖关系图

```
User
├── UserProfile
│   └── UserProject（Canvas 删除时显式清理）
├── ChatSession
│   └── Turn（cascade delete）
├── Canvas
│   ├── OutlineSection（cascade delete）
│   ├── DraftBlock（cascade delete）
│   ├── CanvasVersion（cascade delete）
│   ├── CanvasCitation（cascade delete）
│   └── WorkingMemory（Canvas 删除时显式清理）
├── DeepResearchJob
│   ├── DRJobEvent（cascade delete）
│   ├── DRSectionReview（cascade delete）
│   ├── DRResumeQueue（cascade delete）
│   ├── DRGapSupplement（cascade delete）
│   ├── DRInsight（cascade delete）
│   ├── DRCheckpoint（cascade delete）
│   └── Milvus 临时 collection: job_{job_id}（删除 Job 时即时 drop；24h 兜底清理）
├── IngestJob
│   ├── IngestJobEvent（cascade delete）
│   ├── IngestCheckpoint（cascade delete）
│   └── Paper 记录（ingest 副产物）
│       ├── PDF 文件（raw_papers/ 或 libraries/pdfs/）（Paper 删除时同步删除）
│       └── parsed_data/*.json（Paper 删除时同步删除）
├── ScholarLibrary
│   └── ScholarLibraryPaper（cascade delete）
│       └── libraries/{lib_name}/ 整目录（Library 删除时 shutil.rmtree）
├── Collection（Milvus）
│   ├── Milvus vectors（drop_collection 时删除）
│   └── Paper 记录（delete_collection_papers 时删除，含文件）
├── CollectionLibraryBinding（Collection 删除时联动删除）
├── PaperMetadata / CrossrefCache（持久缓存，按需手动清理）
└── RevokedToken（启动时按过期时间清理）

Redis（每个 Task）
└── task_state（TTL 到期自动删除）
    └── task_events stream（TTL 到期自动删除）
```

---

## 3. 各资源生命周期详述

### 3.1 Canvas（画布）

| 项目 | 内容 |
|------|------|
| 创建时机 | Deep Research 启动时自动创建；或通过 Canvas API 手动创建 |
| 关联关系 | 属于 User；ChatSession 可关联 Canvas |
| 子资源 | OutlineSection、DraftBlock、CanvasVersion、CanvasCitation（均 cascade delete）；WorkingMemory、ChatSession、DR Job（显式清理） |
| 自动清理 | 启动时 `_cleanup_by_age_canvas()`，删除 `updated_at < max_age_days` 且未归档的记录 |
| 手动删除 | `DELETE /canvas/{canvas_id}` → 清理 Session、DR Job、Working Memory、UserProject，再删 Canvas |

**代码位置：**
- `src/api/routes_project.py:80-89`（删除联动）
- `src/utils/storage_cleaner.py`（年龄自动清理）

---

### 3.2 ChatSession（对话会话）

| 项目 | 内容 |
|------|------|
| 创建时机 | POST `/chat/stream` |
| 关联关系 | 可关联 Canvas；属于 User |
| 子资源 | Turn（cascade delete） |
| 自动清理 | 启动时 `_cleanup_by_age_sessions()`，删除 `updated_at < max_age_days` 的记录 |
| 手动删除 | `DELETE /sessions/{session_id}` → 删除 Session + Turn |

**代码位置：**
- `src/collaboration/memory/session_memory.py`
- `src/utils/storage_cleaner.py`

---

### 3.3 WorkingMemory（画布工作记忆）

| 项目 | 内容 |
|------|------|
| 存储位置 | PostgreSQL `working_memory` 表，以 `canvas_id` 为主键 |
| 关联关系 | 属于 Canvas |
| 清理方式 | Canvas 删除时显式调用 `delete_working_memory(canvas_id)` |

**代码位置：**
- `src/collaboration/memory/working_memory.py:20`
- `src/api/routes_project.py:86`

---

### 3.4 IngestJob（导入任务）

| 项目 | 内容 |
|------|------|
| 创建时机 | 文件上传、Scholar 下载触发、Deep Research 触发 |
| 关联关系 | 属于 Collection 和 User |
| 子资源 | IngestJobEvent（cascade delete）、IngestCheckpoint（cascade delete） |
| Checkpoint 清理 | 启动时 TTL 7d 清理（含孤立 Checkpoint） |
| 自动清理 | 启动时 `_cleanup_by_age_ingest_jobs()`，清理终态（done/error/cancelled）超过 `max_age_days` 的记录 |

**代码位置：**
- `src/utils/task_runner.py:132-172`（Checkpoint 清理）
- `src/utils/storage_cleaner.py`（Job 年龄清理）

---

### 3.5 Paper（论文记录）

| 项目 | 内容 |
|------|------|
| 创建时机 | Ingest 完成后写入，来源 = upload / scholar / deep_research |
| 关联关系 | 属于 Collection；可关联 ScholarLibrary |
| 文件资源 | `data/users/{user_id}/raw_papers/` 或 `libraries/{lib}/pdfs/` 下的 PDF 及 `parsed_data/*.json` |
| 删除方式 | 随 Collection 删除，或通过 `DELETE /ingest/collections/{name}/papers/{paper_id}` 单独删除 |
| 文件清理 | `_delete_paper_files()` 在删除 DB 记录前同步删除 PDF 及解析中间产物 |
| 前端保护 | 删除单篇文献前要求输入账号密码确认 |

**代码位置：**
- `src/indexing/paper_store.py`（`_delete_paper_files`、`delete_paper`、`delete_collection_papers`）

---

### 3.6 DeepResearchJob（深度研究任务）

| 项目 | 内容 |
|------|------|
| 创建时机 | POST `/deep-research/start` |
| 关联关系 | 关联 Canvas、ChatSession、User |
| 子资源 | DRJobEvent、DRSectionReview、DRResumeQueue、DRGapSupplement、DRInsight、DRCheckpoint（均 cascade delete）；Milvus 临时 collection `job_{job_id}` |
| Checkpoint 清理 | 启动时 TTL 7d 清理 |
| Milvus 清理 | 手动删除时即时 drop；`_cleanup_by_age_dr_jobs()` 年龄清理时同步 drop；`_cleanup_temporary_collections()` 24h 兜底清理 |
| 手动删除 | `DELETE /deep-research/jobs/{job_id}`（仅允许终态：done/error/cancelled/planning） |
| 自动清理 | 启动时 `_cleanup_by_age_dr_jobs()`，清理终态超过 `max_age_days` 的记录及其 Milvus collection |

**代码位置：**
- `src/collaboration/research/job_store.py:delete_job()`（即时 Milvus drop）
- `src/utils/storage_cleaner.py`（年龄清理 + 24h 兜底）
- `src/utils/task_runner.py`（启动修复、Checkpoint 清理）

---

### 3.7 ScholarLibrary（文献库）

| 项目 | 内容 |
|------|------|
| 创建时机 | Scholar API 手动创建，或绑定 Collection 时自动创建 |
| 关联关系 | 属于 User；可通过 CollectionLibraryBinding 与 Collection 绑定 |
| 子资源 | ScholarLibraryPaper（cascade delete） |
| 文件资源 | `data/users/{user_id}/libraries/{library_name}/` 整目录 |
| 手动删除 | `DELETE /scholar/libraries/{lib_id}` → DB commit 后 `shutil.rmtree(folder_path)` |
| 前端保护 | 永久库删除前要求输入账号密码确认（临时库仅弹窗确认） |

**代码位置：**
- `src/api/routes_scholar.py:1256-1280`
- `src/api/routes_scholar.py:2851-2882`（单个 PDF 文件删除）

---

### 3.8 Collection（Milvus 向量集合）

| 项目 | 内容 |
|------|------|
| 创建时机 | 文件上传时创建；Deep Research 创建 `job_{job_id}` 临时 collection |
| 关联关系 | 属于 User；可与 ScholarLibrary 绑定 |
| 子资源 | Milvus vectors；Paper 记录（含 PDF 文件） |
| 手动删除 | `DELETE /ingest/collections/{name}` → drop Milvus collection + 删 Paper 记录及文件 + 删 Binding |
| 临时 collection 清理 | `job_*` 前缀：启动时 24h 兜底清理；DR Job 删除时即时 drop |
| 前端保护 | 删除知识库前要求输入账号密码确认 |

**代码位置：**
- `src/api/routes_ingest.py`（删除端点）
- `src/utils/storage_cleaner.py`（临时 collection 清理）

---

### 3.9 Redis 任务（Task）

| 项目 | 内容 |
|------|------|
| 创建时机 | 任意 API submit（chat / scholar download） |
| 子资源 | `rag:task:{id}` KV state；`rag:task_events:{id}` 事件流 |
| 自动清理 | Redis TTL（`settings.tasks.task_state_ttl_seconds`） |
| 启动修复 | Chat Task >60s → 标记 error 并释放 slot；Scholar Task >300s 同理 |
| Slot 释放 | 任务进入终态时 `release_slot()` 释放 `rag:active_tasks` / `rag:active_sessions` |

**代码位置：**
- `src/tasks/redis_queue.py`（TTL、slot 释放、启动修复）

---

### 3.10 PaperMetadata / CrossrefCache（持久元数据缓存）

| 项目 | 内容 |
|------|------|
| 存储位置 | PostgreSQL `paper_metadata`、`crossref_cache`、`crossref_cache_by_doi` 表 |
| 用途 | 文献 DOI/标题/作者/期刊/URL/PDF 地址等富化信息，用于文献列表展示和去重 |
| 清理策略 | 有价值的持久缓存，**不自动清理**；提供管理员手动清理接口 |
| 管理员接口 | `DELETE /admin/cache/crossref?older_than_days=N`（Crossref 缓存）<br>`DELETE /admin/cache/paper-metadata`（论文元数据全清） |
| 前端入口 | 管理员设置 → 「存储与缓存清理」节 |

**代码位置：**
- `src/api/routes_auth.py`（`admin_clear_crossref_cache`、`admin_clear_paper_metadata_cache`）
- `frontend/src/components/settings/SettingsModal.tsx`（UI）

---

### 3.11 RevokedToken（已吊销令牌）

| 项目 | 内容 |
|------|------|
| 清理时机 | 应用启动时 `purge_expired_revocations()` |
| 清理条件 | 记录的过期时间已过 |

**代码位置：**
- `src/api/server.py`

---

## 4. 清理机制汇总

### 4.1 启动时自动清理

| 机制 | 触发条件 | 覆盖资源 |
|------|----------|----------|
| 年龄清理：Canvas / Session / UserProject | `storage.cleanup_on_startup = true` | 超过 `max_age_days` 的记录 |
| 年龄清理：IngestJob（终态） | 同上 | `ingest_jobs` + cascade events/checkpoints |
| 年龄清理：DeepResearchJob（终态） | 同上 | `deep_research_jobs` + cascade 子表 + Milvus collection |
| 大小清理：超出 `max_size_gb` 时删最旧 Canvas / Session | 同上 | Canvas、ChatSession |
| 临时 Milvus collection 清理（job_* > 24h） | 同上 | Milvus `job_*` collection |
| 恢复中断的 DR Job（标记 error） | 每次启动 | DeepResearchJob |
| Ingest / DR Checkpoint TTL 清理（> 7d） | 每次启动 | IngestCheckpoint、DRCheckpoint |
| Redis stale task 修复 | 每次启动 | Redis task state |
| RevokedToken 过期清理 | 每次启动 | RevokedToken |

### 4.2 显式 API 删除

| API | 删除内容 |
|-----|----------|
| `DELETE /canvas/{id}` | Canvas + 子表 + Session + DR Job + WorkingMemory + UserProject |
| `DELETE /sessions/{id}` | Session + Turn |
| `DELETE /ingest/collections/{name}` | Milvus collection + Paper 记录 + PDF 文件 + Binding |
| `DELETE /ingest/collections/{name}/papers/{paper_id}` | Paper DB 记录 + PDF 文件 + parsed JSON |
| `DELETE /scholar/libraries/{id}` | Library DB 记录 + LibraryPaper + 整个库目录 |
| `DELETE /scholar/libraries/{id}/papers/{record_id}/pdf` | 单个 PDF 文件（Paper 记录保留） |
| `DELETE /deep-research/jobs/{id}` | DR Job + 全部子表 + Milvus collection |
| `DELETE /admin/cache/crossref` | crossref_cache + crossref_cache_by_doi（支持按天龄过滤） |
| `DELETE /admin/cache/paper-metadata` | paper_metadata 全表 |

### 4.3 手动运维脚本

```bash
# 基础清理（30d 年龄，5GB 大小限制）
python scripts/19_cleanup_storage.py

# 自定义参数
python scripts/19_cleanup_storage.py --max-age 7 --max-size 2 --vacuum

# 仅 vacuum（不删数据）
python scripts/19_cleanup_storage.py --vacuum
```

---

## 5. 配置参考

```json
// config/rag_config.json → storage 节点
{
  "storage": {
    "max_age_days": 30,         // 数据保留天数（Canvas / Session / IngestJob / DR Job）
    "max_size_gb": 5.0,         // DB 大小上限，超出时触发额外清理
    "cleanup_on_startup": true, // 启动时是否执行清理
    "cleanup_batch_size": 100   // 每批删除的记录数
  }
}
```

对应代码：`config/settings.py` → `StorageSettings`

---

*更新日期：2026-03-12*
