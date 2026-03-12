# 资源管理文档

> 本文档描述本项目所有资源的生命周期、现有回收策略及已知缺口。
> 目的：明确哪些资源有完整的回收链路，哪些存在泄漏风险，以及应当如何修复。

---

## 目录

1. [资源总览](#1-资源总览)
2. [资源依赖关系图](#2-资源依赖关系图)
3. [各资源生命周期详述](#3-各资源生命周期详述)
4. [现有清理机制汇总](#4-现有清理机制汇总)
5. [已知缺口（资源泄漏）](#5-已知缺口资源泄漏)
6. [配置参考](#6-配置参考)

---

## 1. 资源总览

本项目涉及四类存储资源：

| 类型 | 实例 | 位置 |
|------|------|------|
| 关系型数据库 | SQLite (rag.db)，21 张表 | `data/rag.db` |
| 向量数据库 | Milvus Collections | Milvus 服务 |
| 文件系统 | PDF、解析中间产物 | `data/users/{user_id}/` |
| 内存 / 队列 | Redis 任务状态、会话内存 | Redis |

---

## 2. 资源依赖关系图

```
User
├── UserProfile
│   └── UserProject (cascade delete)
├── ChatSession
│   └── Turn (cascade delete)
├── Canvas
│   ├── OutlineSection (cascade delete)
│   ├── DraftBlock (cascade delete)
│   ├── CanvasVersion (cascade delete)
│   └── CanvasCitation (cascade delete)
├── DeepResearchJob
│   ├── DRJobEvent (cascade delete)
│   ├── DRSectionReview (cascade delete)
│   ├── DRResumeQueue (cascade delete)
│   ├── DRGapSupplement (cascade delete)
│   ├── DRInsight (cascade delete)
│   ├── DRCheckpoint (cascade delete)
│   └── Milvus 临时 collection: job_{job_id}  ⚠️ 仅 24h 后自动清理
├── IngestJob
│   ├── IngestJobEvent (cascade delete)
│   ├── IngestCheckpoint (cascade delete，启动时 TTL 清理)
│   └── Paper 记录（ingest 副产物）
│       └── PDF 文件（raw_papers/ 或 libraries/pdfs/）  ⚠️ 不随 Paper 删除
├── ScholarLibrary
│   └── ScholarLibraryPaper (cascade delete)
│       └── PDF 文件（libraries/{lib_name}/pdfs/）  ⚠️ 不随 Library 删除
├── Collection（Milvus）
│   ├── Milvus vectors（drop_collection 时删除）
│   └── Paper 记录（delete_collection_papers 时删除）
├── CollectionLibraryBinding（可选联动删除 ScholarLibrary）
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
| 子资源 | OutlineSection、DraftBlock、CanvasVersion、CanvasCitation（均为 cascade delete） |
| 自动清理 | **启动时**：`_cleanup_by_age_canvas()`，删除 `updated_at < 30d` 且未归档的记录 |
| 手动删除 | `DELETE /canvas/{canvas_id}` → `CanvasStore.delete(canvas_id)` |
| 删除联动 | 触发 `SessionStore.delete_sessions_by_canvas_id()` 清理关联 ChatSession |
| 文件清理 | N/A（无文件资源） |
| 状态 | ✅ 完整 |

**代码位置：**
- `src/collaboration/canvas/canvas_manager.py:29-30`
- `src/utils/storage_cleaner.py:35-50`
- `src/collaboration/memory/session_memory.py:386-396`

---

### 3.2 ChatSession（对话会话）

| 项目 | 内容 |
|------|------|
| 创建时机 | POST `/chat/stream` |
| 关联关系 | 可关联 Canvas；属于 User |
| 子资源 | Turn（cascade delete） |
| 自动清理 | **启动时**：`_cleanup_by_age_sessions()`，删除 `updated_at < 30d` 的记录 |
| 手动删除 | `DELETE /sessions/{session_id}` → `SessionStore.delete_session()` |
| 文件清理 | N/A |
| 状态 | ✅ 完整 |

**代码位置：**
- `src/collaboration/memory/session_memory.py:374-396`
- `src/utils/storage_cleaner.py:53-67`

---

### 3.3 IngestJob（导入任务）

| 项目 | 内容 |
|------|------|
| 创建时机 | 文件上传、Scholar 下载触发、Deep Research 触发 |
| 关联关系 | 属于 Collection 和 User |
| 子资源 | IngestJobEvent（cascade delete）、IngestCheckpoint（cascade delete）、Paper 记录（副产物） |
| 自动清理 | 无 Job 记录级别的自动清理；Checkpoint 启动时 TTL 7d 清理 |
| 手动删除 | 无专用 API（⚠️ 缺失） |
| 文件清理 | ⚠️ PDF 文件不随 Job/Paper 删除 |
| 状态 | ⚠️ Job 记录永久堆积；文件不清理 |

**代码位置：**
- `src/utils/task_runner.py:132-149`（checkpoint 清理）
- `src/api/routes_ingest.py`（Job 创建，无删除端点）

---

### 3.4 Paper（论文记录）

| 项目 | 内容 |
|------|------|
| 创建时机 | Ingest 完成后写入，来源 = upload / scholar / deep_research |
| 关联关系 | 属于 Collection；可关联 ScholarLibrary |
| 文件资源 | `data/users/{user_id}/raw_papers/` 或 `libraries/{lib}/pdfs/` |
| 删除方式 | 随 Collection 删除（`delete_collection_papers(collection_name)`） |
| 文件清理 | ⚠️ Paper 记录删除时 PDF 文件**不**同步删除 |
| 状态 | ⚠️ PDF 文件孤立残留 |

**代码位置：**
- `src/indexing/paper_store.py:262-271`

---

### 3.5 DeepResearchJob（深度研究任务）

| 项目 | 内容 |
|------|------|
| 创建时机 | POST `/deep-research/start` |
| 关联关系 | 关联 Canvas、ChatSession、User |
| 子资源 | DRJobEvent、DRSectionReview、DRResumeQueue、DRGapSupplement、DRInsight、DRCheckpoint（均 cascade delete）；Milvus 临时 collection `job_{job_id}` |
| 自动清理 | DR Checkpoint 启动时 TTL 7d 清理；临时 Milvus collection 24h 后清理 |
| 手动删除 | `DELETE /deep-research/jobs/{job_id}`（仅允许终态：done/error/cancelled/planning） |
| 文件清理 | N/A（无直接文件资源） |
| Milvus 清理 | ⚠️ 仅靠 24h TTL 自动清理，无即时删除 |
| 状态 | ⚠️ Job 记录无年龄自动清理；Milvus 临时 collection 仅 24h 清理 |

**代码位置：**
- `src/api/routes_chat.py:4414-4430`（手动删除）
- `src/utils/storage_cleaner.py:87-134`（Milvus 临时 collection）
- `src/utils/task_runner.py:54-192`（启动修复）

---

### 3.6 ScholarLibrary（文献库）

| 项目 | 内容 |
|------|------|
| 创建时机 | Scholar API 手动创建，或绑定 Collection 时自动创建 |
| 关联关系 | 属于 User；可通过 CollectionLibraryBinding 与 Collection 绑定 |
| 子资源 | ScholarLibraryPaper（cascade delete） |
| 文件资源 | `data/users/{user_id}/libraries/{library_name}/pdfs/` 目录 |
| 手动删除 | `DELETE /scholar/libraries/{lib_id}` → 删除 DB 记录，cascade 删除 ScholarLibraryPaper |
| 文件清理 | ⚠️ **Library 目录 `libraries/{library_name}/` 不删除** |
| 自动清理 | 无 |
| 状态 | ⚠️ 严重缺口：删除 Library 后磁盘目录残留 |

**代码位置：**
- `src/api/routes_scholar.py:1257-1270`（删除逻辑，缺少 `shutil.rmtree`）
- `src/api/routes_scholar.py:2851-2882`（单个 PDF 文件删除，仅单文件）

---

### 3.7 Collection（Milvus 向量集合）

| 项目 | 内容 |
|------|------|
| 创建时机 | 文件上传时创建 default collection；Deep Research 创建 `job_{job_id}` 临时 collection |
| 关联关系 | 属于 User；可与 ScholarLibrary 绑定 |
| 子资源 | Milvus vectors；Paper 记录 |
| 手动删除 | `DELETE /ingest/collections/{name}` → drop Milvus collection + delete Paper records + delete binding |
| 临时清理 | `job_*` 前缀 collection：启动时检查 Paper 表年龄，>24h 则 drop |
| 文件清理 | ✅ Milvus drop 时向量数据同步删除；Paper DB 记录同步删除；⚠️ PDF 文件不删 |
| 状态 | ⚠️ Collection 删除后 PDF 文件仍残留 |

**代码位置：**
- `src/api/routes_ingest.py:151-182`（删除）
- `src/utils/storage_cleaner.py:87-134`（临时 collection 24h 清理）

---

### 3.8 Redis 任务（Task）

| 项目 | 内容 |
|------|------|
| 创建时机 | 任意 API submit（chat/scholar download） |
| 子资源 | `rag:task:{id}` KV state；`rag:task_events:{id}` 事件流 |
| 自动清理 | Redis TTL（`settings.tasks.task_state_ttl_seconds`） |
| 启动修复 | Chat Task >60s → 标记 error 并释放 slot；Scholar Task >300s 同理 |
| Slot 释放 | 任务进入终态时 `release_slot()` 释放 `rag:active_tasks` / `rag:active_sessions` |
| 状态 | ✅ 完整（TTL + 启动修复） |

**代码位置：**
- `src/tasks/redis_queue.py:147-150`（slot 释放）
- `src/tasks/redis_queue.py:275-337`（启动修复）

---

### 3.9 Working Memory（会话内存）

| 项目 | 内容 |
|------|------|
| 存储位置 | SQLite `working_memory` 表 |
| 关联关系 | 属于 Session |
| 清理方式 | 随 Session 删除时 cascade 清理（⚠️ 需确认 ORM 关系配置） |
| 状态 | ⚠️ 需确认 Session 删除时是否真实 cascade 清理 |

---

### 3.10 PaperMetadata / CrossrefCache（元数据缓存）

| 项目 | 内容 |
|------|------|
| 存储位置 | SQLite `paper_metadata`、`crossref_cache`、`crossref_cache_by_doi` 表 |
| 清理方式 | **无任何清理机制** |
| 状态 | ⚠️ 缓存表无限增长 |

---

### 3.11 RevokedToken（已吊销令牌）

| 项目 | 内容 |
|------|------|
| 清理时机 | 应用启动时 `purge_expired_revocations()` |
| 清理条件 | 记录的过期时间已过 |
| 状态 | ✅ 完整 |

**代码位置：**
- `src/api/server.py:66-73`

---

## 4. 现有清理机制汇总

### 4.1 启动时自动清理（`src/api/server.py`）

| 机制 | 触发条件 | 代码位置 |
|------|----------|----------|
| 年龄清理：Canvas / Session / UserProject | `storage.cleanup_on_startup = true` | `storage_cleaner.py:137-183` |
| 大小清理：超过 `max_size_gb` 时继续删 Canvas / Session | 同上 | `storage_cleaner.py:184-228` |
| 临时 Milvus collection 清理（job_* > 24h） | 同上 | `storage_cleaner.py:87-134` |
| 恢复中断的 DR Job（标记 error） | 每次启动 | `task_runner.py:54-100` |
| Ingest/DR Checkpoint TTL 清理（> 7d） | 每次启动 | `task_runner.py:132-172` |
| Redis stale task 修复 | 每次启动 | `redis_queue.py:275-337` |
| RevokedToken 过期清理 | 每次启动 | `server.py:66-73` |

### 4.2 显式 API 删除

| API | 删除内容 | 文件清理 |
|-----|----------|----------|
| `DELETE /canvas/{id}` | Canvas + 子表 + 关联 Session | N/A |
| `DELETE /sessions/{id}` | Session + Turn | N/A |
| `DELETE /ingest/collections/{name}` | Milvus collection + Paper 记录 + Binding | ⚠️ 不删 PDF |
| `DELETE /scholar/libraries/{id}` | Library DB 记录 + LibraryPaper | ⚠️ 不删目录 |
| `DELETE /scholar/libraries/{id}/papers/{record_id}/pdf` | 单个 PDF 文件 | ✅ |
| `DELETE /deep-research/jobs/{id}` | DR Job + 全部子表 | N/A |

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

## 5. 已知缺口（资源泄漏）

以下问题已被确认，均为资源无法自动或完整回收的情况：

---

### 缺口 1：删除 ScholarLibrary 后目录不删除 🔴

- **泄漏资源：** `data/users/{user_id}/libraries/{library_name}/` 整个目录
- **触发场景：** 用户删除文献库
- **现有行为：** DB 记录删除，目录保留
- **修复方式：** 在 `routes_scholar.py:1257-1270` 删除 DB 记录后补充 `shutil.rmtree(library_dir, ignore_errors=True)`
- **风险等级：** 高（磁盘持续增长）

---

### 缺口 2：删除 Collection / Paper 后 PDF 文件不删除 🔴

- **泄漏资源：** `data/users/{user_id}/raw_papers/` 及 `libraries/*/pdfs/` 内的 PDF 文件
- **触发场景：** 删除 Collection、或 Collection 被 age 清理
- **现有行为：** Milvus vectors 和 Paper DB 记录删除，PDF 保留
- **修复方式：** 在 `paper_store.py:delete_collection_papers()` 中，删除 DB 前先按 Paper 记录的文件路径字段（`file_path` / `pdf_path`）删除对应文件
- **风险等级：** 高（磁盘持续增长）

---

### 缺口 3：IngestJob 记录永不清理 🟡

- **泄漏资源：** SQLite `ingest_jobs`、`ingest_job_events` 表行
- **触发场景：** 任意 Ingest 完成后
- **现有行为：** Checkpoint 有 TTL，但 Job / Event 记录本身无年龄清理，无删除 API
- **修复方式：** 在 `storage_cleaner.py` 中增加 `_cleanup_by_age_ingest_jobs()`，清理 `status IN (done, error, cancelled) AND updated_at < cutoff`
- **风险等级：** 中（DB 缓慢增长）

---

### 缺口 4：DeepResearchJob 记录无年龄自动清理 🟡

- **泄漏资源：** SQLite `deep_research_jobs` 及其子表（Event / Review / Gap / Insight 等）
- **触发场景：** DR 任务完成后未手动删除
- **现有行为：** 有手动删除 API，但无年龄自动清理
- **修复方式：** 在 `storage_cleaner.py` 中增加 `_cleanup_by_age_dr_jobs()`，清理终态 Job 超过配置年龄的记录
- **风险等级：** 中（DB 缓慢增长）

---

### 缺口 5：Milvus 临时 Collection 仅 24h 清理，无即时释放 🟡

- **泄漏资源：** Milvus `job_{job_id}` collection（DR 任务用）
- **触发场景：** DR 任务删除时
- **现有行为：** 仅在下次启动 + 24h 检查时删除；Job 删除 API 不触发 Milvus drop
- **修复方式：** 在 `routes_chat.py:4414-4430` DR 删除逻辑中，增加 `milvus.client.drop_collection(f"job_{job_id}")` 调用
- **风险等级：** 中（Milvus 空间未及时释放）

---

### 缺口 6：PaperMetadata / Crossref 缓存表无清理 🟢

- **泄漏资源：** SQLite `paper_metadata`、`crossref_cache`、`crossref_cache_by_doi` 表行
- **现有行为：** 无任何 TTL 或清理
- **修复方式：** 增加按 `created_at` 的 TTL 清理（如 90d），或按 LRU 限制条目数量
- **风险等级：** 低（外部 API 缓存，增长相对缓慢）

---

### 缺口 7：用户数据目录无清理 🟢

- **泄漏资源：** `data/users/{user_id}/` 整个目录
- **场景：** 长期不活跃用户或测试账号
- **现有行为：** 无基于用户维度的清理
- **修复方式：** 可选：配合用户删除 API 实现 `shutil.rmtree(user_data_dir)`
- **风险等级：** 低（当前系统用户量少）

---

## 缺口优先级汇总

| 优先级 | 缺口 | 修复复杂度 |
|--------|------|------------|
| 🔴 高 | 缺口 1：Library 目录不删 | 低（1 行代码） |
| 🔴 高 | 缺口 2：PDF 文件不随 Collection/Paper 删除 | 中（需读取文件路径字段） |
| 🟡 中 | 缺口 3：IngestJob 记录永久堆积 | 低（仿现有清理函数） |
| 🟡 中 | 缺口 4：DR Job 无年龄自动清理 | 低（仿现有清理函数） |
| 🟡 中 | 缺口 5：DR Milvus collection 不即时释放 | 低（1 行 drop 调用） |
| 🟢 低 | 缺口 6：元数据缓存表无清理 | 中 |
| 🟢 低 | 缺口 7：用户目录无清理 | 低 |

---

## 6. 配置参考

```json
// config/rag_config.json → storage 节点
{
  "storage": {
    "max_age_days": 30,         // 数据保留天数（Canvas / Session）
    "max_size_gb": 5.0,         // DB 大小上限，超出时触发额外清理
    "cleanup_on_startup": true, // 启动时是否执行清理
    "cleanup_batch_size": 100   // 每批删除的记录数
  }
}
```

对应代码：`config/settings.py` → `StorageSettings`

---

*本文档由代码分析自动生成，日期：2026-03-12。如有代码变更，请同步更新本文档。*
