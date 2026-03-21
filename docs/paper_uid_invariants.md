## 全局唯一论文标识符规范（`paper_uid`）

> 本文档是项目级基础约束。凡是涉及“跨模块标识一篇论文”的设计、实现、存储、接口与数据流，均必须遵守。

### 1. 目标

`paper_uid` 是论文作为学术作品的跨系统稳定标识符。

项目中凡是需要在 `ScholarLibrary`、`CollectionLibraryBinding`、`paper_metadata_store`、检索链路、下载链路、状态链路之间关联同一篇论文时，必须使用同一个 `paper_uid`。

### 2. 为什么必须统一

如果不同模块各自生成论文 ID，会直接导致：

1. 跨模块 join 失败
2. 去重结果不一致
3. 元数据补全过程无法稳定升级
4. 后续 API、缓存、索引、状态表出现多套“论文真相”

因此，`paper_uid` 是整个项目的基础身份约束，而不是某个局部模块的内部细节。

### 3. 生成规则

`paper_uid` 必须按以下优先级链确定性生成：

| 优先级 | 条件 | 格式 | 示例 |
| --- | --- | --- | --- |
| 1 | `normalized_doi` 非空 | `doi:{normalized_doi}` | `doi:10.1038/s41586-023-06345-3` |
| 2 | URL 或 title 中含 arXiv ID | `arxiv:{yymm.nnnnn}` | `arxiv:2306.01234` |
| 3 | PMID 已知 | `pmid:{pmid}` | `pmid:36721000` |
| 4 | 以上均无 | `sha:{sha256(norm_title + "|" + first_author_last_lower + "|" + str(year or ""))[:16]}` | `sha:a3f2b1c4d5e6f789` |

补充要求：

1. arXiv ID 必须去掉版本号后缀
2. `normalized_doi` 必须先经过 `dedup.normalize_doi()` 归一化
3. 结果必须是确定性的，同一论文任何时候都应得到同一个 `paper_uid`

### 4. 唯一权威实现

`paper_uid` 的计算只能通过以下函数：

```python
from src.retrieval.dedup import compute_paper_uid
```

这意味着：

1. 任何模块不得私自实现论文级 ID 生成逻辑
2. 任何模块不得绕过 `compute_paper_uid()` 构造跨模块论文引用键
3. 后续所有新增模块、新增 API、新增服务，只要需要标识一篇论文，就必须调用该函数

### 5. 升级规则

当某篇论文最初只能生成 `sha:*`，但后续补全出了 DOI 时，必须：

1. 重新计算 `paper_uid`
2. 将身份从 `sha:*` 升级为 `doi:*`
3. 同步更新所有依赖该身份的引用、关联和映射

禁止长期保留“旧 sha 身份作为主身份”而不做升级。

### 6. 概念边界

`paper_uid` 不替代以下标识符：

| 标识符 | 用途 | 是否可替代 `paper_uid` |
| --- | --- | --- |
| `paper_id` | collection 内文件标识 | 否 |
| `cite_key` | 单次对话或引文列表中的用户可见引用键 | 否 |
| chunk fingerprint | chunk 级内容去重 | 否 |

这些标识符可以继续存在，但不能拿来充当跨模块论文身份。

### 7. 工程红线

以下行为一律禁止：

1. 在模块内部私自生成另一套论文主键
2. 在模块内部维护第二份论文去重真相表
3. 将局部缓存键、文件名、临时引用键当作跨模块论文身份
4. 在不经过 `compute_paper_uid()` 的情况下拼装论文级 join key

### 8. 存储与真相层要求

论文身份的真相层必须复用现有统一能力，尤其是：

1. `paper_metadata_store` 负责 DOI/title 相关真相索引
2. 共享 SQL 是业务主数据层
3. 任何单模块内存对象、临时缓存对象、局部映射表，都不能成为论文身份的唯一真相

### 9. 适用范围

本约束默认适用于：

1. Scholar / 文献发现相关功能
2. 下载与入库链路
3. 检索与去重链路
4. collection 绑定关系
5. 元数据补全与同步逻辑
6. 后续新增的论文相关 API、服务、任务与状态模型

### 10. 实施要求

以后凡是新增或修改论文相关功能，先检查两个问题：

1. 这里是否在跨模块表示”同一篇论文”？
2. 如果是，是否已经明确使用 `compute_paper_uid()`？

只要这两个问题的答案不清楚，就不能认为实现满足项目约束。

---

### 11. 历史数据回填工具

**脚本**：`scripts/28_backfill_paper_uid.py`

对于在引入 `paper_uid` 之前已存入数据库的历史记录，该脚本会扫描以下三张表并补全 `paper_uid` 字段：

| 表 | 回填策略 |
| --- | --- |
| `paper_metadata` | 从 `doi / title / authors / year` 字段调用 `compute_paper_uid()` 计算 |
| `scholar_library_papers` | 从 `doi / title / authors / year / url` 字段调用 `compute_paper_uid()` 计算 |
| `papers` | 优先从 `paper_metadata` 中继承已有 `paper_uid`；无元数据的行留空（待后续元数据补全后自动填入） |

**常用命令**：

```bash
# 1. 仅扫描统计，不写入（安全检查）
python scripts/28_backfill_paper_uid.py --dry-run

# 2. 正常运行（只补全空值行，幂等）
python scripts/28_backfill_paper_uid.py

# 3. 强制全量重算（当 paper_uid 规则有更新时使用）
python scripts/28_backfill_paper_uid.py --force

# 4. 只处理某一张表
python scripts/28_backfill_paper_uid.py --table paper_metadata
python scripts/28_backfill_paper_uid.py --table scholar_library_papers
python scripts/28_backfill_paper_uid.py --table papers

# 5. 调整批次大小（大库用大批次减少 commit 次数）
python scripts/28_backfill_paper_uid.py --batch-size 500
```

**推荐执行顺序**（首次部署时）：

```bash
# Step 1：先回填 paper_metadata（它是其他表的元数据来源）
python scripts/28_backfill_paper_uid.py --table paper_metadata

# Step 2：回填 scholar_library_papers（独立计算，不依赖 paper_metadata）
python scripts/28_backfill_paper_uid.py --table scholar_library_papers

# Step 3：回填 papers（从 paper_metadata 继承 uid）
python scripts/28_backfill_paper_uid.py --table papers
```

**注意事项**：
- 脚本幂等，可重复运行，默认不覆盖已有 `paper_uid`
- `papers` 表中无元数据（仅有文件名）的行会被跳过，等待后续 `26_backfill_doi.py` 补全元数据后再执行本脚本
- DOI 补全后如需将 `sha:*` 升级为 `doi:*`，使用 `--force` 重跑

---

### 12. 框架一致性自动升级（推荐入口）

**脚本**：`scripts/29_upgrade_framework_consistency.py`

对已有文献库和向量库执行一键升级，是首次部署本规范后唯一需要手动运行的命令。

**向量库（Milvus）处理策略**：

| 数据 | 处理方式 |
| --- | --- |
| 历史 chunk | **无需改写**。`retrieval/service.py` 在查询时通过 `paper_id → paper_metadata` 动态补全 `paper_uid`（运行时透明） |
| 新入库 chunk | **自动写入**。`_build_rows()` 和 `_chunk_embed_upsert_one_doc()` 已将 `paper_uid` 写入 Milvus 动态字段 |

**快速执行**：

```bash
# 标准升级（幂等，可重复运行）
python scripts/29_upgrade_framework_consistency.py

# 升级前预检（只统计，不写入）
python scripts/29_upgrade_framework_consistency.py --dry-run

# DOI 补全后重算（sha → doi 身份升级）
python scripts/29_upgrade_framework_consistency.py --force
```

**完整升级流程（生产环境建议顺序）**：

```bash
# 1. 先补全 DOI（为论文建立最优标识符来源）
python scripts/26_backfill_doi.py

# 2. 执行 paper_uid 框架一致性升级
python scripts/29_upgrade_framework_consistency.py

# 3. 若有新 DOI 补全进来，重算升级
python scripts/29_upgrade_framework_consistency.py --force
```

脚本结束后会自动打印三张表的 `paper_uid` 覆盖率统计。
