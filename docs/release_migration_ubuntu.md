# Ubuntu 发布与迁移手册（完整版）

本文档用于生产发布与跨机器迁移，重点覆盖：系统依赖、Python/前端依赖、配置基线、数据迁移、启动托管与验收。

更新时间：2026-02-19

## 1. 适用范围

- 目标系统：Ubuntu 22.04/24.04
- 部署方式：`docker compose`（Milvus 依赖）+ `systemd`（应用托管）
- 项目路径示例：`/opt/deepsea-rag`

## 2. 系统层依赖（Ubuntu）

建议先安装系统依赖：

```bash
sudo apt update
sudo apt install -y \
  git curl wget unzip ca-certificates gnupg lsb-release \
  build-essential pkg-config python3-dev \
  libssl-dev libffi-dev libsqlite3-dev \
  docker.io docker-compose-plugin
```

可选（仅在服务器需要 headful 浏览器时）：

```bash
sudo apt install -y xvfb
```

## 3. 运行时版本基线

- Python：`3.10+`（推荐 3.10）
- Conda：建议 Miniconda/Anaconda（用于隔离 `deepsea-rag` 环境）
- Node.js：`^20.19.0 || >=22.12.0`
- npm：建议与 Node LTS 配套
- Docker/Compose：可用 `docker compose version` 检查

## 4. 代码与目录

```bash
sudo mkdir -p /opt/deepsea-rag
sudo chown -R "$USER":"$USER" /opt/deepsea-rag
cd /opt/deepsea-rag
git clone <YOUR_REPO_URL> .
```

## 5. Python 环境与依赖

```bash
conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag

# 必须走 Conda 安装 PyTorch 生态
conda install -c pytorch -c conda-forge "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y

# 项目 Python 依赖
pip install -r requirements.txt --no-cache-dir

# 浏览器自动化依赖
playwright install chromium
```

## 6. 前端依赖

```bash
cd frontend
npm ci
npm run build
cd ..
```

## 7. 配置文件（必须项）

```bash
cp config/rag_config.example.json config/rag_config.json
cp config/rag_config.local.example.json config/rag_config.local.json
```

### 7.1 推荐密钥注入方式

优先环境变量（避免明文落盘）：

- `RAG_LLM__OPENAI__API_KEY`
- `RAG_LLM__DEEPSEEK__API_KEY`
- `RAG_LLM__GEMINI__API_KEY`
- `RAG_LLM__CLAUDE__API_KEY`
- `RAG_LLM__KIMI__API_KEY`

### 7.2 发布前关键配置核对

- `config/rag_config.json`
  - `api.host`、`api.port`
  - `llm.default` + provider 列表
  - `storage.max_age_days`、`storage.max_size_gb`
  - `performance.*`（超时/并发）
- `config/rag_config.local.json`
  - 各 provider 密钥
  - `auth.secret_key`（生产必须替换）

## 8. 基础服务（PostgreSQL / Redis / Milvus / MinIO / etcd）

```bash
# 生产 profile
docker compose --profile prod up -d
bash scripts/00_healthcheck_docker.sh
```

## 9. 数据迁移（如从旧机迁移）

建议迁移目录/文件：

- `data/raw_papers/`
- `data/parsed/`
- `data/rag.db`（如仍保留 SQLite 回滚点）
- `data/*.db.bak`（如存在，作为历史迁移备份）
- `artifacts/`（可选）
- Docker PostgreSQL 逻辑备份（推荐 `pg_dump`）
- Docker named volumes：`deepsea-rag-postgres-data`、`deepsea-rag-redis-data`

示例（源机执行）：

```bash
tar -czf deepsea-rag-data.tgz data/raw_papers data/parsed data/rag.db artifacts
```

> 说明：当前版本已统一为单库 `data/rag.db`（SQLModel + Alembic 管理）。若从旧版本迁移，系统启动时会自动尝试将历史多库数据合并到 `rag.db`，并将旧库重命名为 `*.db.bak`。

目标机解压：

```bash
tar -xzf deepsea-rag-data.tgz -C /opt/deepsea-rag
```

## 10. 依赖与版本核对（强烈建议）

```bash
bash scripts/verify_dependencies.sh
```

通过标准：

- `FAIL=0`
- Python 关键约束通过（`transformers/huggingface-hub/fastapi/pydantic/openai/mcp`）

## 11. 初始化与索引（首次或重建）

```bash
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
```

> 若已迁移完成 `data/parsed` 且 Milvus 数据卷完整迁移，可按需跳过部分步骤。

## 12. systemd 托管（推荐）

### 12.1 后端服务

`/etc/systemd/system/deepsea-rag-api.service`

```ini
[Unit]
Description=DeepSea RAG FastAPI Service
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/opt/deepsea-rag
Environment=PYTHONUNBUFFERED=1
Environment=API_HOST=0.0.0.0
Environment=API_PORT=9999
EnvironmentFile=-/opt/deepsea-rag/.env
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag python scripts/08_run_api.py --host 0.0.0.0 --port 9999'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

### 12.2 前端服务（preview 方案，过渡可用）

`/etc/systemd/system/deepsea-rag-frontend.service`

```ini
[Unit]
Description=DeepSea RAG Frontend Preview
After=network.target

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/opt/deepsea-rag/frontend
Environment=NODE_ENV=production
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag npm run preview -- --host 0.0.0.0 --port 5173'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

启用与启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsea-rag-api deepsea-rag-frontend
sudo systemctl start deepsea-rag-api deepsea-rag-frontend
```

说明：

- 不建议写死 `node/npm` 路径；优先通过 conda 环境统一管理。
- 若 systemd 下 `conda` 不在 PATH，请将 `CONDA_EXE` 设置为实际路径（如 `/opt/anaconda3/bin/conda`）。
- 当 Node 位于 `.../envs/deepsea-rag/bin/node` 时，上述 `conda run -n deepsea-rag npm ...` 可直接使用该环境内 node/npm。

### 12.3 Nginx 生产推荐方案（建议）

生产更推荐：**后端由 systemd 托管，前端用 Nginx 托管静态文件**（替代 `npm preview`）。

1) 构建前端：

```bash
cd /opt/deepsea-rag/frontend
conda run -n deepsea-rag npm ci
conda run -n deepsea-rag npm run build
```

2) 安装 Nginx：

```bash
sudo apt update
sudo apt install -y nginx
```

3) Nginx 配置（示例：`/etc/nginx/sites-available/deepsea-rag`）：

```nginx
server {
    listen 80;
    server_name your.domain.com;

    # 前端静态资源
    root /opt/deepsea-rag/frontend/dist;
    index index.html;

    # SPA 路由回退
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 反向代理
    location /api/ {
        proxy_pass http://127.0.0.1:9999/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE/长连接建议配置
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # 直接暴露 docs（可按需加白名单）
    location /docs {
        proxy_pass http://127.0.0.1:9999/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:9999/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

4) 启用站点并检查：

```bash
sudo ln -s /etc/nginx/sites-available/deepsea-rag /etc/nginx/sites-enabled/deepsea-rag
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

5) HTTPS（Let's Encrypt）：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.com
```

> 使用 Nginx 方案后，可以停止前端 preview service，仅保留后端 API systemd。

## 13. 验收清单（Go/No-Go）

- `GET /health` 返回 `ok`
- `GET /health/detailed` 无关键组件失败
- `GET /metrics` 可访问
- `GET /llm/providers` 列表正常
- `POST /chat` 可返回
- `POST /chat/stream` 可流式返回
- `GET /compare/papers` 正常
- `GET /ingest/collections` 正常
- Deep Research 可启动并返回 job_id
- 前端页面可打开并调用后端
- Nginx `nginx -t` 通过且证书自动续期可用（若启用 HTTPS）

## 14. 回滚策略

- 保留旧环境与数据快照
- 新版本发布前备份：
  - `config/rag_config*.json`
  - `data/parsed/`
  - `data/rag.db`
  - `data/*.db.bak`（如存在）
- 回滚时：
  1. 停服务（systemd）
  2. 切回旧代码/旧镜像
  3. 恢复配置与数据快照
  4. 重启并验收关键接口

## 15. 常见发布问题

- `node: command not found`：Node 未安装或 PATH 未配置
- `mcp/trafilatura missing`：未在 `deepsea-rag` 环境执行安装
- API 401：密钥未注入或 `.local.json` 覆盖错误
- 检索为空：Milvus profile 未启动或 collection 未初始化
- Google/Scholar 不稳定：Playwright 依赖/并发参数需调整

## 16. 升级修复记录

### 16.1 绑定文献库 · 本地 PDF 导入时 DOI 获取与改名（2026-03）

**背景与范围**  
本条仅针对「绑定文献库管理 → 选择本地 PDF 文件夹导入」这条链路中，DOI 的获取方式与据此进行的文件改名逻辑；不涉及建库解析、检索或 CrossRef 检索侧用法。

**当前行为**

- **DOI 获取**：由 `src/retrieval/dedup.py` 中 `extract_doi_from_pdf_tiered()` 统一提供，三层顺序、先命中即返回：
  - **Tier 1**：PDF 原生元数据（fitz 读 `/Title` 等，在元数据字符串上做 DOI 正则）。
  - **Tier 2**：**前两页**正文拼接后做 DOI 正则，并从前两页文本中提取候选标题（跳过 Abstract/References/Supplementary 等）；命中 DOI 时返回 `(doi, 候选标题)`。
  - **Tier 3**：用「原生元数据 title 或 Tier 2 候选标题」调 CrossRef（`_crossref_lookup_by_title`），命中则用返回的 DOI。
- **改名逻辑**：`src/api/routes_scholar.py` 中 `import_library_pdfs()` 在拿到 `(doi, title)` 后：
  - 有 DOI → `paper_id = _doi_to_paper_id(ndoi)`，保存为 `{paper_id}.pdf`。
  - 无 DOI → `paper_id = _normalize_to_paper_id(title, [], None)`，再保存。  
  同一逻辑被「从文献库增量建库」前的去重、以及 `_pdf_rename_dedup_library_folder` 等复用。

**已识别问题（已缓解）**

- Tier 2 原仅扫第一页：当第一页为封面（无 DOI）时，正文或版权页上的 DOI 无法被利用；**已改为前两页扫描**。题目原仅依赖原生元数据，**已增加从前两页正文提取候选标题**（排除小节标签）。

**拟升级方向**

- 将 Tier 2 改为**前两页**正文扫描（封面 + 版权/摘要页），减少封面场景漏检。
- **题目提取同步完善**：从前两页正文中选取首条「像标题」的行（长度 10–300、跳过 Abstract/References/Supplementary 等小节标签及 URL/版权行），用于无 DOI 时的命名与 Tier 3 的 CrossRef 查询；Tier 3 在原生元数据无 title 时回退到该候选标题。
- 评估在 Tier 2 与 Tier 3 之间增加一层「正文快速 DOI 扫描」：限定前 N 页（如 3–10 页），并**排除**明显为 References / Bibliography / Supplementary / Supporting Information 的页，避免取到参考文献中的 DOI。
- 保持 Tier 3 的 CrossRef 回退为兜底，不提前替代 PDF 自身信号。

**本次已实现**

- Tier 2 已改为前两页扫描；DOI 命中时同时返回前两页中提取的候选标题（若有）。
- 新增 `_extract_title_from_page_text()`：按行选取首条符合长度与排除规则的行作为候选标题（`_SKIP_TITLE_LABELS` 含 Abstract、References、Bibliography、Supplementary、Supporting Information、Appendix 等）。
- Tier 3 的 title 来源为：原生元数据 title 或上述候选标题，缺则仍不调 CrossRef。

**验证建议**

- 封面页论文：第一页无 DOI、第二页有 DOI，升级后应能通过 Tier 2 命中。
- 第二页为版权/摘要页的论文：同上。
- 无 DOI 的 PDF：仍走 title 归一化命名，无回归。
- 正文无 DOI、仅参考文献页含 DOI：若增加「排除参考文献/Supplementary」逻辑，应不误取引用 DOI。

**影响范围**

- 代码：`src/retrieval/dedup.py`（`extract_doi_from_pdf_tiered`）、可选动 `routes_scholar.py` 仅若改名规则扩展。
- 使用方：入库页「选择本地 PDF 文件夹」导入、文献库文件夹重命名去重、从文献库增量建库前的 DOI 去重。
