# 安装与迁移指南

本文档覆盖 DeepSea RAG 的完整安装流程（Mac 开发环境 + Ubuntu 生产服务器）、配置说明，以及版本升级与数据迁移方法。

---

## 目录

1. [系统要求](#1-系统要求)
2. [Mac 开发环境安装](#2-mac-开发环境安装)
3. [Ubuntu 生产服务器安装](#3-ubuntu-生产服务器安装)
4. [配置说明](#4-配置说明)
5. [首次启动与验证](#5-首次启动与验证)
6. [版本升级与迁移](#6-版本升级与迁移)
7. [数据备份与恢复](#7-数据备份与恢复)
8. [回滚方案](#8-回滚方案)
9. [常见安装问题](#9-常见安装问题)

---

## 1. 系统要求

### 1.1 硬件要求

| 场景 | CPU | RAM | GPU | 存储 |
|------|-----|-----|-----|------|
| Mac 开发（CPU 推理） | 8 核+ | 16 GB+ | 无需 | 50 GB+ SSD |
| Ubuntu 生产（小团队） | 16 核+ | 32 GB+ | NVIDIA 16 GB VRAM+ | 200 GB+ SSD |
| Ubuntu 生产（大规模） | 32 核+ | 64 GB+ | NVIDIA 24 GB VRAM+ × 2 | 500 GB+ NVMe |

### 1.2 软件依赖

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+（推荐 3.10） | 必须 |
| Conda / Miniconda | 任意新版 | 管理 Python 环境 |
| Node.js | `^20.19.0` 或 `>=22.12.0` | Vite 7 引擎要求 |
| Docker Engine | 24.0+ | 运行 Milvus/Redis/PostgreSQL |
| Docker Compose Plugin | 2.20+ | `docker compose`（注意：无连字符） |
| NVIDIA Driver | 525+ | 仅 GPU 生产环境需要 |
| NVIDIA Container Toolkit | 任意新版 | 仅 GPU 生产环境需要 |

### 1.3 Python 依赖关键约束

以下约束**必须**满足，否则会导致 C++ 算子冲突或功能失效：

- `transformers>=4.42.0,<5.0.0`（Docling 兼容性）
- `huggingface-hub==0.36.0`（防止元数据损坏）
- `torch / torchvision / timm`：**必须通过 Conda 安装**，禁止在 `requirements.txt` 中列出
- `fastapi>=0.100.0,<1.0.0`
- `pydantic>=2.0.0,<3.0.0`

---

## 2. Mac 开发环境安装

### 2.1 安装 Conda

```bash
# 下载 Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
# Intel Mac 用：Miniconda3-latest-MacOSX-x86_64.sh

bash Miniconda3-latest-MacOSX-arm64.sh
# 按提示完成安装，重新打开终端
```

### 2.2 创建 Python 环境

```bash
conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag

# 安装 PyTorch（必须通过 Conda，macOS CPU 版）
conda install -c pytorch -c conda-forge \
  "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y
```

### 2.3 安装 Python 依赖

```bash
# 克隆项目
git clone <repo_url> DeepSeaRAG
cd DeepSeaRAG

# 安装依赖
pip install -r requirements.txt --no-cache-dir

# 安装浏览器（学术搜索 + 下载功能依赖）
playwright install chromium
```

### 2.4 启动基础设施

```bash
# 启动 Milvus（CPU 版）+ Redis
# 注意：Mac 使用 dev profile（CPU Milvus）
docker compose --profile dev up -d

# 验证服务状态
docker compose ps
```

预期所有服务状态为 `healthy`：
- `deepsea-milvus`（端口 19530）
- `deepsea-rag-redis`（端口 6379）
- `deepsea-etcd`
- `deepsea-minio`

> **注**：`docker compose --profile dev up -d` 同样会启动 PostgreSQL；开发与生产都以 PostgreSQL 作为正式后端，仅 Milvus 在 `dev/prod` profile 间切换。

### 2.5 安装前端

```bash
cd frontend

# 精确按 lock 文件安装（推荐）
npm ci

# 或普通安装
npm install

cd ..
```

### 2.6 配置

```bash
# 复制配置模板
cp config/rag_config.example.json config/rag_config.json

# 创建本地覆盖配置（填写 API Key，不提交到 Git）
cp config/rag_config.example.json config/rag_config.local.json
```

编辑 `config/rag_config.local.json`，至少填写一个 LLM API Key：

```json
{
  "llm": {
    "platforms": {
      "claude": { "api_key": "sk-ant-xxx" }
    }
  }
}
```

完整配置说明见[第 4 节](#4-配置说明)。

### 2.7 初始化数据库

```bash
# 初始化（创建表并运行迁移）
alembic upgrade head
```

### 2.8 启动服务

```bash
# 终端 1：启动后端
conda activate deepsea-rag
uvicorn src.api.server:app --host 0.0.0.0 --port 9999 --reload

# 终端 2：启动前端
cd frontend && npm run dev
```

访问 `http://localhost:5173` 即可使用。

---

## 3. Ubuntu 生产服务器安装

### 3.1 系统依赖

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  git curl wget unzip ca-certificates gnupg lsb-release \
  build-essential pkg-config python3-dev \
  libssl-dev libffi-dev libsqlite3-dev \
  screen htop iotop
```

### 3.2 安装 Docker

```bash
# 添加 Docker 官方 GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 添加 Docker 软件源
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

# 添加当前用户到 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker
```

### 3.3 安装 NVIDIA Container Toolkit（GPU 环境）

```bash
# 确认 GPU 驱动已安装
nvidia-smi

# 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3.4 安装 Conda 和 Python 环境

```bash
# 下载 Miniconda（Linux）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash && source ~/.bashrc

# 创建环境
conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag

# 安装 PyTorch（GPU CUDA 版）
conda install -c pytorch -c nvidia -c conda-forge \
  "pytorch>=2.6.0" "torchvision>=0.21.0" timm "pytorch-cuda=12.1" -y
```

### 3.5 部署项目

```bash
# 建议部署到 /opt 目录
sudo mkdir -p /opt/deepsea-rag
sudo chown $USER:$USER /opt/deepsea-rag

git clone <repo_url> /opt/deepsea-rag
cd /opt/deepsea-rag

# 安装 Python 依赖
conda activate deepsea-rag
pip install -r requirements.txt --no-cache-dir
playwright install chromium --with-deps  # Ubuntu 需要 --with-deps
```

### 3.6 启动基础设施

```bash
# 生产环境使用 prod profile（GPU Milvus）
docker compose --profile prod up -d

# 查看状态
docker compose ps
docker compose logs -f milvus-prod  # 等待 Milvus 就绪（约 30-60s）
```

### 3.7 安装前端并构建

```bash
# 安装 Node.js（推荐 nvm）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22

# 安装并构建前端
cd /opt/deepsea-rag/frontend
npm ci
npm run build
cd ..
```

### 3.8 配置

```bash
# 创建本地配置（生产环境不提交 API Key）
cp config/rag_config.example.json config/rag_config.json
cp config/rag_config.example.json config/rag_config.local.json

# 编辑本地配置：填写 API Key + 生产数据库配置
nano config/rag_config.local.json
```

关键生产配置项（`rag_config.local.json`）：

```json
{
  "database": {
    "url": "postgresql://rag:your_password@localhost:5433/rag"
  },
  "llm": {
    "platforms": {
      "claude": { "api_key": "sk-ant-xxx" },
      "openai": { "api_key": "sk-proj-xxx" }
    }
  },
  "web_search": {
    "api_key": "tvly-xxx"
  },
  "auth": {
    "secret_key": "your-secure-random-key-64-chars-minimum"
  }
}
```

> **安全警告**：生产环境**必须**修改 `auth.secret_key`，使用随机生成的强密钥：
> ```bash
> python3 -c "import secrets; print(secrets.token_hex(32))"
> ```

### 3.9 初始化数据库

```bash
conda activate deepsea-rag
cd /opt/deepsea-rag
alembic upgrade head
```

### 3.10 配置 systemd 服务

创建后端服务文件：

```bash
sudo nano /etc/systemd/system/deepsea-rag-backend.service
```

```ini
[Unit]
Description=DeepSea RAG Backend
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/deepsea-rag
Environment=PATH=/home/ubuntu/miniconda3/envs/deepsea-rag/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/miniconda3/envs/deepsea-rag/bin/uvicorn \
  src.api.server:app \
  --host 0.0.0.0 \
  --port 9999 \
  --workers 1 \
  --timeout-keep-alive 120
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=deepsea-rag

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsea-rag-backend
sudo systemctl start deepsea-rag-backend
sudo systemctl status deepsea-rag-backend
```

### 3.11 配置 Nginx

安装 Nginx 并配置反向代理（含前端静态文件）：

```bash
sudo apt install -y nginx

sudo nano /etc/nginx/sites-available/deepsea-rag
```

```nginx
server {
    listen 80;
    server_name your.domain.com;

    # 前端静态文件
    root /opt/deepsea-rag/frontend/dist;
    index index.html;

    # 前端路由（SPA fallback）
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 后端 API（含 SSE）
    location /api/ {
        proxy_pass http://127.0.0.1:9999/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";

        # SSE 关键配置
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        keepalive_timeout 3600s;
        chunked_transfer_encoding on;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/deepsea-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

> **HTTPS**：生产环境强烈建议配置 HTTPS（使用 Let's Encrypt / Certbot）。SSE 长连接在 HTTP 下可能被中间代理中断。

### 3.12 配置 HTTPS（推荐）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.com
sudo systemctl reload nginx
```

---

## 4. 配置说明

### 4.1 配置文件加载顺序

```
rag_config.local.json（最高优先，gitignored）
  ↓ 深度合并覆盖
rag_config.json（主配置）
  ↓ 环境变量覆盖
代码默认值（最低优先）
```

### 4.2 必填配置项

以下配置**必须**在 `rag_config.local.json` 中填写：

| 配置路径 | 说明 | 示例 |
|---------|------|------|
| `llm.platforms.<provider>.api_key` | 至少一个 LLM 的 API Key | `"sk-ant-xxx"` |
| `auth.secret_key` | JWT 签名密钥（生产环境） | 随机 64 位 hex |

### 4.3 LLM 提供商配置

| Provider 标识 | 平台 | API Key 获取 |
|--------------|------|-------------|
| `openai` / `openai-thinking` / `openai-mini` | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) |
| `deepseek` / `deepseek-thinking` | DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) |
| `gemini` / `gemini-thinking` / `gemini-vision` / `gemini-flash` | Google | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `claude` / `claude-thinking` / `claude-haiku` | Anthropic | [console.anthropic.com](https://console.anthropic.com/) |
| `kimi` / `kimi-thinking` / `kimi-vision` | Moonshot | [platform.moonshot.cn](https://platform.moonshot.cn/) |
| `sonar` | Perplexity | [docs.perplexity.ai](https://docs.perplexity.ai/) |
| `qwen` / `qwen-thinking` / `qwen-vision` | Alibaba | [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com/) |

### 4.4 可选功能配置

| 功能 | 配置项 | 说明 |
|------|--------|------|
| Tavily 网络搜索 | `web_search.api_key` | 推荐配置，影响 hybrid 检索质量 |
| Semantic Scholar | `semantic_scholar.api_key` | 无 key 也可用，有请求频率限制 |
| NCBI PubMed | `ncbi.api_key` + `ncbi.email` | 无 key 也可用 |
| BrightData 代理 | `content_fetcher.brightdata_api_key` | 用于绕过 PDF 访问限制 |
| 2Captcha 验证码 | `scholar_downloader.twocaptcha_api_key` | 文献下载遇到验证码时使用 |
| CapSolver 验证码 | `capsolver.api_key` | 备选验证码服务 |
| LangSmith Tracing | 环境变量 `LANGCHAIN_API_KEY` | Agent 全链路追踪（可选） |

### 4.5 共享浏览器配置

Playwright 共享浏览器池（`config.shared_browser`）控制浏览器 context 并发：

```json
{
  "shared_browser": {
    "start_headless": true,
    "start_headed": true,
    "headless_context_pool_size": 4,
    "headless_search_reserved_slots": 1,
    "headed_context_pool_size": 2,
    "context_acquire_timeout_seconds": 30
  }
}
```

并发用户多时建议将 `headless_context_pool_size` 调大（如 6-8）。

---

## 5. 首次启动与验证

### 5.1 健康检查

```bash
# 后端健康检查
curl http://localhost:9999/health
# 期望：{"status": "ok", "version": "..."}

# 详细健康检查
curl http://localhost:9999/health/detailed
# 期望：所有 components 均为 ok
```

### 5.2 快速功能验证

```bash
# 1. 登录获取 Token
TOKEN=$(curl -s -X POST http://localhost:9999/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. 测试 Chat（local 模式，无需文档）
curl -X POST http://localhost:9999/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "mode": "local"}'
```

### 5.3 上传测试文档

1. 在前端界面（`http://localhost:5173`）登录
2. 进入文档管理页面，上传一篇 PDF
3. 等待 Ingest 任务完成（SSE 进度条）
4. 在 Chat 页面提问验证检索是否正常

---

## 6. 版本升级与迁移

### 6.1 标准升级流程

```bash
cd /opt/deepsea-rag

# 1. 备份数据（见第 7 节）
./scripts/backup.sh

# 2. 拉取新版本
git fetch origin
git checkout main
git pull

# 3. 停止服务
sudo systemctl stop deepsea-rag-backend

# 4. 更新 Python 依赖
conda activate deepsea-rag
pip install -r requirements.txt --no-cache-dir

# 5. 运行数据库迁移
alembic upgrade head

# 6. 重新构建前端
cd frontend
npm ci
npm run build
cd ..

# 7. 重启服务
sudo systemctl start deepsea-rag-backend
sudo systemctl status deepsea-rag-backend

# 8. 验证
curl http://localhost:9999/health/detailed
```

### 6.2 升级后验证清单

- [ ] `GET /health/detailed` 所有组件状态 ok
- [ ] Chat 功能正常（流式响应、引文显示）
- [ ] Ingest 功能正常（上传 PDF → 入库）
- [ ] Deep Research 功能正常（提交任务 → 进度推送）
- [ ] Scholar 搜索功能正常

### 6.3 数据库 Schema 迁移

新版本若有 Schema 变更，`alembic upgrade head` 会自动应用。

迁移前检查：
```bash
alembic current    # 当前版本
alembic history    # 迁移历史
alembic check      # 检查是否有未应用的迁移
```

若迁移失败：
```bash
alembic downgrade -1    # 回退一个版本
# 或指定版本
alembic downgrade <revision_id>
```

### 6.4 Milvus Collection 迁移

若向量 Schema 发生变更（如 embedding 维度变化），需要重建 collection：

```bash
# 1. 备份 Milvus 数据（见第 7 节）

# 2. 删除旧 collection（谨慎！）
python3 scripts/reset_milvus.py --confirm

# 3. 重新 ingest 所有文档
python3 scripts/batch_ingest.py --dir /path/to/pdfs
```

> **警告**：重建 collection 会删除所有向量数据，需要重新 ingest 全部文档。

### 6.5 历史 SQLite → PostgreSQL 迁移

仅适用于旧版本本地 SQLite 数据迁移；当前正式部署不再以 SQLite 作为默认后端。

```bash
# 1. 启动 PostgreSQL
docker compose up -d postgres

# 2. 导出 SQLite 数据
python3 scripts/migrate_db.py \
  --from sqlite:///data/rag.db \
  --to postgresql://rag:password@localhost:5433/rag

# 3. 修改配置
# 在 rag_config.local.json 中更新 database.url

# 4. 验证迁移结果
python3 scripts/verify_migration.py
```

---

## 7. 数据备份与恢复

### 7.1 备份内容

| 数据 | 位置 | 重要性 |
|------|------|--------|
| PostgreSQL 数据库 | Docker volume `deepsea-rag-postgres-data` / `pg_dump` 导出文件 | 关键 |
| 上传的 PDF 文件 | `data/uploads/` 或配置的存储路径 | 关键 |
| Milvus 数据 | `volumes/milvus/` | 重要（可重建） |
| 配置文件 | `config/rag_config.local.json` | 关键 |
| Canvas 文档 | DB 中 | 随 DB 备份 |

### 7.2 备份脚本

```bash
#!/bin/bash
# 简单备份脚本

BACKUP_DIR="/backups/deepsea-rag/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份 PostgreSQL
docker exec deepsea-rag-postgres \
  pg_dump -U rag rag > "$BACKUP_DIR/rag.sql"

# 备份上传文件
rsync -av data/uploads/ "$BACKUP_DIR/uploads/"

# 备份配置
cp config/rag_config.local.json "$BACKUP_DIR/"

# 备份 Milvus（停服务后）
# docker compose stop milvus-prod
# cp -r volumes/milvus/ "$BACKUP_DIR/milvus/"
# docker compose start milvus-prod

echo "备份完成：$BACKUP_DIR"
```

### 7.3 PostgreSQL 备份

```bash
# 备份
docker exec deepsea-rag-postgres \
  pg_dump -U rag rag > /backups/rag-$(date +%Y%m%d).sql

# 恢复
docker exec -i deepsea-rag-postgres \
  psql -U rag rag < /backups/rag-20260311.sql
```

### 7.4 定时备份（cron）

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /opt/deepsea-rag/scripts/backup.sh >> /var/log/deepsea-backup.log 2>&1
```

---

## 8. 回滚方案

### 8.1 代码回滚

```bash
# 查看最近 tag
git tag -l | sort -V | tail -5

# 回滚到上一个版本
git checkout v7.7

# 重新安装依赖（如有变化）
pip install -r requirements.txt --no-cache-dir

# 回退数据库迁移（如有）
alembic downgrade -1

# 重新构建前端
cd frontend && npm ci && npm run build && cd ..

# 重启服务
sudo systemctl restart deepsea-rag-backend
```

### 8.2 数据库回滚

```bash
# 查看迁移历史
alembic history

# 回退到指定版本
alembic downgrade <revision_id>
```

### 8.3 紧急回滚（从备份恢复）

```bash
# 停服务
sudo systemctl stop deepsea-rag-backend

# 恢复 PostgreSQL
docker exec -i deepsea-rag-postgres \
  psql -U rag rag < /backups/deepsea-rag/<date>/rag.sql

# 恢复上传文件
rsync -av /backups/deepsea-rag/<date>/uploads/ data/uploads/

# 回滚代码
git checkout v7.7

# 启动服务
sudo systemctl start deepsea-rag-backend
```

---

## 9. 常见安装问题

### 9.1 Milvus 启动失败

**症状**：`docker compose ps` 显示 milvus 服务 `unhealthy`

```bash
# 查看日志
docker compose logs milvus-prod

# 常见原因：
# 1. etcd 未就绪 - 等待 30s 后重试
docker compose restart milvus-prod

# 2. volumes/milvus 权限问题
sudo chown -R $USER:$USER volumes/milvus/

# 3. GPU 驱动问题（prod profile）
nvidia-smi   # 验证 GPU 可用
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 9.2 PyTorch 安装冲突

**症状**：`pip install requirements.txt` 时出现 PyTorch 版本冲突

```bash
# 确认 torch 是通过 Conda 安装的
conda list | grep torch

# 若 pip 安装了 torch，先卸载
pip uninstall torch torchvision -y

# 重新用 Conda 安装
conda install -c pytorch -c nvidia "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y
```

### 9.3 transformers 版本冲突

**症状**：`ImportError: cannot import name 'RTDetrImageProcessor'`

```bash
pip install "transformers>=4.42.0,<5.0.0" --force-reinstall
pip install "huggingface-hub==0.36.0" --force-reinstall
```

### 9.4 Playwright 浏览器找不到

**症状**：`BrowserType.launch: Executable doesn't exist`

```bash
# Mac
playwright install chromium

# Ubuntu（需要系统依赖）
playwright install chromium --with-deps
```

### 9.5 Alembic 迁移失败

**症状**：`alembic upgrade head` 报错

```bash
# 检查当前迁移状态
alembic current

# 若数据库是空的（首次安装），先初始化
alembic stamp head  # 仅在全新数据库时使用

# 若迁移时数据库连接被占用，停止应用连接后重试
sudo systemctl stop deepsea-rag-backend
alembic upgrade head
sudo systemctl start deepsea-rag-backend
```

### 9.6 SSE 连接中断（Nginx 代理）

**症状**：SSE 流式响应在几十秒后中断

检查 Nginx 配置中的超时设置：
```nginx
proxy_read_timeout 3600s;
proxy_send_timeout 3600s;
proxy_buffering off;
```

若使用 HTTPS，确保 Nginx 版本 ≥ 1.18（更好的 chunked encoding 支持）。

### 9.7 Redis 连接失败

**症状**：`redis.exceptions.ConnectionError: Error connecting to Redis`

```bash
# 确认 Redis 运行
docker compose ps redis
docker compose logs redis

# 检查配置中的 Redis 地址
grep -r "redis" config/rag_config.json
# 默认：redis://localhost:6379

# 测试连接
redis-cli ping
```

### 9.8 GPU 内存不足

**症状**：CUDA out of memory 错误

```bash
# 减少 Milvus GPU 并发
# 在 docker-compose.yml 中调整 GPU 资源配置

# 或临时降低 BGE 批处理大小
# 在 config/rag_config.json 中调整 search.rerank_batch_size
```
