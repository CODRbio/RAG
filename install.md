# 安装与依赖核对指南

本文档同时覆盖"安装步骤"和"依赖是否安装正确"的核对方法。

如需 Ubuntu 生产发布/迁移，请优先参考：`docs/release_migration_ubuntu.md`。

## 1. 依赖基线

### Python 运行时

- Python：`3.10+`（推荐 3.10）
- PyTorch / Torchvision / timm：必须走 Conda 安装
- 关键约束（见 `requirements.txt`）：
  - `transformers>=4.42.0,<5.0.0`
  - `huggingface-hub==0.36.0`
  - `fastapi>=0.100.0,<1.0.0`
  - `pydantic>=2.0.0,<3.0.0`
  - `openai>=1.0.0,<3.0.0`
  - `anthropic>=0.18.0`
  - `mcp>=1.26.0`
  - `langgraph>=0.2.0`
  - `pymilvus[model]>=2.5.0`

### 前端运行时

- Node.js：`^20.19.0 || >=22.12.0`（由 Vite 7 的引擎要求决定）
- npm：建议使用随 Node LTS 的版本
- 前端锁文件：`frontend/package-lock.json`（lockfileVersion=3）

### 系统依赖（Ubuntu）

```bash
sudo apt install -y \
  git curl wget unzip ca-certificates gnupg lsb-release \
  build-essential pkg-config python3-dev \
  libssl-dev libffi-dev libsqlite3-dev \
  docker.io docker-compose-plugin
```

## 2. 创建环境并安装（后端）

```bash
conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag

# 必须 Conda 安装（不要放进 requirements.txt）
conda install -c pytorch -c conda-forge "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y

# Python 依赖
pip install -r requirements.txt --no-cache-dir

# 浏览器自动化（Google/Scholar 功能依赖）
playwright install chromium
```

## 3. 安装（前端）

```bash
cd frontend
npm install
cd ..
```

若要严格按 lock 还原环境，可用：

```bash
cd frontend
npm ci
cd ..
```

## 4. 环境配置

```bash
# 复制配置模板
cp config/rag_config.example.json config/rag_config.json
cp config/rag_config.local.example.json config/rag_config.local.json
cp .env.example .env

# 编辑 .env，设置 Milvus 地址、设备类型等
# 编辑 config/rag_config.local.json，填入各 LLM provider 的 API Key
```

密钥注入优先级：

1. 环境变量（推荐生产）：`RAG_LLM__OPENAI__API_KEY`、`RAG_LLM__DEEPSEEK__API_KEY` 等
2. `config/rag_config.local.json`（推荐开发）
3. `config/rag_config.json`（不建议写真实密钥）

## 5. 依赖核对（推荐执行）

优先使用一键脚本：

```bash
bash scripts/verify_dependencies.sh
```

### 核对后端关键依赖

```bash
conda run -n deepsea-rag python --version
conda run -n deepsea-rag python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
conda run -n deepsea-rag python -c "import fastapi, mcp; print(fastapi.__version__, mcp.__version__)"
conda run -n deepsea-rag python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
conda run -n deepsea-rag python -c "import langgraph; print(langgraph.__version__)"
```

### 核对前端依赖与 Node 版本

```bash
node -v
npm -v
cd frontend && npm ls --depth=0 && cd ..
```

## 6. 启动前检查

```bash
bash scripts/00_preflight_check.sh
# 开发环境（Mac CPU）
docker compose --profile dev up -d
# 生产环境（Linux GPU）
# docker compose --profile prod up -d
bash scripts/00_healthcheck_docker.sh
```

## 7. 初始化与入库

```bash
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
```

## 8. 启动服务

```bash
# 一键全栈（开发）
bash scripts/start.sh

# 或只启动后端（支持热更新）
python scripts/08_run_api.py --reload

# 或分别启动
bash scripts/start.sh --backend-only
bash scripts/start.sh --frontend-only
```

- 前端：`http://localhost:5173`
- 后端 Swagger：`http://127.0.0.1:9999/docs`

### 生产部署建议（systemd）

发布环境建议使用 `systemd` 托管服务，不建议使用 `start.sh`（其后端默认包含 `--reload`，更适合开发）。
完整生产方案（含 Nginx 静态托管与 `/api` 反向代理）见 `docs/release_migration_ubuntu.md`。

后端 service 示例：

```ini
[Unit]
Description=DeepSea RAG FastAPI Service
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/path/to/RAG
Environment=PYTHONUNBUFFERED=1
Environment=API_HOST=0.0.0.0
Environment=API_PORT=9999
EnvironmentFile=-/path/to/RAG/.env
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag python scripts/08_run_api.py --host 0.0.0.0 --port 9999'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

前端 service 示例：

```ini
[Unit]
Description=DeepSea RAG Frontend Preview
After=network.target

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/path/to/RAG/frontend
Environment=NODE_ENV=production
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag npm run preview -- --host 0.0.0.0 --port 5173'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

启用与管理：

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsea-rag-api deepsea-rag-frontend
sudo systemctl start deepsea-rag-api deepsea-rag-frontend
sudo journalctl -u deepsea-rag-api -f
```

> 若 `conda` 不在 systemd 的 PATH，可把 `CONDA_EXE` 改为实际路径（如 `/opt/anaconda3/bin/conda`）。

## 9. Deep Research 使用说明

当前版本 Deep Research 默认采用"后台任务模式"：

- 点击"开始研究"后，任务在后端持续执行，前端页面关闭/刷新不会自动中断
- 前端通过任务接口自动刷新状态；可在主界面查看进度
- 只有点击"停止任务"或调用取消接口时才会终止任务

### 启动前设置（前端 `⚙` 弹窗）

- `Research Depth`（lite / comprehensive）
- `Output Language`
- `Per-step Models`（scope / plan / research / evaluate / write / verify / synthesize）
- `Strict step model resolution`

以上设置持久化到本地（跨会话保留），对话框内仍可做"本次运行覆盖"。

### 审核与人工介入

- 章节审核：通过此章 / 需要修改 / 重新确认 / 全部通过并触发整合
- 章节缺口补充：
  - `material`：材料线索（文献/URL/数据线索）
  - `direct_info`：直接观点/约束
- 人工材料注入：上传 `pdf/md/txt` 作为临时材料（仅本次任务，不写入持久库）

### 最终整合流程

1. 全部审核通过 → 生成 Abstract
2. 生成 Limitations and Future Directions
3. 聚合 Open Gaps 研究议程
4. 全篇连贯性重写（global coherence refine）
5. 引用保护检查（失败则回退安全版本）
6. Canvas 切换到 `refine` 阶段

### 相关接口

- `POST /deep-research/submit`：提交后台任务，返回 `job_id`
- `GET /deep-research/jobs/{job_id}`：查询任务状态
- `GET /deep-research/jobs/{job_id}/events`：增量拉取进度事件
- `POST /deep-research/jobs/{job_id}/cancel`：停止任务
- `POST /deep-research/jobs/{job_id}/review`：提交章节审核
- `POST /deep-research/jobs/{job_id}/gap-supplement`：提交缺口补充
- `GET /deep-research/jobs/{job_id}/insights`：查看研究洞察

## 10. 常见问题

### Q1: 为什么 requirements.txt 不写 torch？

因为 pip/conda 混装容易导致底层 C++ 算子冲突，项目明确要求 torch 生态由 Conda 安装。

### Q2: 前端安装失败（Node 版本过低）

升级到 Node `20.19+` 或 `22.12+` 后重试 `npm install`。

### Q3: Google/Scholar 不可用

- 确认 `playwright install chromium` 已执行
- 确认 `config/rag_config.json` 中 `google_search.enabled=true`

### Q4: 离线模型如何运行？

```bash
export HF_LOCAL_FILES_ONLY=true
export MODEL_CACHE_ROOT=/path/to/hf_cache
```

### Q5: Milvus 启动失败

```bash
bash scripts/00_healthcheck_docker.sh
docker compose --profile dev logs milvus-dev
```

## 11. 下一步

安装完成后建议按顺序执行：

```bash
bash scripts/verify_dependencies.sh
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
bash scripts/start.sh
```

生产环境请参考 `docs/release_migration_ubuntu.md` 使用 systemd + Nginx 部署。
