# 安装与依赖核对指南

本文档同时覆盖“安装步骤”和“依赖是否安装正确”的核对方法。

如需 Ubuntu 生产发布/迁移，请优先参考：`docs/release_migration_ubuntu.md`。

## 1. 依赖基线（已按当前项目核对）

### Python 运行时

- Python：`3.10+`（推荐 3.10）
- PyTorch / Torchvision / timm：必须走 Conda 安装
- 关键约束（见 `requirements.txt`）：
  - `transformers>=4.42.0,<5.0.0`
  - `huggingface-hub==0.36.0`
  - `fastapi>=0.100.0,<1.0.0`
  - `pydantic>=2.0.0,<3.0.0`
  - `openai>=1.0.0,<3.0.0`
  - `mcp>=1.26.0`

### 前端运行时

- Node.js：`^20.19.0 || >=22.12.0`（由 `frontend` 中 Vite 7 的引擎要求决定）
- npm：建议使用随 Node LTS 的版本
- 前端锁文件：`frontend/package-lock.json`（lockfileVersion=3）

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

## 4. 依赖核对（推荐执行）

推荐优先使用一键脚本：

```bash
bash scripts/verify_dependencies.sh
```

### 核对后端关键依赖

```bash
conda run -n deepsea-rag python --version
conda run -n deepsea-rag python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
conda run -n deepsea-rag python -c "import fastapi, mcp; print(fastapi.__version__, mcp.__version__)"
conda run -n deepsea-rag python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

### 核对前端依赖与 Node 版本

```bash
node -v
npm -v
cd frontend
npm ls --depth=0
cd ..
```

## 5. 启动前检查

```bash
bash scripts/00_preflight_check.sh
# 开发环境
docker compose --profile dev up -d
# 生产环境（Ubuntu 服务器）
# docker compose --profile prod up -d
bash scripts/00_healthcheck_docker.sh
```

## 6. 启动服务

```bash
# 一键全栈
bash scripts/start.sh

# 或只启动后端
python scripts/08_run_api.py --reload
```

### Ubuntu 发布建议（systemd）

发布环境建议使用 `systemd` 托管服务，不建议使用 `start.sh`（其后端命令默认包含 `--reload`，更适合开发）。
如需完整生产方案（含 Nginx 静态托管与 `/api` 反向代理），见 `docs/release_migration_ubuntu.md`。

后端 service（示例路径：`/etc/systemd/system/deepsea-rag-api.service`）：

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

前端 service（示例路径：`/etc/systemd/system/deepsea-rag-frontend.service`）：

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

sudo systemctl status deepsea-rag-api
sudo systemctl status deepsea-rag-frontend
sudo journalctl -u deepsea-rag-api -f
```

说明：

- 若 `conda` 不在 systemd 的 PATH，可把 `Environment=CONDA_EXE=conda` 改为实际路径（例如 `/opt/anaconda3/bin/conda`）。
- 若你的 Node 安装在 `deepsea-rag` 环境（如 `/opt/anaconda3/envs/deepsea-rag/bin/node`），上述写法可直接复用，不需要写死 node/npm 绝对路径。

## 7. 常见问题

### Q1: 为什么 requirements.txt 不写 torch？

因为 pip/conda 混装容易导致底层 C++ 算子冲突，项目明确要求 torch 生态由 Conda 安装。

### Q2: 前端安装失败（Node 版本过低）

升级到 Node `20.19+` 或 `22.12+` 后重试 `npm install`。

### Q3: Google/Scholar 不可用

先确认：

- `playwright install chromium` 已执行
- `config/rag_config.json` 中 `google_search.enabled=true`

### Q4: 离线模型如何运行？

```bash
export HF_LOCAL_FILES_ONLY=true
export MODEL_CACHE_ROOT=/path/to/hf_cache
```

## 8. 下一步

安装完成后建议按顺序执行：

```bash
bash scripts/verify_dependencies.sh
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
# 开发环境
bash scripts/start.sh
# 生产环境建议使用 systemd（见上文）
```

## 9. Deep Research 新交互（后台任务）

当前版本 Deep Research 默认采用“后台任务模式”：

- 点击“开始研究”后，任务在后端持续执行，前端页面关闭/刷新不会自动中断。
- 前端通过任务接口自动刷新状态；可在主界面查看进度。
- 只有点击“停止任务”或调用取消接口时才会终止任务。

相关接口：

- `POST /deep-research/submit`：提交后台任务，返回 `job_id`
- `GET /deep-research/jobs/{job_id}`：查询任务状态
- `GET /deep-research/jobs/{job_id}/events`：增量拉取进度事件
- `POST /deep-research/jobs/{job_id}/cancel`：停止任务

### 启动前设置（前端）

当前版本建议先在输入区 Deep Research 按钮旁的 `⚙` 弹窗设置：

- `Research Depth`（lite/comprehensive）
- `Output Language`
- `Per-step Models`（scope/plan/research/evaluate/write/verify/synthesize）
- `Strict step model resolution`

说明：

- 以上设置持久化到本地（跨会话保留）
- 对话框内仍可做“本次运行覆盖”

### 审核与人工介入（Drafting）

- 章节审核支持：
  - `通过此章`
  - `需要修改`
  - `重新确认`
  - `全部通过并触发整合`（批量）
- 章节缺口支持两种补充：
  - `material`：材料线索（文献/URL/数据线索）
  - `direct_info`：直接观点/约束
- 缺口补充提交后可在 `gap-supplements` 查看状态：
  - `pending`：待采纳
  - `consumed`：已进入重写流程

### 最终整合与保护机制

- 全部审核通过后触发最终整合（synthesize）
- 系统会执行全篇连贯性优化（global refine）
- 若整合后检测到引用/证据标签显著丢失，自动回退安全版本（`citation_guard_fallback`）

### 人工介入（调研不足时）

当研究结果提示覆盖不足（coverage gaps）时，可进行人工介入后继续：

- 文本介入：在确认阶段填写 `user_context`
- 文本模式：
  - `supporting`：补充上下文
  - `direct_injection`：强提示直接注入（高优先级）
- 临时材料：上传 `pdf/md/txt`，通过 `POST /deep-research/context-files` 提取后作为 `user_documents`

说明：上述介入材料仅用于本次 Deep Research，不写入持久本地知识库。
