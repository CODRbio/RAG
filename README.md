# DeepSea RAG

A full-stack Retrieval-Augmented Generation (RAG) platform built for deep-sea science research — and general academic literature workflows. It combines hybrid local+web retrieval, multi-provider LLM support, and a stateful deep research agent to accelerate literature review and knowledge synthesis.

---

## Features

- **Hybrid Retrieval** — dense (BGE-M3) + sparse (BM25) local search fused via RRF, with web aggregation across Tavily, Google Scholar, Semantic Scholar, and PubMed
- **Deep Research Agent** — LangGraph-powered, stateful, resumable literature review pipeline: clarify → outline → per-section research loop → write → verify → human review gate → synthesize
- **Evidence Quota Protection** — `fuse_pools_with_gap_protection()` guarantees that gap-fill and agent-supplemented evidence is not squeezed out by volume, even when the main pool dominates
- **Multi-Provider LLM** — unified dispatch across OpenAI, DeepSeek, Gemini, Claude, Kimi, Perplexity Sonar, and Qwen; thinking-model variants supported
- **Scholar Downloader** — automated PDF acquisition with strategy chain (direct → Playwright browser → Sci-Hub → BrightData → Anna's Archive) and captcha solving (2Captcha, CapSolver)
- **Canvas Editor** — live collaborative document editing with AI expand/condense/refine and inline citation insertion
- **Multi-Document Comparison** — structured analysis across 2–5 papers on configurable dimensions
- **Streaming Everything** — all long-running tasks (ingest, chat, deep research, scholar download) stream progress via SSE with `id:` lines for reconnect/resume
- **Graceful Reliability** — per-task checkpointing, task-level heartbeats, concurrent-slot management, and clean SIGTERM shutdown

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, FastAPI, Uvicorn, SQLModel, Alembic |
| Frontend | React 19, TypeScript, Vite 7, Zustand, Tailwind CSS |
| Vector DB | Milvus 2.5 (dense + sparse, BGE-M3 embedding) |
| Reranker | BGE Reranker, optional ColBERT (jina-colbert-v2) |
| Graph DB | NetworkX (entity/relation extraction) |
| Task Queue | Redis |
| Relational DB | PostgreSQL |
| Agent Framework | LangGraph |
| Browser Automation | Playwright (shared context pool) |
| Observability | OpenTelemetry, Prometheus, LangSmith (optional) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Frontend  (React + Vite)             │
│  Chat | Scholar | Research | Canvas | Compare│
└──────────────────┬──────────────────────────┘
                   │ HTTP / SSE
┌──────────────────▼──────────────────────────┐
│         Backend  (FastAPI)                   │
│  routes_chat  routes_ingest  routes_scholar  │
└────┬──────────────┬──────────────┬───────────┘
     │              │              │
  Chat/DR        Ingest         Scholar
  Worker         Worker         Worker
     │              │              │
┌────▼──────────────▼──────────────▼──────────┐
│  RetrievalService  │  LLMManager  │  DR Agent│
│  (hybrid search,   │  (multi-LLM  │  (LangGraph│
│   gap protection)  │   dispatch)  │   stateful)│
└────────────────────┴──────────────┴──────────┘
     │                                         │
 Milvus  Redis  PostgreSQL  Playwright  BGE Models
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ (3.10 recommended) |
| Conda / Miniconda | any recent |
| Node.js | `^20.19.0` or `>=22.12.0` |
| Docker + Compose Plugin | 24.0+ / 2.20+ |

> **Important**: PyTorch, Torchvision, and timm **must** be installed via Conda. Do not include them in `requirements.txt` — doing so causes C++ operator conflicts.

### 1. Clone & set up Python environment

```bash
git clone <repo-url> DeepSeaRAG
cd DeepSeaRAG

conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag

# Install PyTorch via Conda (macOS CPU)
conda install -c pytorch -c conda-forge "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y

# Install remaining Python dependencies
pip install -r requirements.txt --no-cache-dir

# Install browser for Scholar/download features
playwright install chromium
```

### 2. Start infrastructure

```bash
# macOS dev (CPU Milvus)
docker compose --profile dev up -d

# Ubuntu production (GPU Milvus)
docker compose --profile prod up -d
```

Waits for all services to reach `healthy`: PostgreSQL (5433), Milvus (19530), Redis (6379), etcd, MinIO.

### 3. Configure

```bash
cp config/rag_config.example.json config/rag_config.json
cp config/rag_config.example.json config/rag_config.local.json
```

Edit `config/rag_config.local.json` — at minimum, add one LLM API key:

```json
{
  "database": {
    "url": "postgresql+psycopg://rag:change-me@localhost:5433/rag"
  },
  "llm": {
    "platforms": {
      "claude": { "api_key": "sk-ant-xxx" }
    }
  }
}
```

The local file is git-ignored and takes priority over `rag_config.json`. See [docs/configuration.md](docs/configuration.md) for all options.

### 4. Initialize database

```bash
alembic upgrade head
```

### 5. Start the application

```bash
# Terminal 1 — backend
uvicorn src.api.server:app --host 0.0.0.0 --port 9999 --reload

# Terminal 2 — frontend
cd frontend && npm ci && npm run dev
```

Open **http://localhost:5173**.

---

## Configuration

Configuration is loaded in this priority order (highest wins):

```
UI / API request params
  > config/rag_config.local.json   (local overrides, git-ignored)
  > environment variables
  > config/rag_config.json         (main config)
  > code defaults
```

### API Keys

| Config path | Purpose | Required |
|-------------|---------|----------|
| `llm.platforms.<provider>.api_key` | LLM calls | At least one |
| `web_search.api_key` | Tavily web search | Recommended |
| `semantic_scholar.api_key` | S2 API (rate-limited without) | Optional |
| `ncbi.api_key` + `ncbi.email` | PubMed API | Optional |
| `content_fetcher.brightdata_api_key` | Full-text fetching proxy | Optional |
| `scholar_downloader.twocaptcha_api_key` | Captcha solving | Optional |
| `auth.secret_key` | JWT signing (**change in production**) | Required in prod |

---

## Supported LLM Providers

| Provider ID | Platform | Notes |
|-------------|---------|-------|
| `openai` / `openai-thinking` / `openai-mini` | OpenAI | GPT series; thinking injects `reasoning_effort` |
| `deepseek` / `deepseek-thinking` | DeepSeek | Chat + Reasoner |
| `gemini` / `gemini-thinking` / `gemini-vision` / `gemini-flash` | Google | Vision variants for figure parsing |
| `claude` / `claude-thinking` / `claude-haiku` | Anthropic | Thinking injects `budget_tokens` |
| `kimi` / `kimi-thinking` / `kimi-vision` | Moonshot | K2 series |
| `sonar` | Perplexity | Web-augmented; used for preliminary knowledge |
| `qwen` / `qwen-thinking` / `qwen-vision` | Alibaba | Qwen3 series |

The frontend model selector fetches available models from each platform's API at runtime — no static list to maintain.

---

## Deep Research Workflow

```
User topic
  → Clarify (AI questions + optional Sonar prelim knowledge)
  → User answers
  → Scope (ResearchBrief: scope, criteria, key questions, exclusions, time range)
  → Plan (background search + outline generation)
  → User confirms outline
  → Per-section loop:
       research_node (1+1+1 structured queries + agent supplement)
       evaluate_node (coverage check → gap-fill search if needed)
       generate_claims_node (comprehensive only)
       write_node (three-pool fuse: main 80% + gap 20% + agent 25% quota)
       verify_node (light / medium / severe → rewrite if severe)
  → review_gate_node (human approve / revise / skip)
       revise → targeted agent supplement + integrate rewrite → back to gate
  → synthesize_node (abstract + limitations + coherence refine + citation resolve)
```

### Depth presets

| Parameter | `lite` | `comprehensive` |
|-----------|-------:|----------------:|
| Max research rounds / section | 3 | 5 |
| Max verify rewrites / section | 1 | 2 |
| Coverage threshold | 0.60 | 0.80 |
| write_top_k baseline | 10 | 12 |
| write_top_k cap | 40 | 60 |
| verify severe threshold | 0.45 | 0.35 |

---

## Evidence Retrieval Design

The retrieval system uses a three-pool fusion model with quota protection to prevent gap-fill evidence from being drowned out:

```
main_pool  (main search results)
gap_pool   (eval_supplement — automatic coverage gap fill, 20% quota)
agent_pool (agent_supplement / revise_supplement — agent tool results, 25% quota)

fuse_pools_with_gap_protection(top_k, gap_ratio=0.2, agent_ratio=0.25, multiplier=3.0):
  1. Deduplicate by chunk_id (priority: agent > gap > main)
  2. BGE rerank all candidates together (rerank_k = top_k × 3.0)
  3. Take top_k slice
  4. Soft quota backfill: if gap count < ceil(top_k × 0.2), pull from reranked tail
  5. Return top_k results
```

`effective_write_top_k` derivation:
1. Explicit UI `write_top_k` → `max(preset_baseline, write_top_k)`
2. Explicit UI `step_top_k` → `max(preset_baseline, floor(step_top_k × 1.5))`
3. Neither → `preset_baseline` (lite=10, comprehensive=12)
4. Capped by `search_top_k_write_max`

---

## Project Structure

```
src/
├── api/            FastAPI routes (chat, ingest, scholar, canvas, compare)
├── llm/            LLM manager + agent tools + ReAct loop
├── retrieval/      RetrievalService, hybrid retriever, web search, dedup
├── collaboration/
│   ├── research/   LangGraph Deep Research agent + job store
│   ├── canvas/     Canvas document model
│   ├── memory/     Session / working / persistent memory
│   └── citation/   Citation resolution
├── indexing/       PDF ingest pipeline + embedder
└── tasks/          Redis queue, task state, dispatcher
config/
├── rag_config.json          Main configuration
└── rag_config.example.json  Template (copy to rag_config.local.json)
frontend/src/
├── api/            Axios + SSE clients
├── components/     React components
├── stores/         Zustand state
├── pages/          Page routes
└── i18n/           Chinese / English localization
docs/               Full documentation (see below)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture, module responsibilities, Chat/DR flow, design constraints |
| [docs/developer_guide.md](docs/developer_guide.md) | Dev setup, module conventions, evidence flow, extending tools/providers |
| [docs/api_reference.md](docs/api_reference.md) | All HTTP/SSE endpoints with request/response schemas |
| [docs/installation_and_migration.md](docs/installation_and_migration.md) | Full install guide (Mac + Ubuntu), systemd, Nginx, upgrades, rollback |
| [docs/operations_and_troubleshooting.md](docs/operations_and_troubleshooting.md) | Service management, monitoring, fault diagnosis, emergency checklist |
| [docs/configuration.md](docs/configuration.md) | All config options and environment variables |
| [docs/chat_research_workflow_contract.md](docs/chat_research_workflow_contract.md) | Authoritative spec for current Chat + Deep Research implementation |
| [docs/evidence_retention.md](docs/evidence_retention.md) | Evidence pool mechanics, step_top_k / write_top_k semantics, fuse algorithm |
| [docs/task_management.md](docs/task_management.md) | Long-task contract: slots, heartbeat, checkpoint, SSE resume |
| [install.md](install.md) | Quick install reference + dependency checklist |

---

## Development

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_chat_agent_refusion.py -v -s

# Enable mock LLM (no API calls)
# Set llm.dry_run: true in rag_config.local.json
```

### Key design constraints (do not break)

| Constraint | Reason |
|-----------|--------|
| `eval_supplement` **must** enter gap pool | Otherwise gap-fill evidence competes equally with main results and gets squeezed out |
| `agent_supplement` **must** enter agent pool | Same reason — agent evidence needs its own quota |
| `revise_supplement` **must** be written to section pool | Guarantees citation traceability in synthesize phase |
| `write_stage` **must not** be recycled back into the pool | Prevents write-phase evidence from polluting rerank weights |
| All pool-bound searches use `pool_only=True` | Prevents premature truncation before global rerank |
| No score-threshold filter after fuse | Breaks gap/agent quota protection |
| Chat BGE rerank: max 2 passes | §5¾ (main+gap) + optional agent append — no intermediate rerankings |
| `search_scholar` tool must close aiohttp session in `finally` | Prevents `RuntimeError: Event loop is closed` on agent teardown |

---

## Production Deployment

For Ubuntu production deployment (systemd + Nginx + HTTPS), see [docs/installation_and_migration.md](docs/installation_and_migration.md).

Quick checklist:
- [ ] Replace `auth.secret_key` with a randomly generated 64-character key
- [ ] Use `docker compose --profile prod` (GPU Milvus)
- [ ] Build frontend: `cd frontend && npm ci && npm run build`
- [ ] Configure Nginx with `proxy_buffering off` and `proxy_read_timeout 3600s` for SSE
- [ ] Set up HTTPS (SSE connections may be interrupted by proxies on plain HTTP)
- [ ] Run `alembic upgrade head` before starting the service

---

## License

This project is for internal/research use. Contact the maintainers for licensing questions.
