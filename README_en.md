# DeepSea RAG — User Guide

A full-stack RAG system for scientific research, featuring hybrid retrieval, multi-turn conversation, ReAct Agent, Deep Research automated review generation, canvas collaboration, multi-document comparison, and knowledge graph exploration.

> This document is for **users** — it explains every feature, how to use it, and recommended scenarios.  
> For developer docs, see `docs/developer_guide.md`. For installation, see `install.md`.

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Overview](#system-overview)
- [Core Concept: Data Source Boundary](#core-concept-data-source-boundary)
- [Chat Module](#chat-module)
  - [Basic Conversation & Intent Detection](#basic-conversation--intent-detection)
  - [Retrieval Modes](#retrieval-modes)
  - [Search Source Configuration](#search-source-configuration)
  - [ReAct Agent Mode](#react-agent-mode)
  - [Query Optimizer](#query-optimizer)
  - [Content Fetcher](#content-fetcher)
  - [Year Filtering](#year-filtering)
  - [Advanced Retrieval Parameters](#advanced-retrieval-parameters)
- [Deep Research Module](#deep-research-module)
  - [Workflow Overview](#workflow-overview)
  - [Phase 1: Clarification](#phase-1-clarification)
  - [Phase 2: Outline Confirmation](#phase-2-outline-confirmation)
  - [Phase 3: Background Execution & Monitoring](#phase-3-background-execution--monitoring)
  - [Deep Research Settings](#deep-research-settings)
- [Canvas Collaboration](#canvas-collaboration)
- [Multi-Document Comparison](#multi-document-comparison)
- [Knowledge Graph Explorer](#knowledge-graph-explorer)
- [Document Ingestion](#document-ingestion)
- [Command Palette](#command-palette)
- [Feature Comparison Tables](#feature-comparison-tables)
- [Recommended Use Cases](#recommended-use-cases)
- [FAQ](#faq)

---

## Quick Start

```bash
# 1. Launch services
bash scripts/start.sh

# 2. Open browser
# Frontend: http://localhost:5173
# Backend API docs: http://127.0.0.1:9999/docs
```

First-time checklist:
1. Verify Milvus connection in the sidebar (green = connected)
2. Select a Collection (vector store)
3. Type a question in the input box and press Enter

---

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                       Frontend UI                         │
│  ┌───────────┐  ┌───────────────────┐  ┌──────────────┐  │
│  │  Sidebar   │  │   Chat Window     │  │ Canvas Panel │  │
│  │ · Retrieval│  │  · Conversation   │  │ · Review edit│  │
│  │ · Sources  │  │  · Agent traces   │  │ · Snapshots  │  │
│  │ · Agent    │  │  · Citation cards │  │ · Export     │  │
│  │ · History  │  │  · Evidence stats │  │              │  │
│  └───────────┘  └───────────────────┘  └──────────────┘  │
│  ┌───────────────────────────────────────────────────┐    │
│  │  Input: ⚙ Settings | 🔭 Deep Research | Box | Send │    │
│  └───────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Core Concept: Data Source Boundary

**The search sources you check in the sidebar define the "data source boundary" for the entire system.** Whether it's normal Chat, ReAct Agent, or Deep Research, all retrieval operations strictly respect this boundary:

```
Sidebar Configuration
├── Local RAG toggle ──→ Controls local vector store search
├── Web Search toggle ──→ Controls web search activation
└── Source checkboxes ──→ Limits available web search engines
         │
         ▼
    ┌─────────────────────────────────────┐
    │  All modes share the same boundary   │
    │                                      │
    │  · Normal Chat retrieval             │
    │  · ReAct Agent's toolbox             │
    │  · Every Deep Research search call   │
    │  · Query Optimizer's query targets   │
    └─────────────────────────────────────┘
```

**Unchecked sources are never called**, even when the Agent reasons autonomously.

---

## Chat Module

### Basic Conversation & Intent Detection

Type a question in the input box and press **Enter**. The system runs three-layer intent detection to determine whether your question needs retrieval:

| Layer | Mechanism | Description |
|:-----:|-----------|-------------|
| 1st | Keyword regex | Message contains keywords like "literature", "compare", "which", "find me" → retrieval (zero cost) |
| 2nd | LLM classification | LLM judges `chat` (casual) vs `rag` (needs retrieval), ~300 tokens |
| 3rd | Conservative fallback | If LLM output is unrecognizable → default to retrieval (better safe than sorry) |

How the result affects behavior:

| Detection | Retrieval | Agent | System Prompt |
|:---------:|:---------:|:-----:|:-------------:|
| **chat** (casual) | Skipped | Force-disabled | Pure conversation prompt |
| **rag** (needs retrieval) | Executed | Per your toggle | RAG synthesis prompt |

This means:
- You send "hello" → detected as casual → no retrieval, no Agent, direct answer
- You send "which microbes oxidize methane in cold seeps" → keyword hit → retrieval
- Even with Agent enabled, casual questions will not trigger tool calls

### Retrieval Modes

The system determines retrieval mode from your toggle combination:

| Local RAG | Web Search | Effective Mode | Description |
|:---------:|:----------:|:--------------:|-------------|
| ✅ | ✅ | `hybrid` | Local vector store + web search, results merged and deduplicated |
| ✅ | ❌ | `local` | Local vector store (Milvus) only |
| ❌ | ✅ | `web` | Web search only |
| ❌ | ❌ | `none` | No retrieval, pure LLM conversation |

**Recommendation**: Use `hybrid` for most research scenarios — it combines your local literature with the latest web information.

### Search Source Configuration

In the sidebar **Web Search** section, you can independently toggle each search engine:

| Source | Characteristics | Best For | Default |
|--------|----------------|----------|:-------:|
| **Tavily API** | Fast general web search, broad coverage | General questions, news, tech docs | ✅ On |
| **Google Search** | General search (via Playwright) | Fallback when Tavily is unavailable | ❌ Off |
| **Google Scholar** | Academic paper search, year filtering | Broad academic literature retrieval | ❌ Off |
| **Semantic Scholar** | Academic API, structured metadata | Precise academic search (DOI/authors/abstract) | ❌ Off |
| **NCBI PubMed** | Biomedical database (E-Utilities) | Medicine, biology, genomics, marine ecology | ❌ Off |

Each source has an independent **Top-K** setting (maximum results to fetch from this source, default 3-5).

> **About Web Source Threshold**  
> There is no Threshold slider for web search sources. Different search engine APIs return results without a unified similarity score, making threshold filtering impractical. Web result quality is controlled entirely by the downstream **ColBERT Reranker**.  
> **Similarity Threshold only applies to Local RAG results** (see "Advanced Retrieval Parameters" below).

#### Web Search Data Flow

```
Your per-source Top-K settings
  ↓ Each source fetches independently (concurrent)
  ↓ Cross-source URL deduplication (keeps higher-weight source version)
  ↓ ColBERT Reranker re-ranking
  ↓ Merge with local results
  ↓ Final Top-K truncation
  → Evidence sent to LLM
```

For example, if you enable Tavily (Top-K=5) + Scholar (Top-K=3) + NCBI (Top-K=5), each source independently fetches 5, 3, and 5 results respectively. After deduplication and re-ranking, the Final Top-K setting determines how many total results are kept.

### ReAct Agent Mode

The **ReAct Agent** toggle in the sidebar **Agent Mode** section controls how the LLM behaves.

#### Agent Off (Normal Mode)

```
User question → Intent detection (needs retrieval?)
                  ├─ No → LLM answers directly (pure conversation)
                  └─ Yes → One-shot retrieval (your checked sources) → LLM answers from results
```

- Retrieval runs exactly once
- LLM sees the results and generates an answer directly
- Fast, suitable for simple questions

#### Agent On (ReAct Mode)

```
User question → Intent detection (needs retrieval?)
                  ├─ No → LLM answers directly (Agent is skipped)
                  └─ Yes → Retrieval (your checked sources) → Results injected into System Prompt
                            → Agent reasoning loop (up to 8 rounds)
                                 │  LLM reads existing evidence, judges if sufficient
                                 │  When insufficient, autonomously calls tools
                                 ├→ search_local (local search)
                                 ├→ search_web (Tavily/Google)
                                 ├→ search_scholar (Scholar/Semantic Scholar)
                                 ├→ search_ncbi (PubMed)
                                 ├→ explore_graph (graph exploration)
                                 ├→ compare_papers (paper comparison)
                                 ├→ canvas (canvas operations)
                                 └→ get_citations (citation management)
```

Key details:

- **The Agent's tools are dynamically routed**, not a fixed set. The system selects only relevant tools based on your source checkboxes, message content, and current stage
- **The Agent can only call tools corresponding to your checked sources**. For example, if you only checked Tavily + NCBI, the Agent gets `search_web` and `search_ncbi` only — `search_scholar` will not appear
- The first retrieval round's results are already in the System Prompt. The Agent evaluates whether this evidence is sufficient before initiating additional searches
- Tool calls are visible in real-time in the **Tool Trace** panel

#### Tool Routing Map

| Your checked sources | Agent receives |
|:---:|:---:|
| Tavily and/or Google | `search_web` |
| Google Scholar and/or Semantic Scholar | `search_scholar` |
| NCBI PubMed | `search_ncbi` |
| None checked | No web search tools (only `search_local`) |

Additionally, these tools are auto-mounted based on message content and stage (independent of source checkboxes):

| Tool | Trigger |
|------|---------|
| `explore_graph` | Message mentions relationship, graph, network keywords |
| `compare_papers` / `run_code` | Message mentions comparison, statistics, code keywords |
| `canvas` / `get_citations` | Currently in drafting/refine stage, or message mentions canvas/citation |

#### When to Enable Agent?

| Scenario | Recommendation | Reason |
|----------|:--------------:|--------|
| Simple factual query | Off | One retrieval is enough; Agent adds latency |
| Complex multi-step question | **On** | Agent can iteratively search and cross-validate |
| Multi-source comparison needed | **On** | Agent can call multiple search engines |
| Graph relationship exploration | **On** | Agent can call explore_graph |
| Casual chat / translation | No effect | Intent detection skips retrieval; Agent is not activated |

### Query Optimizer

The **Query Optimizer** toggle in the sidebar **Query Enhancement** section.

When enabled, the system uses an LLM to transform your question into optimized queries for each search engine:

- **Tavily/Google**: Natural language queries
- **Google Scholar**: Academic keyword combinations
- **NCBI PubMed**: MeSH-style medical terminology
- Bilingual query generation (Chinese/English)
- Up to N queries per engine (configurable 1-5, default 3)
- **Only generates queries for your checked sources** — no tokens wasted on unchecked engines

**Recommendation**: Enable. Especially effective for academic searches — significantly improves recall.

### Content Fetcher

The **Content Fetcher** toggle in the sidebar.

When enabled, the system fetches full-text content from web search result URLs instead of just snippets.

- Three-tier fallback: trafilatura → BrightData → Playwright
- Automatic academic domain detection
- Results cached (1 hour TTL)

**Recommendation**: Off by default. Enable only when you need deep full-text analysis — it increases response time.

### Year Filtering

Configured in the Deep Research settings popover (⚙ button) under **Year Window**.

- **Start Year** / **End Year**: Hard filter — only keeps publications within this range
- Applies to: Google Scholar, Semantic Scholar, NCBI PubMed, local RAG (if year metadata exists)
- Tavily and Google do not support native year filtering
- Leave empty = no restriction
- **Shared between Chat and Deep Research**

**Recommendation**: Set a reasonable year window for review writing (e.g., 2015-2025) to avoid outdated literature.

### Advanced Retrieval Parameters

| Parameter | Location | Scope | Description | Default |
|-----------|----------|-------|-------------|:-------:|
| Local Top-K | Sidebar · Retrieval Config | Local RAG | Max chunks returned from local vector retrieval | 5 |
| Similarity Threshold | Sidebar · Merge Params | **Local RAG only** | Local chunks below this similarity are filtered; **does not apply to web results** | 0.5 |
| Final Top-K | Sidebar · Merge Params | Global (Local + Web merged) | Final result count after all sources are merged | 10 |
| Per-source Web Top-K | Sidebar · Web Sources | Per web source | Independent fetch limit for each source, before cross-source dedup + rerank | 3-5 |
| HippoRAG | Sidebar · Retrieval Config | Local RAG | Graph-enhanced retrieval (PageRank expansion) | Off |
| ColBERT Reranker | Sidebar · Retrieval Config | Web + Local results | Re-rank retrieval results (ColBERT model) | On |

#### Parameter Behavior Across Modes

| Parameter | Normal Chat (no Agent) | ReAct Agent | Deep Research |
|-----------|:---------------------:|:-----------:|:-------------:|
| Per-source Web Top-K | ✅ Strictly respected | ❌ Agent tool calls use hardcoded defaults | ✅ Strictly respected |
| Local Top-K | ✅ Strictly respected | ✅ Pre-retrieval respects it; Agent tool calls use defaults | ✅ Respected (auto-scaled) |
| Similarity Threshold | ✅ Strictly respected | ✅ Pre-retrieval respects it; Agent tool calls don't pass it | ✅ Strictly respected |
| Final Top-K | ✅ Strictly respected | ✅ Pre-retrieval respects it; Agent tool calls don't pass it | ✅ Respected (auto-scaled) |

> **Agent Mode Note**: When the ReAct Agent autonomously initiates additional searches (via `search_web`, `search_scholar`, `search_ncbi`, etc.), these tool calls use hardcoded defaults from the tool schemas (`top_k=10` or `limit=5`), **not your sidebar per-source Top-K or Threshold settings**. Only the initial pre-retrieval (the retrieval that runs before the Agent loop starts) fully respects your configuration.

---

## Deep Research Module

Deep Research is the system's flagship feature — it automates the entire research pipeline from topic selection to a complete review manuscript.

**Deep Research strictly respects your data source configuration** — every search call (scope, plan, research, write, verify) only uses your checked sources.

### Workflow Overview

```
Enter topic → Clarify intent → Generate outline → Confirm settings → Background execution → Review complete
               Phase 1          Phase 2           Phase 2            Phase 3
```

Internal execution pipeline:

```
Scope (define boundaries)
  → Plan (generate outline)
    → Research (recursive search per section)    ← uses your checked sources
      → Evaluate (coverage assessment)
        → Write (draft each section)             ← uses your checked sources for evidence
          → Verify (citation verification)       ← uses your checked sources for verification
            → Synthesize (full integration)
```

### How to Start

Three ways to launch Deep Research:

1. **🔭 Deep Research button** (left of input box): Type a topic, then click
2. **Command palette**: Type `/auto your research topic`
3. **⚙ Settings button**: Configure defaults first, then launch

### Phase 1: Clarification

The system generates 2-5 clarification questions to refine the research scope:

- Question types: text input, single-choice, multi-choice
- You can skip any question (the system uses defaults)
- More detailed answers lead to more precise research

Click **"Generate Outline"** to proceed.

### Phase 2: Outline Confirmation

After the system generates a research outline, you can:

| Action | Description |
|--------|-------------|
| **Edit section titles** | Modify text directly |
| **Drag to reorder** | Drag the grip icon to rearrange |
| **Add / remove sections** | Click + or trash icon |
| **Choose research depth** | Lite (5-15 min) or Comprehensive (20-60 min) |
| **Set output language** | Auto / English / Chinese |
| **Override per-step models** | Assign different LLMs to different phases |
| **Configure human intervention** | Choose which phases need manual review |
| **Add context** | Text input or upload temporary files (pdf/md/txt) |

#### Research Depth Comparison

| Dimension | Lite | Comprehensive |
|-----------|------|---------------|
| Estimated time | 5-15 minutes | 20-60 minutes |
| Queries per section | 4 (recall + precision) | 8 (recall + precision) |
| Tiered Top-K | 18 / 10 / 10 | 30 / 15 / 12 |
| Coverage threshold | ≥ 60% | ≥ 80% |
| Best for | Quick survey, initial exploration | Formal review, academic writing |

#### Human Intervention Options

| Phase | Description | Default |
|-------|-------------|:-------:|
| Clarify intent | Answer clarification questions | Required |
| Confirm outline | Edit and approve outline | Required |
| Section review | Review/edit each section after drafting | Optional |
| Refine directives | Provide revision instructions before synthesis | Optional |
| Skip claim generation | Skip core claim extraction before writing | Optional |

Click **"Confirm and Start Research"** to begin execution.

### Phase 3: Background Execution & Monitoring

After submission, the task runs in the background. You can:

- **Safely close the browser**: The task continues on the server
- **Reopen the page**: Task state auto-recovers
- **Monitor in real-time**: Progress logs, coverage curves, cost status
- **Cancel the task**: Click the stop button

The monitoring panel shows:

| Metric | Description |
|--------|-------------|
| Coverage | Evidence coverage per section (0-1) |
| Research rounds | Search iterations per section |
| Cost state | normal → warn → force (forced summary) |
| Self-correction count | Automatic search strategy adjustments |
| Early stop count | Auto-termination when coverage gain plateaus |

Upon completion, the review is automatically written to the canvas, including:
- Full review text (with citation annotations)
- Abstract
- Limitations
- Open Gaps research agenda
- Reference list

### Deep Research Settings

Open the settings popover via the **⚙ button** next to the input box. These settings persist across sessions:

| Setting | Description | Default |
|---------|-------------|:-------:|
| Research Depth | Lite or Comprehensive | Comprehensive |
| Output Language | Auto / English / Chinese | Auto |
| Year Window | Publication year filter (shared with Chat) | No limit |
| Per-step Models | Different LLMs for different phases | scope=sonar-pro, rest=global |
| Strict Mode | Abort if step model fails | Off |
| Skip Claim Generation | Skip claim extraction before writing | Off |

---

## Canvas Collaboration

The canvas is the core workspace for review editing, with four stages:

| Stage | Function |
|-------|----------|
| **Explore** | Browse retrieved evidence and sources |
| **Outline** | Edit the review structure |
| **Drafting** | Write and edit sections |
| **Refine** | Polish, annotate, and finalize |

Supported operations:
- Real-time Markdown editing
- Version snapshots (up to 50 versions)
- AI-assisted paragraph editing
- Citation management and formatting
- Export: Markdown / DOCX / RIS (citations) / JSON

---

## Multi-Document Comparison

Compare 2-5 papers with structured analysis:

1. After generating citations in chat, switch to the **Compare** tab
2. Select papers from chat citations or local library
3. The system generates a comparison matrix: objectives, methods, findings, limitations
4. Includes narrative analysis

---

## Knowledge Graph Explorer

Switch to the **Graph** tab:

- Search entities (species, locations, phenomena, methods, substances, etc.)
- Interactive force-directed graph visualization
- Control exploration depth (number of hops)
- Click nodes for details and associated chunks
- Different entity types shown in different colors

---

## Document Ingestion

Switch to the **Ingest** tab:

1. Upload PDF files (batch supported)
2. Select target Collection
3. Optional: enable table/figure enrichment
4. The system automatically: parses PDF → structured chunking → vector embedding → Milvus indexing → graph building
5. Track ingestion progress in real-time

---

## Command Palette

Type `/` in the input box to open the command palette:

| Command | Function | Example |
|---------|----------|---------|
| `/auto` | Start Deep Research | `/auto deep sea cold seep methane oxidation` |
| `/search` | Execute search | `/search cold seep ecosystem` |
| `/outline` | Generate outline | `/outline` |
| `/draft` | Draft a section | `/draft introduction` |
| `/export` | Export document | `/export` |
| `/status` | View system status | `/status` |

---

## Feature Comparison Tables

### Normal Chat vs ReAct Agent vs Deep Research

| Dimension | Normal Chat | ReAct Agent | Deep Research |
|-----------|:-----------:|:-----------:|:-------------:|
| Data sources | Your checked sources | Your checked sources | Your checked sources |
| Retrieval rounds | 1 | 1 + Agent supplements as needed (up to 8 rounds) | Dozens (recursive per section) |
| Search terms | System generates 1 set | First round by system + Agent supplements | Query optimizer generates batches |
| Tool calling | None | Dynamically routed (based on your config + message) | Built-in full research pipeline |
| Output length | Hundreds of words | Hundreds to thousands | Thousands to tens of thousands (full review) |
| Response time | 5-30 seconds | 10-60 seconds | 5-60 minutes |
| Human intervention | None | None | Optional (clarify/outline/review) |
| Citation management | Basic | Basic | Full (verification + protection) |
| Canvas integration | Optional | Optional | Automatic |
| Background execution | No | No | Yes |
| Intent detection | Auto (casual skips retrieval) | Auto (casual disables Agent) | N/A (always executes) |
| Best for | Quick Q&A | Complex multi-step questions | Review writing |

### Search Source Comparison

| Source | Type | Speed | Academic | Native Year Filter | Best For |
|--------|:----:|:-----:|:--------:|:-----------:|---------|
| Tavily | General | ★★★ | ★☆☆ | ❌ | General questions, tech docs |
| Google | General | ★★☆ | ★☆☆ | ❌ | Tavily fallback |
| Google Scholar | Academic | ★★☆ | ★★★ | ✅ | Broad academic retrieval |
| Semantic Scholar | Academic | ★★★ | ★★★ | ✅ | Precise academic search (DOI) |
| NCBI PubMed | Specialized | ★★★ | ★★★ | ✅ | Biomedicine, genomics, marine ecology |

### Retrieval Enhancement Comparison

| Feature | Effect | Overhead | Recommended |
|---------|--------|----------|:-----------:|
| Query Optimizer | Optimizes query format for your checked engines | +2-5s | ✅ |
| Content Fetcher | Fetches full web page text | +5-15s | As needed |
| HippoRAG | Graph-enhanced retrieval (PageRank) | +1-3s | As needed |
| ColBERT Reranker | Re-ranks results for relevance | +1-2s | ✅ |

---

## Recommended Use Cases

### Scenario 1: Quick Literature Q&A

> "What are the main microorganisms involved in anaerobic methane oxidation in cold seep ecosystems?"

**Recommended config**:
- Local RAG: ✅ | Web Search: ✅
- Sources: Tavily + NCBI PubMed
- Agent: Off
- Query Optimizer: ✅

**Why**: Simple factual query — one retrieval round is sufficient. NCBI precisely targets biomedical literature.

---

### Scenario 2: Multi-Source Cross-Validation

> "Compare findings from different studies on microbial diversity at deep-sea hydrothermal vents"

**Recommended config**:
- Local RAG: ✅ | Web Search: ✅
- Sources: Tavily + Google Scholar + Semantic Scholar + NCBI
- Agent: **On**
- Query Optimizer: ✅

**Why**: Requires multi-source comparison. Agent can supplement the initial retrieval with autonomous searches and cross-validate.

---

### Scenario 3: Review Writing

> "Write a review on carbon cycling in deep-sea cold seep ecosystems"

**Recommended config**:
- Use **Deep Research** (`/auto` command)
- Depth: Comprehensive
- Sources: All enabled
- Year range: 2015-2025
- Human intervention: Enable section review

**Why**: Reviews require systematic literature surveys. Deep Research automates the entire pipeline from topic to final manuscript.

---

### Scenario 4: Quick Survey

> "Get an overview of CRISPR applications in marine microorganisms"

**Recommended config**:
- Use **Deep Research** (`/auto` command)
- Depth: **Lite**
- Sources: NCBI + Google Scholar
- Human intervention: Minimized

**Why**: Lite mode delivers results in 5-15 minutes — ideal for quickly understanding a field.

---

### Scenario 5: Pure Conversation (No Retrieval Needed)

> "Translate this abstract for me" / "Explain what p-value means"

**Recommended config**:
- No configuration changes needed
- The system automatically detects casual intent and skips retrieval and Agent

**Why**: Three-layer intent detection handles this automatically — casual chat, translation, and concept explanations never trigger retrieval.

---

### Scenario 6: Knowledge Graph Exploration

> "Explore relationships between 'methanotrophic archaea' and other entities"

**Recommended config**:
- Switch to the **Graph** tab
- Search for entity keywords
- Adjust exploration depth

**Why**: Graph visualization intuitively reveals entity relationships.

---

## FAQ

**Q: Will Agent mode or Deep Research use search sources I haven't checked?**  
A: No. The sources you check in the sidebar define the "data source boundary" for the entire system. Normal Chat, ReAct Agent, and Deep Research all strictly respect it.

**Q: Can I close the browser during Deep Research execution?**  
A: Yes. The task runs on the server. Reopening the page auto-recovers the task state.

**Q: Does the Agent always get the same set of tools?**  
A: No. The system dynamically routes tools based on your source checkboxes, message content, and current stage. For example, if you haven't checked NCBI, the Agent won't receive the `search_ncbi` tool.

**Q: Why does the Agent sometimes not call any tools?**  
A: The system has three-layer intent detection. If your question is classified as casual, even with Agent enabled, tool calls are skipped and the system answers in pure conversation mode.

**Q: Does Query Optimizer only generate queries for my checked sources?**  
A: Yes. It does not waste tokens generating queries for unchecked engines.

**Q: Does NCBI PubMed require an API key?**  
A: No (default: 3 requests/sec). With an API key configured, the rate increases to 10 requests/sec.

**Q: How do I choose between Lite and Comprehensive?**  
A: Use Lite for quick exploration (5-15 min), Comprehensive for formal writing (20-60 min).

**Q: Does year filtering work for Tavily and Google?**  
A: Tavily and Google do not support native year filtering. Year filtering primarily applies to Google Scholar, Semantic Scholar, NCBI PubMed, and local RAG.

---

*Last updated: 2026-02-19*
