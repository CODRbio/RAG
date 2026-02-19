# Frontend（DeepSea RAG）

前端基于 React + TypeScript + Vite + Zustand + Tailwind CSS，负责聊天交互、画布编辑、配置管理、图谱探索、多文档对比和 Deep Research 工作流。

更新时间：2026-02-19

## 技术栈

| 技术 | 版本 | 用途 |
|---|---|---|
| React | 19.2.0 | UI 框架 |
| TypeScript | ~5.9.3 | 类型系统 |
| Vite | ^7.2.4 | 构建工具 |
| Zustand | ^5.0.11 | 状态管理 |
| Tailwind CSS | ^4.1.18 | 样式框架 |
| React Router DOM | ^7.13.0 | 路由 |
| i18next | — | 国际化（中/英） |
| react-markdown | ^10.1.0 | Markdown 渲染 |
| react-pdf | ^10.3.0 | PDF 预览 |
| react-force-graph-2d | ^1.29.1 | 图谱可视化 |
| Lucide React | — | 图标库 |

## 本地开发

要求：

- Node.js `^20.19.0 || >=22.12.0`
- npm（建议与 Node LTS 配套）

```bash
cd frontend
npm install
npm run dev
```

默认地址：`http://localhost:5173`

## 与后端联调

- 后端默认：`http://127.0.0.1:9999`
- Vite 配置了 `/api` 代理到后端
- 推荐从仓库根目录启动：`bash scripts/start.sh`
- 如需与 lock 文件完全一致，使用 `npm ci` 替代 `npm install`

## 构建

```bash
npm run build     # TypeScript 编译 + Vite 构建
npm run preview   # 预览生产构建
```

## 目录说明

```text
src/
├── main.tsx              # 应用入口
├── App.tsx               # 根组件（路由配置）
├── index.css             # 全局样式
│
├── pages/                # 页面组件
│   ├── ChatPage.tsx      #   主聊天界面
│   ├── IngestPage.tsx    #   文档入库
│   ├── LoginPage.tsx     #   登录认证
│   └── AdminPage.tsx     #   管理后台
│
├── components/           # 业务组件
│   ├── chat/             #   聊天相关
│   │   ├── ChatWindow.tsx
│   │   ├── ChatInput.tsx
│   │   ├── ToolTracePanel.tsx
│   │   └── RetrievalDebugPanel.tsx
│   ├── canvas/           #   画布协作
│   │   ├── CanvasPanel.tsx
│   │   ├── ExploreStage.tsx
│   │   ├── OutlineStage.tsx
│   │   ├── DraftingStage.tsx
│   │   ├── RefineStage.tsx
│   │   ├── StageStepper.tsx
│   │   └── FloatingToolbar.tsx
│   ├── compare/          #   多文档对比
│   │   └── CompareView.tsx
│   ├── graph/            #   图谱可视化
│   │   └── GraphExplorer.tsx
│   ├── workflow/         #   Deep Research 工作流
│   │   ├── DeepResearchDialog.tsx
│   │   ├── DeepResearchSettingsPopover.tsx
│   │   ├── WorkflowStepper.tsx
│   │   ├── CommandPalette.tsx
│   │   ├── IntentModeSelector.tsx
│   │   └── IntentConfirmPopover.tsx
│   ├── research/         #   研究进度
│   │   └── ResearchProgressPanel.tsx
│   ├── settings/         #   设置
│   │   └── SettingsModal.tsx
│   ├── layout/           #   布局
│   │   ├── Header.tsx
│   │   └── Sidebar.tsx
│   └── ui/               #   通用 UI 组件
│       ├── Modal.tsx
│       ├── Toast.tsx
│       └── PdfViewerModal.tsx
│
├── stores/               # Zustand 状态管理
│   ├── index.ts
│   ├── useChatStore.ts       # 聊天状态
│   ├── useCanvasStore.ts     # 画布状态
│   ├── useConfigStore.ts     # 配置（含 Deep Research 设置持久化）
│   ├── useAuthStore.ts       # 认证状态
│   ├── useProjectsStore.ts   # 项目管理
│   ├── useCompareStore.ts    # 对比状态
│   ├── useUIStore.ts         # UI 全局状态
│   └── useToastStore.ts      # 消息提示
│
├── api/                  # 后端接口封装
│   ├── index.ts
│   ├── client.ts             # Axios 实例配置
│   ├── chat.ts               # /chat, /chat/stream
│   ├── canvas.ts             # /canvas/*
│   ├── compare.ts            # /compare/*
│   ├── graph.ts              # /graph/*
│   ├── ingest.ts             # /ingest/*
│   ├── models.ts             # /models/*, /llm/*
│   ├── auth.ts               # /auth/*, /admin/*
│   ├── projects.ts           # /projects/*
│   ├── auto.ts               # /auto-complete
│   └── health.ts             # /health, /health/detailed
│
├── types/                # TypeScript 类型定义
│   └── index.ts
│
└── i18n/                 # 国际化
    ├── index.ts              # i18next 初始化
    └── locales/
        ├── en.json           # English
        └── zh.json           # 中文
```

## 关键交互

### 聊天

- 请求：`POST /chat`、`POST /chat/stream`
- SSE 事件：`meta` → `dashboard` → `tool_trace` → `delta` → `done`
- 工具轨迹面板实时展示 Agent 工具调用过程

### Deep Research

- 启动前设置：`DeepResearchSettingsPopover`（持久化到 localStorage）
- 意图检测：`IntentModeSelector` + `IntentConfirmPopover`
- 研究对话框：`DeepResearchDialog`（澄清 → 大纲确认 → 执行 → 审核）
- 进度面板：`ResearchProgressPanel`（coverage 曲线、成本状态、效率评分）
- 后台任务轮询：`GET /deep-research/jobs/{id}/events`

### 画布协作

- 四阶段流程：Explore → Outline → Drafting → Refine
- 各阶段独立组件，`StageStepper` 导航
- 支持快照、恢复、AI 段落编辑、引用管理

### 多文档对比

- 候选来源：会话引文 + 本地文库搜索
- 结构化对比：`POST /compare`

### 图谱可视化

- `GraphExplorer`：力导向图布局，实体/关系交互

## 开发约定

- 新接口先在 `src/api/` 封装，再接入组件
- 新状态优先进入对应 Zustand store，避免组件内分散状态
- 新增后端字段时同步更新 `src/types/index.ts` 与消费组件
- 新增文案时同步更新 `src/i18n/locales/en.json` 和 `zh.json`
- 提交前至少跑通关键页面：聊天、入库、对比
- CSS 优先使用 Tailwind 工具类，避免自定义 CSS
