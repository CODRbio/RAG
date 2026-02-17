# Frontend（DeepSea RAG）

前端基于 React + TypeScript + Vite + Zustand，负责聊天交互、画布编辑、配置管理、图谱探索和多文档对比。

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
- 推荐从仓库根目录启动：`bash scripts/start.sh`
- 如需与 lock 文件完全一致，使用 `npm ci` 替代 `npm install`

## 目录说明

- `src/pages/`
  - `ChatPage.tsx`
  - `IngestPage.tsx`
  - `LoginPage.tsx`
  - `AdminPage.tsx`
- `src/components/`
  - `chat/`：消息、输入、工具轨迹、检索诊断
  - `canvas/`：画布与编辑辅助
  - `compare/`：多文档对比
  - `graph/`：图谱可视化
  - `layout/`：头部/侧栏配置
  - `workflow/`：研究流程交互
- `src/stores/`
  - `useChatStore`
  - `useConfigStore`
  - `useCanvasStore`
  - `useAuthStore`
  - `useProjectsStore`
  - `useCompareStore`
- `src/api/`：后端接口封装
- `src/types/`：前端类型定义

## 关键交互

- 聊天请求：`POST /chat`、`POST /chat/stream`
- SSE 事件：`meta`、`dashboard`、`tool_trace`、`delta`、`done`
- 对比功能：`/compare/candidates` + `/compare/papers` + `/compare`

## 开发约定

- 新接口先在 `src/api/` 封装，再接入组件
- 新状态优先进入对应 Zustand store，避免组件内分散状态
- 新增后端字段时同步更新 `src/types/` 与消费组件
- 提交前至少跑通关键页面：聊天、入库、对比
