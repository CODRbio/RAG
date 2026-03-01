# 调试面板与后端日志联动

## 什么是调试面板（Debug Panel）

**调试面板**是聊天界面里可选的 **Agent 调试信息展示区**，用于查看单轮对话中 Agent 的详细执行情况，而无需看后端日志。

在侧栏设置中勾选 **「Debug Panel」/「调试面板」** 后，每条助手回复下方会出现可展开的 **Agent Debug** 区块，包含：

| 内容 | 说明 |
|------|------|
| **迭代与工具** | 总迭代次数、工具调用次数、总耗时（LLM + Tools） |
| **是否贡献** | 该轮回答是否用到了 Agent 工具补充的证据（Contributed / No contribution） |
| **时间拆分** | LLM 思考时间 vs 工具执行时间、错误次数 |
| **工具使用统计** | 各工具（search_local / search_web / search_scholar 等）调用次数 |
| **证据块统计** | 预检索块数、Agent 补充块数、被引用的 Agent 块数 |
| **时间线** | 每条工具调用的耗时、参数与结果摘要（可展开） |

数据来源：后端在该轮流式返回的 `agent_debug` 事件；前端仅负责展示，不参与计算。

## 与后端日志的联动

勾选调试面板后，**前端会在该轮请求中带上 `agent_debug_mode: true`**。后端收到后，会在**本请求执行期间**临时将以下 logger 的级别提升为 **DEBUG**：

- `src.retrieval`（检索、SerpAPI/Playwright 等）
- `src.api.routes_chat`（对话流程）
- `src.collaboration.research.agent`（Deep Research Agent）

因此：

- **仅当你在 UI 打开「调试面板」并发送消息时**，该轮请求对应的后端日志会输出 DEBUG 级别（包括 `[retrieval] serpapi/playwright` 等调试行）。
- 不勾选时，请求仍带 `agent_debug_mode: false`（或不传），后端保持原有日志级别，不会多打 DEBUG。

这样可以在需要排查问题时，通过「开调试面板 → 发一条消息」即可在该轮看到完整检索与 Agent 相关 DEBUG 日志，无需改配置文件或重启服务。
