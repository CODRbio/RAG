# Information Download & Gather

本文档描述 RAG 项目中**文献（PDF）下载**与**网页资料全文提取**两大核心能力：流程、验证码处理、以及如何尽量可靠地定位并点击 PDF 下载按钮。

---

## 1. 文献（PDF）下载

### 1.1 整体策略顺序

下载器（`PaperDownloader`，经 `ScholarDownloaderAdapter` 与 RAG 配置对接）按**策略链**依次尝试，直到某一策略成功或全部失败：

| 策略 ID | 说明 |
|--------|------|
| `direct_download` | 直接 HTTP 请求 PDF URL（含 DOI 重定向、直链） |
| `playwright_download` | 浏览器打开页面，查找并点击 PDF 按钮后通过下载事件/流量嗅探保存 |
| `browser_lookup` | 与 playwright_download 共用 `find_and_download_pdf_with_browser` 的完整流程 |
| `sci_hub` | 通过 Sci-Hub 镜像（当前实现为 BrightData + sci-hub.st）获取 PDF |
| `brightdata` | BrightData Web Unlocker 请求目标 URL，得到 PDF 流后落盘 |
| `anna` | Anna's Archive API（需配置 `annas_archive_api_key`） |

策略顺序由配置 `scholar_downloader.default_strategy_order` 决定；未配置时使用上述默认顺序。Academia.edu 等来源会在适配层中适当后置，优先尝试 DOI / Sci-Hub / Anna's。

### 1.2 验证码处理（文献下载流程内）

在**浏览器内打开文章页 / 下载页**时，可能遇到 Cloudflare、reCAPTCHA、hCaptcha、图片验证码等。处理方式统一为：

- **Cloudflare（含 Turnstile）**  
  - 在 `find_and_download_pdf_with_browser` 中，页面稳定后先调用 `solve_cloudflare_if_needed()`。  
  - Turnstile：注入脚本截获 `sitekey`/`callback` 等参数，用 **2Captcha** 求解后把 token 回填页面（`send_token_to_page`）。  
  - 若验证通过后页面直接触发 PDF 下载，会通过下载事件或流量嗅探保存，并返回成功。

- **其他验证码类型（reCAPTCHA v2/v3、hCaptcha、FunCaptcha、图片/文字验证码）**  
  - 使用统一抽象 `CaptchaSolver`（`src/retrieval/downloader/captcha_solver.py`）：  
    - **Cloudflare/Turnstile** → 仅用 2Captcha（与上面一致）。  
    - **其余类型** → 优先 **CapSolver API**，失败则 **2Captcha** 回退。  
  - 流程：`detect_captcha_type` → `extract_captcha_params` → `CaptchaSolver.solve()` → `apply_captcha_token`。  
  - 文献下载器内部也实现了与 `captcha_page_runner.run_captcha_flow` 同构的检测/参数提取/回填逻辑，保证在“打开文章页 → 解验证码 → 再找 PDF 按钮”的顺序下工作。

- **配置**  
  - 2Captcha：`scholar_downloader.twocaptcha_api_key` 或 `content_fetcher.two_captcha_api_key`。  
  - CapSolver：顶层 `capsolver.api_key` 或 `scholar_downloader.capsolver_api_key`。  
  - 可选浏览器扩展：`scholar_downloader.capsolver_extension_path`（CapSolver 扩展，用于部分场景）。

验证码解决失败时，不会立刻放弃整次下载，会继续尝试多步站点流程、通用选择器点击等后续逻辑。

### 1.3 PDF 按钮查找：核心思路

目标：在文章页或中间页上**稳定地找到“主文 PDF”的下载入口**（链接或按钮），并区分补充材料、参考文献等无关链接。

#### 1.3.1 选择器来源与优先级

- **配置文件**：`config/pdf_selectors.json`，是“如何找 PDF 按钮”的**唯一权威配置**。  
- **选择器链构建**（`_build_selector_list`）：  
  1. **站点专属选择器**（最高优先级）：根据当前页面 URL 的 `site_url_patterns` 匹配到某一站点（如 `springer_nature`、`elsevier`、`wiley` 等），使用该站点在 `site_specific` 下配置的 CSS 选择器列表。  
  2. **通用选择器**（按固定顺序追加）：  
     - `generic.direct_links`（如 `a[href$=".pdf"]`、`a[href*="/doi/pdf/"]`）  
     - `generic.buttons`（如 `button:has-text("Download PDF")`、`button[aria-label*="PDF"]`）  
     - `generic.links`（如 `a[title*="PDF"]`、`a:has-text("Download PDF")`）  
     - `generic.text_matchers`（如 `:text("View PDF")`）

因此：**同一 URL 会先尝试该站点专属选择器，再尝试通用选择器**，避免把“下载补充材料”等误当成主文。

#### 1.3.2 多步站点（multi_step_sites）

部分出版商需要**多步操作**才能到达 PDF（例如先点“Download”，再在第二页点“Article PDF”）。`pdf_selectors.json` 中的 `multi_step_sites` 为这类站点定义了：

- `detect_contains`：URL 包含哪些字符串时视为该站点。  
- `steps`：每一步的 `selector` 与 `wait`（秒）。  
  执行时按顺序：等待元素 `attached` → 优先选可见元素 → `scroll_into_view_if_needed` → 短窗 `expect_download` 探测点击是否触发下载；若触发则立即保存并返回；未触发则进入下一步或跳转/弹窗兜底。

当前已配置的多步站点示例：

- **wiley**：先点 coolBar 上的 PDF/Download，再点 pdfdirect 或 ePDF 链接；并带 Wiley 专属兜底（从当前 (e)pdf 页构造 `pdfdirect?...&download=true` 直链）。  
- **elife**：先点下载按钮，再点 “Article PDF”。  
- **zootaxa**（mapress）：先 `a.obj_galley_link.pdf`，再 `a.download[download]`。  
- **atypon_publishers**（science.org、pnas、tandfonline 等）：先 PDF 入口，再 “Proceed” / “Accept and Download” 等。  
- **researchgate**：先 PDF 下载按钮，再 “Download without” / “Continue to download” 等。

多步流程在**通用 Cloudflare 处理之后**执行：先 `solve_cloudflare_if_needed`，再按 URL 匹配 `multi_step_sites` 执行对应 steps。

#### 1.3.3 启发式打分与可选 LLM 精排

- **ActionableElement / Candidate**：从页面收集到的每个“可能是 PDF 入口”的节点（链接、按钮）会带上一系列特征：`selector`、`text`、`href`、是否固定定位、在文档中的位置比例、祖先路径（id/class）、是否在视口等。  
- **启发式规则**（`PDFExtractor.score_and_filter`，`ScoreConfig`）：  
  - 正向：固定/置顶、主区域/侧栏、文案含 “download pdf”“full text” 等、href 含 `/doi/pdf/`、`.pdf` 等。  
  - 负向：处于 “references”“footer”“supplement” 等区域、文案或 href 含 “supplement”“citation”“appendix” 等。  
- **可选 LLM 精排**：在启发式粗排后，对 top-N 候选调用轻量 LLM（prompt 见 `prompts/downloader_candidate_rerank.txt`），返回 `best_candidate_id` 与 `confidence`；仅当置信度不低于 `ScoreConfig.min_confidence_to_accept` 时才采纳，否则不强行选一个，避免误点。

这样可以在**不依赖站点名称**的情况下，对未知站点或配置未覆盖的页面仍有一定“猜主文 PDF”的能力。

#### 1.3.4 其他提高成功率的机制

- **响应流嗅探**：在 `find_and_download_pdf_with_browser` 中注册 `page.on("response", ...)`，对 `content-type: application/pdf` 或 URL 以 `.pdf` 结尾的响应做 body 读取并缓存到内存。即便点击没有触发 Playwright 的 `download` 事件（例如一次性 token、XHR 下载），也能从嗅探到的二进制流落盘，避免“按钮点对了但没接到文件”。  
- **下载事件与弹窗**：除当前页的 `page.on("download")` 外，通过 `page.context.on("page", ...)` 对新开的 tab/窗口也绑定下载监听，避免 `target="_blank"` 打开的页面上的下载被漏掉。  
- **Cookie / 同意条**：在关键步骤后调用 `_handle_cookie_consent()`，关闭常见的 Cookie 同意弹窗，减少对“下载”按钮的遮挡。  
- **站点专属逻辑**：如 Springer 等待 `.c-pdf-download` 等区域加载；The Innovation 使用 `_collect_innovation_pdf_buttons` 等，对已知站点做小幅增强。  
- **Sci-Hub 解析**：当使用 Sci-Hub 策略时，从返回的 HTML 中解析 PDF 地址：优先 `<meta name="citation_pdf_url">`，其次 iframe `src`、`a[href*=".pdf"]`、以及 `/download/...pdf` 的正则，再请求该 URL（经 BrightData 等）得到 PDF。

### 1.4 配置与入口小结（文献下载）

- **选择器与多步**：`config/pdf_selectors.json`（`site_specific`、`generic`、`site_url_patterns`、`multi_step_sites`）。  
- **验证码**：`scholar_downloader.twocaptcha_api_key`、`capsolver.api_key`、可选 `capsolver_extension_path`。  
- **代理 / 超时**：`scholar_downloader.proxy`、`timeouts`（如 `button_appear_timeout`、`download_event_timeout`、`captcha_timeout`）。  
- **策略顺序**：`scholar_downloader.default_strategy_order`。  
- 实现入口：`src/retrieval/downloader/paper_downloader_refactored.py`（`find_and_download_pdf_with_browser` / `_find_and_download_pdf_with_browser_impl`）、`adapter.py`（策略编排与 RAG 配置映射）。

---

## 2. 网页资料全文提取（Web Content Fetcher）

### 2.1 作用与集成方式

- **作用**：对**搜索引擎/学术搜索返回的摘要**进行“全文抓取”：访问落地页，提取正文文本，回写到 hit 的 `content`，并标记 `metadata.content_type = "full_text"`，供后续检索/融合使用。  
- **集成点**：在 `RetrievalService` 的混合检索中，**Phase 1** 仅做 provider 搜索（快速）；**Phase 2** 在需要时对 raw hits 调用 `WebContentFetcher.enrich_results()`（可能较慢）。是否启用、以及是否用 LLM 预判“哪些 URL 需要抓全文”，由请求参数 `use_content_fetcher`（如 `"on"`、`"off"`、`"auto"`）控制。

### 2.2 三级提取策略（自动降级）

对**单个 URL** 的抓取采用三级策略，按顺序尝试，任一成功即返回正文：

1. **trafilatura（纯 HTTP）**  
   - 使用 `trafilatura.fetch_url` + `trafilatura.extract` 抓取并提取正文。  
   - 无需浏览器，速度快，适合无强反爬的普通网页。

2. **Playwright 浏览器**  
   - 优先从项目的 headless context 池借出上下文；不可用时启动临时浏览器。  
   - 对页面应用 stealth（`apply_stealth_to_page`），然后 `goto` 目标 URL。  
   - **验证码**：若配置了 `two_captcha_api_key` 或 `capsolver_api_key`，会构建 `CaptchaSolver` 并调用 **`run_captcha_flow`**（与文献下载共用 `captcha_page_runner`）：  
     - Cloudflare/Turnstile → 仅 2Captcha。  
     - 其他类型 → CapSolver 优先，2Captcha 回退。  
   - 抓取到的是渲染后的 HTML，再用 trafilatura（或 BeautifulSoup 降级）提取正文。

3. **BrightData Web Unlocker**  
   - 通过 BrightData API 请求目标 URL，拿回 HTML 后仍用 trafilatura 提取正文。  
   - 用于反爬较重、但不需要交互的页面；需配置 `content_fetcher.brightdata_api_key`。

同 URL 在进程内有 **in-flight 合并**（并发请求同一 URL 只发一次）；并支持 **TTL 内存缓存** 与 **磁盘 SQLite 缓存**（可配置 TTL、promote 阈值），减少重复抓取。

### 2.3 URL 过滤

- **is_qualifying_url(url, only_academic=False)**：  
  - 排除：非 http(s)、搜索引擎、社交媒体、视频站、登录页子串、二进制/多媒体扩展名（含 `.pdf`）等。  
  - `only_academic=True` 时，仅保留学术域名白名单（如 scholar、arxiv、pubmed、doi、各大出版社、biorxiv、researchgate 等）。  
- 在 `enrich_results` 中，只会对通过过滤的 URL 发起抓取。

### 2.4 验证码处理（网页提取）

- 仅在 **Playwright** 这一级使用验证码逻辑。  
- 流程：`page.goto` → 短等待 → **`run_captcha_flow(page, captcha_solver, two_captcha_api_key, ...)`**。  
- 与文献下载一致：Turnstile 用 2Captcha；其他类型用 `CaptchaSolver`（CapSolver 优先，2Captcha 回退）。  
- 配置：`content_fetcher.two_captcha_api_key`、`content_fetcher.capsolver_api_key`、`content_fetcher.captcha_timeout_seconds`。  
- 单页总时长上限：若同时配置了 BrightData 与 2Captcha，为 2 分钟，否则 1 分钟（与 `timeout_seconds` 取小）。

### 2.5 可选 LLM 预判（Lazy Fetching）

- **evaluate_snippets_need_fetch**：对 Phase 1 结果中来源为 scholar/google 的条目，用 LLM 判断“摘要是否足够”，仅对返回的 URL 列表做全文抓取，减少无效请求。  
- Prompt：`web_content_fetch_decide.txt` / `web_content_fetch_decide_system.txt`；模型返回 `{ "urls_to_fetch": ["url1", ...] }`。  
- 若 LLM 调用失败，降级为对候选 URL 全量抓取。

### 2.6 配置与入口小结（网页提取）

- **配置**：`config/settings.py` 下 `content_fetcher`（enabled、only_academic、timeout_seconds、brightdata_api_key、two_captcha_api_key、capsolver_api_key、cache、disk_cache 等）。  
- **实现**：`src/retrieval/web_content_fetcher.py`（`WebContentFetcher`、`fetch_content`、`enrich_results`、`evaluate_snippets_need_fetch`）。  
- **调用**：`src/retrieval/service.py` 在 Phase 2 根据 `use_content_fetcher` 调用 `enrich_results`。

---

## 3. 验证码与 PDF 按钮：设计要点总结

- **验证码**  
  - 文献下载与网页提取共用：`captcha_solver.py`（类型检测、CapSolver/2Captcha 路由）、`captcha_page_runner.py`（`run_captcha_flow`、Turnstile 参数截获与回填）。  
  - Cloudflare/Turnstile 统一用 2Captcha；其余类型 CapSolver 优先、2Captcha 兜底，保证在“验证码通过后再找按钮/再取正文”的顺序下执行。

- **PDF 按钮**  
  - **优先站点化**：`config/pdf_selectors.json` 的 `site_specific` + `site_url_patterns`，确保已知站点用已知选择器，减少误点补充材料/引用。  
  - **多步流程**：`multi_step_sites` 明确多步顺序与每步选择器，并配合短窗 `expect_download` 与站点兜底（如 Wiley pdfdirect）。  
  - **通用兜底**：`generic` 的 direct_links / buttons / links / text_matchers，对未知站点仍可尝试。  
  - **启发式 + 可选 LLM**：先规则打分过滤危险区与补充材料，再可选 LLM 精排，仅在高置信度时采纳，避免“瞎点”。  
  - **网络与 UI 双收**：响应流嗅探 + 下载事件 + 新页面下载监听，确保“点对了”时即使未触发标准下载事件也能从内存/新 tab 拿到 PDF。

以上能力共同保证：在验证码处理后的文献下载与网页资料提取中，既能尽量找准 PDF 按钮，又能在页面结构变化或未知站点时有一定鲁棒性。
