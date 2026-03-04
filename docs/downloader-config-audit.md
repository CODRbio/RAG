# Downloader config audit: RAG config vs original config vs UI

## What the downloader (PaperDownloader) reads from config

| Config key | Used by | In RAG config? | In UI? | Note |
|------------|---------|----------------|--------|------|
| **api_keys.annas_archive** | PaperDownloader | Yes (`scholar_downloader.annas_archive_api_key`) | No | Adapter passes via initial_config. |
| **api_keys.twocaptcha** | PaperDownloader | Yes (`scholar_downloader.twocaptcha_api_key` or `content_fetcher.two_captcha_api_key`) | No | Adapter passes via initial_config. |
| **api_keys.brightdata** | PaperDownloader | Yes (`content_fetcher.brightdata_api_key`) | No | Adapter passes via initial_config. |
| **api_keys.deepseek / anthropic** | LLMAssistant | Yes (via `llm.default` + platforms) | No | Adapter passes via initial_config. |
| **llm.provider, llm.model** | LLMAssistant | Yes (from `llm.default` + resolve_model) | No | Adapter passes via initial_config. |
| **downloader.timeouts** | PaperDownloader | **No** | No | Downloader uses built-in defaults (~20 timeout params). |
| **downloader.experience_store_path** | PaperDownloader | **No** | No | Downloader defaults to `{download_dir}/.experience_store.json`. |
| **downloader.proxy** | BrowserManager (via PaperDownloader) | Yes (`scholar_downloader.proxy`) | No | **Not wired**: adapter stores it but PaperDownloader never passes it to launch. |
| **downloader.capsolver_extension_path** | BrowserManager | Yes (`scholar_downloader.capsolver_extension_path`) | No | **Not wired**: browser_manager uses hardcoded `./Extension-capsolver`. |
| **annas_keyword_max_pages** | PaperDownloader | **No** | No | Downloader uses default 5. |
| **capsolver.enabled / api_key** | — | Yes (top-level `capsolver`) | No | Not used by PaperDownloader; may be for other flows. |

## RAG config blocks (rag_config.json + rag_config.local.json)

- **scholar_downloader**: enabled, download_dir, annas_archive_api_key, twocaptcha_api_key, capsolver_extension_path, proxy, max_concurrent_downloads, show_browser, persist_browser, auto_ingest_after_download → all read by settings; only **database.url** is exposed in UI (advanced settings), not scholar_downloader.
- **content_fetcher**: brightdata_api_key, two_captcha_api_key, … → used by adapter for api_keys; not in UI.

## Summary of gaps

1. **In RAG but not passed to downloader**: `scholar_downloader.proxy`, `scholar_downloader.capsolver_extension_path` (adapter has them but downloader/browser_manager do not use them).
2. **Used by downloader but not in RAG**: `downloader.timeouts`, `downloader.experience_store_path`, `annas_keyword_max_pages`.
3. **Not in UI**: All scholar_downloader and content_fetcher keys (only database URL is in advanced settings).

After fixes: wire proxy and capsolver_extension_path into initial_config and into the downloader/browser launch; add optional scholar_downloader keys (experience_store_path, annas_keyword_max_pages) to RAG config and settings.
