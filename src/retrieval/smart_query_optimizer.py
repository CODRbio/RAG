"""
LLM 驱动的智能查询优化器。

针对不同搜索引擎生成合适的搜索词：
- Google Scholar / Semantic Scholar: 学术关键词
- Google: 通用关键词
- Tavily: 自然语言问句

支持中英文双语（中文输入时生成英文查询）。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.log import get_logger
from src.retrieval.query_optimizer import optimize_query

logger = get_logger(__name__)


def _get_smart_optimizer_config() -> Dict[str, Any]:
    """从 config 读取 smart_optimizer 配置（settings 中 WebSearchConfig 无此字段，从 JSON 读）"""
    default = _default_smart_config()
    try:
        config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            ws = raw.get("web_search") or {}
            so = ws.get("smart_optimizer")
            if isinstance(so, dict):
                return {
                    "enabled": so.get("enabled", default["enabled"]),
                    "llm_provider": (so.get("llm_provider") or default["llm_provider"]).strip(),
                    "max_queries_per_provider": min(int(so.get("max_queries_per_provider", 3)), 5),
                    "enable_bilingual": so.get("enable_bilingual", default["enable_bilingual"]),
                    "fallback_to_simple": so.get("fallback_to_simple", default["fallback_to_simple"]),
                }
    except Exception:
        pass
    return default


def _default_smart_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "llm_provider": "deepseek",
        "max_queries_per_provider": 3,
        "enable_bilingual": True,
        "fallback_to_simple": True,
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 对象"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _normalize_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(q).strip() for q in raw if q and str(q).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _fallback_en_query(provider: str, query: str) -> str:
    provider = (provider or "").lower()
    if provider in ("scholar", "semantic"):
        return query
    if provider == "tavily":
        return f"Overview of {query}"
    if provider == "google":
        return query
    return query


def _sanitize_query(provider: str, q: str) -> str:
    """Remove generic suffixes that can hurt snippet specificity for some engines."""
    text = (q or "").strip()
    p = (provider or "").lower()
    if not text:
        return text
    if p in ("scholar", "google", "semantic"):
        text = re.sub(r"\s+(review|survey|overview)\s*$", "", text, flags=re.I).strip()
    return text


class SmartQueryOptimizer:
    """
    LLM 驱动的智能查询优化器。

    为每个搜索引擎生成 2-3 个不同角度的搜索词，支持中英双语。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _get_smart_optimizer_config()
        self._llm_client = None

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", True))

    def _get_llm_client(
        self,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ):
        """
        获取 LLM 客户端。
        优先使用外部传入的 llm_provider（跟随 UI 选择），否则回退到 config 中的固定值。
        """
        # 如果外部指定了 provider，每次创建新 client（不缓存，因为 provider 可能变化）
        if llm_provider:
            try:
                from src.llm import LLMManager
                cfg_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
                manager = LLMManager.from_json(str(cfg_path))
                if not manager.is_available(llm_provider):
                    logger.warning(f"Smart optimizer: requested provider {llm_provider} not available, fallback")
                    return self._get_llm_client()  # 回退到默认
                return manager.get_client(llm_provider)
            except Exception as e:
                logger.warning(f"Smart optimizer: failed to use provider {llm_provider}: {e}")
                return self._get_llm_client()  # 回退到默认

        # 默认：使用 config 中的 provider（缓存）
        if self._llm_client is not None:
            return self._llm_client
        try:
            from src.llm import LLMManager
            cfg_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
            manager = LLMManager.from_json(str(cfg_path))
            provider = (self._config.get("llm_provider") or "deepseek").strip()
            if not manager.is_available(provider):
                logger.warning(f"Smart optimizer: LLM provider {provider} not available")
                return None
            self._llm_client = manager.get_client(provider)
            return self._llm_client
        except Exception as e:
            logger.warning(f"Smart optimizer: failed to load LLM client: {e}")
            return None

    def optimize(
        self,
        query: str,
        providers: List[str],
        max_queries_per_provider: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        为每个 provider 生成一组搜索词。

        Args:
            query: 用户原始查询
            providers: 要使用的来源列表，如 ["scholar", "tavily", "google", "semantic"]

        Returns:
            {
                "scholar": ["deep sea cold seep biodiversity", "cold seep ecosystem"],
                "tavily": ["What are deep sea cold seeps?"],
                "google": ["deep sea cold seep characteristics"],
                "semantic": ["cold seep chemosynthesis biodiversity"]
            }
            每个 key 只包含当前 providers 中存在的来源；每个 value 为 1~max_queries_per_provider 个查询串。
        """
        query = (query or "").strip()
        if not query:
            return {p: [query] for p in providers}

        if not self.enabled:
            return self._fallback_optimize(query, providers)

        client = self._get_llm_client(llm_provider=llm_provider, model_override=model_override)
        if not client:
            return self._fallback_optimize(query, providers)

        max_per = self._config.get("max_queries_per_provider", 3)
        if isinstance(max_queries_per_provider, int):
            max_per = min(max(max_queries_per_provider, 1), 5)
        bilingual = self._config.get("enable_bilingual", True)
        is_zh_input = _is_chinese(query)

        provider_list = ", ".join(providers)
        bilingual_instruction = ""
        if bilingual and is_zh_input:
            bilingual_instruction = (
                f" The input is Chinese. You MUST generate exactly {max_per} Chinese queries and "
                f"{max_per} English queries for each provider. Return them separately under keys "
                "\"zh\" and \"en\" for each provider."
            )

        prompt = f"""You are a search query optimizer. Given a user query and a list of search engines, generate 1-{max_per} optimized search queries **per engine**. Output ONLY a valid JSON object, no markdown.

**User query:** {query}

**Search engines to optimize for:** {provider_list}

**Rules per engine:**
- **scholar**: Academic keywords, short keyword phrases (3-6 words). Avoid generic suffixes like "review/survey/overview". Example: "deep sea cold seep biodiversity", "cold seep chemosynthesis mechanism".
- **semantic**: Precise academic terms, no natural language. Example: "cold seep chemosynthesis", "cold seep ecosystem diversity".
- **google**: General keywords, concise. Avoid generic suffixes like "overview/review". Example: "deep sea cold seep characteristics", "cold seep definition".
- **tavily**: Natural language questions or full sentences. Example: "What are deep sea cold seeps and where do they occur?"
{bilingual_instruction}

**Output format:**
- If input is Chinese: a JSON object with keys exactly: scholar, semantic, google, tavily. Each value is an object with keys **zh** and **en**, each an array of exactly {max_per} queries.
- Otherwise: each value is an array of 1-{max_per} query strings.
Only include keys for engines that are in the list above; omit others.

Example (Chinese input, providers include scholar and tavily):
{{"scholar": {{"zh": ["深海冷泉 生物多样性", "冷泉 生态", "冷泉 形成 机制"], "en": ["deep sea cold seep biodiversity", "cold seep ecology", "cold seep formation mechanism"]}}, "tavily": {{"zh": ["深海冷泉是什么？", "深海冷泉的生态作用是什么？", "深海冷泉如何形成？"], "en": ["What is a deep sea cold seep?", "What is the ecological role of cold seeps?", "How do cold seeps form?"]}}}}

Example (English input, providers include scholar and tavily):
{{"scholar": ["cold seep biodiversity", "deep sea cold seep ecosystem"], "tavily": ["What is a deep sea cold seep?"]}}
"""

        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": "You output only valid JSON. No markdown, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                model=model_override or None,
                max_tokens=800,
            )
            text = (resp.get("final_text") or "").strip()
            data = _extract_json_object(text)
            if not data or not isinstance(data, dict):
                return self._fallback_optimize(query, providers)

            out: Dict[str, List[str]] = {}
            for p in providers:
                raw = data.get(p)
                if bilingual and is_zh_input:
                    if isinstance(raw, dict):
                        zh = _normalize_list(raw.get("zh"))
                        en = _normalize_list(raw.get("en"))
                    else:
                        # 兼容旧格式：按语言检测拆分
                        items = _normalize_list(raw)
                        zh = [q for q in items if _is_chinese(q)]
                        en = [q for q in items if not _is_chinese(q)]

                    if not zh:
                        zh = [query]
                    if not en:
                        en = [_fallback_en_query(p, query)]

                    # 补齐到每种语言 max_per 条
                    while len(zh) < max_per:
                        zh.append(zh[-1])
                    while len(en) < max_per:
                        en.append(en[-1])

                    zh = zh[:max_per]
                    en = en[:max_per]
                    out[p] = [_sanitize_query(p, q) for q in (zh + en)]
                else:
                    queries = _normalize_list(raw)[:max_per]
                    if not queries:
                        queries = [query]
                    out[p] = [_sanitize_query(p, q) for q in queries]
            return out
        except Exception as e:
            logger.warning(f"Smart query optimizer LLM call failed: {e}")
            return self._fallback_optimize(query, providers)

    def _fallback_optimize(self, query: str, providers: List[str]) -> Dict[str, List[str]]:
        """使用简单规则优化器作为 fallback"""
        if not self._config.get("fallback_to_simple", True):
            return {p: [query] for p in providers}
        return {
            p: [optimize_query(p, query) or query]
            for p in providers
        }


# 全局单例（懒加载配置）
_smart_optimizer_instance: Optional[SmartQueryOptimizer] = None


def get_smart_query_optimizer(config: Optional[Dict[str, Any]] = None) -> SmartQueryOptimizer:
    global _smart_optimizer_instance
    if _smart_optimizer_instance is None:
        _smart_optimizer_instance = SmartQueryOptimizer(config=config)
    return _smart_optimizer_instance
