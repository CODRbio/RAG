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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.log import get_logger
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()
from src.retrieval.query_optimizer import optimize_query

logger = get_logger(__name__)


# ── 时效性关键词（启发式）────────────────────────────────────────────────────

_FRESH_RE_EN = re.compile(
    r"\b(latest|recent(?:ly)?|newest|current(?:ly)?|today|this\s+week|this\s+month|"
    r"this\s+year|20(24|25|26)|breaking|trending|just\s+released?|"
    r"new\s+(?:research|study|paper|report|development)|state[\s-]of[\s-]the[\s-]art\s+2025|"
    r"cutting[\s-]edge|hot\s+topic)\b",
    re.IGNORECASE,
)
_FRESH_RE_ZH = re.compile(
    r"(最新|近期|最近|今年|今天|本年|当前|新进展|最新进展|新研究|最新研究|"
    r"最新动态|最新消息|最新进展|最近发展|刚发布|刚出|新出)"
)


def _is_fresh_query_heuristic(query: str) -> bool:
    """启发式检测查询是否具有时效性（LLM 路由的前置保险）。"""
    text = query or ""
    return bool(_FRESH_RE_EN.search(text) or _FRESH_RE_ZH.search(text))


# ── 路由计划数据类 ────────────────────────────────────────────────────────────

@dataclass
class RoutingPlan:
    """
    代价感知路由计划。

    primary:   首选引擎列表（先并发执行）
    fallback:  当 primary 结果 < min_results 时自动启动的备选引擎列表
    queries:   {engine: [query, ...]}，覆盖 primary + fallback 所有引擎
    is_fresh:  是否为时效性查询（true → tavily 出现在 primary）
    min_results: 判定"结果充足"的最低条数，不足则触发 fallback
    """

    primary: List[str] = field(default_factory=list)
    fallback: List[str] = field(default_factory=list)
    queries: Dict[str, List[str]] = field(default_factory=dict)
    is_fresh: bool = False
    min_results: int = 3


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


# ── Pydantic Response Models (结构化输出) ────────────────────────────────────

class _DynamicQueryResponse(BaseModel):
    """查询优化响应：引擎名作为动态 key，接受任意额外字段。"""
    model_config = ConfigDict(extra="allow")


class _RoutingPlanLLMResponse(BaseModel):
    """路由计划 LLM 响应结构。"""
    is_fresh: bool = False
    primary: Dict[str, Any] = Field(default_factory=dict)
    fallback: Dict[str, Any] = Field(default_factory=dict)


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
    if provider in ("scholar", "semantic", "ncbi"):
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
    if p in ("scholar", "google", "semantic", "ncbi"):
        text = re.sub(r"\s+(review|survey|overview)\s*$", "", text, flags=re.I).strip()
    return text


# Engines that do not support Chinese meaningfully; always use English queries only.
_ENGLISH_ONLY_ENGINES = frozenset({"ncbi", "semantic", "scholar"})


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
                return manager.get_lite_client(llm_provider)
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
            self._llm_client = manager.get_lite_client(provider)
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
        auto_route: bool = False,
    ) -> Dict[str, List[str]]:
        """
        为搜索引擎生成优化查询词，支持两种工作模式：

        普通模式 (auto_route=False)：
            为 providers 中的**每个**引擎生成查询，前端手动选择，不改变引擎列表。

        代价感知路由模式 (auto_route=True)：
            LLM 从 providers（候选池）中**选出最合适的引擎子集**，
            并只为选中的引擎生成查询。返回 dict 的 key 即为实际选用的引擎。
            调用方应根据返回 keys 收窄实际搜索范围，避免滥用高代价 API。

        Returns:
            {engine: [query, ...], ...}
            auto_route=True 时只含被选中的引擎；False 时含全部 providers。
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
            eng_only = [p for p in providers if p.strip().lower() in _ENGLISH_ONLY_ENGINES]
            zh_en = [p for p in providers if p.strip().lower() not in _ENGLISH_ONLY_ENGINES]
            parts = []
            if eng_only:
                parts.append(
                    f"For {', '.join(eng_only)}: output a plain array of {max_per} English query strings only (no Chinese)."
                )
            if zh_en:
                parts.append(
                    f"For {', '.join(zh_en)}: output an object with keys \"zh\" and \"en\", each an array of {max_per} queries."
                )
            bilingual_instruction = " " + " ".join(parts) if parts else ""

        # 代价感知路由 (auto_route=True) 当前未使用：统一用 get_routing_plan()；
        # 此处保留分支但复用 optimizer_normal（原 optimizer_auto_route.txt 已移至 prompts/backup）。
        prompt = _pm.render(
            "optimizer_normal.txt",
            query=query,
            provider_list=provider_list,
            bilingual_instruction=bilingual_instruction,
            max_per=max_per,
        )

        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("optimizer_system.txt")},
                    {"role": "user", "content": prompt},
                ],
                model=model_override or None,

                response_model=_DynamicQueryResponse,
            )
            parsed_resp: Optional[_DynamicQueryResponse] = resp.get("parsed_object")
            if parsed_resp is None:
                raw_text = (resp.get("final_text") or "").strip()
                if raw_text:
                    parsed_resp = _DynamicQueryResponse.model_validate_json(raw_text)
            data: Optional[Dict[str, Any]] = parsed_resp.model_dump() if parsed_resp is not None else None
            if not data or not isinstance(data, dict):
                return self._fallback_optimize(query, providers)

            # auto_route：直接以 LLM 返回的 keys 为准（只处理候选集内的引擎）
            target_providers = (
                [p for p in providers if p in data] if auto_route else providers
            )
            if not target_providers:
                logger.warning("Auto-route: LLM 未选中任何候选引擎，回退到全量")
                return self._fallback_optimize(query, providers)

            out: Dict[str, List[str]] = {}
            for p in target_providers:
                raw = data.get(p)
                p_lower = p.strip().lower()
                if bilingual and is_zh_input:
                    if isinstance(raw, dict):
                        zh = _normalize_list(raw.get("zh"))
                        en = _normalize_list(raw.get("en"))
                    else:
                        items = _normalize_list(raw)
                        zh = [q for q in items if _is_chinese(q)]
                        en = [q for q in items if not _is_chinese(q)]

                    if not en:
                        en = [_fallback_en_query(p, query)]
                    while len(en) < max_per:
                        en.append(en[-1])
                    en = en[:max_per]

                    if p_lower in _ENGLISH_ONLY_ENGINES:
                        out[p] = [_sanitize_query(p, q) for q in en]
                    else:
                        if not zh:
                            zh = [query]
                        while len(zh) < max_per:
                            zh.append(zh[-1])
                        zh = zh[:max_per]
                        out[p] = [_sanitize_query(p, q) for q in (zh + en)]
                else:
                    queries = _normalize_list(raw)[:max_per]
                    if not queries:
                        queries = [query]
                    out[p] = [_sanitize_query(p, q) for q in queries]

            if auto_route:
                logger.info(f"代价感知路由选定引擎: {list(out.keys())}")
            return out
        except Exception as e:
            logger.warning(f"Smart query optimizer LLM call failed: {e}")
            return self._fallback_optimize(query, providers)

    def get_routing_plan(
        self,
        query: str,
        candidate_providers: List[str],
        max_queries_per_provider: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> RoutingPlan:
        """
        生成代价感知路由计划。

        LLM 从候选引擎中选出：
        - primary：1-2 个最匹配的引擎（并发首发）
        - fallback：1 个备选引擎（primary 结果 < min_results 时启用）

        时效性保险：若启发式检测到查询含"最新/recent/latest"等，
        即使 LLM 未选 tavily，也会将其插入 primary（如果在候选池中）。
        """
        query = (query or "").strip()
        if not query or not candidate_providers:
            return RoutingPlan(primary=candidate_providers, queries={p: [query] for p in candidate_providers})

        is_fresh = _is_fresh_query_heuristic(query)
        max_per = self._config.get("max_queries_per_provider", 2)
        if isinstance(max_queries_per_provider, int):
            max_per = min(max(max_queries_per_provider, 1), 4)
        is_zh = _is_chinese(query)

        client = self._get_llm_client(llm_provider=llm_provider, model_override=model_override)
        if not client:
            return self._fallback_routing_plan(query, candidate_providers, is_fresh)

        bilingual_note = ""
        if is_zh:
            eng_only = [x for x in candidate_providers if str(x).strip().lower() in _ENGLISH_ONLY_ENGINES]
            zh_en = [x for x in candidate_providers if str(x).strip().lower() not in _ENGLISH_ONLY_ENGINES]
            parts = []
            if eng_only:
                parts.append(
                    f"For {', '.join(eng_only)} use a plain array of English queries only."
                )
            if zh_en:
                parts.append(
                    f"For {', '.join(zh_en)} use {{\"zh\": [...], \"en\": [...]}}."
                )
            bilingual_note = " " + " ".join(parts) if parts else ""

        provider_list = ", ".join(candidate_providers)
        prompt = _pm.render(
            "optimizer_routing_plan.txt",
            query=query,
            provider_list=provider_list,
            max_per=max_per,
            bilingual_note=bilingual_note,
        )

        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("optimizer_routing_plan_system.txt")},
                    {"role": "user", "content": prompt},
                ],
                model=model_override or None,

                response_model=_RoutingPlanLLMResponse,
            )
            parsed_plan: Optional[_RoutingPlanLLMResponse] = resp.get("parsed_object")
            if parsed_plan is None:
                raw_text = (resp.get("final_text") or "").strip()
                if raw_text:
                    parsed_plan = _RoutingPlanLLMResponse.model_validate_json(raw_text)
            data: Optional[Dict[str, Any]] = parsed_plan.model_dump() if parsed_plan is not None else None
            if not data or not isinstance(data, dict):
                return self._fallback_routing_plan(query, candidate_providers, is_fresh)

            llm_is_fresh = bool(data.get("is_fresh", is_fresh))
            # 时效性保险：启发式检测到 fresh 但 LLM 遗漏时强制插入 tavily
            if is_fresh and not llm_is_fresh:
                logger.info("时效性保险触发: 启发式检测为 fresh，覆盖 LLM 路由")
                llm_is_fresh = True

            raw_primary = data.get("primary") or {}
            raw_fallback = data.get("fallback") or {}
            if not isinstance(raw_primary, dict):
                raw_primary = {}
            if not isinstance(raw_fallback, dict):
                raw_fallback = {}
            candidate_set = {
                str(p).strip().lower()
                for p in candidate_providers
                if str(p).strip()
            }

            # 时效性保险：fresh 但 primary 没有 tavily → 强插
            primary_keys_norm = {str(k).strip().lower() for k in raw_primary.keys()}
            if llm_is_fresh and "tavily" in candidate_set and "tavily" not in primary_keys_norm:
                # 生成一个简单的时效性查询
                fresh_q = f"latest news {query}" if not _is_chinese(query) else f"最新进展 {query}"
                raw_primary["tavily"] = [fresh_q]
                for k in list(raw_fallback.keys()):
                    if str(k).strip().lower() == "tavily":
                        raw_fallback.pop(k, None)

            def _parse_engine_queries(raw: Any, engine: str) -> List[str]:
                eng_norm = engine.strip().lower()
                if is_zh:
                    if isinstance(raw, dict):
                        zh = _normalize_list(raw.get("zh"))
                        en = _normalize_list(raw.get("en"))
                    else:
                        items = _normalize_list(raw)
                        zh = [q for q in items if _is_chinese(q)]
                        en = [q for q in items if not _is_chinese(q)]
                    if not en:
                        en = [_fallback_en_query(engine, query)]
                    en = (en * max_per)[:max_per]
                    if eng_norm in _ENGLISH_ONLY_ENGINES:
                        return [_sanitize_query(engine, q) for q in en]
                    if not zh:
                        zh = [query]
                    zh = (zh * max_per)[:max_per]
                    return [_sanitize_query(engine, q) for q in (zh + en)]
                else:
                    qs = _normalize_list(raw)[:max_per]
                    return [_sanitize_query(engine, q) for q in (qs or [query])]

            queries: Dict[str, List[str]] = {}
            primary_engines: List[str] = []
            for eng, raw_q in raw_primary.items():
                eng_norm = str(eng).strip().lower()
                if eng_norm not in candidate_set or eng_norm in primary_engines:
                    continue
                queries[eng_norm] = _parse_engine_queries(raw_q, eng_norm)
                primary_engines.append(eng_norm)

            fallback_engines: List[str] = []
            for eng, raw_q in raw_fallback.items():
                eng_norm = str(eng).strip().lower()
                if eng_norm not in candidate_set or eng_norm in primary_engines or eng_norm in fallback_engines:
                    continue
                queries[eng_norm] = _parse_engine_queries(raw_q, eng_norm)
                fallback_engines.append(eng_norm)

            if not primary_engines:
                return self._fallback_routing_plan(query, candidate_providers, llm_is_fresh)

            plan = RoutingPlan(
                primary=primary_engines,
                fallback=fallback_engines,
                queries=queries,
                is_fresh=llm_is_fresh,
                min_results=3,
            )
            logger.info(
                f"路由计划: primary={plan.primary}, fallback={plan.fallback}, "
                f"is_fresh={plan.is_fresh}"
            )
            return plan

        except Exception as e:
            logger.warning(f"get_routing_plan LLM call failed: {e}")
            return self._fallback_routing_plan(query, candidate_providers, is_fresh)

    def _fallback_routing_plan(
        self,
        query: str,
        providers: List[str],
        is_fresh: bool = False,
    ) -> RoutingPlan:
        """规则回退：基于简单规则生成路由计划（无需 LLM）。"""
        # 时效性：tavily 优先
        if is_fresh and "tavily" in providers:
            primary = ["tavily"]
            fallback = [p for p in ("scholar", "google", "ncbi") if p in providers][:1]
        elif "ncbi" in providers:
            primary = ["ncbi"]
            fallback = [p for p in ("scholar", "tavily") if p in providers][:1]
        elif "scholar" in providers:
            primary = ["scholar"]
            fallback = [p for p in ("tavily", "google") if p in providers][:1]
        elif "tavily" in providers:
            primary = ["tavily"]
            fallback = [p for p in ("google", "scholar") if p in providers][:1]
        else:
            primary = providers[:1]
            fallback = providers[1:2]

        queries = self._fallback_optimize(query, primary + fallback)
        return RoutingPlan(
            primary=primary,
            fallback=fallback,
            queries=queries,
            is_fresh=is_fresh,
            min_results=3,
        )

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
