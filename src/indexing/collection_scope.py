"""
每个 collection 的覆盖范围摘要（scope summary）。
建库时或入库完成后由 LLM 快速生成，用于「查询与本地库是否匹配」判断；支持后续刷新。
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from src.log import get_logger
from src.utils.prompt_manager import PromptManager
from src.utils.context_limits import summarize_if_needed, COLLECTION_SCOPE_MAX_CHARS

_pm = PromptManager()
logger = get_logger(__name__)

# 存储路径：data/collection_scope.json
_SCOPE_FILE: Optional[Path] = None


def _scope_file() -> Path:
    global _SCOPE_FILE
    if _SCOPE_FILE is None:
        from config.settings import settings
        _SCOPE_FILE = settings.path.data / "collection_scope.json"
    return _SCOPE_FILE


def _load_data() -> dict:
    p = _scope_file()
    if not p.exists():
        return {"collections": {}}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("collection_scope load failed: %s", e)
        return {"collections": {}}
    if "collections" not in data:
        data["collections"] = {}
    return data


def _save_data(data: dict) -> None:
    p = _scope_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_scope(collection_name: str) -> Optional[str]:
    """获取某 collection 的 scope 摘要；不存在返回 None。"""
    meta = get_scope_meta(collection_name)
    return meta.get("scope_summary") if meta else None


def get_scope_meta(collection_name: str) -> Optional[dict]:
    """获取某 collection 的 scope 元数据（scope_summary, updated_at）；不存在返回 None。"""
    if not (collection_name or "").strip():
        return None
    data = _load_data()
    coll = data["collections"].get((collection_name or "").strip())
    if not coll or not isinstance(coll, dict):
        return None
    summary = (coll.get("scope_summary") or "").strip() or None
    if summary is None:
        return None
    return {"scope_summary": summary, "updated_at": coll.get("updated_at")}


def set_scope(collection_name: str, scope_summary: str) -> None:
    """写入某 collection 的 scope 摘要。"""
    name = (collection_name or "").strip()
    if not name:
        return
    from datetime import datetime
    data = _load_data()
    data["collections"][name] = {
        "scope_summary": (scope_summary or "").strip(),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_data(data)
    logger.info("collection_scope set for %s", name)


def get_document_titles_for_collection(
    collection_name: str,
    max_titles: Optional[int] = None,
) -> List[str]:
    """
    获取指定集合内所有文档的题目/文件名列表（用于生成 scope 摘要）。
    优先使用 paper_metadata 中的 title，无则用 filename。max_titles 为 None 时返回全部。
    """
    name = (collection_name or "").strip()
    if not name:
        return []
    try:
        from src.indexing.paper_store import list_papers
        papers = list_papers(name)
    except Exception as e:
        logger.warning("get_document_titles_for_collection list_papers failed: %s", e)
        return []
    titles: List[str] = []
    try:
        from src.indexing.paper_metadata_store import paper_meta_store
        for p in papers:
            if max_titles is not None and len(titles) >= max_titles:
                break
            paper_id = (p.get("paper_id") or "").strip()
            filename = (p.get("filename") or "").strip()
            title = ""
            if paper_id:
                meta = paper_meta_store.get(paper_id)
                if meta and isinstance(meta.get("title"), str) and meta["title"].strip():
                    title = meta["title"].strip()
            if not title and filename:
                title = filename
            if title:
                titles.append(title)
    except Exception as e:
        logger.warning("get_document_titles_for_collection meta lookup failed: %s", e)
        for p in (papers[:max_titles] if max_titles is not None else papers):
            fn = (p.get("filename") or "").strip()
            if fn:
                titles.append(fn)
    return titles


def generate_scope_summary(
    collection_name: str,
    sample_texts: Optional[List[str]] = None,
    llm_client: Any = None,
    max_sample_chars: int = 6000,
) -> str:
    """
    用 LLM 快速生成该 collection 的覆盖范围摘要（1–2 句）。
    sample_texts 可选，来自建库/入库时的文档片段；无则仅根据 collection 名称推断。
    """
    name = (collection_name or "").strip()
    if not name:
        return ""

    if not llm_client:
        logger.debug("generate_scope_summary: no llm_client, skip")
        return ""

    sample_block = ""
    if sample_texts:
        parts = []
        total = 0
        for t in sample_texts:
            if not isinstance(t, str) or not t.strip():
                continue
            part = t.strip()
            if total + len(part) > max_sample_chars:
                part = part[: max_sample_chars - total] + "…"
                parts.append(part)
                total += len(part)
                break
            parts.append(part)
            total += len(part)
        sample_block = "\n---\n".join(parts) if parts else "(无)"
    else:
        sample_block = "(无样本，仅根据库名推断)"

    prompt = _pm.render(
        "collection_scope_generate.txt",
        collection_name=name,
        sample_block=sample_block,
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("collection_scope_generate_system.txt")},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.get("final_text") or "").strip()
        if text and len(text) <= 2000:
            return text
    except Exception as e:
        logger.warning("generate_scope_summary LLM failed: %s", e)
    return ""


def summarize_new_materials(
    sample_texts: List[str],
    llm_client: Any,
    max_chars: int = COLLECTION_SCOPE_MAX_CHARS,
) -> str:
    """将本次入库的文档片段归纳为 1–2 句「本批材料摘要」，用于与已有 scope 合并。输入总长超 40k 时先总结再送入。"""
    if not sample_texts or not llm_client:
        return ""
    parts = []
    total = 0
    part_max = 4000
    for t in sample_texts:
        if not isinstance(t, str) or not t.strip():
            continue
        raw = t.strip()
        part = (raw[:part_max] + "…" if len(raw) > part_max else raw)
        parts.append(part)
        total += len(part)
        if total >= max_chars:
            break
    if not parts:
        return ""
    sample_block = "\n---\n".join(parts)
    if len(sample_block) > max_chars:
        sample_block = summarize_if_needed(
            sample_block, max_chars, llm_client=None, purpose="collection_scope_new_materials"
        )
    prompt = _pm.render("collection_scope_new_materials_summary.txt", sample_block=sample_block)
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("collection_scope_new_materials_summary_system.txt")},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.get("final_text") or "").strip()
        if text and len(text) <= 800:
            return text
    except Exception as e:
        logger.warning("summarize_new_materials LLM failed: %s", e)
    return ""


def update_scope_with_new_materials(
    collection_name: str,
    new_materials_summary: str,
    llm_client: Any,
) -> str:
    """
    根据「本次加入材料的摘要」与「已有 scope」做合并总结，写入更新后的 scope。
    若当前无已有 scope，则用 new_materials_summary 作为初始 scope。
    返回更新后的 scope 字符串，失败返回空串。
    """
    name = (collection_name or "").strip()
    if not name or not llm_client:
        return ""
    new_summary = (new_materials_summary or "").strip()
    if not new_summary:
        return ""

    existing = get_scope(name) or "（尚无）"

    prompt = _pm.render(
        "collection_scope_update.txt",
        existing_scope=existing,
        new_materials_summary=new_summary,
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("collection_scope_update_system.txt")},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.get("final_text") or "").strip()
        if text and len(text) <= 2000:
            set_scope(name, text)
            return text
        if existing == "（尚无）" and new_summary:
            set_scope(name, new_summary)
            return new_summary
    except Exception as e:
        logger.warning("update_scope_with_new_materials LLM failed: %s", e)
        if existing == "（尚无）" and new_summary:
            set_scope(name, new_summary)
            return new_summary
    return ""
