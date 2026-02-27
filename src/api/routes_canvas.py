"""
Canvas API：CRUD、大纲/草稿、快照、导出、AI 编辑。
"""

import uuid
from datetime import datetime
from pathlib import Path
import re
import requests

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.routes_auth import get_optional_user_id

from src.api.schemas import (
    CanvasAIEditRequest,
    CanvasAIEditResponse,
    CanvasCreateRequest,
    CanvasRefineRequest,
    CanvasRefineResponse,
    CanvasResponse,
    CanvasUpdateRequest,
    CanvasVersionItem,
    CitationFilterRequest,
    CitationFilterResponse,
    CitationResponse,
    DraftBlockSchema,
    DraftUpsertRequest,
    ExportResponse,
    OutlineSectionSchema,
    OutlineUpsertRequest,
)
from src.collaboration.canvas.canvas_manager import (
    create_canvas,
    create_snapshot,
    delete_canvas,
    delete_canvas_citation,
    export_canvas,
    filter_canvas_citations,
    get_canvas,
    get_canvas_citations,
    list_snapshots,
    restore_snapshot,
    update_canvas,
    upsert_draft,
    upsert_outline,
)
from src.collaboration.canvas.models import DraftBlock, OutlineSection
from src.collaboration.citation.formatter import format_bibtex, format_reference_list, format_ris
from src.collaboration.memory.session_memory import get_session_store
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

router = APIRouter(prefix="/canvas", tags=["canvas"])


def _canvas_to_response(c) -> CanvasResponse:
    from src.api.schemas import AnnotationSchema, ResearchBriefSchema

    # 序列化 annotations
    annotations_out = []
    for ann in (c.annotations or []):
        annotations_out.append(AnnotationSchema(
            id=ann.id,
            section_id=ann.section_id,
            target_text=ann.target_text,
            directive=ann.directive,
            status=ann.status,
            created_at=ann.created_at.isoformat() if hasattr(ann.created_at, 'isoformat') else str(ann.created_at),
        ))

    # 序列化 research_brief
    brief_out = None
    if c.research_brief:
        rb = c.research_brief
        brief_out = ResearchBriefSchema(
            scope=rb.scope,
            success_criteria=rb.success_criteria,
            key_questions=rb.key_questions,
            exclusions=rb.exclusions,
            time_range=rb.time_range,
            source_priority=rb.source_priority,
            action_plan=getattr(rb, 'action_plan', ''),
        )

    # 序列化 citation_pool
    citations_out = [
        {
            "id": cit.id,
            "cite_key": cit.cite_key or cit.id,
            "title": cit.title,
            "authors": cit.authors or [],
            "year": cit.year,
            "doi": cit.doi,
            "url": cit.url,
            "bibtex": cit.bibtex,
        }
        for cit in c.citation_pool.values()
    ]

    return CanvasResponse(
        id=c.id,
        session_id=c.session_id,
        topic=c.topic,
        working_title=c.working_title,
        abstract=c.abstract,
        keywords=c.keywords,
        stage=c.stage,
        refined_markdown=getattr(c, "refined_markdown", "") or "",
        outline=[
            {"id": s.id, "title": s.title, "level": s.level, "order": s.order, "parent_id": s.parent_id, "status": s.status, "guidance": s.guidance}
            for s in c.outline
        ],
        drafts={
            sid: {
                "section_id": b.section_id,
                "content_md": b.content_md,
                "version": b.version,
                "used_fragment_ids": b.used_fragment_ids,
                "used_citation_ids": b.used_citation_ids,
                "updated_at": b.updated_at.isoformat(),
            }
            for sid, b in c.drafts.items()
        },
        citation_pool=citations_out,
        identified_gaps=c.identified_gaps or [],
        user_directives=c.user_directives or [],
        annotations=annotations_out,
        research_brief=brief_out,
        skip_draft_review=getattr(c, 'skip_draft_review', False),
        skip_refine_review=getattr(c, 'skip_refine_review', False),
        version=c.version,
    )


@router.post("", response_model=CanvasResponse)
def canvas_create(
    body: CanvasCreateRequest,
    user_id: str | None = Depends(get_optional_user_id),
) -> CanvasResponse:
    c = create_canvas(session_id=body.session_id, topic=body.topic, user_id=user_id or "")
    if body.session_id:
        store = get_session_store()
        if store.get_session_meta(body.session_id):
            store.update_session_meta(body.session_id, {"canvas_id": c.id})
    return _canvas_to_response(c)


@router.get("/{canvas_id}", response_model=CanvasResponse)
def canvas_get(canvas_id: str) -> CanvasResponse:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    return _canvas_to_response(c)


@router.patch("/{canvas_id}", response_model=CanvasResponse)
def canvas_update(canvas_id: str, body: CanvasUpdateRequest) -> CanvasResponse:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    fields = body.model_dump(exclude_unset=True)
    update_canvas(canvas_id, **fields)
    c = get_canvas(canvas_id)
    return _canvas_to_response(c)


@router.delete("/{canvas_id}")
def canvas_delete(canvas_id: str) -> dict:
    if not delete_canvas(canvas_id):
        raise HTTPException(status_code=404, detail="canvas not found")
    return {"ok": True, "canvas_id": canvas_id}


@router.post("/{canvas_id}/outline", response_model=CanvasResponse)
def canvas_upsert_outline(canvas_id: str, body: OutlineUpsertRequest) -> CanvasResponse:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    sections = [
        OutlineSection(
            id=s.id or str(uuid.uuid4())[:8],
            title=s.title,
            level=s.level,
            order=s.order,
            parent_id=s.parent_id,
            status=s.status,
            guidance=s.guidance,
        )
        for s in body.sections
    ]
    upsert_outline(canvas_id, sections)
    c = get_canvas(canvas_id)
    return _canvas_to_response(c)


@router.post("/{canvas_id}/drafts", response_model=CanvasResponse)
def canvas_upsert_draft(canvas_id: str, body: DraftUpsertRequest) -> CanvasResponse:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    b = body.block
    block = DraftBlock(
        section_id=b.section_id,
        content_md=b.content_md,
        version=b.version,
        used_fragment_ids=b.used_fragment_ids,
        used_citation_ids=b.used_citation_ids,
        updated_at=datetime.now(),
    )
    upsert_draft(canvas_id, block)
    c = get_canvas(canvas_id)
    return _canvas_to_response(c)


@router.post("/{canvas_id}/snapshot")
def canvas_snapshot(canvas_id: str) -> dict:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    try:
        ver = create_snapshot(canvas_id)
        return {"ok": True, "canvas_id": canvas_id, "version_number": ver}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{canvas_id}/restore/{version_number}")
def canvas_restore(canvas_id: str, version_number: int) -> dict:
    if not restore_snapshot(canvas_id, version_number):
        raise HTTPException(status_code=404, detail="canvas or version not found")
    return {"ok": True, "canvas_id": canvas_id, "restored_version": version_number}


@router.get("/{canvas_id}/snapshots", response_model=list[CanvasVersionItem])
def canvas_list_snapshots(canvas_id: str, limit: int = Query(50, ge=1, le=200)) -> list[CanvasVersionItem]:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    rows = list_snapshots(canvas_id, limit=limit)
    return [CanvasVersionItem(**r) for r in rows]


@router.get("/{canvas_id}/export", response_model=ExportResponse)
def canvas_export(canvas_id: str, format: str = "json") -> ExportResponse:
    """
    导出画布。
    - format=json: 返回完整 JSON 结构
    - format=markdown: 返回 Markdown 格式（大纲 + 草稿 + 引用）
    """
    import json as json_lib
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    if format == "markdown":
        if getattr(c, "refined_markdown", "").strip():
            content = c.refined_markdown
        else:
            # 生成 Markdown 格式
            lines = []
            if c.working_title:
                lines.append(f"# {c.working_title}\n")
            if c.abstract:
                lines.append(f"## 摘要\n\n{c.abstract}\n")
            if c.outline:
                lines.append("## 大纲\n")
                for s in c.outline:
                    indent = "  " * (s.level - 1)
                    lines.append(f"{indent}- {s.title}")
                lines.append("")
            if c.drafts:
                lines.append("## 正文\n")
                for section_id, block in c.drafts.items():
                    if block.content_md:
                        lines.append(block.content_md)
                        lines.append("")
            citations = get_canvas_citations(canvas_id)
            if citations:
                lines.append("## 参考文献\n")
                lines.append(format_reference_list(citations))
            content = "\n".join(lines)
    else:
        # JSON 格式：返回完整数据
        try:
            data = export_canvas(canvas_id)
            content = json_lib.dumps(data, ensure_ascii=False, indent=2)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return ExportResponse(
        format=format,
        content=content,
        session_id=c.session_id or "",
        canvas_id=canvas_id,
    )


@router.post("/{canvas_id}/refine-full", response_model=CanvasRefineResponse)
def canvas_refine_full(canvas_id: str, body: CanvasRefineRequest) -> CanvasRefineResponse:
    """对全文进行再次精炼，支持多轮迭代与回退。"""
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")

    source_md = (body.content_md or "").strip()
    if not source_md:
        source_md = (getattr(c, "refined_markdown", "") or "").strip()
    if not source_md:
        # fallback to exported markdown
        lines = []
        if c.working_title:
            lines.append(f"# {c.working_title}\n")
        if c.abstract:
            lines.append(f"## 摘要\n\n{c.abstract}\n")
        for _, block in c.drafts.items():
            if block.content_md:
                lines.append(block.content_md)
                lines.append("")
        source_md = "\n".join(lines).strip()
    if not source_md:
        raise HTTPException(status_code=400, detail="empty canvas content, nothing to refine")

    # Merge persistent + per-run directives
    directives: list[str] = []
    directives.extend(getattr(c, "user_directives", []) or [])
    directives.extend([d.strip() for d in (body.directives or []) if str(d or "").strip()])
    directives = list(dict.fromkeys(directives))
    directives_block = "\n".join(f"- {d}" for d in directives[:20]) if directives else "(none)"

    prompt_doc, locked_placeholders, locked_applied, locked_skipped = _prepare_locked_placeholders(
        source_md,
        body.locked_ranges or [],
    )
    locked_rule_line = "No locked segments in this run."
    if locked_placeholders:
        locked_rule_line = (
            "Keep every [[[LOCKED_SEGMENT_n]]] token exactly unchanged and in-place; "
            "these are protected sections and must not be edited."
        )

    cjk_n = len(re.findall(r"[\u4e00-\u9fff]", source_md))
    latin_n = len(re.findall(r"[A-Za-z]", source_md))
    if cjk_n >= latin_n:
        lang_hint = "Write in Chinese (中文)."
    else:
        lang_hint = "Write in English."

    from src.llm.llm_manager import get_manager
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client()

    prompt = _pm.render(
        "canvas_refine_document.txt",
        lang_hint=lang_hint,
        locked_rule_line=locked_rule_line,
        directives_block=directives_block,
        prompt_doc=prompt_doc,
    )

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": _pm.render("canvas_refine_system.txt")},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5000,
            timeout_seconds=300,
        )
        edited = (resp.get("final_text") or "").strip()
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="LLM refine timeout: document may be too long or provider is slow; please retry with narrower directives or shorter content.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM refine failed: {e}")

    if not edited:
        edited = source_md
    else:
        edited, lock_guard_triggered, lock_guard_message = _restore_locked_placeholders(
            edited,
            source_md,
            locked_placeholders,
        )
        if lock_guard_triggered:
            return CanvasRefineResponse(
                edited_markdown=source_md,
                snapshot_version=None,
                locked_applied=locked_applied,
                locked_skipped=locked_skipped,
                lock_guard_triggered=True,
                lock_guard_message=lock_guard_message,
            )

        # Full-refine fallback guard: never accept output that drops existing citation/evidence tags.
        edited, guard_triggered, _ = _citation_guard(source_md, edited)
        if guard_triggered:
            edited = source_md

    snapshot_ver = None
    if body.save_snapshot_before:
        try:
            snapshot_ver = create_snapshot(canvas_id)
        except Exception:
            snapshot_ver = None

    update_canvas(
        canvas_id,
        refined_markdown=edited,
        stage="refine",
        version=int(getattr(c, "version", 1) or 1) + 1,
    )

    return CanvasRefineResponse(
        edited_markdown=edited,
        snapshot_version=snapshot_ver,
        locked_applied=locked_applied,
        locked_skipped=locked_skipped,
        lock_guard_triggered=False,
        lock_guard_message="",
    )


@router.post("/{canvas_id}/citations/filter", response_model=CitationFilterResponse)
def canvas_filter_citations(canvas_id: str, body: CitationFilterRequest) -> CitationFilterResponse:
    """筛选引用池：keep_keys 仅保留指定引用；remove_keys 删除指定引用。"""
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    if body.keep_keys is not None:
        removed = filter_canvas_citations(canvas_id, body.keep_keys)
        remaining = body.keep_keys
    elif body.remove_keys is not None:
        existing_keys = [cit.cite_key or cit.id for cit in get_canvas_citations(canvas_id)]
        keep_keys = [k for k in existing_keys if k not in body.remove_keys]
        removed = filter_canvas_citations(canvas_id, keep_keys)
        remaining = [cit.cite_key or cit.id for cit in get_canvas_citations(canvas_id)]
    else:
        raise HTTPException(status_code=400, detail="keep_keys or remove_keys is required")
    return CitationFilterResponse(removed_count=removed, remaining_keys=remaining)


@router.delete("/{canvas_id}/citations/{cite_key}")
def canvas_delete_citation(canvas_id: str, cite_key: str) -> dict:
    """删除指定 cite_key 的引用。"""
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    if not delete_canvas_citation(canvas_id, cite_key):
        raise HTTPException(status_code=404, detail="citation not found")
    return {"ok": True, "canvas_id": canvas_id, "removed_cite_key": cite_key}


@router.get("/{canvas_id}/citations")
def canvas_citations(
    canvas_id: str,
    format: str = Query("both", description="bibtex | text | ris | both"),
) -> dict:
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")
    citations = get_canvas_citations(canvas_id)
    if format == "bibtex":
        return {"format": "bibtex", "content": format_bibtex(citations)}
    if format == "text":
        return {"format": "text", "content": format_reference_list(citations)}
    if format == "ris":
        return {"format": "ris", "content": format_ris(citations)}
    return {
        "format": "both",
        "bibtex": format_bibtex(citations),
        "reference_list": format_reference_list(citations),
        "citations": [
            CitationResponse(
                cite_key=c.cite_key or c.id,
                title=c.title,
                authors=c.authors or [],
                year=c.year,
                doi=c.doi,
                url=c.url,
                bibtex=c.bibtex,
            )
            for c in citations
        ],
    }


# ---- AI Edit ----

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"


def _extract_bracket_tags(text: str) -> list[str]:
    # Keep simple and conservative: treat bracketed tokens as evidence/citation-like tags.
    tags = re.findall(r"\[[^\[\]\n]{1,80}\]", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _citation_guard(source_text: str, edited_text: str) -> tuple[str, bool, str]:
    source_tags = _extract_bracket_tags(source_text)
    if not source_tags:
        return edited_text, False, ""
    missing = [t for t in source_tags if t not in edited_text]
    if missing:
        preview = ", ".join(missing[:5])
        return (
            source_text,
            True,
            f"Detected citation/evidence loss, rolled back this edit. Missing tags: {preview}",
        )
    return edited_text, False, ""


def _prepare_locked_placeholders(
    source_text: str,
    locked_ranges: list[dict],
) -> tuple[str, dict[str, str], int, int]:
    if not locked_ranges:
        return source_text, {}, 0, 0

    normalized: list[tuple[int, int, str]] = []
    skipped = 0
    total_len = len(source_text)

    for raw in locked_ranges:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        try:
            start = int(raw.get("start", -1))
            end = int(raw.get("end", -1))
            expected = str(raw.get("text", ""))
        except Exception:
            skipped += 1
            continue
        if start < 0 or end <= start or end > total_len:
            skipped += 1
            continue
        seg = source_text[start:end]
        if expected and seg != expected:
            skipped += 1
            continue
        normalized.append((start, end, seg))

    if not normalized:
        return source_text, {}, 0, skipped

    normalized.sort(key=lambda x: x[0])
    non_overlap: list[tuple[int, int, str]] = []
    last_end = -1
    for item in normalized:
        if item[0] < last_end:
            skipped += 1
            continue
        non_overlap.append(item)
        last_end = item[1]

    if not non_overlap:
        return source_text, {}, 0, skipped

    placeholders: dict[str, str] = {}
    parts: list[str] = []
    cursor = 0
    for i, (start, end, seg) in enumerate(non_overlap):
        token = f"[[[LOCKED_SEGMENT_{i}]]]"
        parts.append(source_text[cursor:start])
        parts.append(token)
        placeholders[token] = seg
        cursor = end
    parts.append(source_text[cursor:])
    return "".join(parts), placeholders, len(non_overlap), skipped


def _restore_locked_placeholders(
    edited_text: str,
    source_text: str,
    placeholders: dict[str, str],
) -> tuple[str, bool, str]:
    if not placeholders:
        return edited_text, False, ""
    missing = [t for t in placeholders.keys() if t not in edited_text]
    if missing:
        preview = ", ".join(missing[:3])
        return (
            source_text,
            True,
            f"Locked segments were modified/removed by model, rolled back this full refine. Missing tokens: {preview}",
        )
    restored = edited_text
    for token, seg in placeholders.items():
        restored = restored.replace(token, seg)
    return restored, False, ""


_AI_EDIT_PROMPTS = {
    "rewrite": _pm.render("canvas_edit_rewrite.txt"),
    "expand": _pm.render("canvas_edit_expand.txt"),
    "condense": _pm.render("canvas_edit_condense.txt"),
    "add_citations": _pm.render("canvas_edit_add_citations.txt"),
    "targeted_refine": _pm.render("canvas_edit_targeted_refine.txt"),
}


@router.post("/{canvas_id}/ai-edit", response_model=CanvasAIEditResponse)
def canvas_ai_edit(canvas_id: str, body: CanvasAIEditRequest) -> CanvasAIEditResponse:
    """AI 段落级编辑：重写/扩展/精简/添加引用"""
    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")

    action = body.action
    if action not in _AI_EDIT_PROMPTS:
        raise HTTPException(status_code=400, detail=f"unknown action: {action}, expected: {list(_AI_EDIT_PROMPTS.keys())}")

    # 获取 LLM 客户端
    from src.llm.llm_manager import get_manager
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client()

    # 可选：检索补充资料（用于 add_citations）
    retrieval_context = ""
    citations_added: list[str] = []
    if body.search_mode != "none" and action == "add_citations":
        try:
            from src.retrieval.service import get_retrieval_service
            svc = get_retrieval_service()
            pack = svc.search(query=body.section_text[:200], mode=body.search_mode, top_k=5)
            if pack.chunks:
                refs = []
                for ch in pack.chunks:
                    cite_key = ch.chunk_id or ch.doc_id
                    refs.append(f"[{cite_key}] {ch.text[:200]}")
                    citations_added.append(cite_key)
                retrieval_context = "\n\n可用参考资料：\n" + "\n".join(refs)
        except Exception:
            pass

    # 构建 prompt
    system = _AI_EDIT_PROMPTS[action]
    if retrieval_context:
        system += retrieval_context

    directive_block = ""
    if (body.directive or "").strip():
        directive_block = f"\n\n用户定向指令：\n{body.directive.strip()}"

    user_msg = body.section_text
    if body.context:
        user_msg = f"上下文：\n{body.context}\n\n需要编辑的段落：\n{body.section_text}{directive_block}"
    elif directive_block:
        user_msg = f"需要编辑的段落：\n{body.section_text}{directive_block}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat(messages)
        edited_text = (resp.get("final_text") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")

    guard_triggered = False
    guard_message = ""
    if body.preserve_citations:
        edited_text, guard_triggered, guard_message = _citation_guard(body.section_text, edited_text)

    # 自动创建快照
    try:
        create_snapshot(canvas_id)
    except Exception:
        pass

    return CanvasAIEditResponse(
        edited_text=edited_text,
        citations_added=citations_added,
        citation_guard_triggered=guard_triggered,
        citation_guard_message=guard_message,
    )
