"""
自动完成综述服务。

根据检索结果和 LLM 自动执行：检索 -> 生成大纲 -> 逐章写作 -> 返回完整 Markdown。
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.collaboration.canvas.canvas_manager import (
    create_canvas,
    get_canvas,
    upsert_draft,
    upsert_outline,
    update_canvas,
)
from src.collaboration.canvas.models import DraftBlock, OutlineSection, SurveyCanvas
from src.collaboration.citation.manager import resolve_response_citations, sync_evidence_to_canvas
from src.collaboration.citation.formatter import format_reference_list
from src.retrieval.service import RetrievalService, get_retrieval_service


@dataclass
class AutoCompleteResult:
    """自动完成结果"""

    session_id: str = ""
    canvas_id: str = ""
    markdown: str = ""
    outline: List[str] = field(default_factory=list)
    citations: List[Any] = field(default_factory=list)  # Citation 对象列表
    total_time_ms: float = 0.0
    dashboard: Optional[Dict[str, Any]] = None  # 研究进度仪表盘（Agent 模式）


def _parse_outline_from_llm(text: str) -> List[Dict[str, Any]]:
    """从 LLM 返回的文本解析大纲结构。"""
    sections: List[Dict[str, Any]] = []
    lines = (text or "").strip().split("\n")
    order = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 匹配 "1. 引言" / "1.1 背景" / "- 引言" / "* 背景"
        m = re.match(r"^([\d\.\-\*]+)\s*(.+)$", line)
        if m:
            prefix, title = m.group(1), m.group(2).strip()
            level = 1
            if re.match(r"^\d", prefix):
                parts = prefix.split(".")
                level = min(len(parts), 3)
            sections.append({"title": title, "level": level, "order": order})
            order += 1
    return sections


class AutoCompleteService:
    """自动完成综述服务"""

    def __init__(
        self,
        llm_client: Any,
        retrieval_service: Optional[RetrievalService] = None,
        max_sections: int = 6,
        max_words_per_section: int = 500,
        include_abstract: bool = True,
    ):
        self.llm = llm_client
        self.retrieval = retrieval_service
        self.max_sections = max_sections
        self.max_words_per_section = max_words_per_section
        self.include_abstract = include_abstract

    def complete(
        self,
        topic: str,
        canvas_id: Optional[str] = None,
        session_id: str = "",
        search_mode: str = "hybrid",
        existing_outline: Optional[List[OutlineSection]] = None,
        user_id: str = "",
        filters: Optional[Dict[str, Any]] = None,
        clarification_answers: Optional[Dict[str, str]] = None,
        output_language: Optional[str] = None,
        step_models: Optional[Dict[str, Optional[str]]] = None,
        use_agent: bool = False,
    ) -> AutoCompleteResult:
        """
        自动完成综述（Deep Research 引擎）。

        Args:
            topic: 综述主题
            canvas_id: 可选，已有画布 ID
            session_id: 可选，会话 ID
            search_mode: 检索模式 local | web | hybrid
            existing_outline: 可选，已有大纲（跳过生成大纲）
            user_id: 用户 ID
            filters: UI 检索参数透传（web_providers, llm_provider, model_override, final_top_k 等）
            clarification_answers: 澄清问题回答 {question_id: answer_text}
            use_agent: 是否使用递归研究 Agent（LangGraph）

        Returns:
            AutoCompleteResult
        """
        t0 = time.perf_counter()
        result = AutoCompleteResult(session_id=session_id, canvas_id=canvas_id or "")

        # ── 新引擎：递归研究 Agent ──
        if use_agent:
            return self._run_agent(topic, canvas_id, session_id, user_id,
                                   search_mode, filters, clarification_answers, output_language, step_models, t0)
        filters = filters or {}
        collection = (filters.get("collection") or "").strip() or None
        retrieval = self.retrieval or get_retrieval_service(collection=collection, top_k=15)

        # 从 filters 取 top_k，Deep Research 默认更大
        main_top_k = filters.get("final_top_k") or 15
        section_top_k = max(main_top_k // 2, 10)

        # 1. 检索主题资料（透传完整 filters）
        pack = retrieval.search(
            query=topic, mode=search_mode, top_k=main_top_k, filters=filters or None,
        )
        context_str = pack.to_context_string(max_chunks=min(main_top_k, 20))

        # 2. 创建或获取画布
        if canvas_id:
            canvas = get_canvas(canvas_id)
            if canvas is None:
                canvas = create_canvas(session_id=session_id, topic=topic, user_id=user_id)
                result.canvas_id = canvas.id
        else:
            canvas = create_canvas(session_id=session_id, topic=topic, user_id=user_id)
            result.canvas_id = canvas.id

        # 3. 同步引用到画布
        sync_evidence_to_canvas(result.canvas_id, pack)

        # 4. 生成或使用大纲（融入澄清回答）
        if existing_outline and len(existing_outline) > 0:
            outline_sections = existing_outline
        else:
            outline_sections = self._generate_outline(topic, context_str, clarification_answers)
            if outline_sections:
                upsert_outline(result.canvas_id, outline_sections)
            else:
                # fallback: 单章
                outline_sections = [
                    OutlineSection(id="s1", title="正文", level=1, order=0),
                ]
                upsert_outline(result.canvas_id, outline_sections)

        result.outline = [s.title for s in outline_sections]
        sorted_sections = sorted(outline_sections, key=lambda s: (s.order, s.level))
        doc_key_to_cite_key: Dict[str, str] = {}
        existing_cite_keys: set[str] = set()
        cited_pool: Dict[str, Any] = {}

        # 5. 逐章写作
        markdown_parts: List[str] = []
        markdown_parts.append(f"# {topic}\n")

        if self.include_abstract:
            abstract = self._generate_abstract(topic, context_str)
            if abstract:
                markdown_parts.append(f"## 摘要\n{abstract}\n")

        for section in sorted_sections[: self.max_sections]:
            section_query = f"{topic} {section.title}"
            section_pack = retrieval.search(
                query=section_query, mode=search_mode,
                top_k=section_top_k, filters=filters or None,
            )
            section_context = section_pack.to_context_string(max_chunks=min(section_top_k, 15))

            content = self._generate_section(
                topic=topic,
                section_title=section.title,
                context=section_context,
            )
            if content:
                # 章节级 hash 引文后处理，并跨章节复用 cite_key 映射
                content, section_citations, _ = resolve_response_citations(
                    content,
                    section_pack.chunks,
                    doc_key_to_cite_key=doc_key_to_cite_key,
                    existing_cite_keys=existing_cite_keys,
                    include_unreferenced_documents=False,
                )
                for c in section_citations:
                    key = c.cite_key or c.id
                    if key and key not in cited_pool:
                        cited_pool[key] = c
                sync_evidence_to_canvas(result.canvas_id, section_pack)
                markdown_parts.append(f"## {section.title}\n{content}\n")

                block = DraftBlock(
                    section_id=section.id,
                    content_md=content,
                    version=1,
                    used_fragment_ids=[],
                    used_citation_ids=[],
                )
                upsert_draft(result.canvas_id, block)

        # 6. 参考文献
        if cited_pool:
            markdown_parts.append("## 参考文献\n")
            markdown_parts.append(format_reference_list(list(cited_pool.values())).strip())
        else:
            canvas = get_canvas(result.canvas_id)
            if canvas and canvas.citation_pool:
                markdown_parts.append("## 参考文献\n")
                markdown_parts.append(format_reference_list(list(canvas.citation_pool.values())).strip())

        result.markdown = "\n".join(markdown_parts).strip() + "\n"
        update_canvas(result.canvas_id, stage="refine")
        result.total_time_ms = (time.perf_counter() - t0) * 1000
        # 最终引用列表（优先使用正文替换后实际引用到的文档级 citation）
        if cited_pool:
            result.citations = list(cited_pool.values())
        else:
            final_canvas = get_canvas(result.canvas_id)
            if final_canvas and final_canvas.citation_pool:
                result.citations = list(final_canvas.citation_pool.values())
        return result

    def _generate_outline(
        self, topic: str, context: str,
        clarification_answers: Optional[Dict[str, str]] = None,
    ) -> List[OutlineSection]:
        """LLM 生成大纲（可融入用户澄清回答）"""
        answers_block = ""
        if clarification_answers:
            lines = [f"- {k}: {v}" for k, v in clarification_answers.items() if v]
            if lines:
                answers_block = "\n用户补充信息：\n" + "\n".join(lines) + "\n"

        prompt = f"""根据以下参考资料，为综述主题「{topic}」生成大纲结构。
{answers_block}
要求：
1. 列出 3-6 个主要章节，每行一个
2. 格式示例：1. 引言  2. 研究背景  3. 核心内容  4. 展望
3. 只输出章节列表，不要其他文字

参考资料：
{context[:3000] if len(context) > 3000 else context}
"""
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术写作助手，只返回大纲列表。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            )
            text = (resp.get("final_text") or "").strip()
            parsed = _parse_outline_from_llm(text)
            sections = []
            for i, p in enumerate(parsed[: self.max_sections]):
                sections.append(
                    OutlineSection(
                        id=f"s{i+1}",
                        title=p.get("title", f"章节{i+1}"),
                        level=p.get("level", 1),
                        order=i,
                        status="todo",
                    )
                )
            return sections
        except Exception:
            return []

    def _generate_abstract(self, topic: str, context: str) -> str:
        """LLM 生成摘要"""
        prompt = f"""根据以下参考资料，为综述主题「{topic}」撰写 150-250 字的摘要。
只输出摘要内容，不要其他文字。

参考资料：
{context[:2500] if len(context) > 2500 else context}
"""
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术写作助手。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
            )
            return (resp.get("final_text") or "").strip()
        except Exception:
            return ""

    def _generate_section(
        self,
        topic: str,
        section_title: str,
        context: str,
    ) -> str:
        """LLM 生成章节内容"""
        prompt = f"""根据以下参考资料，撰写综述「{topic}」中「{section_title}」章节的内容。
要求：
1. 字数约 {self.max_words_per_section} 字
2. 使用学术语言，逻辑清晰
3. 引用时使用方括号标记；若参考资料中有 [ref_hash] 请直接引用该标记（后续会自动替换为正式 cite_key）
4. 只输出章节正文，不要标题

参考资料：
{context}
"""
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术写作助手，基于参考资料撰写综述章节。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1200,
            )
            return (resp.get("final_text") or "").strip()
        except Exception:
            return ""

    def _run_agent(
        self,
        topic: str,
        canvas_id: Optional[str],
        session_id: str,
        user_id: str,
        search_mode: str,
        filters: Optional[Dict[str, Any]],
        clarification_answers: Optional[Dict[str, str]],
        output_language: Optional[str],
        step_models: Optional[Dict[str, Optional[str]]],
        t0: float,
    ) -> AutoCompleteResult:
        """使用递归研究 Agent (LangGraph) 执行 Deep Research"""
        from src.collaboration.research.agent import run_deep_research

        agent_result = run_deep_research(
            topic=topic,
            llm_client=self.llm,
            canvas_id=canvas_id,
            session_id=session_id,
            user_id=user_id,
            search_mode=search_mode,
            filters=filters,
            max_iterations=self.max_sections * 5,
            clarification_answers=clarification_answers,
            output_language=output_language or "auto",
            step_models=step_models,
        )

        # 更新 Canvas
        cid = agent_result.get("canvas_id", canvas_id or "")
        if cid:
            try:
                update_canvas(cid, markdown=agent_result["markdown"])
            except Exception:
                pass

        return AutoCompleteResult(
            session_id=session_id,
            canvas_id=cid,
            markdown=agent_result["markdown"],
            outline=agent_result.get("outline", []),
            citations=agent_result.get("citations", []),
            total_time_ms=(time.perf_counter() - t0) * 1000,
            dashboard=agent_result.get("dashboard"),
        )
