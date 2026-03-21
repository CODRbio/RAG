from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlmodel import Session, select

from config.settings import settings
from src.collaboration.citation.manager import (
    chunk_to_citation,
    merge_citations_by_document,
    resolve_response_citations,
)
from src.db.engine import get_engine
from src.db.models import Paper, PaperMetadata, ScholarLibrary, ScholarLibraryPaper
from src.indexing.assistant_artifact_store import (
    list_resource_annotations,
    sync_annotation_vector,
    upsert_media_vectors,
)
from src.indexing.paper_metadata_store import paper_meta_store
from src.log import get_logger
from src.parser.pdf_parser import PDFProcessor, ParserConfig
from src.retrieval.evidence import EvidenceChunk, EvidencePack
from src.services.collection_library_binding_service import resolve_bound_library_for_collection
from src.services.global_graph_service import get_global_graph_service
from src.utils.path_manager import PathManager

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
_SUMMARY_SECTION_KEYWORDS = ("abstract", "introduction", "result", "discussion", "conclusion", "finding")
_ASPECT_KEYWORDS = {
    "objective": ("objective", "motivation", "goal", "aim", "abstract", "introduction"),
    "methodology": ("method", "approach", "experiment", "materials", "model", "design"),
    "key_findings": ("result", "finding", "discussion", "conclusion"),
    "limitations": ("limitation", "future work", "threat", "discussion"),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _serialize_citation(citation: Any) -> Dict[str, Any]:
    anchors = []
    for anchor in getattr(citation, "anchors", []) or []:
        anchors.append(
            {
                "chunk_id": getattr(anchor, "chunk_id", "") or "",
                "page_num": getattr(anchor, "page_num", None),
                "bbox": getattr(anchor, "bbox", None),
                "snippet": getattr(anchor, "snippet", None),
            }
        )
    primary = anchors[0] if anchors else {}
    return {
        "cite_key": getattr(citation, "cite_key", None) or getattr(citation, "id", ""),
        "chunk_id": primary.get("chunk_id") or getattr(citation, "chunk_id", None),
        "title": getattr(citation, "title", "") or "",
        "authors": getattr(citation, "authors", None) or [],
        "year": getattr(citation, "year", None),
        "doc_id": getattr(citation, "doc_id", None),
        "url": getattr(citation, "url", None),
        "pdf_url": getattr(citation, "pdf_url", None),
        "doi": getattr(citation, "doi", None),
        "bbox": primary.get("bbox") or getattr(citation, "bbox", None),
        "page_num": primary.get("page_num") or getattr(citation, "page_num", None),
        "anchors": anchors,
        "provider": getattr(citation, "provider", None),
    }


def _chunk_to_serializable(chunk: EvidenceChunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "score": chunk.score,
        "source_type": chunk.source_type,
        "doc_title": chunk.doc_title,
        "authors": chunk.authors or [],
        "year": chunk.year,
        "url": chunk.url,
        "pdf_url": chunk.pdf_url,
        "doi": chunk.doi,
        "paper_uid": chunk.paper_uid,
        "page_num": chunk.page_num + 1 if chunk.page_num is not None else None,
        "section_title": chunk.section_title,
        "evidence_type": chunk.evidence_type,
        "bbox": chunk.bbox,
        "provider": chunk.provider,
    }


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()) if len(tok) >= 2}


def _score_chunk(query: str, chunk: EvidenceChunk) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    text = " ".join(
        part for part in [
            chunk.text,
            chunk.doc_title or "",
            chunk.section_title or "",
            chunk.evidence_type or "",
        ]
        if part
    )
    c_tokens = _tokenize(text)
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    bonus = 0.0
    if chunk.evidence_type == "figure":
        bonus += 0.5
    if chunk.evidence_type == "annotation":
        bonus -= 0.25
    return float(overlap) + bonus


def _is_summary_section(section_title: Optional[str]) -> bool:
    title = (section_title or "").lower()
    return any(key in title for key in _SUMMARY_SECTION_KEYWORDS)


class ReferenceAssistantService:
    def __init__(self) -> None:
        self.graph_service = get_global_graph_service()

    def _normalize_scope(
        self,
        *,
        user_id: str,
        scope: Optional[Dict[str, Any]] = None,
        default_collection: Optional[str] = None,
    ) -> Dict[str, str]:
        scope_type = str((scope or {}).get("scope_type") or "").strip().lower()
        scope_key = str((scope or {}).get("scope_key") or "").strip()
        if not scope_type:
            if default_collection:
                return {"user_id": user_id, "scope_type": "collection", "scope_key": default_collection}
            return {
                "user_id": user_id,
                "scope_type": "global",
                "scope_key": (settings.collection.global_ or "global").strip() or "global",
            }
        if scope_type == "collection":
            if not scope_key and default_collection:
                scope_key = default_collection
            if not scope_key:
                raise ValueError("collection scope requires scope_key")
        elif scope_type == "library":
            if not scope_key:
                raise ValueError("library scope requires scope_key")
        elif scope_type == "global":
            scope_key = scope_key or "global"
        else:
            raise ValueError(f"unsupported scope_type: {scope_type}")
        return {"user_id": user_id, "scope_type": scope_type, "scope_key": scope_key}

    def _resolve_default_scope(self, user_id: str, collection_name: Optional[str]) -> Dict[str, str]:
        if collection_name:
            return {"user_id": user_id, "scope_type": "collection", "scope_key": collection_name}
        return self._normalize_scope(user_id=user_id, scope=None, default_collection=None)

    def _resolve_parsed_json(self, paper: Dict[str, Any]) -> Optional[Path]:
        user_id = paper["user_id"]
        paper_id = paper["paper_id"]
        candidates: List[Path] = []
        library_name = paper.get("library_name")
        if library_name:
            candidates.append(PathManager.get_user_library_parsed_path(user_id, library_name) / paper_id / "enriched.json")
        for root in PathManager.get_user_all_library_parsed_paths(user_id):
            candidates.append(root / paper_id / "enriched.json")
        candidates.append(PathManager.get_user_parsed_path(user_id) / paper_id / "enriched.json")
        candidates.append(settings.path.data / "parsed" / paper_id / "enriched.json")
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
        return None

    def locate_paper(
        self,
        locator: Dict[str, Any],
        *,
        user_id: str,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        paper_uid = str(locator.get("paper_uid") or "").strip()
        paper_id = str(locator.get("paper_id") or "").strip()
        collection = str(locator.get("collection") or "").strip()
        normalized_scope = self._normalize_scope(user_id=user_id, scope=scope, default_collection=collection or None)

        def _match_stmt(stmt):
            if normalized_scope["scope_type"] == "collection":
                stmt = stmt.where(Paper.collection == normalized_scope["scope_key"])
            elif normalized_scope["scope_type"] == "library":
                stmt = stmt.where(Paper.library_id == _safe_int(normalized_scope["scope_key"], 0))
            return stmt.where(Paper.user_id == user_id).order_by(Paper.created_at.desc())

        with Session(get_engine()) as session:
            row: Optional[Paper] = None
            if paper_uid:
                row = session.exec(_match_stmt(select(Paper).where(Paper.paper_uid == paper_uid))).first()
                if row is None:
                    stmt = select(Paper).where(Paper.paper_uid == paper_uid, Paper.user_id == user_id).order_by(Paper.created_at.desc())
                    row = session.exec(stmt).first()
            if row is None and paper_id:
                stmt = select(Paper).where(Paper.paper_id == paper_id)
                if collection:
                    stmt = stmt.where(Paper.collection == collection)
                row = session.exec(_match_stmt(stmt)).first()
                if row is None:
                    row = session.exec(
                        select(Paper).where(Paper.paper_id == paper_id, Paper.user_id == user_id).order_by(Paper.created_at.desc())
                    ).first()
            if row is None and paper_uid:
                lib_stmt = select(ScholarLibraryPaper).where(ScholarLibraryPaper.paper_uid == paper_uid)
                if normalized_scope["scope_type"] == "library":
                    lib_stmt = lib_stmt.where(ScholarLibraryPaper.library_id == _safe_int(normalized_scope["scope_key"], 0))
                lib_row = session.exec(lib_stmt.order_by(ScholarLibraryPaper.added_at.desc())).first()
                if lib_row is not None and (lib_row.collection_paper_id or "").strip():
                    row = session.exec(
                        select(Paper).where(
                            Paper.user_id == user_id,
                            Paper.paper_id == lib_row.collection_paper_id,
                        ).order_by(Paper.created_at.desc())
                    ).first()
            if row is None:
                raise ValueError("paper not found in current scope")

            meta = paper_meta_store.get(row.paper_id) or {}
            library_name = None
            if getattr(row, "library_id", None) is not None:
                lib = session.get(ScholarLibrary, row.library_id)
                if lib is not None and lib.user_id == user_id:
                    library_name = lib.name
            result = {
                "user_id": user_id,
                "collection": row.collection,
                "paper_id": row.paper_id,
                "paper_uid": (getattr(row, "paper_uid", "") or meta.get("paper_uid") or paper_uid).strip(),
                "file_path": row.file_path or "",
                "filename": row.filename or "",
                "library_id": getattr(row, "library_id", None),
                "library_paper_id": getattr(row, "library_paper_id", None),
                "source": getattr(row, "source", "") or "",
                "library_name": library_name,
                "doc_metadata": meta,
            }
            parsed_json = self._resolve_parsed_json(result)
            result["parsed_json"] = str(parsed_json) if parsed_json else ""
            return result

    def _load_doc(self, paper: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        parsed_json = Path(str(paper.get("parsed_json") or ""))
        if not parsed_json.exists():
            raise ValueError("parsed enriched.json not found")
        with open(parsed_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        doc = PDFProcessor.load(parsed_json)
        return doc, raw

    def _doc_metadata(self, paper: Dict[str, Any], raw_doc: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(paper.get("doc_metadata") or {})
        doc_meta = raw_doc.get("doc_metadata") if isinstance(raw_doc.get("doc_metadata"), dict) else {}
        meta.update({k: v for k, v in doc_meta.items() if v})
        meta.setdefault("title", meta.get("title") or paper["paper_id"])
        meta.setdefault("paper_uid", paper.get("paper_uid") or "")
        return meta

    def _annotation_chunks(self, paper: Dict[str, Any], doc_meta: Dict[str, Any]) -> List[EvidenceChunk]:
        paper_uid = (paper.get("paper_uid") or "").strip()
        if not paper_uid:
            return []
        rows = list_resource_annotations(user_id=paper["user_id"], paper_uid=paper_uid, status="active", limit=200)
        out: List[EvidenceChunk] = []
        for row in rows:
            locator = row.get_target_locator()
            text_parts = []
            if row.target_text:
                text_parts.append(row.target_text)
            if row.directive:
                text_parts.append(row.directive)
            text = "\n".join(part for part in text_parts if part.strip()).strip()
            if not text:
                continue
            out.append(
                EvidenceChunk(
                    chunk_id=f"annotation:{row.id}",
                    doc_id=paper["paper_id"],
                    text=text,
                    score=0.0,
                    source_type="dense",
                    doc_title=doc_meta.get("title"),
                    authors=doc_meta.get("authors"),
                    year=doc_meta.get("year"),
                    doi=doc_meta.get("doi"),
                    paper_uid=paper_uid,
                    page_num=locator.get("page") if isinstance(locator.get("page"), int) else None,
                    section_title=str(locator.get("section_path") or ""),
                    evidence_type="annotation",
                    bbox=locator.get("bbox") if isinstance(locator.get("bbox"), list) else None,
                    provider="local",
                )
            )
        return out

    def _doc_chunks(self, paper: Dict[str, Any], doc: Any, raw_doc: Dict[str, Any], doc_meta: Dict[str, Any]) -> List[EvidenceChunk]:
        chunks: List[EvidenceChunk] = []
        for idx, block in enumerate(getattr(doc, "content_flow", []) or []):
            block_type = getattr(getattr(block, "block_type", None), "value", None) or str(getattr(block, "block_type", "") or "")
            section_title = " > ".join(getattr(block, "heading_path", []) or [])
            page_num = getattr(block, "page_index", None)
            bbox = list(getattr(block, "bbox", []) or []) or None
            if getattr(block, "text", None):
                text = str(block.text).strip()
                if text:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{paper['paper_id']}:{getattr(block, 'block_id', idx)}:text",
                            doc_id=paper["paper_id"],
                            text=text,
                            score=0.0,
                            source_type="dense",
                            doc_title=doc_meta.get("title"),
                            authors=doc_meta.get("authors"),
                            year=doc_meta.get("year"),
                            url=(doc_meta.get("extra") or {}).get("url") if isinstance(doc_meta.get("extra"), dict) else None,
                            pdf_url=(doc_meta.get("extra") or {}).get("pdf_url") if isinstance(doc_meta.get("extra"), dict) else None,
                            doi=doc_meta.get("doi"),
                            paper_uid=paper.get("paper_uid"),
                            page_num=page_num,
                            section_title=section_title,
                            evidence_type="text",
                            bbox=bbox,
                            provider="local",
                        )
                    )
            table_data = getattr(block, "table_data", None)
            if table_data is not None:
                table_text = getattr(table_data, "markdown", None) or getattr(getattr(block, "enrichment", None), "semantic_summary", None) or ""
                table_text = str(table_text).strip()
                if table_text:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{paper['paper_id']}:{getattr(block, 'block_id', idx)}:table",
                            doc_id=paper["paper_id"],
                            text=table_text,
                            score=0.0,
                            source_type="dense",
                            doc_title=doc_meta.get("title"),
                            authors=doc_meta.get("authors"),
                            year=doc_meta.get("year"),
                            doi=doc_meta.get("doi"),
                            paper_uid=paper.get("paper_uid"),
                            page_num=page_num,
                            section_title=section_title,
                            evidence_type="table",
                            bbox=bbox,
                            provider="local",
                        )
                    )
            figure_data = getattr(block, "figure_data", None)
            if figure_data is not None:
                caption = str(getattr(figure_data, "caption", "") or "").strip()
                if caption:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{paper['paper_id']}:{getattr(block, 'block_id', idx)}:caption",
                            doc_id=paper["paper_id"],
                            text=caption,
                            score=0.0,
                            source_type="dense",
                            doc_title=doc_meta.get("title"),
                            authors=doc_meta.get("authors"),
                            year=doc_meta.get("year"),
                            doi=doc_meta.get("doi"),
                            paper_uid=paper.get("paper_uid"),
                            page_num=page_num,
                            section_title=section_title,
                            evidence_type="figure",
                            bbox=bbox,
                            provider="local",
                        )
                    )
                analysis_parts = []
                enrichment = getattr(block, "enrichment", None)
                interpretation = getattr(enrichment, "interpretation", None) if enrichment is not None else None
                if interpretation is not None:
                    if getattr(interpretation, "description", None):
                        analysis_parts.append(interpretation.description)
                    if getattr(interpretation, "key_findings", None):
                        analysis_parts.extend(str(item) for item in interpretation.key_findings if str(item).strip())
                    if getattr(interpretation, "evidence", None):
                        analysis_parts.extend(str(item) for item in interpretation.evidence if str(item).strip())
                if getattr(figure_data, "ocr_text", None):
                    analysis_parts.append(str(figure_data.ocr_text))
                analysis_text = "\n".join(part for part in analysis_parts if part.strip()).strip()
                if analysis_text:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{paper['paper_id']}:{getattr(block, 'block_id', idx)}:analysis",
                            doc_id=paper["paper_id"],
                            text=analysis_text,
                            score=0.0,
                            source_type="dense",
                            doc_title=doc_meta.get("title"),
                            authors=doc_meta.get("authors"),
                            year=doc_meta.get("year"),
                            doi=doc_meta.get("doi"),
                            paper_uid=paper.get("paper_uid"),
                            page_num=page_num,
                            section_title=section_title,
                            evidence_type="figure",
                            bbox=bbox,
                            provider="local",
                        )
                    )
        chunks.extend(self._annotation_chunks(paper, doc_meta))
        return chunks

    def _fallback_citations(self, chunks: List[EvidenceChunk], max_docs: int = 4) -> List[Dict[str, Any]]:
        base_chunks = [chunk for chunk in chunks if chunk.evidence_type != "annotation"]
        if not base_chunks:
            return []
        citations = merge_citations_by_document([chunk_to_citation(chunk) for chunk in base_chunks[: max_docs * 2]])
        return [_serialize_citation(c) for c in citations[:max_docs]]

    def _build_evidence_summary(self, chunks: List[EvidenceChunk]) -> Dict[str, Any]:
        docs = {chunk.doc_id for chunk in chunks if chunk.doc_id}
        by_type = Counter(chunk.evidence_type or "unknown" for chunk in chunks)
        return {
            "total_chunks": len(chunks),
            "total_documents": len(docs),
            "evidence_type_breakdown": dict(sorted(by_type.items())),
        }

    def _llm_answer(
        self,
        *,
        task: str,
        prompt: str,
        chunks: List[EvidenceChunk],
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        if not chunks:
            return "", [], False
        try:
            from src.llm.llm_manager import get_manager

            pack = EvidencePack(query=task, chunks=chunks)
            context = pack.to_context_string(max_chunks=min(len(chunks), 8))
            manager = get_manager(str(_CONFIG_PATH))
            client = manager.get_client(llm_provider or None)
            system = (
                "You are an academic paper assistant. "
                "Use only the provided evidence. "
                "For every substantive claim, cite one or more evidence markers like [ref:xxxxxxxx]. "
                "If the evidence is insufficient, say so explicitly."
            )
            response = client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Task: {task}\n\n{prompt}\n\nEvidence:\n{context}"},
                ],
                model=model_override or None,
            )
            text = str(response.get("final_text") or "").strip()
            if not text:
                return "", [], False
            resolved, citations, _ = resolve_response_citations(text, chunks)
            return resolved, [_serialize_citation(citation) for citation in citations], True
        except Exception as exc:
            logger.debug("reference assistant llm fallback task=%s err=%s", task, exc)
            return "", [], False

    def summarize_paper(
        self,
        locator: Dict[str, Any],
        *,
        user_id: str,
        scope: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        paper = self.locate_paper(locator, user_id=user_id, scope=scope)
        doc, raw_doc = self._load_doc(paper)
        doc_meta = self._doc_metadata(paper, raw_doc)
        chunks = self._doc_chunks(paper, doc, raw_doc, doc_meta)
        selected = sorted(
            chunks,
            key=lambda chunk: (
                0 if _is_summary_section(chunk.section_title) else 1,
                0 if chunk.evidence_type == "figure" else 1,
                chunk.page_num or 9999,
            ),
        )[:8]
        prompt = (
            "Produce a deep single-paper reading summary in Markdown with short sections for "
            "objective, methodology, key findings, and figures/media observations."
        )
        if question:
            prompt += f"\nAlso address this focus question: {question}"
        summary_text, citations, used_llm = self._llm_answer(
            task="single paper summary",
            prompt=prompt,
            chunks=selected,
            llm_provider=llm_provider,
            model_override=model_override,
        )
        if not summary_text:
            lines = [f"# {doc_meta.get('title') or paper['paper_id']}"]
            global_summary = str(raw_doc.get("global_summary") or "").strip()
            if global_summary:
                lines.append(global_summary)
            figure_chunks = [chunk for chunk in selected if chunk.evidence_type == "figure"]
            if figure_chunks:
                lines.append("")
                lines.append("## Figure And Media Signals")
                for chunk in figure_chunks[:3]:
                    lines.append(f"- {chunk.text[:260]}")
            text_chunks = [chunk for chunk in selected if chunk.evidence_type == "text"]
            if text_chunks:
                lines.append("")
                lines.append("## Core Evidence")
                for chunk in text_chunks[:3]:
                    label = chunk.section_title or "paper section"
                    lines.append(f"- {label}: {chunk.text[:260]}")
            if question:
                lines.append("")
                lines.append(f"## Focus Question\n{question}")
            summary_text = "\n".join(lines).strip()
        return {
            "paper_card": {
                "paper_id": paper["paper_id"],
                "paper_uid": paper.get("paper_uid"),
                "collection": paper["collection"],
                "title": doc_meta.get("title") or paper["paper_id"],
                "authors": doc_meta.get("authors") or [],
                "year": doc_meta.get("year"),
                "doi": doc_meta.get("doi"),
            },
            "summary_md": summary_text,
            "citations": citations or self._fallback_citations(selected),
            "evidence_summary": self._build_evidence_summary(selected),
            "evidence_chunks": [_chunk_to_serializable(chunk) for chunk in selected],
            "used_llm": used_llm,
        }

    def ask_paper(
        self,
        locator: Dict[str, Any],
        *,
        user_id: str,
        question: str,
        scope: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("question is empty")
        paper = self.locate_paper(locator, user_id=user_id, scope=scope)
        doc, raw_doc = self._load_doc(paper)
        doc_meta = self._doc_metadata(paper, raw_doc)
        chunks = self._doc_chunks(paper, doc, raw_doc, doc_meta)
        ranked = sorted(chunks, key=lambda chunk: _score_chunk(question, chunk), reverse=True)
        selected = [chunk for chunk in ranked[:8] if _score_chunk(question, chunk) > 0]
        if not selected:
            selected = ranked[:6]
        answer_text, citations, used_llm = self._llm_answer(
            task="paper question answering",
            prompt=f"Answer the question about this paper.\nQuestion: {question}",
            chunks=selected,
            llm_provider=llm_provider,
            model_override=model_override,
        )
        if not answer_text:
            if not selected:
                answer_text = "Insufficient evidence in the current paper context."
            else:
                lines = [f"Question: {question}", "", "Evidence-based answer:"]
                for chunk in selected[:4]:
                    label = chunk.section_title or chunk.evidence_type or "paper"
                    lines.append(f"- {label}: {chunk.text[:280]}")
                answer_text = "\n".join(lines)
        return {
            "paper_card": {
                "paper_id": paper["paper_id"],
                "paper_uid": paper.get("paper_uid"),
                "collection": paper["collection"],
                "title": doc_meta.get("title") or paper["paper_id"],
                "authors": doc_meta.get("authors") or [],
                "year": doc_meta.get("year"),
                "doi": doc_meta.get("doi"),
            },
            "answer_md": answer_text,
            "citations": citations or self._fallback_citations(selected),
            "evidence_summary": self._build_evidence_summary(selected),
            "evidence_chunks": [_chunk_to_serializable(chunk) for chunk in selected],
            "used_llm": used_llm,
        }

    def _extract_aspect_text(self, raw_doc: Dict[str, Any], aspect: str) -> str:
        wanted = _ASPECT_KEYWORDS.get(aspect, (aspect,))
        content_flow = raw_doc.get("content_flow")
        if not isinstance(content_flow, list):
            content_flow = []
        snippets: List[str] = []
        for block in content_flow:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            heading_path = " ".join(str(item) for item in (block.get("heading_path") or [])).lower()
            if any(keyword in heading_path for keyword in wanted):
                snippets.append(text[:280])
            if len(snippets) >= 2:
                break
        if snippets:
            return "\n".join(snippets)
        return str(raw_doc.get("global_summary") or "").strip()[:280]

    def compare_papers(
        self,
        paper_uids: List[str],
        *,
        user_id: str,
        aspects: Optional[List[str]] = None,
        scope: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        if len(paper_uids) < 2:
            raise ValueError("compare requires at least 2 papers")
        aspects = [str(aspect).strip() for aspect in (aspects or list(_ASPECT_KEYWORDS.keys())) if str(aspect).strip()]
        resolved: List[Dict[str, Any]] = []
        evidence_chunks: List[EvidenceChunk] = []
        for paper_uid in paper_uids[:5]:
            paper = self.locate_paper({"paper_uid": paper_uid}, user_id=user_id, scope=scope)
            doc, raw_doc = self._load_doc(paper)
            doc_meta = self._doc_metadata(paper, raw_doc)
            paper["doc"] = doc
            paper["raw_doc"] = raw_doc
            paper["doc_meta"] = doc_meta
            resolved.append(paper)
            evidence_chunks.extend(self._doc_chunks(paper, doc, raw_doc, doc_meta)[:2])

        matrix: Dict[str, Dict[str, str]] = {}
        for aspect in aspects:
            matrix[aspect] = {}
            for paper in resolved:
                matrix[aspect][paper["paper_uid"]] = self._extract_aspect_text(paper["raw_doc"], aspect)

        prompt_lines = ["Compare the following papers in Markdown. Focus on differences and similarities."]
        for paper in resolved:
            prompt_lines.append(f"Paper: {paper['doc_meta'].get('title') or paper['paper_uid']}")
            for aspect in aspects:
                prompt_lines.append(f"{aspect}: {matrix[aspect][paper['paper_uid']]}")
            prompt_lines.append("")
        narrative, citations, used_llm = self._llm_answer(
            task="multi-paper comparison",
            prompt="\n".join(prompt_lines),
            chunks=evidence_chunks[:8],
            llm_provider=llm_provider,
            model_override=model_override,
        )
        if not narrative:
            lines = ["# Paper Comparison"]
            for aspect in aspects:
                lines.append(f"## {aspect}")
                for paper in resolved:
                    title = paper["doc_meta"].get("title") or paper["paper_uid"]
                    lines.append(f"- {title}: {matrix[aspect][paper['paper_uid']] or '(no strong evidence extracted)'}")
            narrative = "\n".join(lines)

        papers_payload = []
        for paper in resolved:
            papers_payload.append(
                {
                    "paper_id": paper["paper_id"],
                    "paper_uid": paper["paper_uid"],
                    "title": paper["doc_meta"].get("title") or paper["paper_id"],
                    "year": paper["doc_meta"].get("year"),
                    "doi": paper["doc_meta"].get("doi"),
                }
            )
        return {
            "papers": papers_payload,
            "comparison_matrix": matrix,
            "narrative": narrative,
            "citations": citations or self._fallback_citations(evidence_chunks, max_docs=len(resolved)),
            "evidence_summary": self._build_evidence_summary(evidence_chunks),
            "used_llm": used_llm,
        }

    def _build_parser_config(
        self,
        *,
        output_dir: Path,
        llm_text_provider: Optional[str] = None,
        llm_vision_provider: Optional[str] = None,
        llm_text_model: Optional[str] = None,
        llm_vision_model: Optional[str] = None,
    ) -> ParserConfig:
        cfg = ParserConfig.from_json(_CONFIG_PATH)
        cfg.output_dir = str(output_dir)
        cfg.enrich_tables = False
        cfg.enrich_figures = True
        if llm_text_provider:
            cfg.llm_text_provider = llm_text_provider
        if llm_vision_provider:
            cfg.llm_vision_provider = llm_vision_provider
        if llm_text_model:
            cfg.llm_text_model = llm_text_model
        if llm_vision_model:
            cfg.llm_vision_model = llm_vision_model
        return cfg

    def _doc_has_media(self, doc: Any) -> bool:
        for block in getattr(doc, "content_flow", []) or []:
            figure_data = getattr(block, "figure_data", None)
            if figure_data is None:
                continue
            if getattr(figure_data, "image_path", None):
                return True
            enrichment = getattr(block, "enrichment", None)
            if getattr(getattr(enrichment, "interpretation", None), "description", None):
                return True
        return False

    def _update_media_stats(self, paper: Dict[str, Any], doc: Any) -> None:
        figure_count = 0
        figure_success = 0
        for block in getattr(doc, "content_flow", []) or []:
            if getattr(block, "figure_data", None) is None:
                continue
            figure_count += 1
            enrichment = getattr(block, "enrichment", None)
            interpretation = getattr(enrichment, "interpretation", None) if enrichment is not None else None
            if getattr(getattr(block, "figure_data", None), "image_path", None) or interpretation is not None:
                figure_success += 1
        with Session(get_engine()) as session:
            row = session.exec(
                select(Paper).where(
                    Paper.user_id == paper["user_id"],
                    Paper.collection == paper["collection"],
                    Paper.paper_id == paper["paper_id"],
                )
            ).first()
            if row is None:
                return
            row.enrich_figures_enabled = 1
            row.figure_count = figure_count
            row.figure_success = figure_success
            session.add(row)
            session.commit()

    def analyze_paper_media(
        self,
        paper_uids: List[str],
        *,
        user_id: str,
        scope: Optional[Dict[str, Any]] = None,
        force_reparse: bool = False,
        upsert_vectors_enabled: bool = True,
        llm_text_provider: Optional[str] = None,
        llm_vision_provider: Optional[str] = None,
        llm_text_model: Optional[str] = None,
        llm_vision_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        from src.llm.llm_manager import get_manager

        manager = get_manager(str(_CONFIG_PATH))
        results: List[Dict[str, Any]] = []
        for paper_uid in paper_uids:
            paper = self.locate_paper({"paper_uid": paper_uid}, user_id=user_id, scope=scope)
            parsed_json = Path(str(paper.get("parsed_json") or ""))
            parsed_dir = parsed_json.parent if parsed_json.exists() else None
            if parsed_dir is None:
                parsed_dir = PathManager.get_user_parsed_path(user_id) / paper["paper_id"]
            cfg = self._build_parser_config(
                output_dir=parsed_dir,
                llm_text_provider=llm_text_provider,
                llm_vision_provider=llm_vision_provider,
                llm_text_model=llm_text_model,
                llm_vision_model=llm_vision_model,
            )
            processor = PDFProcessor(config=cfg, llm_manager=manager)
            doc = None
            raw_doc: Dict[str, Any] = {}
            if not force_reparse and parsed_json.exists():
                try:
                    doc, raw_doc = self._load_doc(paper)
                except Exception:
                    doc = None
            if doc is None or not self._doc_has_media(doc):
                file_path = Path(str(paper.get("file_path") or ""))
                if not file_path.exists():
                    raise ValueError(f"pdf file not found for paper_uid={paper_uid}")
                doc = processor.process(file_path, output_dir=parsed_dir, skip_enrichment=False)
                with open(parsed_dir / "enriched.json", "r", encoding="utf-8") as f:
                    raw_doc = json.load(f)
                paper["parsed_json"] = str(parsed_dir / "enriched.json")
            doc_meta = self._doc_metadata(paper, raw_doc)
            vector_result = {"deleted_count": 0, "upserted_count": 0}
            if upsert_vectors_enabled:
                vector_result = upsert_media_vectors(
                    paper["collection"],
                    doc=doc,
                    paper_id=paper["paper_id"],
                    doc_metadata=doc_meta,
                )
            self._update_media_stats(paper, doc)
            rows = list_resource_annotations(user_id=user_id, paper_uid=paper_uid, status="active", limit=200)
            for row in rows:
                sync_annotation_vector(row, collection=paper["collection"])
            figure_count = sum(1 for block in getattr(doc, "content_flow", []) or [] if getattr(block, "figure_data", None) is not None)
            analysis_count = sum(
                1 for block in getattr(doc, "content_flow", []) or []
                if getattr(block, "figure_data", None) is not None
                and (
                    getattr(getattr(getattr(block, "enrichment", None), "interpretation", None), "description", None)
                    or getattr(getattr(block, "figure_data", None), "ocr_text", None)
                )
            )
            results.append(
                {
                    "paper_uid": paper_uid,
                    "paper_id": paper["paper_id"],
                    "collection": paper["collection"],
                    "parsed_json": paper["parsed_json"],
                    "figure_count": figure_count,
                    "analysis_count": analysis_count,
                    "vector_upserted_count": vector_result.get("upserted_count", 0),
                }
            )
        return {
            "mode": "media-analysis",
            "items": results,
            "summary_md": "\n".join(
                f"- {item['paper_uid']}: figures={item['figure_count']}, analyses={item['analysis_count']}, vectors={item['vector_upserted_count']}"
                for item in results
            ),
            "citations": [],
            "provenance": [{"source": "parser"}, {"source": "milvus"}],
        }

    def _scope_local_paper_uids(self, *, user_id: str, scope: Dict[str, Any]) -> set[str]:
        normalized_scope = self._normalize_scope(user_id=user_id, scope=scope)
        papers = self.graph_service.fact_builder.collect_scope_papers(self.graph_service._normalize_scope(normalized_scope))
        return {str(paper.get("paper_uid") or "").strip() for paper in papers if str(paper.get("paper_uid") or "").strip()}

    def _discovery_from_subgraph(
        self,
        *,
        graph_type: str,
        response_mode: str,
        seeds: Dict[str, Any],
        user_id: str,
        scope: Dict[str, Any],
        question: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        subgraph = self.graph_service.query_subgraph(
            graph_type=graph_type,
            scope=self._normalize_scope(user_id=user_id, scope=scope),
            seeds=seeds,
            depth=2,
            limit=max(20, limit * 4),
        )
        summary = self.graph_service.summarize_subgraph(
            graph_type=graph_type,
            scope=self._normalize_scope(user_id=user_id, scope=scope),
            seeds=seeds,
            depth=2,
            question=question,
            max_items=limit,
        )
        nodes = list(subgraph.get("nodes") or [])
        local_uids = self._scope_local_paper_uids(user_id=user_id, scope=scope)
        items: List[Dict[str, Any]] = []
        if graph_type == "citation":
            inbound = defaultdict(int)
            for edge in subgraph.get("edges") or []:
                if edge.get("relation") == "cites":
                    inbound[str(edge.get("target") or "")] += 1
            for node in nodes:
                node_id = str(node.get("id") or "")
                if not node_id.startswith("paper:"):
                    continue
                paper_uid = node_id.split("paper:", 1)[-1]
                if not paper_uid or paper_uid in seeds.get("paper_uids", []) or paper_uid in local_uids:
                    continue
                items.append(
                    {
                        "paper_uid": paper_uid,
                        "title": node.get("label") or paper_uid,
                        "score": inbound.get(node_id, 0),
                        "reason": "referenced_by_seed_subgraph",
                    }
                )
            items.sort(key=lambda item: item.get("score", 0), reverse=True)
        else:
            top_nodes = sorted(nodes, key=lambda item: item.get("pagerank", 0.0), reverse=True)
            for node in top_nodes:
                node_id = str(node.get("id") or "")
                if graph_type == "author" and not node_id.startswith("author:"):
                    continue
                if graph_type == "institution" and not node_id.startswith("institution:"):
                    continue
                items.append(
                    {
                        "node_id": node_id,
                        "label": node.get("label") or node_id,
                        "type": node.get("type"),
                        "score": node.get("pagerank", 0.0),
                    }
                )
        return {
            "mode": response_mode,
            "items": items[:limit],
            "summary_md": summary.get("summary") or "",
            "citations": [],
            "provenance": summary.get("provenance") or [],
        }

    def discover(
        self,
        mode: str,
        *,
        user_id: str,
        seeds: Dict[str, Any],
        scope: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Canonical normalisation: accept kebab-case (from frontend URL) or underscore (from tool schema)
        mode = (mode or "").strip().lower().replace("-", "_")
        scope_payload = self._normalize_scope(user_id=user_id, scope=scope, default_collection=None)
        options = dict(options or {})
        limit = max(1, min(_safe_int(options.get("limit"), 10), 20))
        if mode == "experts":
            return self._discovery_from_subgraph(
                graph_type="author",
                response_mode=mode,
                seeds=seeds,
                user_id=user_id,
                scope=scope_payload,
                question=options.get("question"),
                limit=limit,
            )
        if mode == "institutions":
            return self._discovery_from_subgraph(
                graph_type="institution",
                response_mode=mode,
                seeds=seeds,
                user_id=user_id,
                scope=scope_payload,
                question=options.get("question"),
                limit=limit,
            )
        if mode == "missing_core":
            return self._discovery_from_subgraph(
                graph_type="citation",
                response_mode=mode,
                seeds=seeds,
                user_id=user_id,
                scope=scope_payload,
                question=options.get("question"),
                limit=limit,
            )
        if mode == "forward_tracking":
            items: List[Dict[str, Any]] = []
            for paper_uid in seeds.get("paper_uids") or []:
                try:
                    paper = self.locate_paper({"paper_uid": paper_uid}, user_id=user_id, scope=scope_payload)
                except Exception:
                    continue
                meta = paper_meta_store.get(paper["paper_id"]) or {}
                work = self.graph_service.enricher._resolve_openalex_work(meta)
                if not isinstance(work, dict):
                    continue
                cited_api_url = str(work.get("cited_by_api_url") or "").strip()
                if not cited_api_url:
                    continue
                if "per-page=" not in cited_api_url:
                    sep = "&" if "?" in cited_api_url else "?"
                    cited_api_url = f"{cited_api_url}{sep}per-page={limit}"
                data = self.graph_service.enricher._openalex_fetch_json(cited_api_url)
                results = data.get("results") if isinstance(data, dict) else []
                for item in results or []:
                    if not isinstance(item, dict):
                        continue
                    items.append(
                        {
                            "seed_paper_uid": paper_uid,
                            "title": item.get("title") or "",
                            "year": item.get("publication_year"),
                            "doi": item.get("doi"),
                            "reason": "openalex_cited_by",
                        }
                    )
            items.sort(key=lambda item: item.get("year") or 0, reverse=True)
            return {
                "mode": mode,
                "items": items[:limit],
                "summary_md": "\n".join(
                    f"- {item.get('title') or '(untitled)'} ({item.get('year') or 'n/a'}) <- {item.get('seed_paper_uid')}"
                    for item in items[:limit]
                ),
                "citations": [],
                "provenance": [{"source": "openalex", "kind": "cited_by"}],
            }
        raise ValueError(f"unsupported discovery mode: {mode}")


_reference_assistant_service: Optional[ReferenceAssistantService] = None


def get_reference_assistant_service() -> ReferenceAssistantService:
    global _reference_assistant_service
    if _reference_assistant_service is None:
        _reference_assistant_service = ReferenceAssistantService()
    return _reference_assistant_service
