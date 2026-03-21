from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import networkx as nx
from sqlmodel import Session, delete as sql_delete, select

from config.settings import settings
from src.db.engine import get_engine
from src.db.models import (
    CollectionLibraryBinding,
    GraphFact,
    GraphSnapshot,
    Paper,
    PaperMetadata,
    ScholarLibrary,
    ScholarLibraryPaper,
)
from src.log import get_logger
from src.retrieval.dedup import compute_paper_uid, normalize_doi, normalize_title
from src.retrieval.semantic_scholar import semantic_scholar_searcher

logger = get_logger(__name__)

VALID_GRAPH_TYPES = frozenset({"entity", "citation", "author", "institution"})
VALID_SCOPE_TYPES = frozenset({"global", "collection", "library"})

_MAX_WORK_CACHE = 2_000  # per-service-instance OpenAlex response cache ceiling

_global_graph_service: Optional["GlobalGraphService"] = None


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return default
    return default


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _openalex_cfg() -> Any:
    cfg = getattr(settings, "openalex", None)
    if cfg is not None:
        return cfg

    class _Fallback:
        enabled = False
        api_key = ""
        base_url = "https://api.openalex.org"
        timeout_seconds = 10

    return _Fallback()


def _openalex_base_works_url() -> str:
    cfg = _openalex_cfg()
    return f"{(getattr(cfg, 'base_url', 'https://api.openalex.org') or 'https://api.openalex.org').rstrip('/')}/works"


def _node_label_from_id(node_id: str) -> str:
    if ":" not in node_id:
        return node_id
    return node_id.split(":", 1)[1]


def _paper_node_id(paper_uid: str) -> str:
    return f"paper:{paper_uid}"


def _author_node_id(name: str, external_id: Optional[str] = None) -> str:
    raw = (external_id or "").strip()
    if raw:
        raw = raw.rstrip("/").split("/")[-1]
        return f"author:{raw}"
    return f"author:{_normalize_token(name)}"


def _institution_node_id(name: str, external_id: Optional[str] = None) -> str:
    raw = (external_id or "").strip()
    if raw:
        raw = raw.rstrip("/").split("/")[-1]
        return f"institution:{raw}"
    return f"institution:{_normalize_token(name)}"


def _paper_uid_from_ref(item: Any) -> Optional[str]:
    if isinstance(item, str):
        value = item.strip()
        return value or None
    if not isinstance(item, dict):
        return None
    if item.get("paper_uid"):
        return str(item.get("paper_uid")).strip() or None
    doi = normalize_doi(item.get("doi"))
    title = (item.get("title") or "").strip()
    authors = item.get("authors") if isinstance(item.get("authors"), list) else None
    year = _as_int(item.get("year"), 0) or None
    url = (item.get("url") or "").strip() or None
    pmid = (item.get("pmid") or "").strip() or None
    if not (doi or title or pmid):
        return None
    return compute_paper_uid(
        doi=doi or None,
        title=title or None,
        authors=authors,
        year=year,
        url=url,
        pmid=pmid,
    )


def _chunks(lst: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive fixed-size chunks from *lst*."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _extract_authorships(extra: Dict[str, Any], authors: List[str]) -> List[Dict[str, Any]]:
    raw_authorships = extra.get("authorships")
    if isinstance(raw_authorships, list) and raw_authorships:
        out: List[Dict[str, Any]] = []
        for item in raw_authorships:
            if not isinstance(item, dict):
                continue
            author_obj = item.get("author") or {}
            name = (
                (author_obj.get("display_name") or "").strip()
                or (item.get("author_name") or "").strip()
            )
            if not name:
                continue
            insts = []
            for inst in (item.get("institutions") or []):
                if not isinstance(inst, dict):
                    continue
                inst_name = (inst.get("display_name") or inst.get("name") or "").strip()
                if not inst_name:
                    continue
                insts.append(
                    {
                        "name": inst_name,
                        "id": (inst.get("id") or inst.get("ror") or "").strip() or None,
                    }
                )
            out.append(
                {
                    "author_name": name,
                    "author_id": (author_obj.get("id") or item.get("authorId") or "").strip() or None,
                    "institutions": insts,
                }
            )
        if out:
            return out

    institutions = extra.get("institutions")
    norm_insts: List[Dict[str, Any]] = []
    if isinstance(institutions, list):
        for inst in institutions:
            if isinstance(inst, dict):
                name = (inst.get("display_name") or inst.get("name") or "").strip()
                if name:
                    norm_insts.append({"name": name, "id": (inst.get("id") or inst.get("ror") or "").strip() or None})
            elif isinstance(inst, str) and inst.strip():
                norm_insts.append({"name": inst.strip(), "id": None})

    out = []
    for name in authors:
        if not name:
            continue
        out.append({"author_name": name, "author_id": None, "institutions": list(norm_insts)})
    return out


@dataclass(frozen=True)
class GraphScope:
    user_id: str
    scope_type: str
    scope_key: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "scope_type": self.scope_type,
            "scope_key": self.scope_key,
        }


class EntityGraphAdapter:
    def _get_hippo(self):
        try:
            from src.graph.hippo_rag import get_hippo_rag

            graph_path = settings.path.data / "hippo_graph.json"
            if not graph_path.exists():
                return None
            return get_hippo_rag(graph_path)
        except Exception as e:
            logger.warning("Entity graph load failed: %s", e)
            return None

    def list_snapshots(self, scope: GraphScope) -> List[Dict[str, Any]]:
        graph_path = settings.path.data / "hippo_graph.json"
        if not graph_path.exists():
            return []
        version = int(graph_path.stat().st_mtime)
        return [
            {
                "user_id": scope.user_id,
                "scope_type": scope.scope_type,
                "scope_key": scope.scope_key,
                "graph_type": "entity",
                "snapshot_version": version,
                "status": "ready",
                "storage_path": str(graph_path),
                "node_count": 0,
                "edge_count": 0,
                "built_from_revision": str(version),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        ]

    def stats(self) -> Dict[str, Any]:
        hippo = self._get_hippo()
        if hippo is None:
            return {
                "available": False,
                "total_nodes": 0,
                "total_edges": 0,
                "entity_count": 0,
                "chunk_count": 0,
                "entity_types": {},
                "snapshot_version": None,
            }
        stats = hippo.stats()
        graph_path = settings.path.data / "hippo_graph.json"
        return {
            "available": True,
            **stats,
            "snapshot_version": int(graph_path.stat().st_mtime) if graph_path.exists() else None,
        }

    def build_subgraph(self, seeds: List[str], depth: int, limit: int) -> Dict[str, Any]:
        hippo = self._get_hippo()
        if hippo is None:
            return {
                "nodes": [],
                "edges": [],
                "metrics": {"node_count": 0, "edge_count": 0, "top_nodes": [], "bridge_nodes": []},
                "snapshot_version": None,
                "provenance": [{"source": "hipporag", "status": "missing"}],
            }
        if not seeds:
            seeds = sorted(
                hippo.entities.keys(),
                key=lambda name: len(hippo.entities[name].mentions),
                reverse=True,
            )[:1]
        seed_set = {s for s in seeds if s in hippo.G}
        if not seed_set:
            return {
                "nodes": [],
                "edges": [],
                "metrics": {"node_count": 0, "edge_count": 0, "top_nodes": [], "bridge_nodes": []},
                "snapshot_version": None,
                "provenance": [{"source": "hipporag", "status": "seed_missing"}],
            }

        visited: set[str] = set()
        queue = deque((seed, 0) for seed in seed_set)
        while queue:
            node, dist = queue.popleft()
            if node in visited or dist > max(1, depth):
                continue
            visited.add(node)
            if len(visited) >= max(1, limit):
                continue
            for _, nbr, _ in hippo.G.edges(node, data=True):
                if nbr not in visited:
                    queue.append((nbr, dist + 1))
            for nbr, _, _ in hippo.G.in_edges(node, data=True):
                if nbr not in visited:
                    queue.append((nbr, dist + 1))

        nodes = []
        for node_id in visited:
            attrs = hippo.G.nodes.get(node_id, {})
            entity_info = hippo.entities.get(node_id)
            node_type = attrs.get("type", "ENTITY")
            if entity_info is not None:
                node_type = entity_info.type
            nodes.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "label": node_id,
                    "is_seed": node_id in seed_set,
                    "paper_id": attrs.get("paper_id", ""),
                }
            )

        edges = []
        for src, dst, data in hippo.G.edges(data=True):
            if src in visited and dst in visited:
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "relation": data.get("relation", ""),
                        "weight": data.get("weight", 1),
                        "provenance": {"source": "hipporag"},
                    }
                )

        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], relation=edge["relation"], weight=edge["weight"])
        pagerank = nx.pagerank(G) if G.number_of_nodes() else {}
        top_nodes = sorted(
            (
                {"id": node["id"], "type": node["type"], "score": round(float(pagerank.get(node["id"], 0.0)), 6)}
                for node in nodes
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:10]
        graph_path = settings.path.data / "hippo_graph.json"
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "top_nodes": top_nodes,
                "bridge_nodes": [],
            },
            "snapshot_version": int(graph_path.stat().st_mtime) if graph_path.exists() else None,
            "provenance": [{"source": "hipporag", "status": "ok"}],
        }


class AcademicFactBuilder:
    def collect_scope_papers(self, scope: GraphScope) -> List[Dict[str, Any]]:
        with Session(get_engine()) as session:
            out: Dict[str, Dict[str, Any]] = {}

            def _merge_paper(record: Dict[str, Any]) -> None:
                paper_uid = (record.get("paper_uid") or "").strip()
                if not paper_uid:
                    return
                existing = out.get(paper_uid) or {}
                merged = dict(existing)
                merged.update({k: v for k, v in record.items() if v not in (None, "", [], {})})
                if "authors" not in merged:
                    merged["authors"] = []
                merged["authors"] = list(merged.get("authors") or [])
                merged["extra"] = dict(merged.get("extra") or {})
                out[paper_uid] = merged

            def _meta_to_record(
                *,
                user_id: str,
                collection_name: Optional[str],
                library_id: Optional[int],
                library_paper_id: Optional[int],
                paper_id: Optional[str],
                meta_row: Optional[PaperMetadata],
                fallback_title: str = "",
                fallback_authors: Optional[List[str]] = None,
                fallback_year: Optional[int] = None,
                fallback_doi: str = "",
                fallback_url: str = "",
                fallback_pdf_url: str = "",
            ) -> Dict[str, Any]:
                extra = _safe_json_loads(getattr(meta_row, "extra", None), {}) if meta_row is not None else {}
                if fallback_url and not extra.get("url"):
                    extra["url"] = fallback_url
                if fallback_pdf_url and not extra.get("pdf_url"):
                    extra["pdf_url"] = fallback_pdf_url
                authors = _safe_json_loads(getattr(meta_row, "authors", None), []) if meta_row is not None else []
                if not authors:
                    authors = list(fallback_authors or [])
                title = (getattr(meta_row, "title", "") or fallback_title or "").strip()
                doi = normalize_doi((getattr(meta_row, "doi", "") if meta_row is not None else "") or fallback_doi or "")
                year = getattr(meta_row, "year", None) if meta_row is not None else fallback_year
                paper_uid = (getattr(meta_row, "paper_uid", "") if meta_row is not None else "") or compute_paper_uid(
                    doi=doi or None,
                    title=title or None,
                    authors=authors,
                    year=year,
                    url=(extra.get("url") or extra.get("pdf_url") or None),
                    pmid=(extra.get("pmid") or None),
                )
                return {
                    "user_id": user_id,
                    "collection_name": collection_name or "",
                    "library_id": library_id,
                    "library_paper_id": library_paper_id,
                    "paper_id": paper_id or "",
                    "paper_uid": paper_uid,
                    "doi": doi,
                    "title": title,
                    "authors": list(authors or []),
                    "year": year,
                    "extra": dict(extra or {}),
                }

            if scope.scope_type == "collection":
                paper_rows = session.exec(
                    select(Paper).where(
                        Paper.user_id == scope.user_id,
                        Paper.collection == scope.scope_key,
                    )
                ).all()
                for row in paper_rows:
                    meta_row = session.get(PaperMetadata, row.paper_id)
                    _merge_paper(
                        _meta_to_record(
                            user_id=scope.user_id,
                            collection_name=row.collection,
                            library_id=getattr(row, "library_id", None),
                            library_paper_id=getattr(row, "library_paper_id", None),
                            paper_id=row.paper_id,
                            meta_row=meta_row,
                        )
                    )

                binding = session.exec(
                    select(CollectionLibraryBinding).where(
                        CollectionLibraryBinding.user_id == scope.user_id,
                        CollectionLibraryBinding.collection_name == scope.scope_key,
                    )
                ).first()
                if binding is not None:
                    lib_rows = session.exec(
                        select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == binding.library_id)
                    ).all()
                    for row in lib_rows:
                        meta_row = (
                            session.get(PaperMetadata, row.collection_paper_id)
                            if (row.collection_paper_id or "").strip()
                            else None
                        )
                        _merge_paper(
                            _meta_to_record(
                                user_id=scope.user_id,
                                collection_name=row.collection_name or scope.scope_key,
                                library_id=row.library_id,
                                library_paper_id=row.id,
                                paper_id=row.collection_paper_id or None,
                                meta_row=meta_row,
                                fallback_title=row.title,
                                fallback_authors=row.get_authors(),
                                fallback_year=row.year,
                                fallback_doi=row.doi,
                                fallback_url=row.url,
                                fallback_pdf_url=row.pdf_url,
                            )
                        )
            elif scope.scope_type == "library":
                library_id = _as_int(scope.scope_key, 0)
                lib = session.get(ScholarLibrary, library_id)
                if lib is None or lib.user_id != scope.user_id:
                    return []
                lib_rows = session.exec(
                    select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == library_id)
                ).all()
                for row in lib_rows:
                    meta_row = (
                        session.get(PaperMetadata, row.collection_paper_id)
                        if (row.collection_paper_id or "").strip()
                        else None
                    )
                    _merge_paper(
                        _meta_to_record(
                            user_id=scope.user_id,
                            collection_name=row.collection_name or "",
                            library_id=row.library_id,
                            library_paper_id=row.id,
                            paper_id=row.collection_paper_id or None,
                            meta_row=meta_row,
                            fallback_title=row.title,
                            fallback_authors=row.get_authors(),
                            fallback_year=row.year,
                            fallback_doi=row.doi,
                            fallback_url=row.url,
                            fallback_pdf_url=row.pdf_url,
                        )
                    )
            else:
                paper_rows = session.exec(
                    select(Paper).where(Paper.user_id == scope.user_id)
                ).all()
                for row in paper_rows:
                    meta_row = session.get(PaperMetadata, row.paper_id)
                    _merge_paper(
                        _meta_to_record(
                            user_id=scope.user_id,
                            collection_name=row.collection,
                            library_id=getattr(row, "library_id", None),
                            library_paper_id=getattr(row, "library_paper_id", None),
                            paper_id=row.paper_id,
                            meta_row=meta_row,
                        )
                    )

                lib_ids = [
                    int(lib_id)
                    for lib_id in session.exec(
                        select(ScholarLibrary.id).where(ScholarLibrary.user_id == scope.user_id)
                    ).all()
                ]
                if lib_ids:
                    lib_rows = session.exec(
                        select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id.in_(lib_ids))
                    ).all()
                    for row in lib_rows:
                        meta_row = (
                            session.get(PaperMetadata, row.collection_paper_id)
                            if (row.collection_paper_id or "").strip()
                            else None
                        )
                        _merge_paper(
                            _meta_to_record(
                                user_id=scope.user_id,
                                collection_name=row.collection_name or "",
                                library_id=row.library_id,
                                library_paper_id=row.id,
                                paper_id=row.collection_paper_id or None,
                                meta_row=meta_row,
                                fallback_title=row.title,
                                fallback_authors=row.get_authors(),
                                fallback_year=row.year,
                                fallback_doi=row.doi,
                                fallback_url=row.url,
                                fallback_pdf_url=row.pdf_url,
                            )
                        )
            return list(out.values())

    def compute_revision(self, graph_type: str, scope: GraphScope, papers: List[Dict[str, Any]]) -> str:
        serializable = []
        for paper in sorted(papers, key=lambda item: item.get("paper_uid") or ""):
            extra = dict(paper.get("extra") or {})
            serializable.append(
                {
                    "paper_uid": paper.get("paper_uid"),
                    "doi": paper.get("doi"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors") or [],
                    "year": paper.get("year"),
                    "collection_name": paper.get("collection_name"),
                    "library_id": paper.get("library_id"),
                    "references": extra.get("references") or extra.get("referenced_works") or extra.get("reference_uids") or [],
                    "authorships": extra.get("authorships") or [],
                    "institutions": extra.get("institutions") or [],
                }
            )
        payload = {
            "graph_type": graph_type,
            "scope": scope.as_dict(),
            "papers": serializable,
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def build_local_facts(self, graph_type: str, scope: GraphScope, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        local_by_uid = {p.get("paper_uid"): p for p in papers if p.get("paper_uid")}
        local_by_doi = {
            normalize_doi(p.get("doi")): p.get("paper_uid")
            for p in papers
            if normalize_doi(p.get("doi"))
        }
        local_by_title = {
            normalize_title(p.get("title") or ""): p.get("paper_uid")
            for p in papers
            if normalize_title(p.get("title") or "")
        }

        def add_fact(
            src_node_id: str,
            src_node_type: str,
            src_label: str,
            relation_type: str,
            dst_node_id: str,
            dst_node_type: str,
            dst_label: str,
            *,
            weight: float = 1.0,
            provenance: Optional[Dict[str, Any]] = None,
        ) -> None:
            facts.append(
                {
                    "user_id": scope.user_id,
                    "scope_type": scope.scope_type,
                    "scope_key": scope.scope_key,
                    "graph_type": graph_type,
                    "src_node_id": src_node_id,
                    "src_node_type": src_node_type,
                    "src_label": src_label,
                    "relation_type": relation_type,
                    "dst_node_id": dst_node_id,
                    "dst_node_type": dst_node_type,
                    "dst_label": dst_label,
                    "weight": float(weight),
                    "provenance_json": json.dumps(
                        provenance or {"source": "local", "kind": "scope_materialized"},
                        ensure_ascii=False,
                    ),
                }
            )

        for paper in papers:
            paper_uid = (paper.get("paper_uid") or "").strip()
            if not paper_uid:
                continue
            paper_id = _paper_node_id(paper_uid)
            paper_label = (paper.get("title") or paper_uid).strip()
            authorships = _extract_authorships(dict(paper.get("extra") or {}), list(paper.get("authors") or []))

            if graph_type == "author":
                for authorship in authorships:
                    author_name = (authorship.get("author_name") or "").strip()
                    if not author_name:
                        continue
                    author_id = _author_node_id(author_name, authorship.get("author_id"))
                    add_fact(
                        author_id,
                        "author",
                        author_name,
                        "authored",
                        paper_id,
                        "paper",
                        paper_label,
                        provenance={"source": "local", "kind": "metadata_authorship"},
                    )

            elif graph_type == "institution":
                for authorship in authorships:
                    author_name = (authorship.get("author_name") or "").strip()
                    if not author_name:
                        continue
                    author_id = _author_node_id(author_name, authorship.get("author_id"))
                    # Store authored edge so the snapshot builder can derive
                    # co-institution (collaborates_with) purely from facts.
                    add_fact(
                        author_id,
                        "author",
                        author_name,
                        "authored",
                        paper_id,
                        "paper",
                        paper_label,
                        provenance={"source": "local", "kind": "metadata_authorship"},
                    )
                    for inst in (authorship.get("institutions") or []):
                        inst_name = (inst.get("name") or "").strip()
                        if not inst_name:
                            continue
                        inst_id = _institution_node_id(inst_name, inst.get("id"))
                        add_fact(
                            author_id,
                            "author",
                            author_name,
                            "affiliated_with",
                            inst_id,
                            "institution",
                            inst_name,
                            provenance={"source": "local", "kind": "metadata_affiliation"},
                        )

            elif graph_type == "citation":
                extra = dict(paper.get("extra") or {})
                refs = (
                    extra.get("references")
                    or extra.get("reference_uids")
                    or extra.get("referenced_papers")
                    or extra.get("referenced_works")
                    or []
                )
                for item in refs:
                    target_uid = None
                    if isinstance(item, str) and item.startswith("https://openalex.org/"):
                        continue
                    if isinstance(item, dict):
                        target_uid = _paper_uid_from_ref(item)
                        if target_uid is None:
                            ref_doi = normalize_doi(item.get("doi"))
                            ref_title = normalize_title(item.get("title") or "")
                            target_uid = local_by_doi.get(ref_doi) or local_by_title.get(ref_title)
                    else:
                        target_uid = _paper_uid_from_ref(item)
                    if not target_uid and isinstance(item, str):
                        stripped = item.strip()
                        target_uid = local_by_doi.get(normalize_doi(stripped)) or local_by_title.get(normalize_title(stripped))
                    if not target_uid or target_uid == paper_uid or target_uid not in local_by_uid:
                        continue
                    target = local_by_uid[target_uid]
                    add_fact(
                        paper_id,
                        "paper",
                        paper_label,
                        "cites",
                        _paper_node_id(target_uid),
                        "paper",
                        (target.get("title") or target_uid).strip(),
                        provenance={"source": "local", "kind": "metadata_reference"},
                    )
        return facts


class AcademicGraphEnricher:
    def __init__(self) -> None:
        self._work_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _openalex_headers(self) -> Dict[str, str]:
        cfg = _openalex_cfg()
        headers = {"Accept": "application/json", "User-Agent": "DeepSea-RAG/1.0"}
        api_key = (getattr(cfg, "api_key", "") or "").strip()
        if api_key and "@" not in api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _openalex_timeout(self) -> int:
        cfg = _openalex_cfg()
        return int(getattr(cfg, "timeout_seconds", 10) or 10)

    def _openalex_add_auth(self, url: str) -> str:
        cfg = _openalex_cfg()
        api_key = (getattr(cfg, "api_key", "") or "").strip()
        if "@" in api_key:
            joiner = "&" if "?" in url else "?"
            return f"{url}{joiner}mailto={quote(api_key)}"
        return url

    def _openalex_fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        cfg = _openalex_cfg()
        if not getattr(cfg, "enabled", False):
            return None
        url = self._openalex_add_auth(url)
        try:
            req = Request(url, headers=self._openalex_headers(), method="GET")
            with urlopen(req, timeout=self._openalex_timeout()) as resp:
                if getattr(resp, "status", 200) != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _openalex_fetch_work_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        ndoi = normalize_doi(doi)
        if not ndoi:
            return None
        key = f"doi:{ndoi}"
        if key in self._work_cache:
            return self._work_cache[key]
        base_url = _openalex_base_works_url()
        data = self._openalex_fetch_json(f"{base_url}/doi:{quote(ndoi, safe='/')}")
        if len(self._work_cache) >= _MAX_WORK_CACHE:
            self._work_cache.pop(next(iter(self._work_cache)), None)
        self._work_cache[key] = data
        return data

    def _openalex_lookup_work_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        title = (title or "").strip()
        if len(title) < 10:
            return None
        key = f"title:{normalize_title(title)}"
        if key in self._work_cache:
            return self._work_cache[key]
        qs = urlencode(
            {
                "search": title,
                "select": "id,doi,title,authorships,publication_year,primary_location,ids,referenced_works",
                "per_page": 1,
            }
        )
        data = self._openalex_fetch_json(f"{_openalex_base_works_url()}?{qs}")
        work = None
        if isinstance(data, dict):
            results = data.get("results") or []
            work = results[0] if results else None
        if len(self._work_cache) >= _MAX_WORK_CACHE:
            self._work_cache.pop(next(iter(self._work_cache)), None)
        self._work_cache[key] = work
        return work

    def _openalex_fetch_work_by_id(self, work_id: str) -> Optional[Dict[str, Any]]:
        if not work_id:
            return None
        work_id = work_id.strip()
        key = f"id:{work_id}"
        if key in self._work_cache:
            return self._work_cache[key]
        if work_id.startswith("https://openalex.org/"):
            url = work_id
        else:
            url = f"{_openalex_base_works_url()}/{work_id}"
        data = self._openalex_fetch_json(url)
        if len(self._work_cache) >= _MAX_WORK_CACHE:
            self._work_cache.pop(next(iter(self._work_cache)), None)
        self._work_cache[key] = data
        return data

    def _openalex_fetch_works_batch(self, work_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple OpenAlex works in one batch request — avoids N+1 serial HTTP calls.

        Returns a mapping from full OpenAlex URL → work dict for every successfully
        fetched work.  Works not found in the API are simply absent from the result.
        """
        if not work_ids:
            return {}
        cfg = _openalex_cfg()
        if not getattr(cfg, "enabled", False):
            return {}
        # Map short W-ID → original URL so we can key the result dict by URL.
        short_to_url: Dict[str, str] = {}
        for raw in work_ids:
            url = str(raw).strip()
            short = url.rstrip("/").split("/")[-1] if url.startswith("https://openalex.org/") else url
            if short:
                short_to_url[short] = url
        if not short_to_url:
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for chunk in _chunks(list(short_to_url.keys()), 50):
            filter_str = "|".join(chunk)
            qs = urlencode(
                {
                    "filter": f"openalex_id:{filter_str}",
                    "select": "id,doi,title,authorships,publication_year,ids",
                    "per_page": len(chunk),
                }
            )
            data = self._openalex_fetch_json(f"{_openalex_base_works_url()}?{qs}")
            if not isinstance(data, dict):
                continue
            for item in (data.get("results") or []):
                if not isinstance(item, dict):
                    continue
                item_id = (item.get("id") or "").strip()
                if item_id:
                    result[item_id] = item
        return result

    def _resolve_openalex_work(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doi = normalize_doi(paper.get("doi"))
        if doi:
            work = self._openalex_fetch_work_by_doi(doi)
            if work:
                return work
        title = (paper.get("title") or "").strip()
        if title:
            return self._openalex_lookup_work_by_title(title)
        return None

    def _semantic_fallback(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = normalize_doi(paper.get("doi")) or (paper.get("title") or "").strip()
        if not query:
            return []
        try:
            results = asyncio.run(semantic_scholar_searcher.search(query=query, limit=1))
        except Exception:
            return []
        if not results:
            return []
        meta = (results[0].get("metadata") or {}) if isinstance(results[0], dict) else {}
        authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
        facts: List[Dict[str, Any]] = []
        paper_uid = paper.get("paper_uid")
        if not paper_uid:
            return facts
        paper_id = _paper_node_id(paper_uid)
        paper_label = (paper.get("title") or paper_uid).strip()
        for author_name in authors:
            clean_name = str(author_name).strip()
            if not clean_name:
                continue
            facts.append(
                {
                    "user_id": paper.get("user_id") or "default",
                    "scope_type": paper.get("scope_type") or "global",
                    "scope_key": paper.get("scope_key") or "global",
                    "graph_type": "author",
                    "src_node_id": _author_node_id(clean_name),
                    "src_node_type": "author",
                    "src_label": clean_name,
                    "relation_type": "authored",
                    "dst_node_id": paper_id,
                    "dst_node_type": "paper",
                    "dst_label": paper_label,
                    "weight": 1.0,
                    "provenance_json": json.dumps({"source": "semantic_scholar", "kind": "fallback_search"}, ensure_ascii=False),
                }
            )
        return facts

    def build_facts(self, graph_type: str, scope: GraphScope, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for paper in papers:
            work = self._resolve_openalex_work(paper)
            if work is None:
                if graph_type == "author":
                    enriched = self._semantic_fallback(
                        {
                            **paper,
                            "user_id": scope.user_id,
                            "scope_type": scope.scope_type,
                            "scope_key": scope.scope_key,
                        }
                    )
                    facts.extend(enriched)
                continue
            paper_uid = paper.get("paper_uid")
            if not paper_uid:
                continue
            paper_id = _paper_node_id(paper_uid)
            paper_label = (paper.get("title") or paper_uid).strip()
            authorships = work.get("authorships") or []
            if graph_type == "author":
                for authorship in authorships:
                    if not isinstance(authorship, dict):
                        continue
                    author_obj = authorship.get("author") or {}
                    author_name = (author_obj.get("display_name") or "").strip()
                    if not author_name:
                        continue
                    facts.append(
                        {
                            "user_id": scope.user_id,
                            "scope_type": scope.scope_type,
                            "scope_key": scope.scope_key,
                            "graph_type": graph_type,
                            "src_node_id": _author_node_id(author_name, author_obj.get("id")),
                            "src_node_type": "author",
                            "src_label": author_name,
                            "relation_type": "authored",
                            "dst_node_id": paper_id,
                            "dst_node_type": "paper",
                            "dst_label": paper_label,
                            "weight": 1.0,
                            "provenance_json": json.dumps({"source": "openalex", "kind": "authorship"}, ensure_ascii=False),
                        }
                    )
            elif graph_type == "institution":
                for authorship in authorships:
                    if not isinstance(authorship, dict):
                        continue
                    author_obj = authorship.get("author") or {}
                    author_name = (author_obj.get("display_name") or "").strip()
                    if not author_name:
                        continue
                    author_id = _author_node_id(author_name, author_obj.get("id"))
                    for inst in (authorship.get("institutions") or []):
                        if not isinstance(inst, dict):
                            continue
                        inst_name = (inst.get("display_name") or "").strip()
                        if not inst_name:
                            continue
                        facts.append(
                            {
                                "user_id": scope.user_id,
                                "scope_type": scope.scope_type,
                                "scope_key": scope.scope_key,
                                "graph_type": graph_type,
                                "src_node_id": author_id,
                                "src_node_type": "author",
                                "src_label": author_name,
                                "relation_type": "affiliated_with",
                                "dst_node_id": _institution_node_id(inst_name, inst.get("id") or inst.get("ror")),
                                "dst_node_type": "institution",
                                "dst_label": inst_name,
                                "weight": 1.0,
                                "provenance_json": json.dumps({"source": "openalex", "kind": "authorship_institution"}, ensure_ascii=False),
                            }
                        )
            elif graph_type == "citation":
                ref_ids = [str(r) for r in (work.get("referenced_works") or [])[:20]]
                ref_works_map = self._openalex_fetch_works_batch(ref_ids)
                for ref in ref_ids:
                    ref_work = ref_works_map.get(str(ref))
                    if not isinstance(ref_work, dict):
                        continue
                    target_uid = compute_paper_uid(
                        doi=normalize_doi(ref_work.get("doi") or ""),
                        title=(ref_work.get("title") or "").strip() or None,
                        authors=[
                            ((a.get("author") or {}).get("display_name") or "").strip()
                            for a in (ref_work.get("authorships") or [])
                            if isinstance(a, dict)
                        ],
                        year=_as_int(ref_work.get("publication_year"), 0) or None,
                        pmid=((ref_work.get("ids") or {}).get("pmid") or "").replace(
                            "https://pubmed.ncbi.nlm.nih.gov/",
                            "",
                        ).strip("/") or None,
                    )
                    if not target_uid or target_uid == paper_uid:
                        continue
                    target_label = (ref_work.get("title") or target_uid).strip()
                    facts.append(
                        {
                            "user_id": scope.user_id,
                            "scope_type": scope.scope_type,
                            "scope_key": scope.scope_key,
                            "graph_type": graph_type,
                            "src_node_id": paper_id,
                            "src_node_type": "paper",
                            "src_label": paper_label,
                            "relation_type": "cites",
                            "dst_node_id": _paper_node_id(target_uid),
                            "dst_node_type": "paper",
                            "dst_label": target_label,
                            "weight": 1.0,
                            "provenance_json": json.dumps({"source": "openalex", "kind": "referenced_work"}, ensure_ascii=False),
                        }
                    )
        return facts


class _FactView:
    """Lightweight proxy over a raw fact dict.

    Provides the same attribute interface as ``GraphFact`` so that
    ``_build_academic_snapshot_payload`` can operate on in-memory dicts
    without a SQL round-trip after ``_replace_scope_facts``.
    """

    __slots__ = (
        "src_node_id",
        "src_node_type",
        "src_label",
        "relation_type",
        "dst_node_id",
        "dst_node_type",
        "dst_label",
        "weight",
        "provenance_json",
    )

    def __init__(self, d: Dict[str, Any]) -> None:
        self.src_node_id: str = d["src_node_id"]
        self.src_node_type: str = d["src_node_type"]
        self.src_label: str = d["src_label"]
        self.relation_type: str = d["relation_type"]
        self.dst_node_id: str = d["dst_node_id"]
        self.dst_node_type: str = d["dst_node_type"]
        self.dst_label: str = d["dst_label"]
        self.weight: float = float(d.get("weight") or 1.0)
        self.provenance_json: str = d.get("provenance_json") or "{}"

    def get_provenance(self) -> Dict[str, Any]:
        return _safe_json_loads(self.provenance_json, {})


class GlobalGraphService:
    def __init__(self) -> None:
        self.entity_adapter = EntityGraphAdapter()
        self.fact_builder = AcademicFactBuilder()
        self.enricher = AcademicGraphEnricher()
        self._locks: Dict[Tuple[str, str, str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, graph_type: str, scope: GraphScope) -> threading.Lock:
        key = (scope.user_id, scope.scope_type, scope.scope_key, graph_type)
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _normalize_scope(self, scope: Dict[str, Any]) -> GraphScope:
        user_id = str(scope.get("user_id") or "default").strip() or "default"
        scope_type = str(scope.get("scope_type") or "global").strip().lower() or "global"
        scope_key = str(scope.get("scope_key") or "").strip()
        if scope_type not in VALID_SCOPE_TYPES:
            raise ValueError(f"unsupported scope_type: {scope_type}")
        if scope_type == "global":
            scope_key = scope_key or "global"
        if scope_type == "library":
            scope_key = str(_as_int(scope_key, 0))
            if scope_key == "0":
                raise ValueError("library scope requires numeric scope_key")
        if scope_type == "collection" and not scope_key:
            raise ValueError("collection scope requires scope_key")
        return GraphScope(user_id=user_id, scope_type=scope_type, scope_key=scope_key)

    def _validate_graph_type(self, graph_type: str) -> str:
        graph_type = (graph_type or "").strip().lower()
        if graph_type not in VALID_GRAPH_TYPES:
            raise ValueError(f"unsupported graph_type: {graph_type}")
        return graph_type

    def _scope_hash(self, scope: GraphScope) -> str:
        return hashlib.sha1(
            f"{scope.user_id}|{scope.scope_type}|{scope.scope_key}".encode("utf-8")
        ).hexdigest()[:16]

    def _snapshot_dir(self, graph_type: str, scope: GraphScope) -> Path:
        return settings.path.data / "graph_snapshots" / graph_type / self._scope_hash(scope)

    def _snapshot_path(self, graph_type: str, scope: GraphScope, version: int) -> Path:
        return self._snapshot_dir(graph_type, scope) / f"{version}.json"

    def _snapshot_row_to_dict(self, row: GraphSnapshot) -> Dict[str, Any]:
        return {
            "user_id": row.user_id,
            "scope_type": row.scope_type,
            "scope_key": row.scope_key,
            "graph_type": row.graph_type,
            "snapshot_version": row.snapshot_version,
            "status": row.status,
            "storage_path": row.storage_path,
            "node_count": row.node_count,
            "edge_count": row.edge_count,
            "built_from_revision": row.built_from_revision,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _graph_to_payload(
        self,
        graph_type: str,
        G: nx.Graph,
        snapshot_version: int,
        provenance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pagerank: Dict[str, float] = {}
        bridge_nodes: List[str] = []
        if G.number_of_nodes():
            try:
                pagerank = nx.pagerank(nx.DiGraph(G))
            except Exception:
                pagerank = {}
            try:
                bridge_nodes = list(nx.articulation_points(G.to_undirected()))
            except Exception:
                bridge_nodes = []

        nodes = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "type": attrs.get("type", ""),
                    "label": attrs.get("label", _node_label_from_id(node_id)),
                    "pagerank": round(float(pagerank.get(node_id, 0.0)), 6),
                    "degree": int(G.degree(node_id)),
                }
            )
        nodes.sort(key=lambda item: (item.get("label") or item["id"]))

        edges = []
        for src, dst, attrs in G.edges(data=True):
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "relation": attrs.get("relation", ""),
                    "weight": float(attrs.get("weight", 1.0)),
                    "provenance": attrs.get("provenance", {}),
                }
            )

        top_nodes = sorted(
            (
                {"id": node["id"], "type": node["type"], "score": node["pagerank"]}
                for node in nodes
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:10]
        return {
            "graph_type": graph_type,
            "snapshot_version": snapshot_version,
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "top_nodes": top_nodes,
                "bridge_nodes": bridge_nodes[:10],
            },
            "provenance": provenance,
        }

    def _load_snapshot_payload(self, row: GraphSnapshot) -> Dict[str, Any]:
        path = Path(row.storage_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _replace_scope_facts(self, graph_type: str, scope: GraphScope, facts: List[Dict[str, Any]]) -> None:
        now = _now_iso()
        with Session(get_engine()) as session:
            session.exec(
                sql_delete(GraphFact).where(
                    GraphFact.user_id == scope.user_id,
                    GraphFact.scope_type == scope.scope_type,
                    GraphFact.scope_key == scope.scope_key,
                    GraphFact.graph_type == graph_type,
                )
            )
            for fact in facts:
                session.add(
                    GraphFact(
                        user_id=scope.user_id,
                        scope_type=scope.scope_type,
                        scope_key=scope.scope_key,
                        graph_type=graph_type,
                        src_node_id=fact["src_node_id"],
                        src_node_type=fact["src_node_type"],
                        src_label=fact["src_label"],
                        relation_type=fact["relation_type"],
                        dst_node_id=fact["dst_node_id"],
                        dst_node_type=fact["dst_node_type"],
                        dst_label=fact["dst_label"],
                        weight=float(fact.get("weight", 1.0)),
                        provenance_json=fact.get("provenance_json") or "{}",
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()

    def _query_scope_facts(self, graph_type: str, scope: GraphScope) -> List[GraphFact]:
        with Session(get_engine()) as session:
            return list(
                session.exec(
                    select(GraphFact).where(
                        GraphFact.user_id == scope.user_id,
                        GraphFact.scope_type == scope.scope_type,
                        GraphFact.scope_key == scope.scope_key,
                        GraphFact.graph_type == graph_type,
                    )
                ).all()
            )

    def _query_latest_snapshot_rows(self, graph_type: str, scope: GraphScope) -> List[GraphSnapshot]:
        with Session(get_engine()) as session:
            return list(
                session.exec(
                    select(GraphSnapshot)
                    .where(
                        GraphSnapshot.user_id == scope.user_id,
                        GraphSnapshot.scope_type == scope.scope_type,
                        GraphSnapshot.scope_key == scope.scope_key,
                        GraphSnapshot.graph_type == graph_type,
                    )
                    .order_by(GraphSnapshot.snapshot_version.desc())
                ).all()
            )

    def _build_academic_snapshot_payload(
        self,
        graph_type: str,
        scope: GraphScope,
        snapshot_version: int,
        papers: List[Dict[str, Any]],
        facts: List[Any],  # GraphFact rows or _FactView proxies
    ) -> Dict[str, Any]:
        G = nx.DiGraph()
        provenance: List[Dict[str, Any]] = []
        for row in facts:
            prov = row.get_provenance()
            provenance.append(prov)
            G.add_node(
                row.src_node_id,
                type=row.src_node_type,
                label=row.src_label or _node_label_from_id(row.src_node_id),
            )
            G.add_node(
                row.dst_node_id,
                type=row.dst_node_type,
                label=row.dst_label or _node_label_from_id(row.dst_node_id),
            )
            G.add_edge(
                row.src_node_id,
                row.dst_node_id,
                relation=row.relation_type,
                weight=row.weight,
                provenance=prov,
            )

        if graph_type == "author":
            authors_by_paper: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for row in facts:
                if row.relation_type != "authored":
                    continue
                authors_by_paper[row.dst_node_id].append((row.src_node_id, row.src_label))
            pair_weights: Dict[Tuple[str, str], int] = defaultdict(int)
            pair_labels: Dict[Tuple[str, str], Tuple[str, str]] = {}
            for entries in authors_by_paper.values():
                ids = sorted({item[0]: item[1] for item in entries}.items(), key=lambda item: item[0])
                for idx, (src_id, src_label) in enumerate(ids):
                    for dst_id, dst_label in ids[idx + 1:]:
                        key = (src_id, dst_id)
                        pair_weights[key] += 1
                        pair_labels[key] = (src_label, dst_label)
            for (src_id, dst_id), weight in pair_weights.items():
                src_label, dst_label = pair_labels[(src_id, dst_id)]
                G.add_node(src_id, type="author", label=src_label)
                G.add_node(dst_id, type="author", label=dst_label)
                G.add_edge(
                    src_id,
                    dst_id,
                    relation="co_author",
                    weight=float(weight),
                    provenance={"source": "derived", "kind": "shared_papers", "shared_paper_count": weight},
                )

        elif graph_type == "institution":
            # Derive collaborates_with from the stored facts only.
            # build_local_facts (institution) emits both "authored" (author→paper)
            # and "affiliated_with" (author→institution) edges, so we can reconstruct
            # co-institution relationships without re-processing the papers list.
            author_to_papers: Dict[str, List[str]] = defaultdict(list)
            author_to_insts: Dict[str, List[str]] = defaultdict(list)
            for row in facts:
                if row.relation_type == "authored":
                    author_to_papers[row.src_node_id].append(row.dst_node_id)
                elif row.relation_type == "affiliated_with":
                    author_to_insts[row.src_node_id].append(row.dst_node_id)
            paper_to_insts: Dict[str, set] = defaultdict(set)
            for author_id, paper_nodes in author_to_papers.items():
                for paper_node in paper_nodes:
                    for inst_id in author_to_insts.get(author_id, []):
                        paper_to_insts[paper_node].add(inst_id)
            for _paper_node, insts in paper_to_insts.items():
                inst_list = sorted(insts)
                for idx, src_id in enumerate(inst_list):
                    for dst_id in inst_list[idx + 1:]:
                        if G.has_edge(src_id, dst_id) and G[src_id][dst_id].get("relation") == "collaborates_with":
                            G[src_id][dst_id]["weight"] = float(G[src_id][dst_id].get("weight", 1.0)) + 1.0
                        else:
                            src_label = (G.nodes.get(src_id) or {}).get("label", src_id)
                            dst_label = (G.nodes.get(dst_id) or {}).get("label", dst_id)
                            G.add_node(src_id, type="institution", label=src_label)
                            G.add_node(dst_id, type="institution", label=dst_label)
                            G.add_edge(
                                src_id,
                                dst_id,
                                relation="collaborates_with",
                                weight=1.0,
                                provenance={"source": "derived", "kind": "shared_papers"},
                            )

        elif graph_type == "citation":
            for src, dst, attrs in list(G.edges(data=True)):
                if attrs.get("relation") != "cites":
                    continue
                if not G.has_edge(dst, src):
                    G.add_edge(
                        dst,
                        src,
                        relation="cited_by",
                        weight=attrs.get("weight", 1.0),
                        provenance={"source": "derived", "kind": "reverse_cites"},
                    )

        return self._graph_to_payload(
            graph_type=graph_type,
            G=G,
            snapshot_version=snapshot_version,
            provenance=[
                json.loads(item)
                for item in sorted(
                    {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in provenance}
                )
            ],
        )

    def _materialize_snapshot(
        self,
        graph_type: str,
        scope: GraphScope,
        current_revision: str,
        snapshot_version: int,
    ) -> Dict[str, Any]:
        papers = self.fact_builder.collect_scope_papers(scope)
        local_facts = self.fact_builder.build_local_facts(graph_type, scope, papers)
        enriched_facts = self.enricher.build_facts(graph_type, scope, papers)
        deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for fact in local_facts + enriched_facts:
            key = (fact["src_node_id"], fact["relation_type"], fact["dst_node_id"])
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = fact
            else:
                existing_weight = float(existing.get("weight", 1.0))
                incoming_weight = float(fact.get("weight", 1.0))
                if incoming_weight > existing_weight:
                    deduped[key] = fact
        # Build the graph payload from in-memory deduped facts — no SQL round-trip.
        fact_views = [_FactView(d) for d in deduped.values()]
        payload = self._build_academic_snapshot_payload(
            graph_type=graph_type,
            scope=scope,
            snapshot_version=snapshot_version,
            papers=papers,
            facts=fact_views,
        )
        payload["built_from_revision"] = current_revision
        payload["scope"] = scope.as_dict()
        # Persist facts after building to keep the hot path shorter.
        self._replace_scope_facts(graph_type, scope, list(deduped.values()))
        return payload

    def _cleanup_old_snapshot_files(self, graph_type: str, scope: GraphScope, keep_version: int) -> None:
        """Delete JSON files for old snapshot versions in the scope directory.

        Only the file for *keep_version* is retained; all others (including
        orphaned ``.tmp`` files) are removed so disk usage stays bounded.
        """
        snap_dir = self._snapshot_dir(graph_type, scope)
        if not snap_dir.exists():
            return
        for f in snap_dir.iterdir():
            if f.suffix not in (".json", ".tmp"):
                continue
            try:
                ver = int(f.stem)
            except ValueError:
                # Remove stray .tmp files regardless of name.
                if f.suffix == ".tmp":
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue
            if ver != keep_version:
                try:
                    f.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug("failed to delete old snapshot file %s: %s", f, exc)

    def ensure_snapshot(self, graph_type: str, scope: Dict[str, Any], refresh: bool = False) -> Dict[str, Any]:
        graph_type = self._validate_graph_type(graph_type)
        normalized_scope = self._normalize_scope(scope)

        if graph_type == "entity":
            snapshots = self.entity_adapter.list_snapshots(normalized_scope)
            if snapshots:
                return snapshots[0]
            return {
                **normalized_scope.as_dict(),
                "graph_type": "entity",
                "snapshot_version": None,
                "status": "missing",
                "storage_path": "",
                "node_count": 0,
                "edge_count": 0,
                "built_from_revision": "",
                "error_message": "",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }

        papers = self.fact_builder.collect_scope_papers(normalized_scope)
        current_revision = self.fact_builder.compute_revision(graph_type, normalized_scope, papers)
        rows = self._query_latest_snapshot_rows(graph_type, normalized_scope)
        latest_ready = next((row for row in rows if row.status == "ready"), None)
        latest_row = rows[0] if rows else None
        if (
            not refresh
            and latest_ready is not None
            and latest_row is not None
            and latest_row.id == latest_ready.id
            and latest_ready.built_from_revision == current_revision
        ):
            return self._snapshot_row_to_dict(latest_ready)

        lock = self._lock_for(graph_type, normalized_scope)
        with lock:
            rows = self._query_latest_snapshot_rows(graph_type, normalized_scope)
            latest_ready = next((row for row in rows if row.status == "ready"), None)
            latest_row = rows[0] if rows else None
            if (
                not refresh
                and latest_ready is not None
                and latest_row is not None
                and latest_row.id == latest_ready.id
                and latest_ready.built_from_revision == current_revision
            ):
                return self._snapshot_row_to_dict(latest_ready)

            with Session(get_engine()) as session:
                version = (
                    session.exec(
                        select(GraphSnapshot.snapshot_version)
                        .where(
                            GraphSnapshot.user_id == normalized_scope.user_id,
                            GraphSnapshot.scope_type == normalized_scope.scope_type,
                            GraphSnapshot.scope_key == normalized_scope.scope_key,
                            GraphSnapshot.graph_type == graph_type,
                        )
                        .order_by(GraphSnapshot.snapshot_version.desc())
                    ).first()
                    or 0
                ) + 1
                row = GraphSnapshot(
                    user_id=normalized_scope.user_id,
                    scope_type=normalized_scope.scope_type,
                    scope_key=normalized_scope.scope_key,
                    graph_type=graph_type,
                    snapshot_version=version,
                    status="building",
                    storage_path="",
                    node_count=0,
                    edge_count=0,
                    built_from_revision=current_revision,
                    error_message="",
                    created_at=_now_iso(),
                    updated_at=_now_iso(),
                )
                session.add(row)
                session.commit()

            try:
                payload = self._materialize_snapshot(
                    graph_type=graph_type,
                    scope=normalized_scope,
                    current_revision=current_revision,
                    snapshot_version=version,
                )
                path = self._snapshot_path(graph_type, normalized_scope, version)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write: temp file + os.replace so a crash mid-write
                # never leaves a partial JSON visible to readers.
                tmp_path = path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(str(tmp_path), str(path))
                with Session(get_engine()) as session:
                    stmt = select(GraphSnapshot).where(
                        GraphSnapshot.user_id == normalized_scope.user_id,
                        GraphSnapshot.scope_type == normalized_scope.scope_type,
                        GraphSnapshot.scope_key == normalized_scope.scope_key,
                        GraphSnapshot.graph_type == graph_type,
                        GraphSnapshot.snapshot_version == version,
                    )
                    snap = session.exec(stmt).first()
                    if snap is not None:
                        snap.status = "ready"
                        snap.storage_path = str(path)
                        snap.node_count = int(payload.get("metrics", {}).get("node_count", 0))
                        snap.edge_count = int(payload.get("metrics", {}).get("edge_count", 0))
                        snap.built_from_revision = current_revision
                        snap.updated_at = _now_iso()
                        session.add(snap)
                        session.commit()
                        self._cleanup_old_snapshot_files(graph_type, normalized_scope, version)
                        return self._snapshot_row_to_dict(snap)
            except Exception as e:
                logger.warning(
                    "graph snapshot build failed graph_type=%s scope=%s:%s err=%s",
                    graph_type,
                    normalized_scope.scope_type,
                    normalized_scope.scope_key,
                    e,
                )
                with Session(get_engine()) as session:
                    stmt = select(GraphSnapshot).where(
                        GraphSnapshot.user_id == normalized_scope.user_id,
                        GraphSnapshot.scope_type == normalized_scope.scope_type,
                        GraphSnapshot.scope_key == normalized_scope.scope_key,
                        GraphSnapshot.graph_type == graph_type,
                        GraphSnapshot.snapshot_version == version,
                    )
                    snap = session.exec(stmt).first()
                    if snap is not None:
                        snap.status = "error"
                        snap.error_message = str(e)
                        snap.updated_at = _now_iso()
                        session.add(snap)
                        session.commit()
                if latest_ready is not None:
                    return self._snapshot_row_to_dict(latest_ready)
                raise

    def list_snapshots(self, graph_type: str, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        graph_type = self._validate_graph_type(graph_type)
        normalized_scope = self._normalize_scope(scope)
        if graph_type == "entity":
            return self.entity_adapter.list_snapshots(normalized_scope)
        return [
            self._snapshot_row_to_dict(row)
            for row in self._query_latest_snapshot_rows(graph_type, normalized_scope)
        ]

    def graph_stats(self, graph_type: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        graph_type = self._validate_graph_type(graph_type)
        normalized_scope = self._normalize_scope(scope)
        if graph_type == "entity":
            return self.entity_adapter.stats()
        # Read the latest ready snapshot row without triggering a rebuild.
        snapshot_rows = self._query_latest_snapshot_rows(graph_type, normalized_scope)
        latest_ready = next((row for row in snapshot_rows if row.status == "ready"), None)
        snapshot_meta = self._snapshot_row_to_dict(latest_ready) if latest_ready else {}
        rows = self._query_scope_facts(graph_type, normalized_scope)
        node_ids = {row.src_node_id for row in rows} | {row.dst_node_id for row in rows}
        relation_counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            relation_counts[row.relation_type] += 1
        return {
            "available": bool(snapshot_meta.get("storage_path") or rows),
            "graph_type": graph_type,
            "scope": normalized_scope.as_dict(),
            "fact_count": len(rows),
            "total_nodes": len(node_ids),
            "total_edges": len(rows),
            "relation_counts": dict(sorted(relation_counts.items())),
            "snapshot_version": snapshot_meta.get("snapshot_version"),
            "snapshot_status": snapshot_meta.get("status"),
        }

    def _coerce_seed_node_ids(self, seeds: Optional[Dict[str, Any]]) -> List[str]:
        if not seeds:
            return []
        node_ids = [str(v).strip() for v in (seeds.get("node_ids") or []) if str(v).strip()]
        paper_uids = [str(v).strip() for v in (seeds.get("paper_uids") or []) if str(v).strip()]
        node_ids.extend(_paper_node_id(uid) for uid in paper_uids)
        deduped: List[str] = []
        seen = set()
        for node_id in node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            deduped.append(node_id)
        return deduped

    def _extract_subgraph(
        self,
        payload: Dict[str, Any],
        seeds: List[str],
        depth: int,
        limit: int,
    ) -> Dict[str, Any]:
        if not payload.get("nodes"):
            return {
                **payload,
                "nodes": [],
                "edges": [],
            }
        G = nx.DiGraph()
        node_map = {node["id"]: dict(node) for node in payload.get("nodes") or [] if node.get("id")}
        for node_id, node in node_map.items():
            G.add_node(node_id, **node)
        for edge in payload.get("edges") or []:
            src = edge.get("source")
            dst = edge.get("target")
            if src in node_map and dst in node_map:
                G.add_edge(src, dst, **edge)

        if not seeds:
            seeds = [item["id"] for item in sorted(payload.get("metrics", {}).get("top_nodes", []), key=lambda x: x.get("score", 0), reverse=True)[:1] if item.get("id")]
        seeds = [seed for seed in seeds if seed in G]
        if not seeds:
            nodes = list(node_map.values())[: max(1, limit)]
            allowed = {node["id"] for node in nodes}
            edges = [edge for edge in payload.get("edges") or [] if edge.get("source") in allowed and edge.get("target") in allowed]
            return {**payload, "nodes": nodes, "edges": edges}

        visited: List[str] = []
        seen = set()
        queue = deque((seed, 0) for seed in seeds)
        while queue and len(visited) < max(1, limit):
            node_id, dist = queue.popleft()
            if node_id in seen or dist > max(1, depth):
                continue
            seen.add(node_id)
            visited.append(node_id)
            for _, nbr in G.out_edges(node_id):
                if nbr not in seen:
                    queue.append((nbr, dist + 1))
            for nbr, _ in G.in_edges(node_id):
                if nbr not in seen:
                    queue.append((nbr, dist + 1))

        allowed = set(visited)
        nodes = []
        for node_id in visited:
            node = dict(node_map[node_id])
            node["is_seed"] = node_id in seeds
            nodes.append(node)
        edges = [
            edge
            for edge in payload.get("edges") or []
            if edge.get("source") in allowed and edge.get("target") in allowed
        ]
        return {**payload, "nodes": nodes, "edges": edges}

    def query_subgraph(
        self,
        graph_type: str,
        scope: Dict[str, Any],
        seeds: Optional[Dict[str, Any]],
        depth: int,
        limit: int,
        snapshot_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        graph_type = self._validate_graph_type(graph_type)
        normalized_scope = self._normalize_scope(scope)
        seed_node_ids = self._coerce_seed_node_ids(seeds)
        depth = max(1, min(int(depth or 1), 3))
        limit = max(1, min(int(limit or 50), 300))

        if graph_type == "entity":
            return self.entity_adapter.build_subgraph(seed_node_ids, depth=depth, limit=limit)

        snapshot_meta = self.ensure_snapshot(graph_type, normalized_scope.as_dict(), refresh=False)
        rows = self._query_latest_snapshot_rows(graph_type, normalized_scope)
        target_row = None
        if snapshot_version is not None:
            target_row = next((row for row in rows if row.snapshot_version == snapshot_version and row.status == "ready"), None)
            if target_row is None:
                raise ValueError(
                    f"snapshot_version={snapshot_version} not found or not ready for "
                    f"graph_type={graph_type} scope={normalized_scope.scope_type}/{normalized_scope.scope_key}"
                )
        if target_row is None:
            target_row = next((row for row in rows if row.status == "ready"), None)
        if target_row is None:
            return {
                "nodes": [],
                "edges": [],
                "metrics": {"node_count": 0, "edge_count": 0, "top_nodes": [], "bridge_nodes": []},
                "snapshot_version": snapshot_meta.get("snapshot_version"),
                "provenance": [{"source": "graph_service", "status": "empty"}],
            }
        payload = self._load_snapshot_payload(target_row)
        result = self._extract_subgraph(payload, seed_node_ids, depth=depth, limit=limit)
        result["snapshot_version"] = target_row.snapshot_version
        return result

    def summarize_subgraph(
        self,
        graph_type: str,
        scope: Dict[str, Any],
        seeds: Optional[Dict[str, Any]],
        *,
        depth: int = 1,
        question: Optional[str] = None,
        max_items: Optional[int] = None,
        snapshot_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        graph_type = self._validate_graph_type(graph_type)
        normalized_scope = self._normalize_scope(scope)
        subgraph = self.query_subgraph(
            graph_type=graph_type,
            scope=normalized_scope.as_dict(),
            seeds=seeds,
            depth=depth,
            limit=max_items or 50,
            snapshot_version=snapshot_version,
        )
        nodes = list(subgraph.get("nodes") or [])
        edges = list(subgraph.get("edges") or [])
        top_nodes = sorted(nodes, key=lambda item: item.get("pagerank", 0.0), reverse=True)[: max(1, min(max_items or 5, 10))]
        bridge_nodes = list(subgraph.get("metrics", {}).get("bridge_nodes") or [])[: max(1, min(max_items or 5, 10))]
        lines = [
            f"# {graph_type.title()} Graph Summary",
            f"- scope: {normalized_scope.scope_type}:{normalized_scope.scope_key}",
            f"- snapshot_version: {subgraph.get('snapshot_version')}",
            f"- nodes: {len(nodes)}",
            f"- edges: {len(edges)}",
        ]
        if question:
            lines.append(f"- question: {question.strip()}")
        lines.append("")
        lines.append("## Core Nodes")
        if top_nodes:
            for node in top_nodes:
                lines.append(
                    f"- {node.get('label') or node['id']} ({node.get('type') or 'node'})"
                    f" pagerank={node.get('pagerank', 0.0):.4f} degree={node.get('degree', 0)}"
                )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Key Relations")
        if edges:
            for edge in edges[: max(1, min(max_items or 8, 12))]:
                src = next((n.get("label") for n in nodes if n["id"] == edge.get("source")), edge.get("source"))
                dst = next((n.get("label") for n in nodes if n["id"] == edge.get("target")), edge.get("target"))
                lines.append(
                    f"- {src} --[{edge.get('relation')}]--> {dst} (weight={float(edge.get('weight', 1.0)):.1f})"
                )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Bridge Nodes")
        if bridge_nodes:
            for node_id in bridge_nodes:
                label = next((n.get("label") for n in nodes if n["id"] == node_id), node_id)
                lines.append(f"- {label}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Provenance")
        prov_lines = []
        seen = set()
        for item in (subgraph.get("provenance") or [])[:20]:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    item = {"source": item}
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            source = item.get("source") or "unknown"
            kind = item.get("kind")
            suffix = f" ({kind})" if kind else ""
            prov_lines.append(f"- {source}{suffix}")
        if prov_lines:
            lines.extend(prov_lines)
        else:
            lines.append("- none")
        return {
            "summary": "\n".join(lines),
            "snapshot_version": subgraph.get("snapshot_version"),
            "provenance": subgraph.get("provenance") or [],
            "subgraph": subgraph,
        }

    def mark_scope_stale(
        self,
        scope: Dict[str, Any],
        reason: str,
        graph_types: Optional[Iterable[str]] = None,
    ) -> int:
        normalized_scope = self._normalize_scope(scope)
        target_types = [
            self._validate_graph_type(graph_type)
            for graph_type in (graph_types or ("citation", "author", "institution"))
            if graph_type != "entity"
        ]
        if not target_types:
            return 0
        now = _now_iso()
        count = 0
        with Session(get_engine()) as session:
            rows = list(
                session.exec(
                    select(GraphSnapshot).where(
                        GraphSnapshot.user_id == normalized_scope.user_id,
                        GraphSnapshot.scope_type == normalized_scope.scope_type,
                        GraphSnapshot.scope_key == normalized_scope.scope_key,
                        GraphSnapshot.graph_type.in_(target_types),
                        GraphSnapshot.status.in_(["ready", "building", "rebuilding"]),
                    )
                ).all()
            )
            for row in rows:
                row.status = "stale"
                row.error_message = reason[:500]
                row.updated_at = now
                session.add(row)
                count += 1
            session.commit()
        return count


def get_global_graph_service() -> GlobalGraphService:
    global _global_graph_service
    if _global_graph_service is None:
        _global_graph_service = GlobalGraphService()
    return _global_graph_service


def mark_graph_scope_stale(
    *,
    user_id: str,
    scope_type: str,
    scope_key: str,
    reason: str,
    graph_types: Optional[Iterable[str]] = None,
) -> int:
    try:
        return get_global_graph_service().mark_scope_stale(
            {"user_id": user_id, "scope_type": scope_type, "scope_key": scope_key},
            reason=reason,
            graph_types=graph_types,
        )
    except Exception as e:
        logger.debug(
            "mark_graph_scope_stale failed user=%s scope=%s:%s reason=%s err=%s",
            user_id,
            scope_type,
            scope_key,
            reason,
            e,
        )
        return 0
