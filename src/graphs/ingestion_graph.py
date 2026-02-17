"""
LangGraph 离线入库图：解析 PDF → 切块 → 向量化 → Milvus upsert → HippoRAG 建图
支持 chunk_id 主键 true upsert，重复跑同一批文档不产生重复数据。
"""

import json
from pathlib import Path
from typing import TypedDict, Literal, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    RunnableConfig = dict  # type: ignore

from src.chunking.chunker import ChunkConfig, chunk_blocks
from src.graph.hippo_rag import HippoRAG


def _truncate(content: str, max_len: int = 65000) -> str:
    return content[:max_len] if len(content) > max_len else content


class IngestionState(TypedDict, total=False):
    pdf_paths: list[str]
    current_index: int
    parsed_paths: dict[str, str]
    total_chunks: int
    total_upserted: int
    errors: list[dict]
    build_graph: bool
    run_id: str
    artifact_path: str | None


def _list_pdfs(state: IngestionState, *, config: RunnableConfig) -> dict:
    raw_papers = Path(config["configurable"]["raw_papers_path"])
    max_docs = config["configurable"].get("max_docs")
    paths = sorted(str(p) for p in raw_papers.glob("*.pdf"))
    if max_docs is not None:
        paths = paths[: max_docs]
    return {
        "pdf_paths": paths,
        "current_index": 0,
        "parsed_paths": {},
        "total_chunks": 0,
        "total_upserted": 0,
    }


def _parse_one_pdf(state: IngestionState, *, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    processor = cfg["processor"]
    skip_enrichment = cfg.get("skip_enrichment", True)
    parsed_dir = Path(cfg["parsed_dir"])
    pdf_paths = state["pdf_paths"]
    idx = state["current_index"]
    pdf_path = Path(pdf_paths[idx])
    paper_id = pdf_path.stem
    output_dir = parsed_dir / paper_id
    errors = list(state.get("errors") or [])

    try:
        processor.process(
            pdf_path,
            output_dir=output_dir,
            skip_enrichment=skip_enrichment,
        )
        parsed_paths = dict(state.get("parsed_paths") or {})
        parsed_paths[paper_id] = str(output_dir)
        return {"parsed_paths": parsed_paths}
    except Exception as e:
        errors.append({"paper_id": paper_id, "stage": "parse", "error": str(e)})
        return {"errors": errors}


def _chunk_embed_upsert_one_doc(state: IngestionState, *, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    embedder = cfg["embedder"]
    milvus = cfg["milvus"]
    collection_name = cfg["collection_name"]
    chunk_config = cfg.get("chunk_config") or {}
    pdf_paths = state["pdf_paths"]
    idx = state["current_index"]
    paper_id = Path(pdf_paths[idx]).stem
    parsed_paths = state.get("parsed_paths") or {}
    json_path = Path(parsed_paths.get(paper_id, "")) / "enriched.json"
    errors = list(state.get("errors") or [])

    out = {
        "total_chunks": state.get("total_chunks", 0),
        "total_upserted": state.get("total_upserted", 0),
        "current_index": idx + 1,
    }

    if not json_path.exists():
        errors.append({"paper_id": paper_id, "stage": "chunk_embed_upsert", "error": f"missing {json_path}"})
        return {**out, "errors": errors}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc.get("doc_id", paper_id)
        content_flow = doc.get("content_flow", [])
        ccfg = ChunkConfig(
            target_chars=chunk_config.get("target_chars", 1000),
            min_chars=chunk_config.get("min_chars", 200),
            max_chars=chunk_config.get("max_chars", 1800),
            overlap_sentences=chunk_config.get("overlap_sentences", 2),
            table_rows_per_chunk=chunk_config.get("table_rows_per_chunk", 10),
        )
        chunks = chunk_blocks(content_flow, doc_id=doc_id, config=ccfg)

        rows = []
        for c in chunks:
            text = _truncate(c.text)
            meta = c.meta or {}
            page_range = meta.get("page_range", [0, 0])
            page = page_range[0] if isinstance(page_range, (list, tuple)) else meta.get("page", 0)
            rows.append({
                "paper_id": doc_id,
                "chunk_id": c.chunk_id,
                "content": text,
                "raw_content": text,
                "domain": "global",
                "content_type": c.content_type,
                "chunk_type": ",".join(meta.get("block_types", []))[:64] or "paragraph",
                "section_path": str(meta.get("section_path", ""))[:512],
                "page": int(page) if isinstance(page, (int, float)) else 0,
                "_text_for_embed": text,
            })

        if not rows:
            return out

        texts = [r["_text_for_embed"] for r in rows]
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = embedder.encode(batch)
            for k, j in enumerate(range(i, min(i + batch_size, len(rows)))):
                rows[j]["dense_vector"] = emb["dense"][k].tolist()
                sp = emb["sparse"]._getrow(k).tocoo()
                rows[j]["sparse_vector"] = {int(col): float(val) for col, val in zip(sp.col, sp.data)}
        for r in rows:
            del r["_text_for_embed"]

        upsert_batch = 100
        for start in range(0, len(rows), upsert_batch):
            batch = rows[start : start + upsert_batch]
            milvus.upsert(collection_name, batch)

        out["total_chunks"] = out["total_chunks"] + len(rows)
        out["total_upserted"] = out["total_upserted"] + len(rows)
    except Exception as e:
        errors.append({"paper_id": paper_id, "stage": "chunk_embed_upsert", "error": str(e)})
        out["errors"] = errors

    return out


def _route_after_upsert(state: IngestionState) -> Literal["parse_one_pdf", "build_hippo_graph", "write_artifact"]:
    pdf_paths = state.get("pdf_paths") or []
    idx = state.get("current_index", 0)
    if idx < len(pdf_paths):
        return "parse_one_pdf"
    if state.get("build_graph"):
        return "build_hippo_graph"
    return "write_artifact"


def _build_hippo_graph(state: IngestionState, *, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    parsed_dir = Path(cfg["parsed_dir"])
    graph_output_path = Path(cfg["graph_output_path"])
    chunk_config = cfg.get("chunk_config") or {}
    ccfg = ChunkConfig(
        target_chars=chunk_config.get("target_chars", 1000),
        min_chars=chunk_config.get("min_chars", 200),
        max_chars=chunk_config.get("max_chars", 1800),
        overlap_sentences=chunk_config.get("overlap_sentences", 2),
        table_rows_per_chunk=chunk_config.get("table_rows_per_chunk", 10),
    )
    hippo = HippoRAG()
    hippo.build_from_parsed_docs(parsed_dir, use_llm=False, chunk_config=ccfg)
    graph_output_path.parent.mkdir(parents=True, exist_ok=True)
    hippo.save(graph_output_path)
    return {}


def _write_artifact(state: IngestionState, *, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    run_id = state.get("run_id") or cfg.get("run_id", "")
    artifacts_dir = Path(cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"06_ingest_{run_id}.json"
    artifact = {
        "run_id": run_id,
        "input_count": len(state.get("pdf_paths") or []),
        "total_chunks": state.get("total_chunks", 0),
        "total_upserted": state.get("total_upserted", 0),
        "errors": state.get("errors") or [],
        "build_graph": state.get("build_graph", False),
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return {"artifact_path": str(artifact_path)}


def _route_after_list(state: IngestionState) -> Literal["parse_one_pdf", "write_artifact"]:
    if state.get("pdf_paths"):
        return "parse_one_pdf"
    return "write_artifact"


def build_ingestion_graph(*, checkpointer: Any = None):
    """构建离线入库图。checkpointer 为 SqliteSaver 实例时支持可续跑；由调用方 with SqliteSaver.from_conn_string(path) as cp 传入。"""
    builder = StateGraph(IngestionState)

    builder.add_node("list_pdfs", _list_pdfs)
    builder.add_node("parse_one_pdf", _parse_one_pdf)
    builder.add_node("chunk_embed_upsert_one_doc", _chunk_embed_upsert_one_doc)
    builder.add_node("build_hippo_graph", _build_hippo_graph)
    builder.add_node("write_artifact", _write_artifact)

    builder.add_edge(START, "list_pdfs")
    builder.add_conditional_edges("list_pdfs", _route_after_list)
    builder.add_edge("parse_one_pdf", "chunk_embed_upsert_one_doc")
    builder.add_conditional_edges("chunk_embed_upsert_one_doc", _route_after_upsert)
    builder.add_edge("build_hippo_graph", "write_artifact")
    builder.add_edge("write_artifact", END)

    return builder.compile(checkpointer=checkpointer)
