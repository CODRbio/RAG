"""
数据入库 API：文件上传、解析、chunk、embedding、入库；集合管理。
"""

import hashlib
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])
_INGEST_CANCEL_EVENTS: dict[str, threading.Event] = {}
_INGEST_CANCEL_LOCK = threading.Lock()


def _request_cancel(job_id: str) -> None:
    with _INGEST_CANCEL_LOCK:
        ev = _INGEST_CANCEL_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _INGEST_CANCEL_EVENTS[job_id] = ev
        ev.set()


def _is_cancel_requested(job_id: str) -> bool:
    with _INGEST_CANCEL_LOCK:
        ev = _INGEST_CANCEL_EVENTS.get(job_id)
    return bool(ev and ev.is_set())


def _clear_cancel_event(job_id: str) -> None:
    with _INGEST_CANCEL_LOCK:
        _INGEST_CANCEL_EVENTS.pop(job_id, None)


# ============================================================
# Collections
# ============================================================

@router.get("/collections")
def list_collections() -> dict:
    """列出 Milvus 中已有的集合及行数"""
    from src.indexing.milvus_ops import milvus
    try:
        names: list = milvus.client.list_collections()
    except Exception as e:
        logger.error("list_collections failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Milvus 不可用: {e}")
    result = []
    for name in sorted(names):
        try:
            cnt = milvus.count(name)
        except Exception:
            cnt = -1
        result.append({"name": name, "count": cnt})
    return {"collections": result}


@router.post("/collections")
def create_collection(body: dict) -> dict:
    """创建新集合（v2 schema: chunk_id 主键，支持 upsert），并生成覆盖范围摘要（可后续刷新）"""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    recreate = bool(body.get("recreate", False))
    from src.indexing.milvus_ops import milvus
    try:
        milvus.create_collection(name, recreate=recreate, schema_version="v2")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 建库时用当前模型根据库名生成 scope 摘要，供「查询与库是否匹配」使用
    try:
        from src.indexing.collection_scope import generate_scope_summary, set_scope
        from src.llm.llm_manager import get_manager
        _config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
        manager = get_manager(str(_config_path))
        client = manager.get_client(None)
        summary = generate_scope_summary(name, sample_texts=None, llm_client=client)
        if summary:
            set_scope(name, summary)
    except Exception as e:
        logger.debug("create_collection scope summary failed (non-fatal): %s", e)
    return {"ok": True, "name": name}


@router.delete("/collections/{name}")
def delete_collection(name: str) -> dict:
    """删除指定集合（不可恢复）"""
    from src.indexing.milvus_ops import milvus
    try:
        if not milvus.client.has_collection(name):
            raise HTTPException(status_code=404, detail=f"集合 '{name}' 不存在")
        milvus.client.drop_collection(name)
        # 同时清理 paper 元数据
        try:
            from src.indexing.paper_store import delete_collection_papers
            delete_collection_papers(name)
        except Exception as pe:
            logger.warning("paper_store cleanup failed for '%s': %s", name, pe)
        logger.info("Collection '%s' deleted", name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "name": name}


@router.get("/collections/{name}/scope")
def get_collection_scope(name: str) -> dict:
    """获取指定集合的覆盖范围摘要（建库/入库或刷新时生成）。"""
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    from src.indexing.collection_scope import get_scope_meta
    meta = get_scope_meta(name)
    if meta is None:
        return {"ok": True, "name": name, "scope_summary": None, "updated_at": None}
    return {"ok": True, "name": name, **meta}


@router.put("/collections/{name}/scope")
def update_collection_scope(name: str, body: dict = Body(...)) -> dict:
    """编辑并保存指定集合的覆盖范围摘要。body: { "scope_summary": "..." }"""
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    from src.indexing.milvus_ops import milvus
    if not milvus.client.has_collection(name):
        raise HTTPException(status_code=404, detail=f"集合 '{name}' 不存在")
    scope_summary = (body.get("scope_summary") or "").strip()
    from src.indexing.collection_scope import set_scope, get_scope_meta
    set_scope(name, scope_summary)
    meta = get_scope_meta(name)
    return {"ok": True, "name": name, "scope_summary": scope_summary or None, "updated_at": meta.get("updated_at") if meta else None}


@router.post("/collections/{name}/scope-refresh")
def refresh_collection_scope(name: str, body: Optional[dict] = Body(None)) -> dict:
    """刷新指定集合的覆盖范围摘要。默认根据集合内文档题目用 LLM 生成；body 可选: { "sample_texts": ["..."] } 覆盖默认。"""
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    from src.indexing.milvus_ops import milvus
    if not milvus.client.has_collection(name):
        raise HTTPException(status_code=404, detail=f"集合 '{name}' 不存在")
    sample_texts = None
    if body and isinstance(body.get("sample_texts"), list):
        sample_texts = [str(t).strip() for t in body["sample_texts"] if t and str(t).strip()][:20]
    if not sample_texts:
        from src.indexing.collection_scope import get_document_titles_for_collection
        titles = get_document_titles_for_collection(name, max_titles=None)
        if titles:
            sample_texts = ["Document titles (use these to infer scope):\n- " + "\n- ".join(titles)]
    try:
        from src.indexing.collection_scope import generate_scope_summary, set_scope
        from src.llm.llm_manager import get_manager
        config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
        manager = get_manager(str(config_path))
        client = manager.get_client(None)
        summary = generate_scope_summary(
            name, sample_texts=sample_texts, llm_client=client, max_sample_chars=40000
        )
        if summary:
            set_scope(name, summary)
        return {"ok": True, "name": name, "scope_summary": summary or ""}
    except Exception as e:
        logger.warning("scope-refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Papers (文件级管理)
# ============================================================

@router.get("/collections/{name}/papers")
def list_papers_in_collection(name: str) -> dict:
    """列出指定集合中已入库的文件列表"""
    from src.indexing.paper_store import list_papers
    papers = list_papers(name)
    return {"collection": name, "papers": papers}


@router.delete("/collections/{name}/papers/{paper_id:path}")
def delete_paper_from_collection(name: str, paper_id: str) -> dict:
    """
    删除指定文件：从 Milvus 删除该 paper_id 的全部 chunks，并移除元数据记录。
    """
    from src.indexing.milvus_ops import milvus
    from src.indexing.paper_store import delete_paper

    # 1. 从 Milvus 删除该 paper_id 的所有 chunks
    deleted_count = 0
    try:
        if milvus.client.has_collection(name):
            # Milvus delete by filter
            result = milvus.client.delete(
                collection_name=name,
                filter=f'paper_id == "{paper_id}"',
            )
            deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
            logger.info("Deleted %s chunks for paper_id='%s' from '%s'", deleted_count, paper_id, name)
    except Exception as e:
        logger.error("Milvus delete failed for paper_id='%s': %s", paper_id, e)
        raise HTTPException(status_code=500, detail=f"Milvus 删除失败: {e}")

    # 2. 从 SQLite 删除元数据
    delete_paper(name, paper_id)

    return {"ok": True, "collection": name, "paper_id": paper_id, "deleted_chunks": deleted_count}


# ============================================================
# Upload
# ============================================================

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    collection: str = Form(""),
) -> dict:
    """
    上传文件（PDF 等），保存到 data/raw_papers/，返回文件信息列表。
    """
    raw_papers = settings.path.raw_papers
    raw_papers.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        # 安全文件名
        safe_name = Path(f.filename).name
        dest = raw_papers / safe_name
        # 如果重名，加 uuid 后缀
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = raw_papers / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        content_hash = hashlib.sha256(content).hexdigest()
        saved.append({
            "filename": safe_name,
            "path": str(dest),
            "size": len(content),
            "content_hash": content_hash,
        })
        logger.info("Uploaded: %s -> %s (%d bytes)", safe_name, dest, len(content))

    return {"uploaded": saved, "count": len(saved)}


# ============================================================
# Process (Parse + Chunk + Embed + Upsert) with SSE progress
# ============================================================

def _normalize_process_body(body: dict) -> dict:
    file_paths = body.get("file_paths") or []
    if not file_paths:
        raise HTTPException(status_code=400, detail="file_paths 不能为空")
    collection_name = (body.get("collection") or "").strip() or settings.collection.global_
    content_hashes = body.get("content_hashes") or {}
    skip_enrichment = body.get("skip_enrichment", True)
    enrich_tables = body.get("enrich_tables", not skip_enrichment)
    enrich_figures = body.get("enrich_figures", not skip_enrichment)
    actual_skip = bool(skip_enrichment and (not enrich_tables) and (not enrich_figures))
    return {
        "file_paths": file_paths,
        "collection_name": collection_name,
        "content_hashes": content_hashes,
        "enrich_tables": bool(enrich_tables),
        "enrich_figures": bool(enrich_figures),
        "actual_skip": actual_skip,
        "llm_text_provider": (body.get("llm_text_provider") or "").strip() or None,
        "llm_text_model": (body.get("llm_text_model") or "").strip() or None,
        "llm_text_concurrency": body.get("llm_text_concurrency"),
        "llm_vision_provider": (body.get("llm_vision_provider") or "").strip() or None,
        "llm_vision_model": (body.get("llm_vision_model") or "").strip() or None,
        "llm_vision_concurrency": body.get("llm_vision_concurrency"),
    }


def _emit_job_event(job_id: str, event: str, data: dict) -> None:
    from src.indexing.ingest_job_store import append_event
    append_event(job_id, event, data)


def _finalize_cancelled_job(
    job_id: str,
    *,
    total: int,
    processed_files: int,
    failed_files: int,
    total_chunks: int,
    total_upserted: int,
    current_file: str = "",
) -> None:
    from src.indexing.ingest_job_store import get_job, update_job

    job = get_job(job_id)
    if job and job.get("status") == "cancelled":
        return
    _emit_job_event(
        job_id,
        "cancelled",
        {
            "message": "任务已取消",
            "current_file": current_file,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_files": total,
        },
    )
    _emit_job_event(
        job_id,
        "done",
        {
            "cancelled": True,
            "total_files": total,
            "total_chunks": total_chunks,
            "total_upserted": total_upserted,
            "errors": [],
        },
    )
    update_job(
        job_id,
        status="cancelled",
        processed_files=processed_files,
        failed_files=failed_files,
        total_chunks=total_chunks,
        total_upserted=total_upserted,
        current_stage="cancelled",
        message="任务已取消",
        finished_at=time.time(),
    )


def _run_ingest_job(job_id: str, cfg: dict) -> None:
    from src.chunking.chunker import ChunkConfig, chunk_blocks
    from src.indexing.embedder import embedder
    from src.indexing.ingest_job_store import update_job
    from src.indexing.milvus_ops import milvus

    file_paths = cfg["file_paths"]
    collection_name = cfg["collection_name"]
    content_hashes = cfg["content_hashes"]
    enrich_tables = cfg["enrich_tables"]
    enrich_figures = cfg["enrich_figures"]
    actual_skip = cfg["actual_skip"]
    llm_text_provider = cfg["llm_text_provider"]
    llm_text_model = cfg["llm_text_model"]
    llm_text_concurrency = cfg["llm_text_concurrency"]
    llm_vision_provider = cfg["llm_vision_provider"]
    llm_vision_model = cfg["llm_vision_model"]
    llm_vision_concurrency = cfg["llm_vision_concurrency"]

    update_job(job_id, status="running", message="任务已启动")
    total = len(file_paths)
    total_chunks = 0
    total_upserted = 0
    errors = []

    try:
        milvus.create_collection(collection_name, recreate=False, schema_version="v2")
    except Exception as e:
        msg = f"集合创建失败: {e}"
        _emit_job_event(job_id, "error", {"message": msg})
        _emit_job_event(job_id, "done", {"total_files": total, "total_chunks": 0, "total_upserted": 0, "errors": [{"stage": "init", "error": str(e)}]})
        update_job(job_id, status="error", error_message=str(e), message=msg, finished_at=time.time())
        return

    config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
    try:
        from src.parser.pdf_parser import PDFProcessor, ParserConfig
        parser_cfg = ParserConfig.from_json(config_path) if config_path.exists() else ParserConfig()
        parser_cfg.enrich_tables = enrich_tables
        parser_cfg.enrich_figures = enrich_figures
        if llm_text_provider:
            parser_cfg.llm_text_provider = llm_text_provider
        parser_cfg.llm_text_model = llm_text_model
        if llm_text_concurrency is not None:
            try:
                parser_cfg.llm_text_concurrency = max(1, int(llm_text_concurrency))
            except Exception:
                pass
        if llm_vision_provider:
            parser_cfg.llm_vision_provider = llm_vision_provider
        parser_cfg.llm_vision_model = llm_vision_model
        if llm_vision_concurrency is not None:
            try:
                parser_cfg.llm_vision_concurrency = max(1, int(llm_vision_concurrency))
            except Exception:
                pass
        llm_manager = None
        if not actual_skip:
            try:
                from src.llm import LLMManager
                llm_manager = LLMManager.from_json(str(config_path))
            except Exception:
                pass
        processor = PDFProcessor(config=parser_cfg, llm_manager=llm_manager)
    except Exception as e:
        msg = f"处理器初始化失败: {e}"
        _emit_job_event(job_id, "error", {"message": msg})
        _emit_job_event(job_id, "done", {"total_files": total, "total_chunks": 0, "total_upserted": 0, "errors": [{"stage": "init", "error": str(e)}]})
        update_job(job_id, status="error", error_message=str(e), message=msg, finished_at=time.time())
        return

    chunk_cfg = ChunkConfig(
        target_chars=settings.chunk.target_chars,
        min_chars=settings.chunk.min_chars,
        max_chars=settings.chunk.max_chars,
        overlap_sentences=settings.chunk.overlap_sentences,
        table_rows_per_chunk=settings.chunk.table_rows_per_chunk,
    )
    parsed_dir = settings.path.parsed
    processed_files = 0
    failed_files = 0
    sample_texts_for_scope: List[str] = []  # 收集片段供入库完成后生成 scope 摘要
    max_scope_samples = 10

    _emit_job_event(job_id, "start", {
        "total": total,
        "collection": collection_name,
        "enrich_tables": enrich_tables,
        "enrich_figures": enrich_figures,
    })

    if _is_cancel_requested(job_id):
        _finalize_cancelled_job(
            job_id,
            total=total,
            processed_files=0,
            failed_files=0,
            total_chunks=0,
            total_upserted=0,
        )
        return

    for idx, fpath in enumerate(file_paths):
        pdf_path = Path(fpath)
        paper_id = pdf_path.stem
        file_name = pdf_path.name
        if _is_cancel_requested(job_id):
            _finalize_cancelled_job(
                job_id,
                total=total,
                processed_files=processed_files,
                failed_files=failed_files,
                total_chunks=total_chunks,
                total_upserted=total_upserted,
                current_file=file_name,
            )
            return
        update_job(job_id, current_file=file_name, current_stage="parsing", message=f"解析 {file_name}...")
        _emit_job_event(job_id, "progress", {
            "file": file_name,
            "index": idx,
            "total": total,
            "stage": "parsing",
            "message": f"解析 {file_name}...",
        })

        output_dir = parsed_dir / paper_id
        try:
            enrich_queue = queue.Queue()

            def put_enrich_progress(kind_key: str, payload: dict) -> None:
                kind = "table" if kind_key == "enrich_table" else "figure"
                enrich_queue.put({
                    "file": file_name,
                    "kind": kind,
                    "index": payload.get("index"),
                    "total": payload.get("total"),
                    "status": payload.get("status", ""),
                    "message": payload.get("message"),
                })

            parse_timeout = 600
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    processor.process,
                    pdf_path,
                    output_dir=output_dir,
                    skip_enrichment=actual_skip,
                    progress_callback=put_enrich_progress if not actual_skip else None,
                )
                start_ts = time.time()
                while True:
                    if _is_cancel_requested(job_id):
                        future.cancel()
                        _finalize_cancelled_job(
                            job_id,
                            total=total,
                            processed_files=processed_files,
                            failed_files=failed_files,
                            total_chunks=total_chunks,
                            total_upserted=total_upserted,
                            current_file=file_name,
                        )
                        return
                    try:
                        while True:
                            ev = enrich_queue.get_nowait()
                            _emit_job_event(job_id, "enrich_progress", ev)
                    except queue.Empty:
                        pass
                    try:
                        future.result(timeout=5)
                        break
                    except FuturesTimeoutError:
                        elapsed = int(time.time() - start_ts)
                        if elapsed > parse_timeout:
                            future.cancel()
                            raise TimeoutError(f"PDF 解析超时 ({elapsed}s > {parse_timeout}s)")
                        _emit_job_event(job_id, "heartbeat", {
                            "file": file_name,
                            "stage": "parsing",
                            "elapsed": elapsed,
                            "message": f"解析中 {file_name}... ({elapsed}s)",
                        })
            try:
                while True:
                    ev = enrich_queue.get_nowait()
                    _emit_job_event(job_id, "enrich_progress", ev)
            except queue.Empty:
                pass
            logger.info("Parsed: %s -> %s", file_name, output_dir)
        except Exception as e:
            logger.error("Parse failed for %s: %s", file_name, e)
            errors.append({"file": file_name, "stage": "parse", "error": str(e)})
            _emit_job_event(job_id, "file_error", {"file": file_name, "stage": "parse", "error": str(e)})
            processed_files += 1
            failed_files += 1
            update_job(job_id, processed_files=processed_files, failed_files=failed_files, current_stage="parse_error")
            continue

        update_job(job_id, current_stage="chunking", message=f"切块 {file_name}...")
        _emit_job_event(job_id, "progress", {
            "file": file_name,
            "index": idx,
            "total": total,
            "stage": "chunking",
            "message": f"切块 {file_name}...",
        })

        json_path = output_dir / "enriched.json"
        if not json_path.exists():
            errors.append({"file": file_name, "stage": "chunk", "error": "enriched.json 不存在"})
            _emit_job_event(job_id, "file_error", {"file": file_name, "stage": "chunk", "error": "enriched.json 不存在"})
            processed_files += 1
            failed_files += 1
            update_job(job_id, processed_files=processed_files, failed_files=failed_files, current_stage="chunk_error")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            doc_id = doc.get("doc_id", paper_id)
            enrich_meta = doc.get("enrichment_meta") or {}
            table_count = int(enrich_meta.get("table_count", 0) or 0)
            figure_count = int(enrich_meta.get("figure_count", 0) or 0)
            table_success = int(enrich_meta.get("table_success", 0) or 0)
            figure_success = int(enrich_meta.get("figure_success", 0) or 0)
            content_flow = doc.get("content_flow", [])
            doc_claims = doc.get("claims") or []
            doc_metadata = doc.get("doc_metadata") or {}
            chunks = chunk_blocks(content_flow, doc_id=doc_id, config=chunk_cfg, claims=doc_claims)
        except Exception as e:
            errors.append({"file": file_name, "stage": "chunk", "error": str(e)})
            _emit_job_event(job_id, "file_error", {"file": file_name, "stage": "chunk", "error": str(e)})
            processed_files += 1
            failed_files += 1
            update_job(job_id, processed_files=processed_files, failed_files=failed_files, current_stage="chunk_error")
            continue

        if not chunks:
            _emit_job_event(job_id, "file_done", {"file": file_name, "chunks": 0, "upserted": 0})
            processed_files += 1
            update_job(job_id, processed_files=processed_files, current_stage="done")
            continue

        update_job(job_id, current_stage="embedding", message=f"向量化 {file_name} ({len(chunks)} chunks)...")
        _emit_job_event(job_id, "progress", {
            "file": file_name,
            "index": idx,
            "total": total,
            "stage": "embedding",
            "message": f"向量化 {file_name} ({len(chunks)} chunks)...",
        })

        try:
            rows = _build_rows(chunks, doc_id, collection_name, doc_metadata=doc_metadata)
            # 写入 DOI/Title 到 SQLite 持久化存储（供跨源去重使用）
            if doc_metadata.get("doi") or doc_metadata.get("title"):
                _update_paper_metadata(doc_id, doc_metadata)
            texts = [r.pop("_text_for_embed") for r in rows]
            if len(sample_texts_for_scope) < max_scope_samples and texts:
                for t in texts[:3]:
                    if len(sample_texts_for_scope) >= max_scope_samples:
                        break
                    if t and isinstance(t, str) and t.strip():
                        sample_texts_for_scope.append(t.strip()[:4000])
            batch_size = 32
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for bi, i in enumerate(range(0, len(texts), batch_size)):
                if _is_cancel_requested(job_id):
                    _finalize_cancelled_job(
                        job_id,
                        total=total,
                        processed_files=processed_files,
                        failed_files=failed_files,
                        total_chunks=total_chunks,
                        total_upserted=total_upserted,
                        current_file=file_name,
                    )
                    return
                batch = texts[i:i + batch_size]
                logger.info("Embedding %s batch %d/%d (%d texts)", file_name, bi + 1, total_batches, len(batch))
                emb = embedder.encode(batch)
                for k, j in enumerate(range(i, min(i + batch_size, len(rows)))):
                    rows[j]["dense_vector"] = emb["dense"][k].tolist()
                    sp = emb["sparse"]._getrow(k).tocoo()
                    rows[j]["sparse_vector"] = {int(col): float(val) for col, val in zip(sp.col, sp.data)}
        except Exception as e:
            logger.error("Embed failed for %s: %s\n%s", file_name, e, traceback.format_exc())
            errors.append({"file": file_name, "stage": "embed", "error": str(e)})
            _emit_job_event(job_id, "file_error", {"file": file_name, "stage": "embed", "error": str(e)})
            processed_files += 1
            failed_files += 1
            update_job(job_id, processed_files=processed_files, failed_files=failed_files, current_stage="embed_error")
            continue

        update_job(job_id, current_stage="indexing", message=f"入库 {file_name} ({len(rows)} rows)...")
        _emit_job_event(job_id, "progress", {
            "file": file_name,
            "index": idx,
            "total": total,
            "stage": "indexing",
            "message": f"入库 {file_name} ({len(rows)} rows)...",
        })

        try:
            upsert_batch = 100
            total_ubatches = (len(rows) + upsert_batch - 1) // upsert_batch
            for ui, start in enumerate(range(0, len(rows), upsert_batch)):
                if _is_cancel_requested(job_id):
                    _finalize_cancelled_job(
                        job_id,
                        total=total,
                        processed_files=processed_files,
                        failed_files=failed_files,
                        total_chunks=total_chunks,
                        total_upserted=total_upserted,
                        current_file=file_name,
                    )
                    return
                batch = rows[start:start + upsert_batch]
                logger.info("Upsert %s batch %d/%d (%d rows)", file_name, ui + 1, total_ubatches, len(batch))
                milvus.upsert(collection_name, batch)
            total_chunks += len(chunks)
            total_upserted += len(rows)
            logger.info("Upsert done for %s: %d chunks, %d rows", file_name, len(chunks), len(rows))
            try:
                from src.indexing.paper_store import upsert_paper
                upsert_paper(
                    collection=collection_name,
                    paper_id=doc_id,
                    filename=file_name,
                    file_path=str(fpath),
                    file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                    chunk_count=len(chunks),
                    row_count=len(rows),
                    enrich_tables_enabled=enrich_tables,
                    enrich_figures_enabled=enrich_figures,
                    table_count=table_count,
                    figure_count=figure_count,
                    table_success=table_success,
                    figure_success=figure_success,
                    status="done",
                    content_hash=content_hashes.get(fpath, ""),
                )
            except Exception as pe:
                logger.warning("paper_store write failed: %s", pe)
        except Exception as e:
            logger.error("Upsert failed for %s: %s\n%s", file_name, e, traceback.format_exc())
            errors.append({"file": file_name, "stage": "upsert", "error": str(e)})
            _emit_job_event(job_id, "file_error", {"file": file_name, "stage": "upsert", "error": str(e)})
            try:
                from src.indexing.paper_store import upsert_paper
                upsert_paper(
                    collection=collection_name,
                    paper_id=doc_id,
                    filename=file_name,
                    file_path=str(fpath),
                    enrich_tables_enabled=enrich_tables,
                    enrich_figures_enabled=enrich_figures,
                    status="error",
                    error_message=str(e),
                    content_hash=content_hashes.get(fpath, ""),
                )
            except Exception:
                pass
            processed_files += 1
            failed_files += 1
            update_job(job_id, processed_files=processed_files, failed_files=failed_files, current_stage="upsert_error")
            continue

        _emit_job_event(job_id, "file_done", {"file": file_name, "chunks": len(chunks), "upserted": len(rows)})
        processed_files += 1
        update_job(
            job_id,
            processed_files=processed_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_upserted=total_upserted,
            current_stage="done",
            message=f"已完成 {processed_files}/{total}",
        )

    # 入库完成后：本批材料摘要 + 与已有 scope 合并，增量更新（不重算全量）
    if total_upserted > 0 and collection_name and sample_texts_for_scope:
        try:
            from src.indexing.collection_scope import summarize_new_materials, update_scope_with_new_materials
            from src.llm.llm_manager import get_manager
            config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
            manager = get_manager(str(config_path))
            client = manager.get_client(None)
            new_summary = summarize_new_materials(sample_texts_for_scope, client)
            if new_summary:
                update_scope_with_new_materials(collection_name, new_summary, client)
                logger.info("ingest job done: scope updated (incremental) for %s", collection_name)
        except Exception as e:
            logger.debug("ingest scope update failed (non-fatal): %s", e)

    _emit_job_event(job_id, "done", {
        "total_files": total,
        "total_chunks": total_chunks,
        "total_upserted": total_upserted,
        "errors": errors,
    })
    update_job(
        job_id,
        status="done",
        processed_files=processed_files,
        failed_files=failed_files,
        total_chunks=total_chunks,
        total_upserted=total_upserted,
        current_stage="done",
        finished_at=time.time(),
        message=f"完成: {processed_files}/{total}",
    )


def _run_ingest_job_safe(job_id: str, cfg: dict) -> None:
    from src.indexing.ingest_job_store import update_job
    try:
        _run_ingest_job(job_id, cfg)
    except Exception as e:
        logger.error("ingest job failed unexpectedly job_id=%s err=%s\n%s", job_id, e, traceback.format_exc())
        _emit_job_event(job_id, "error", {"message": str(e)})
        _emit_job_event(
            job_id,
            "done",
            {"total_files": len(cfg.get("file_paths") or []), "total_chunks": 0, "total_upserted": 0, "errors": [{"stage": "job", "error": str(e)}]},
        )
        update_job(job_id, status="error", error_message=str(e), finished_at=time.time(), message=f"异常终止: {e}")
    finally:
        _clear_cancel_event(job_id)


@router.post("/process")
def process_files(body: dict) -> JSONResponse:
    """创建入库任务并立即返回 job_id（Worker 将自动领取并执行）。"""
    from src.indexing.ingest_job_store import create_job

    cfg = _normalize_process_body(body)
    job = create_job(cfg["collection_name"], cfg, total_files=len(cfg["file_paths"]))
    job_id = job.get("job_id")
    return JSONResponse({"ok": True, "job_id": job_id})


@router.get("/jobs")
def list_ingest_jobs(limit: int = 20, status: Optional[str] = None) -> dict:
    from src.indexing.ingest_job_store import list_jobs
    jobs = list_jobs(limit=limit, status=status)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
def get_ingest_job(job_id: str) -> dict:
    from src.indexing.ingest_job_store import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {"job": job}


@router.post("/jobs/{job_id}/cancel")
def cancel_ingest_job(job_id: str) -> dict:
    from src.indexing.ingest_job_store import get_job, update_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")

    current_status = str(job.get("status") or "")
    if current_status in {"done", "error", "cancelled"}:
        return {"ok": True, "job_id": job_id, "status": current_status}

    if current_status == "pending":
        update_job(job_id, status="cancelled", message="任务已取消（未启动）", finished_at=time.time())
        _emit_job_event(job_id, "cancelled", {"job_id": job_id, "message": "任务已取消（未启动）"})
        _emit_job_event(job_id, "done", {"cancelled": True, "total_files": 0, "total_chunks": 0, "total_upserted": 0, "errors": []})
        return {"ok": True, "job_id": job_id, "status": "cancelled"}

    _request_cancel(job_id)
    update_job(job_id, message="收到取消请求，正在停止任务...")
    _emit_job_event(job_id, "cancel_requested", {"job_id": job_id, "message": "已请求取消"})
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


@router.get("/jobs/{job_id}/events")
def stream_ingest_job_events(job_id: str, after_id: int = 0) -> StreamingResponse:
    from src.indexing.ingest_job_store import get_job, list_events

    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="job 不存在")

    def event_stream():
        cursor = max(0, int(after_id))
        idle_ticks = 0
        while True:
            events = list_events(job_id, after_id=cursor, limit=500)
            if events:
                idle_ticks = 0
                for ev in events:
                    cursor = max(cursor, int(ev["event_id"]))
                    yield _sse(ev["event"], ev["data"])
                continue

            job = get_job(job_id)
            if not job:
                yield _sse("error", {"message": "job 不存在"})
                break

            if job.get("status") in {"done", "error", "cancelled"}:
                # 终态时再做一次拉取，确保最后事件不丢
                final_events = list_events(job_id, after_id=cursor, limit=500)
                for ev in final_events:
                    cursor = max(cursor, int(ev["event_id"]))
                    yield _sse(ev["event"], ev["data"])
                break

            idle_ticks += 1
            if idle_ticks % 5 == 0:
                yield _sse(
                    "heartbeat",
                    {
                        "job_id": job_id,
                        "file": job.get("current_file", ""),
                        "stage": job.get("current_stage", ""),
                        "message": job.get("message", "waiting"),
                    },
                )
            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _truncate(content: str, max_len: int = 65000) -> str:
    return content[:max_len] if len(content) > max_len else content


def _update_paper_metadata(doc_id: str, doc_metadata: dict) -> None:
    """写入论文 DOI/Title 到 SQLite 持久化存储（供跨源去重使用）"""
    try:
        from src.indexing.paper_metadata_store import paper_meta_store
        paper_meta_store.upsert(
            paper_id=doc_id,
            doi=doc_metadata.get("doi"),
            title=doc_metadata.get("title"),
            source="ingestion",
        )
    except Exception as e:
        logger.warning("Failed to update paper_metadata store: %s", e)


def _get_field_max_lengths(collection_name: str) -> dict:
    """查询集合 schema 获取各 VARCHAR 字段的 max_length"""
    defaults = {
        "paper_id": 250, "chunk_id": 120, "domain": 30,
        "content_type": 30, "chunk_type": 30, "section_path": 500,
    }
    try:
        from src.indexing.milvus_ops import milvus
        info = milvus.client.describe_collection(collection_name)
        for field in info.get("fields", []):
            name = field.get("name", "")
            params = field.get("params", {})
            if name in defaults and "max_length" in params:
                defaults[name] = int(params["max_length"]) - 2  # 留 2 字符安全余量
    except Exception:
        pass
    return defaults


def _build_rows(chunks, doc_id: str, collection_name: str = "",
                 doc_metadata: dict | None = None) -> list:
    """将 chunks 转为 Milvus upsert 行格式，截断上限动态匹配集合 schema"""
    limits = _get_field_max_lengths(collection_name) if collection_name else {}
    l_paper = limits.get("paper_id", 250)
    l_chunk = limits.get("chunk_id", 120)
    l_ctype = limits.get("content_type", 30)
    l_cktype = limits.get("chunk_type", 30)
    l_sp = limits.get("section_path", 500)

    dm = doc_metadata or {}
    doi = dm.get("doi") or ""
    doc_title = dm.get("title") or ""

    rows = []
    for c in chunks:
        text = _truncate(c.text)
        meta = c.meta or {}
        page_range = meta.get("page_range", [0, 0])
        page = page_range[0] if isinstance(page_range, (list, tuple)) else meta.get("page", 0)
        sp_raw = meta.get("section_path", "")
        if isinstance(sp_raw, (list, tuple)):
            sp_raw = " > ".join(str(s) for s in sp_raw)
        row = {
            "paper_id": str(doc_id)[:l_paper],
            "chunk_id": str(c.chunk_id)[:l_chunk],
            "content": text,
            "raw_content": text,
            "domain": "global",
            "content_type": str(c.content_type or "text")[:l_ctype],
            "chunk_type": ",".join(meta.get("block_types", []))[:l_cktype] or "paragraph",
            "section_path": str(sp_raw)[:l_sp],
            "page": int(page) if isinstance(page, (int, float)) else 0,
            "_text_for_embed": text,
        }
        if doi:
            row["doi"] = doi
        if doc_title:
            row["doc_title"] = doc_title
        rows.append(row)
    return rows


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
