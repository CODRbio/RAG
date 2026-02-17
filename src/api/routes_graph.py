"""
知识图谱 API：统计、实体列表、邻居查询。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


def _get_hippo():
    """获取 HippoRAG 实例"""
    try:
        from src.graph.hippo_rag import get_hippo_rag
        graph_path = settings.path.data / "hippo_graph.json"
        if not graph_path.exists():
            return None
        return get_hippo_rag(graph_path)
    except Exception as e:
        logger.warning(f"HippoRAG 加载失败: {e}")
        return None


def _query_chunk_in_collection(collection_name: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    from src.indexing.milvus_ops import milvus
    escaped_chunk_id = chunk_id.replace('"', '\\"')

    try:
        rows = milvus.query(
            collection_name,
            filter=f'chunk_id == "{escaped_chunk_id}"',
            output_fields=["chunk_id", "paper_id", "content", "raw_content", "section_path", "page", "content_type", "chunk_type"],
            limit=1,
        )
    except Exception as e:
        logger.warning("query chunk failed collection=%s chunk=%s err=%s", collection_name, chunk_id, e)
        return None

    if not rows:
        return None
    row = rows[0] if isinstance(rows, list) else rows
    if not isinstance(row, dict):
        return None
    return {
        "collection": collection_name,
        "chunk_id": row.get("chunk_id") or chunk_id,
        "paper_id": row.get("paper_id") or "",
        "content": (row.get("raw_content") or row.get("content") or ""),
        "section_path": row.get("section_path") or "",
        "page": row.get("page"),
        "content_type": row.get("content_type") or "",
        "chunk_type": row.get("chunk_type") or "",
    }


def _find_enriched_json_by_paper_id(paper_id: str) -> Optional[Path]:
    parsed_dir = settings.path.data / "parsed"
    direct = parsed_dir / paper_id / "enriched.json"
    if direct.exists():
        return direct
    candidates = list(parsed_dir.glob(f"*{paper_id}*/enriched.json"))
    if candidates:
        return candidates[0]
    return None


def _query_chunk_from_parsed(paper_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    json_path = _find_enriched_json_by_paper_id(paper_id)
    if not json_path:
        return None
    try:
        from src.chunking.chunker import ChunkConfig, chunk_blocks

        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = str(doc.get("doc_id") or paper_id)
        content_flow = doc.get("content_flow") or []
        claims = doc.get("claims") or []

        # 先按 settings 配置切一次，再按默认配置切一次，兼容历史构建批次差异
        cfg_candidates = [
            ChunkConfig(
                target_chars=settings.chunk.target_chars,
                min_chars=settings.chunk.min_chars,
                max_chars=settings.chunk.max_chars,
                overlap_sentences=settings.chunk.overlap_sentences,
                table_rows_per_chunk=settings.chunk.table_rows_per_chunk,
            ),
            ChunkConfig(),
        ]
        for cfg in cfg_candidates:
            chunks = chunk_blocks(content_flow, doc_id=doc_id, config=cfg, claims=claims)
            for c in chunks:
                if str(c.chunk_id) != chunk_id:
                    continue
                meta = c.meta or {}
                page_range = meta.get("page_range") or [None, None]
                page = page_range[0] if isinstance(page_range, (list, tuple)) and page_range else None
                return {
                    "collection": "parsed_fallback",
                    "chunk_id": str(c.chunk_id),
                    "paper_id": doc_id,
                    "content": c.text or "",
                    "section_path": meta.get("section_path") or "",
                    "page": page,
                    "content_type": c.content_type or "",
                    "chunk_type": ",".join(meta.get("block_types", [])) or "",
                }
    except Exception as e:
        logger.warning("query chunk from parsed failed paper=%s chunk=%s err=%s", paper_id, chunk_id, e)
        return None
    return None


@router.get("/stats")
def graph_stats() -> Dict[str, Any]:
    """获取图谱统计信息"""
    hippo = _get_hippo()
    if hippo is None:
        return {
            "available": False,
            "total_nodes": 0,
            "total_edges": 0,
            "entity_count": 0,
            "chunk_count": 0,
            "entity_types": {},
        }
    stats = hippo.stats()
    return {"available": True, **stats}


@router.get("/entities")
def graph_entities(
    entity_type: Optional[str] = Query(None, description="按类型过滤: SPECIES|LOCATION|PHENOMENON|METHOD|SUBSTANCE"),
    limit: int = Query(200, ge=1, le=1000, description="最大返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    q: Optional[str] = Query(None, description="按名称搜索（子串匹配）"),
) -> Dict[str, Any]:
    """获取实体列表"""
    hippo = _get_hippo()
    if hippo is None:
        raise HTTPException(status_code=404, detail="图谱未加载")

    entities = []
    for name, entity in hippo.entities.items():
        if entity_type and entity.type != entity_type:
            continue
        if q and q.lower() not in name.lower():
            continue
        entities.append({
            "name": name,
            "type": entity.type,
            "mention_count": len(entity.mentions),
        })

    # 按 mention_count 降序排序
    entities.sort(key=lambda e: -e["mention_count"])
    total = len(entities)
    page = entities[offset: offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "entities": page}


@router.get("/neighbors/{entity_name}")
def graph_neighbors(
    entity_name: str,
    depth: int = Query(1, ge=1, le=3, description="扩展深度 (1-3)"),
) -> Dict[str, Any]:
    """
    获取指定实体的邻居子图。

    返回以 entity_name 为中心的 nodes + edges 数据（可直接用于前端图谱渲染）。
    """
    hippo = _get_hippo()
    if hippo is None:
        raise HTTPException(status_code=404, detail="图谱未加载")

    G = hippo.G
    if entity_name not in G:
        raise HTTPException(status_code=404, detail=f"实体 '{entity_name}' 不存在")

    # BFS 收集 n-hop 邻居
    visited: set = set()
    frontier = {entity_name}
    edges_set: set = set()

    for _ in range(depth):
        next_frontier: set = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            for _, neighbor, data in G.edges(node, data=True):
                edges_set.add((node, neighbor, data.get("relation", ""), data.get("weight", 1)))
                next_frontier.add(neighbor)
            # 也查入边
            for predecessor, _, data in G.in_edges(node, data=True):
                edges_set.add((predecessor, node, data.get("relation", ""), data.get("weight", 1)))
                next_frontier.add(predecessor)
        frontier = next_frontier - visited

    # 所有涉及的节点
    all_node_ids = set()
    for s, t, *_ in edges_set:
        all_node_ids.add(s)
        all_node_ids.add(t)
    all_node_ids.add(entity_name)

    # 构建 nodes
    nodes = []
    for nid in all_node_ids:
        node_data = G.nodes.get(nid, {})
        ntype = node_data.get("type", "ENTITY")
        # 从 entities 表获取更多信息
        entity_info = hippo.entities.get(nid)
        if entity_info:
            ntype = entity_info.type
        nodes.append({
            "id": nid,
            "type": ntype,
            "paper_id": node_data.get("paper_id", ""),
            "is_center": nid == entity_name,
        })

    # 构建 edges
    edges = [
        {"source": s, "target": t, "relation": rel, "weight": w}
        for s, t, rel, w in edges_set
    ]

    return {
        "center": entity_name,
        "depth": depth,
        "nodes": nodes,
        "edges": edges,
    }


@router.get("/chunk/{chunk_id}")
def graph_chunk_detail(
    chunk_id: str,
    collection: Optional[str] = Query(None, description="优先查询的集合名"),
    paper_id: Optional[str] = Query(None, description="可选，前端透传用于展示"),
) -> Dict[str, Any]:
    """获取 chunk 详情文本与元数据（用于图谱节点点击弹窗）。"""
    from src.indexing.milvus_ops import milvus

    candidates: List[str] = []
    if collection:
        candidates.append(collection)

    try:
        for name in milvus.client.list_collections():
            if name not in candidates:
                candidates.append(name)
    except Exception:
        # 无法列集合时至少尝试默认集合配置
        for name in settings.collection.all():
            if name not in candidates:
                candidates.append(name)

    for cname in candidates:
        found = _query_chunk_in_collection(cname, chunk_id)
        if found:
            # 追加图上相邻实体，便于解释该 chunk 的关系来源
            related_entities: List[str] = []
            hippo = _get_hippo()
            if hippo is not None and chunk_id in hippo.G:
                for predecessor, _, _ in hippo.G.in_edges(chunk_id, data=True):
                    ptype = hippo.G.nodes.get(predecessor, {}).get("type", "ENTITY")
                    if ptype != "CHUNK":
                        related_entities.append(str(predecessor))
            found["related_entities"] = sorted(set(related_entities))[:30]
            if paper_id and not found.get("paper_id"):
                found["paper_id"] = paper_id
            return found

    # 回退：Milvus 不存在时，尝试从 parsed/enriched.json 重建并匹配 chunk_id
    if paper_id:
        fallback = _query_chunk_from_parsed(paper_id, chunk_id)
        if fallback:
            related_entities: List[str] = []
            hippo = _get_hippo()
            if hippo is not None and chunk_id in hippo.G:
                for predecessor, _, _ in hippo.G.in_edges(chunk_id, data=True):
                    ptype = hippo.G.nodes.get(predecessor, {}).get("type", "ENTITY")
                    if ptype != "CHUNK":
                        related_entities.append(str(predecessor))
            fallback["related_entities"] = sorted(set(related_entities))[:30]
            return fallback

    raise HTTPException(status_code=404, detail=f"chunk '{chunk_id}' 未找到")
