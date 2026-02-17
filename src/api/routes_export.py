"""
导出 API：按 canvas 输出 Markdown。

支持通过 cite_key_format 参数控制引用格式：
- numeric: [1], [2], [3]
- hash: [a3f7b2c91e04]
- author_date: [Smith2023]
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import ExportRequest, ExportResponse
from src.collaboration.canvas.canvas_manager import get_canvas
from src.collaboration.export.formatter import export_canvas_markdown
from src.collaboration.memory.session_memory import get_session_store

router = APIRouter(prefix="/export", tags=["export"])


@router.post("", response_model=ExportResponse)
def export_canvas(body: ExportRequest) -> ExportResponse:
    canvas_id = body.canvas_id or ""
    if not canvas_id and body.session_id:
        meta = get_session_store().get_session_meta(body.session_id)
        if meta:
            canvas_id = meta.get("canvas_id") or ""
    if not canvas_id:
        raise HTTPException(status_code=400, detail="canvas_id or session_id is required")

    c = get_canvas(canvas_id)
    if c is None:
        raise HTTPException(status_code=404, detail="canvas not found")

    fmt = (body.format or "markdown").lower()
    if fmt != "markdown":
        raise HTTPException(status_code=400, detail="only markdown format is supported")

    # 支持 cite_key_format 参数覆盖配置
    cite_key_format = body.cite_key_format
    if cite_key_format and cite_key_format not in ("numeric", "hash", "author_date"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid cite_key_format: {cite_key_format}, must be one of: numeric, hash, author_date"
        )

    content = export_canvas_markdown(c, cite_key_format=cite_key_format)
    return ExportResponse(format="markdown", content=content, canvas_id=canvas_id, session_id=body.session_id or "")
