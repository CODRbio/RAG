"""
自动完成综述 API：POST /auto-complete
"""

from pathlib import Path

from fastapi import APIRouter

from src.api.schemas import AutoCompleteRequest, AutoCompleteResponse
from src.collaboration.auto_complete import AutoCompleteService
from src.llm.llm_manager import get_manager
from src.retrieval.service import get_retrieval_service

router = APIRouter(tags=["auto-complete"])

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"


@router.post("/auto-complete", response_model=AutoCompleteResponse)
def auto_complete(body: AutoCompleteRequest) -> AutoCompleteResponse:
    """
    自动完成综述：根据主题检索 -> 生成大纲 -> 逐章写作 -> 返回完整 Markdown。
    """
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client()
    retrieval = get_retrieval_service(top_k=15)

    svc = AutoCompleteService(
        llm_client=client,
        retrieval_service=retrieval,
        max_sections=body.max_sections,
        include_abstract=True,
    )
    result = svc.complete(
        topic=body.topic,
        canvas_id=body.canvas_id,
        session_id=body.session_id or "",
        search_mode=body.search_mode,
    )
    return AutoCompleteResponse(
        session_id=result.session_id,
        canvas_id=result.canvas_id,
        markdown=result.markdown,
        outline=result.outline,
        citations=result.citations,
        total_time_ms=result.total_time_ms,
    )
