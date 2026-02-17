"""引用管理：生成 cite_key、同步到 Canvas、格式化输出。"""

from src.collaboration.citation.manager import (
    CiteKeyGenerator,
    resolve_response_citations,
    sync_evidence_to_canvas,
    chunk_to_citation,
)
from src.collaboration.citation.formatter import (
    format_bibtex,
    format_reference_list,
)

__all__ = [
    "CiteKeyGenerator",
    "resolve_response_citations",
    "sync_evidence_to_canvas",
    "chunk_to_citation",
    "format_bibtex",
    "format_reference_list",
]
