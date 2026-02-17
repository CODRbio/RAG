"""
Canvas 导出：Markdown 生成。

支持通过 cite_key_format 参数控制引用格式：
- numeric: [1], [2], [3]
- hash: [a3f7b2c91e04]
- author_date: [Smith2023]
"""

from typing import List, Literal, Optional

from src.collaboration.canvas.models import Citation, OutlineSection, SurveyCanvas
from src.collaboration.citation.formatter import (
    format_bibtex,
    format_reference_list,
    citations_to_markdown_list,
)
from src.collaboration.citation.manager import CiteKeyGenerator


def _sorted_outline(sections: List[OutlineSection]) -> List[OutlineSection]:
    return sorted(sections, key=lambda s: (s.order, s.level, s.title))


def _format_outline_markdown(sections: List[OutlineSection]) -> List[str]:
    lines: List[str] = []
    for s in _sorted_outline(sections):
        indent = "  " * max(s.level - 1, 0)
        title = s.title or "未命名章节"
        lines.append(f"{indent}- {title}")
    return lines


def _section_header(level: int, title: str) -> str:
    lvl = min(max(level, 1), 4)
    return "#" * (lvl + 1) + f" {title}"


def _regenerate_cite_keys(
    citations: List[Citation],
    format: Literal["numeric", "hash", "author_date"],
) -> List[Citation]:
    """重新生成所有 citations 的 cite_key（用于导出时切换格式）。"""
    gen = CiteKeyGenerator(format=format)
    result = []
    for c in citations:
        new_key = gen.generate(c)
        # 创建新的 Citation 对象，保留原有字段但更新 cite_key
        new_citation = Citation(
            id=c.id,
            title=c.title,
            authors=c.authors,
            year=c.year,
            doc_id=c.doc_id,
            url=c.url,
            doi=c.doi,
            bibtex=c.bibtex,
            cite_key=new_key,
            created_at=c.created_at,
        )
        result.append(new_citation)
    return result


def export_canvas_markdown(
    canvas: SurveyCanvas,
    cite_key_format: Optional[str] = None,
    include_bibtex: bool = False,
) -> str:
    """
    导出 Canvas 为 Markdown 格式。

    Args:
        canvas: SurveyCanvas 对象
        cite_key_format: 引用键格式（numeric/hash/author_date），None 则使用原有 cite_key
        include_bibtex: 是否在参考文献后附加 BibTeX

    Returns:
        Markdown 文本
    """
    title = canvas.working_title or canvas.topic or "未命名综述"
    lines: List[str] = [f"# {title}", ""]

    if canvas.abstract:
        lines.extend(["## 摘要", canvas.abstract.strip(), ""])

    if canvas.keywords:
        lines.extend(["## 关键词", ", ".join(canvas.keywords), ""])

    if canvas.outline:
        lines.append("## 大纲")
        lines.extend(_format_outline_markdown(canvas.outline))
        lines.append("")

    if canvas.drafts:
        lines.append("## 正文")
        for s in _sorted_outline(canvas.outline):
            block = canvas.drafts.get(s.id)
            if not block or not block.content_md.strip():
                continue
            lines.append(_section_header(s.level, s.title or "未命名章节"))
            lines.append(block.content_md.strip())
            lines.append("")

    citations = list(canvas.citation_pool.values())
    if citations:
        # 如果指定了格式，重新生成 cite_key
        if cite_key_format and cite_key_format in ("numeric", "hash", "author_date"):
            citations = _regenerate_cite_keys(citations, cite_key_format)

        lines.append("## 参考文献")
        lines.append(format_reference_list(citations, use_cite_key=True).strip())
        lines.append("")

        if include_bibtex:
            lines.append("## BibTeX")
            lines.append("```bibtex")
            lines.append(format_bibtex(citations).strip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines).strip() + "\n"
