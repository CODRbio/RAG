"""
引用格式化：BibTeX 与参考文献段落。

支持三种 cite_key 格式的输出：
- numeric: [1], [2], [3]
- hash: [a3f7b2c91e04]
- author_date: [Smith2023], [Smith2023a]
"""

from typing import List, Literal, Optional

from src.collaboration.canvas.models import Citation


def format_bibtex(citations: List[Citation]) -> str:
    """
    输出 BibTeX 块。

    使用 Citation 的 cite_key 作为 BibTeX 条目键。
    """
    lines = []
    for c in citations:
        if c.bibtex:
            lines.append(c.bibtex.strip())
            continue
        key = c.cite_key or c.id
        author = " and ".join(c.authors) if c.authors else ""
        title = (c.title or "").replace("{", "{{").replace("}", "}}")
        year = str(c.year) if c.year else ""
        url = c.url or ""
        doi_line = f"\n  doi = {{{c.doi}}}," if c.doi else ""
        lines.append(
            f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{author}}},\n  year = {{{year}}},{doi_line}\n  url = {{{url}}}\n}}"
        )
    return "\n\n".join(lines) if lines else ""


def format_reference_list(
    citations: List[Citation],
    use_cite_key: bool = True,
    style: Literal["apa", "numeric", "custom"] = "custom",
) -> str:
    """
    输出参考文献段落。

    Args:
        citations: 引用列表
        use_cite_key: 是否使用 cite_key 作为标签（否则使用数字序号）
        style: 格式风格
            - "apa": 类 APA 格式
            - "numeric": 数字序号
            - "custom": 自定义格式（使用 cite_key）

    Returns:
        格式化的参考文献文本
    """
    lines = []
    for i, c in enumerate(citations, 1):
        author_part = ", ".join(c.authors) if c.authors else "佚名"
        year_part = f" ({c.year})" if c.year else ""
        title_part = c.title or "(无标题)"
        url_part = f" {c.url}" if c.url else ""
        doi_part = f" https://doi.org/{c.doi}" if c.doi and not c.url else ""

        if style == "numeric" or not use_cite_key:
            label = str(i)
        else:
            label = c.cite_key or c.id or str(i)

        if style == "apa":
            # APA 风格: Author (Year). Title. URL
            lines.append(f"[{label}] {author_part}{year_part}. {title_part}.{url_part or doi_part}")
        else:
            # 自定义风格
            lines.append(f"[{label}] {author_part}{year_part}. {title_part}.{url_part or doi_part}")

    return "\n\n".join(lines) if lines else ""


def _normalize_doi(doi: Optional[str]) -> str:
    value = (doi or "").strip()
    if not value:
        return ""
    lower = value.lower()
    if lower.startswith("https://doi.org/"):
        return value[16:].strip()
    if lower.startswith("http://doi.org/"):
        return value[15:].strip()
    if lower.startswith("doi:"):
        return value[4:].strip()
    return value


def format_ris(citations: List[Citation]) -> str:
    """
    输出 RIS 引文格式。

    字段覆盖：
    - TY (固定 GEN)
    - TI
    - AU (可多行)
    - PY
    - DO
    - UR
    - ER
    """
    records: list[str] = []
    for c in citations:
        lines: list[str] = ["TY  - GEN"]
        title = (c.title or "").strip()
        if title:
            lines.append(f"TI  - {title}")
        for author in (c.authors or []):
            a = str(author or "").strip()
            if a:
                lines.append(f"AU  - {a}")
        if c.year:
            lines.append(f"PY  - {c.year}")
        doi = _normalize_doi(c.doi)
        if doi:
            lines.append(f"DO  - {doi}")
        url = (c.url or "").strip()
        if url:
            lines.append(f"UR  - {url}")
        lines.append("ER  -")
        records.append("\n".join(lines))
    return "\n\n".join(records) + ("\n" if records else "")


def format_inline_citation(cite_key: str, style: Literal["bracket", "parenthetical"] = "bracket") -> str:
    """
    格式化行内引用标记。

    Args:
        cite_key: 引用键
        style: 格式风格
            - "bracket": [Smith2023]
            - "parenthetical": (Smith, 2023)

    Returns:
        格式化的引用标记
    """
    if style == "parenthetical":
        # 尝试解析 author_date 格式
        import re
        match = re.match(r"^([A-Za-z]+)(\d{4})([a-z]?)$", cite_key)
        if match:
            author, year, suffix = match.groups()
            return f"({author}, {year}{suffix})"
        return f"({cite_key})"
    return f"[{cite_key}]"


def citations_to_markdown_list(citations: List[Citation]) -> str:
    """
    将引用列表转换为 Markdown 列表格式。

    Returns:
        Markdown 格式的引用列表
    """
    lines = []
    for c in citations:
        key = c.cite_key or c.id
        author_part = ", ".join(c.authors) if c.authors else "佚名"
        year_part = f" ({c.year})" if c.year else ""
        title_part = c.title or "(无标题)"
        
        # 带链接的标题
        if c.url:
            title_md = f"[{title_part}]({c.url})"
        elif c.doi:
            title_md = f"[{title_part}](https://doi.org/{c.doi})"
        else:
            title_md = title_part
        
        lines.append(f"- **[{key}]** {author_part}{year_part}. {title_md}")
    
    return "\n".join(lines) if lines else ""
