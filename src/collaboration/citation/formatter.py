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


def _format_authors_full(authors: List[str]) -> str:
    """Format author list for reference list (full names, APA-like).

    1 author:  Wang, X.
    2 authors: Wang, X. and Li, Y.
    3+ authors: Wang, X., Li, Y., and Zhang, Z.
    """
    if not authors:
        return "Anonymous"

    def _one(author: str) -> str:
        parts = [p for p in str(author or "").replace(",", " ").split() if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        last = parts[-1]
        initials = " ".join(f"{p[0].upper()}." for p in parts[:-1] if p)
        return f"{last}, {initials}" if initials else last

    formatted = [_one(a) for a in authors if str(a or "").strip()]
    formatted = [a for a in formatted if a]
    if not formatted:
        return "Anonymous"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def format_reference_list(
    citations: List[Citation],
    use_cite_key: bool = True,
    style: Literal["apa", "ieee", "numeric", "custom"] = "custom",
) -> str:
    """Output reference list in academic format.

    Default (custom) style produces APA-like entries:
      [Wang et al., 2011] Wang, X., Li, Y., and Zhang, Z. (2011). Title. URL

    Args:
        citations: citation objects (ordered by first appearance)
        use_cite_key: use cite_key as label; False falls back to numeric
        style: apa | ieee | numeric | custom
    """
    import re as _re

    lines = []
    for i, c in enumerate(citations, 1):
        year_part = f"{c.year}" if c.year else "n.d."
        title_part = c.title or "(Untitled)"
        doi_norm = _normalize_doi(c.doi)
        link_part = c.url or (f"https://doi.org/{doi_norm}" if doi_norm else "")

        if style in ("numeric", "ieee") or not use_cite_key:
            label = str(i)
        else:
            label = c.cite_key or c.id or str(i)

        # Web-only sources (Web1, Web2, ...) — compact format: just title + URL
        is_web_key = bool(_re.fullmatch(r"Web\d+", label, _re.IGNORECASE))
        if is_web_key:
            if link_part:
                lines.append(f"[{label}] {title_part}. {link_part}")
            else:
                lines.append(f"[{label}] {title_part}.")
            continue

        author_full = _format_authors_full(c.authors or [])
        link_suffix = f" {link_part}" if link_part else ""

        if style == "apa":
            lines.append(f"{author_full} ({year_part}). {title_part}.{link_suffix}")
        elif style in ("ieee", "numeric"):
            lines.append(f"[{label}] {author_full}, \"{title_part}\", {year_part}.{link_suffix}")
        else:
            lines.append(f"[{label}] {author_full} ({year_part}). {title_part}.{link_suffix}")

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
    """Format in-text citation marker.

    With the new academic cite_key format (e.g., "Wang et al., 2011"),
    bracket style produces: [Wang et al., 2011]
    parenthetical style produces: (Wang et al., 2011)
    """
    if style == "parenthetical":
        return f"({cite_key})"
    return f"[{cite_key}]"


def citations_to_markdown_list(citations: List[Citation]) -> str:
    """Convert citation list to Markdown bullet format with hyperlinked titles."""
    lines = []
    for c in citations:
        key = c.cite_key or c.id
        author_full = _format_authors_full(c.authors or [])
        year_part = f" ({c.year})" if c.year else ""
        title_part = c.title or "(Untitled)"

        if c.url:
            title_md = f"[{title_part}]({c.url})"
        elif c.doi:
            doi_norm = _normalize_doi(c.doi)
            title_md = f"[{title_part}](https://doi.org/{doi_norm})" if doi_norm else title_part
        else:
            title_md = title_part

        lines.append(f"- **[{key}]** {author_full}{year_part}. {title_md}")

    return "\n".join(lines) if lines else ""
