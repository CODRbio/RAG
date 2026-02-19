"""
PDF Parser - EnrichedDoc 完整解析流程

设计原则：
- Docling First：主解析引擎
- Deterministic Before LLM：几何匹配、上下文关联用确定性算法
- Anti-Hallucination：表格摘要必须基于 computed_stats
- Fail-Safe：失败标记 needs_review=True，保留 provenance
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ConfigDict, Field

from src.utils.prompt_manager import PromptManager

_pm = PromptManager()
logger = logging.getLogger(__name__)


class _AnyJSONObject(BaseModel):
    model_config = ConfigDict(extra="allow")


class _FigureEnrichmentResponse(BaseModel):
    figure_type: str = "unknown"
    description: str = ""
    qa_pairs: list[dict[str, Any]] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

# ============================================================
# ENUMS
# ============================================================


class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST = "list"


class LayoutType(str, Enum):
    SINGLE_COLUMN = "single_column"
    DOUBLE_COLUMN = "double_column"
    MIXED = "mixed"


class CaptionMatchMethod(str, Enum):
    GRAVITY_DOWN = "gravity_down"
    GRAVITY_UP = "gravity_up"
    GRAVITY_SIDE = "gravity_side"
    CROSS_PAGE = "cross_page"
    NONE = "none"


class EnrichmentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# SCHEMA (dataclass)
# ============================================================


@dataclass
class SourceMeta:
    filename: str
    file_hash: Optional[str] = None
    num_pages: int = 0


@dataclass
class ParseMeta:
    parser_version: str
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class EnrichmentMeta:
    table_count: int = 0
    figure_count: int = 0
    table_success: int = 0
    figure_success: int = 0
    table_failed: int = 0
    figure_failed: int = 0
    table_skipped: int = 0
    figure_skipped: int = 0


@dataclass
class PageData:
    page_index: int
    width: float
    height: float
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN
    column_boundaries: Optional[list[float]] = None


@dataclass
class CaptionMatch:
    method: CaptionMatchMethod
    score: float
    source_block_id: Optional[str] = None


@dataclass
class TableData:
    markdown: str
    structured: list[list[str]]
    row_count: int
    col_count: int
    title: Optional[str] = None
    footnotes: list[str] = field(default_factory=list)
    computed_stats: Optional[dict[str, dict[str, float]]] = None


@dataclass
class FigureData:
    image_path: str
    image_hash: str
    width_px: int
    height_px: int
    caption: Optional[str] = None
    caption_match: Optional[CaptionMatch] = None
    ocr_text: Optional[list[str]] = None


@dataclass
class FigureInterpretation:
    figure_type: str
    description: str
    qa_pairs: list[dict[str, str]]
    key_findings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class BlockEnrichment:
    status: EnrichmentStatus
    semantic_summary: Optional[str] = None
    interpretation: Optional[FigureInterpretation] = None
    error_message: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    input_hash: Optional[str] = None


@dataclass
class BlockProvenance:
    source: str
    input_hash: Optional[str] = None


@dataclass
class HeadingNode:
    level: int
    text: str
    reading_order: int
    children: list["HeadingNode"] = field(default_factory=list)


@dataclass
class ContentBlock:
    block_id: str
    block_type: BlockType
    page_index: int
    bbox: tuple[float, float, float, float]
    reading_order: int = -1
    column_index: Optional[int] = None
    heading_path: list[str] = field(default_factory=list)
    text: Optional[str] = None
    table_data: Optional[TableData] = None
    figure_data: Optional[FigureData] = None
    enrichment: Optional[BlockEnrichment] = None
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    provenance: Optional[BlockProvenance] = None


@dataclass
class EnrichedDoc:
    doc_id: str
    source: SourceMeta
    global_summary: Optional[str] = None
    pages: list[PageData] = field(default_factory=list)
    content_flow: list[ContentBlock] = field(default_factory=list)
    hierarchy: list[HeadingNode] = field(default_factory=list)
    parse_meta: ParseMeta = field(default_factory=lambda: ParseMeta(parser_version="1.0.0"))
    enrichment_meta: Optional[EnrichmentMeta] = None
    claims: list[dict] = field(default_factory=list)  # ClaimExtractor 提取的核心声明
    doc_metadata: Optional[dict] = None  # DOI / title / authors / year


# ============================================================
# CONFIG
# ============================================================


@dataclass
class ParserConfig:
    parser_version: str = "1.0.0"
    docling_ocr_enabled: bool = False
    coordinate_origin: str = "top_left"
    column_gap_min_ratio: float = 0.02
    column_gap_max_ratio: float = 0.15
    gravity_max_v_down_ratio: float = 0.06
    gravity_min_x_overlap: float = 0.30
    gravity_cross_page_penalty: float = 0.20
    table_title_patterns: list[str] = field(
        default_factory=lambda: [r"^Table\s*\d+", r"^Tab\.\s*\d+", r"^表\s*\d+"]
    )
    table_footnote_patterns: list[str] = field(
        default_factory=lambda: [r"^Note:", r"^\*", r"^注[：:]", r"^Source:"]
    )
    caption_patterns: list[str] = field(
        default_factory=lambda: [r"^(Fig\.|Figure)\s*\d+", r"^图\s*\d+"]
    )
    ocr_enabled: bool = False
    llm_text_provider: str = "deepseek"
    llm_vision_provider: str = "gemini-vision"
    llm_text_model: Optional[str] = None
    llm_vision_model: Optional[str] = None
    llm_text_concurrency: int = 1
    llm_vision_concurrency: int = 1
    llm_text_max_tokens: int = 500
    llm_vision_max_tokens: int = 1024
    llm_json_repair_max_tokens: int = 800
    llm_temperature: float = 0.1
    llm_max_retries: int = 2
    llm_call_timeout_seconds: int = 90
    enrich_tables: bool = True
    enrich_figures: bool = True
    output_dir: str = "./output"
    bottom_page_ratio: float = 0.85
    full_width_ratio: float = 0.60
    table_title_distance_pt: float = 72.0
    table_footnote_distance_pt: float = 48.0

    @classmethod
    def from_json(cls, path: str | Path) -> "ParserConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        p = raw.get("parser", {})
        return cls(
            parser_version=p.get("version", "1.0.0"),
            docling_ocr_enabled=p.get("docling_ocr", False),
            column_gap_min_ratio=p.get("column_gap_min_ratio", 0.02),
            column_gap_max_ratio=p.get("column_gap_max_ratio", 0.15),
            gravity_max_v_down_ratio=p.get("gravity_max_v_down_ratio", 0.06),
            gravity_min_x_overlap=p.get("gravity_min_x_overlap", 0.30),
            gravity_cross_page_penalty=p.get("gravity_cross_page_penalty", 0.20),
            table_title_patterns=p.get("table_title_patterns", [r"^Table\s*\d+", r"^表\s*\d+"]),
            table_footnote_patterns=p.get("table_footnote_patterns", [r"^Note:", r"^\*"]),
            caption_patterns=p.get("caption_patterns", [r"^(Fig\.|Figure)\s*\d+", r"^图\s*\d+"]),
            llm_text_provider=p.get("llm_text_provider", "deepseek"),
            llm_vision_provider=p.get("llm_vision_provider", "gemini-vision"),
            llm_text_model=p.get("llm_text_model"),
            llm_vision_model=p.get("llm_vision_model"),
            llm_text_concurrency=max(1, int(p.get("llm_text_concurrency", 1) or 1)),
            llm_vision_concurrency=max(1, int(p.get("llm_vision_concurrency", 1) or 1)),
            llm_text_max_tokens=int(p.get("llm_text_max_tokens", 500)),
            llm_vision_max_tokens=int(p.get("llm_vision_max_tokens", 1024)),
            llm_json_repair_max_tokens=int(p.get("llm_json_repair_max_tokens", 800)),
            llm_temperature=p.get("llm_temperature", 0.1),
            llm_max_retries=p.get("llm_max_retries", 2),
            llm_call_timeout_seconds=int(p.get("llm_call_timeout_seconds", 90)),
            enrich_tables=bool(p.get("enrich_tables", True)),
            enrich_figures=bool(p.get("enrich_figures", True)),
        )


# ============================================================
# UTILS
# ============================================================


def compute_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_bbox(
    bbox: tuple[float, float, float, float], page_height: float, source_origin: str = "top_left"
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if source_origin == "bottom_left":
        y0, y1 = page_height - y1, page_height - y0
    return (x0, y0, x1, y1)


def parse_numeric(text: str) -> Optional[float]:
    if not text or not isinstance(text, str):
        return None
    text = text.strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100
        except ValueError:
            return None
    if "±" in text:
        text = text.split("±")[0].strip()
    try:
        return float(re.sub(r"[^\d.eE+-]", "", text) or "0")
    except ValueError:
        return None


def compute_table_stats(structured: list[list[str]]) -> dict[str, dict[str, float]]:
    """只处理 >50% 行是数值的列。"""
    result: dict[str, dict[str, float]] = {}
    if not structured or len(structured) < 2:
        return result
    headers = structured[0]
    rows = structured[1:]
    for col_idx, col_name in enumerate(headers):
        col_name = str(col_name or f"col_{col_idx}").strip()
        vals: list[float] = []
        for row in rows:
            if col_idx < len(row):
                v = parse_numeric(row[col_idx])
                if v is not None:
                    vals.append(v)
        if len(vals) >= len(rows) * 0.5 and vals:
            result[col_name] = {
                "min": min(vals),
                "max": max(vals),
                "mean": statistics.mean(vals),
                "n": len(vals),
            }
    return result


def _pattern_match(text: str, patterns: list[str]) -> bool:
    text = str(text or "")
    for pat in patterns:
        if re.search(pat, text.strip(), re.IGNORECASE):
            return True
    return False


# ============================================================
# DoclingParser
# ============================================================


class DoclingParser:
    def __init__(self, config: ParserConfig):
        self.config = config

    def parse(
        self, pdf_path: str | Path
    ) -> tuple[list[ContentBlock], list[PageData], dict[str, Any]]:
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
        except ImportError as e:
            raise ImportError(f"docling 未安装: {e}. pip install docling")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.docling_ocr_enabled
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        doc = result.document

        blocks: list[ContentBlock] = []
        pages: list[PageData] = []
        block_counter = 0

        # 收集页面尺寸
        pages_dict = getattr(doc, "pages", None) or {}
        if isinstance(pages_dict, dict):
            for page_no, page_item in sorted(pages_dict.items(), key=lambda x: x[0]):
                size = getattr(page_item, "size", None)
                if size:
                    w = getattr(size, "width", 612) or 612
                    h = getattr(size, "height", 792) or 792
                else:
                    w, h = 612, 792
                pages.append(PageData(page_index=int(page_no) - 1, width=w, height=h))
        if not pages:
            pages = [PageData(page_index=0, width=612, height=792)]

        def get_bbox_and_page(elem: Any) -> tuple[tuple[float, float, float, float], int]:
            prov = getattr(elem, "prov", None) or []
            if prov:
                p = prov[0] if isinstance(prov, (list, tuple)) else prov
                bbox = getattr(p, "bbox", None)
                page_no = int(getattr(p, "page_no", 1)) - 1
                if bbox:
                    l_ = getattr(bbox, "l", 0) or 0
                    t_ = getattr(bbox, "t", 0) or 0
                    r_ = getattr(bbox, "r", 0) or 0
                    b_ = getattr(bbox, "b", 0) or 0
                    origin = getattr(bbox, "coord_origin", "TOPLEFT") or "TOPLEFT"
                    ph = pages[page_no].height if page_no < len(pages) else 792
                    return normalize_bbox((l_, t_, r_, b_), ph, "bottom_left" if "BOTTOM" in str(origin).upper() else "top_left"), page_no
            return (0, 0, 100, 100), 0

        def make_block_id() -> str:
            nonlocal block_counter
            block_counter += 1
            return f"blk_{block_counter:04d}"

        # Texts
        texts = getattr(doc, "texts", []) or []
        for t in texts:
            label = str(getattr(t, "label", "text")).lower()
            txt = str(getattr(t, "text", "") or "")
            bbox_norm, page_ix = get_bbox_and_page(t)
            bt = BlockType.TEXT
            if "section" in label or "heading" in label or "title" in label:
                bt = BlockType.HEADING
            elif label == "caption":
                bt = BlockType.CAPTION
            elif label == "footnote":
                bt = BlockType.FOOTNOTE
            elif "list" in label:
                bt = BlockType.LIST
            blocks.append(
                ContentBlock(
                    block_id=make_block_id(),
                    block_type=bt,
                    page_index=page_ix,
                    bbox=bbox_norm,
                    reading_order=-1,
                    text=txt,
                    provenance=BlockProvenance(source="docling"),
                )
            )

        # Tables
        tables = getattr(doc, "tables", []) or []
        for tbl in tables:
            bbox_norm, page_ix = get_bbox_and_page(tbl)
            md = ""
            structured: list[list[str]] = []
            try:
                if hasattr(tbl, "export_to_dataframe"):
                    df = tbl.export_to_dataframe(doc=doc)
                    if df is not None and not df.empty:
                        md = df.to_markdown() if hasattr(df, "to_markdown") else str(df)
                        cols = df.columns.tolist()
                        if cols and isinstance(cols[0], tuple):
                            cols = [" ".join(str(x) for x in c) for c in cols]
                        cols = [str(c) for c in cols]
                        structured = [cols]
                        for _, row in df.iterrows():
                            structured.append([str(v) if v is not None and str(v) != "nan" else "" for v in row.tolist()])
                if not md and hasattr(tbl, "export_to_markdown"):
                    try:
                        md = tbl.export_to_markdown(doc=doc) or ""
                    except TypeError:
                        md = tbl.export_to_markdown() or ""
            except Exception:
                pass
            if not structured and md:
                lines = [ln for ln in md.split("\n") if "|" in ln]
                for ln in lines:
                    cells = [c.strip() for c in ln.split("|") if c.strip() and c.strip() != "---"]
                    if cells:
                        structured.append(cells)
            row_count = len(structured)
            col_count = max(len(r) for r in structured) if structured else 0
            td = TableData(
                markdown=md,
                structured=structured,
                row_count=row_count,
                col_count=col_count,
                computed_stats=compute_table_stats(structured),
            )
            blocks.append(
                ContentBlock(
                    block_id=make_block_id(),
                    block_type=BlockType.TABLE,
                    page_index=page_ix,
                    bbox=bbox_norm,
                    reading_order=-1,
                    table_data=td,
                    provenance=BlockProvenance(source="docling"),
                )
            )

        # Pictures
        pictures = getattr(doc, "pictures", []) or []
        for pic in pictures:
            bbox_norm, page_ix = get_bbox_and_page(pic)
            blocks.append(
                ContentBlock(
                    block_id=make_block_id(),
                    block_type=BlockType.FIGURE,
                    page_index=page_ix,
                    bbox=bbox_norm,
                    reading_order=-1,
                    figure_data=FigureData(
                        image_path="",
                        image_hash="",
                        width_px=0,
                        height_px=0,
                    ),
                    provenance=BlockProvenance(source="docling"),
                )
            )

        raw_output = {"text_count": len(texts), "table_count": len(tables), "picture_count": len(pictures)}
        return blocks, pages, raw_output


# ============================================================
# LayoutAnalyzer
# ============================================================


class LayoutAnalyzer:
    def __init__(self, config: ParserConfig):
        self.config = config

    def analyze(
        self, blocks: list[ContentBlock], pages: list[PageData]
    ) -> tuple[list[ContentBlock], list[PageData]]:
        # 按页分组
        by_page: dict[int, list[ContentBlock]] = {}
        for b in blocks:
            by_page.setdefault(b.page_index, []).append(b)

        for page_ix, page in enumerate(pages):
            if page_ix not in by_page:
                continue
            page_blocks = by_page[page_ix]
            page_width = page.width
            page_height = page.height

            # 双栏检测：只用 TEXT/HEADING
            text_blocks = [b for b in page_blocks if b.block_type in (BlockType.TEXT, BlockType.HEADING)]
            if len(text_blocks) >= 4:
                centers_x = sorted([(b.bbox[0] + b.bbox[2]) / 2 for b in text_blocks])
                max_gap, gap_pos = 0, 0
                for i in range(1, len(centers_x)):
                    gap = centers_x[i] - centers_x[i - 1]
                    if gap > max_gap:
                        max_gap, gap_pos = gap, (centers_x[i] + centers_x[i - 1]) / 2
                gap_ratio = max_gap / page_width if page_width else 0
                pos_ratio = gap_pos / page_width if page_width else 0.5
                if (
                    self.config.column_gap_min_ratio < gap_ratio < self.config.column_gap_max_ratio
                    and 0.35 < pos_ratio < 0.65
                ):
                    page.layout_type = LayoutType.DOUBLE_COLUMN
                    page.column_boundaries = [gap_pos]

            # 分配 column_index
            col_bound = page.column_boundaries
            full_width = page_width * self.config.full_width_ratio
            for b in page_blocks:
                w = b.bbox[2] - b.bbox[0]
                if page.layout_type == LayoutType.DOUBLE_COLUMN and col_bound and w < full_width:
                    cx = (b.bbox[0] + b.bbox[2]) / 2
                    b.column_index = 0 if cx < col_bound[0] else 1
                else:
                    b.column_index = 0

        # 阅读顺序
        def key_fn(b: ContentBlock):
            p = pages[b.page_index] if b.page_index < len(pages) else pages[0]
            col = b.column_index or 0
            return (b.page_index, col, b.bbox[1], b.bbox[0])

        sorted_blocks = sorted(blocks, key=key_fn)
        for i, b in enumerate(sorted_blocks):
            b.reading_order = i

        return sorted_blocks, pages


# ============================================================
# ContextLinker
# ============================================================


class ContextLinker:
    def __init__(self, config: ParserConfig):
        self.config = config

    def link_all(
        self, blocks: list[ContentBlock], pages: list[PageData]
    ) -> tuple[list[ContentBlock], list[HeadingNode]]:
        blocks = self._build_hierarchy(blocks)
        blocks = self._link_table_context(blocks, pages)
        blocks = self._link_figure_caption(blocks, pages)
        hierarchy = self._extract_hierarchy(blocks)
        return blocks, hierarchy

    def _build_hierarchy(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        headings = [(b, i) for i, b in enumerate(blocks) if b.block_type == BlockType.HEADING]
        headings_sorted = sorted(headings, key=lambda x: x[0].reading_order)

        def level_from_text(t: str) -> int:
            m = re.match(r"^(\d+)(\.\d+)*", (t or "").strip())
            if m:
                return len(m.group(0).split("."))
            return 1

        current_path: list[str] = []
        current_levels: list[int] = []
        heading_idx = 0

        for b in sorted(blocks, key=lambda x: x.reading_order):
            while heading_idx < len(headings_sorted) and headings_sorted[heading_idx][0].reading_order <= b.reading_order:
                h = headings_sorted[heading_idx][0]
                lv = level_from_text(h.text or "")
                while current_levels and current_levels[-1] >= lv:
                    current_path.pop()
                    current_levels.pop()
                current_path.append((h.text or "").strip())
                current_levels.append(lv)
                heading_idx += 1
            b.heading_path = current_path.copy()
        return blocks

    def _extract_hierarchy(self, blocks: list[ContentBlock]) -> list[HeadingNode]:
        headings = [b for b in blocks if b.block_type == BlockType.HEADING]
        headings = sorted(headings, key=lambda x: x.reading_order)

        def level_from_text(t: str) -> int:
            m = re.match(r"^(\d+)(\.\d+)*", (t or "").strip())
            if m:
                return len(m.group(0).split("."))
            return 1

        nodes: list[HeadingNode] = []
        stack: list[tuple[HeadingNode, int]] = []
        for h in headings:
            lv = level_from_text(h.text or "")
            node = HeadingNode(level=lv, text=(h.text or "").strip(), reading_order=h.reading_order)
            while stack and stack[-1][1] >= lv:
                stack.pop()
            if stack:
                stack[-1][0].children.append(node)
            else:
                nodes.append(node)
            stack.append((node, lv))
        return nodes

    def _link_table_context(
        self, blocks: list[ContentBlock], pages: list[PageData]
    ) -> list[ContentBlock]:
        for b in blocks:
            if b.block_type != BlockType.TABLE or not b.table_data:
                continue
            page = pages[b.page_index] if b.page_index < len(pages) else None
            if not page:
                continue
            page_h = page.height
            x0, y0, x1, y1 = b.bbox
            dist_pt = self.config.table_title_distance_pt
            fn_dist = self.config.table_footnote_distance_pt

            # Title 向上
            candidates_above: list[tuple[ContentBlock, float]] = []
            for other in blocks:
                if other.block_type not in (BlockType.TEXT, BlockType.CAPTION) or not other.text:
                    continue
                if other.page_index != b.page_index:
                    continue
                oy0 = other.bbox[1]
                oy1 = other.bbox[3]
                if oy1 <= y0 and y0 - oy1 <= dist_pt:
                    candidates_above.append((other, y0 - oy1))
            candidates_above.sort(key=lambda x: x[1])
            for other, _ in candidates_above:
                if _pattern_match(other.text, self.config.table_title_patterns):
                    b.table_data.title = other.text.strip()
                    break

            # Footnote 向下
            footnotes: list[str] = []
            for other in blocks:
                if other.block_type != BlockType.TEXT or not other.text:
                    continue
                if other.page_index != b.page_index:
                    continue
                oy0 = other.bbox[1]
                if oy0 >= y1 and oy0 - y1 <= fn_dist:
                    if _pattern_match(other.text, self.config.table_footnote_patterns):
                        footnotes.append(other.text.strip())
            if footnotes:
                b.table_data.footnotes = footnotes

            # 跨页脚注
            if not footnotes and y1 > page_h * self.config.bottom_page_ratio:
                next_page_ix = b.page_index + 1
                if next_page_ix < len(pages):
                    top_blocks = [
                        o for o in blocks
                        if o.page_index == next_page_ix and o.block_type == BlockType.TEXT and o.text
                    ][:10]
                    for o in top_blocks:
                        if _pattern_match(o.text, self.config.table_footnote_patterns):
                            b.table_data.footnotes.append(o.text.strip())
                            break
        return blocks

    def _link_figure_caption(
        self, blocks: list[ContentBlock], pages: list[PageData]
    ) -> list[ContentBlock]:
        picture_blocks = [b for b in blocks if b.block_type == BlockType.FIGURE and b.figure_data]
        caption_candidates: list[ContentBlock] = [
            b for b in blocks
            if (
                b.block_type == BlockType.CAPTION
                or _pattern_match(b.text or "", self.config.caption_patterns)
                or (10 <= len((b.text or "").strip()) <= 600 and b.block_type == BlockType.TEXT)
            )
            and b.text
        ]
        if not picture_blocks:
            return blocks

        # 评分并为每个 picture 选择最佳 caption
        assignments: dict[str, tuple[ContentBlock, float, CaptionMatchMethod]] = {}
        for cap in caption_candidates:
            page = pages[cap.page_index] if cap.page_index < len(pages) else None
            if not page:
                continue
            page_h = page.height
            page_w = page.width
            cx0, cy0, cx1, cy1 = cap.bbox
            cap_w = cx1 - cx0
            cap_col = cap.column_index

            max_v_down = min(72, page_h * self.config.gravity_max_v_down_ratio)
            max_v_up = min(48, page_h * 0.04)

            for pic in picture_blocks:
                if pic.page_index != cap.page_index:
                    continue
                fx0, fy0, fx1, fy1 = pic.bbox
                fig_w = fx1 - fx0
                is_full_width = fig_w > page_w * self.config.full_width_ratio

                # 同栏约束（跨栏图例外）
                if not is_full_width and cap_col is not None and pic.column_index is not None:
                    if cap_col != pic.column_index:
                        continue

                # caption 在图下方（更常见）
                if cy0 >= fy1:
                    v_dist = cy0 - fy1
                    if v_dist > max_v_down:
                        continue
                    method = CaptionMatchMethod.GRAVITY_DOWN
                    penalty = 0.0
                # caption 在图上方
                elif fy0 >= cy1:
                    v_dist = fy0 - cy1
                    if v_dist > max_v_up:
                        continue
                    method = CaptionMatchMethod.GRAVITY_UP
                    penalty = 0.15
                else:
                    continue

                x_overlap = max(0, min(fx1, cx1) - max(fx0, cx0))
                x_overlap_ratio = x_overlap / min(fig_w, cap_w) if min(fig_w, cap_w) > 0 else 0
                if x_overlap_ratio < self.config.gravity_min_x_overlap:
                    continue

                score = 1.2 * x_overlap_ratio - 0.008 * v_dist - penalty
                if pic.block_id not in assignments or score > assignments[pic.block_id][1]:
                    assignments[pic.block_id] = (cap, score, method)

        # 按 caption 聚合图片块
        cap_to_pics: dict[str, list[ContentBlock]] = {}
        cap_scores: dict[str, tuple[float, CaptionMatchMethod]] = {}
        for pic in picture_blocks:
            if pic.block_id not in assignments:
                pic.needs_review = True
                pic.review_reasons.append("caption_not_matched")
                continue
            cap, score, method = assignments[pic.block_id]
            cap_to_pics.setdefault(cap.block_id, []).append(pic)
            prev = cap_scores.get(cap.block_id)
            if not prev or score > prev[0]:
                cap_scores[cap.block_id] = (score, method)

        merged_blocks: list[ContentBlock] = []
        assigned_picture_ids = set(assignments.keys())

        for cap in caption_candidates:
            pics = cap_to_pics.get(cap.block_id, [])
            if not pics:
                continue
            x0 = min(p.bbox[0] for p in pics)
            y0 = min(p.bbox[1] for p in pics)
            x1 = max(p.bbox[2] for p in pics)
            y1 = max(p.bbox[3] for p in pics)
            reading_order = min(p.reading_order for p in pics)
            score, method = cap_scores.get(cap.block_id, (0.0, CaptionMatchMethod.NONE))

            merged_blocks.append(
                ContentBlock(
                    block_id=f"{cap.block_id}_fig",
                    block_type=BlockType.FIGURE,
                    page_index=cap.page_index,
                    bbox=(x0, y0, x1, y1),
                    reading_order=reading_order,
                    column_index=cap.column_index,
                    heading_path=cap.heading_path.copy(),
                    figure_data=FigureData(
                        image_path="",
                        image_hash="",
                        width_px=0,
                        height_px=0,
                        caption=str(cap.text or "").strip(),
                        caption_match=CaptionMatch(
                            method=method,
                            score=score,
                            source_block_id=cap.block_id,
                        ),
                    ),
                    provenance=BlockProvenance(source="docling"),
                )
            )

        # 移除已归并的碎片图块，保留未匹配的图块
        remaining_blocks = [b for b in blocks if b.block_id not in assigned_picture_ids]
        remaining_blocks.extend(merged_blocks)
        remaining_blocks.sort(key=lambda b: b.reading_order)
        return remaining_blocks


# ============================================================
# FigureExtractor
# ============================================================


class FigureExtractor:
    def __init__(self, config: ParserConfig):
        self.config = config

    def extract_figures(
        self,
        pdf_path: str | Path,
        figure_blocks: list[ContentBlock],
        output_dir: str | Path,
    ) -> list[ContentBlock]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            for b in figure_blocks:
                if b.figure_data:
                    b.needs_review = True
                    b.review_reasons.append("pymupdf_not_installed")
            return figure_blocks

        output_dir = Path(output_dir)
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        for i, block in enumerate(figure_blocks):
            if block.block_type != BlockType.FIGURE or not block.figure_data:
                continue
            fd = block.figure_data
            page = doc[block.page_index]
            x0, y0, x1, y1 = block.bbox
            # PyMuPDF 坐标系为左上原点，直接使用归一化后的 bbox
            rect = fitz.Rect(x0, y0, x1, y1)
            rect = rect & page.rect
            try:
                pix = page.get_pixmap(clip=rect, dpi=150)
                img_path = assets_dir / f"fig_{i + 1:03d}.png"
                pix.save(str(img_path))
                with open(img_path, "rb") as f:
                    fd.image_hash = compute_hash(f.read())
                fd.image_path = f"assets/{img_path.name}"
                fd.width_px = pix.width
                fd.height_px = pix.height
            except Exception:
                block.needs_review = True
                block.review_reasons.append("figure_extract_failed")
        doc.close()
        return figure_blocks


# ============================================================
# LLMEnricher
# ============================================================

FIGURE_JSON_SCHEMA = """{
  "figure_type": "line_plot|bar_chart|diagram|photo|map|multi-panel figure|...",
  "description": "1-2 sentences for overall summary",
  "key_findings": ["finding1", "finding2", "finding3"],
  "evidence": ["panel/legend/axis cues supporting findings", "..."],
  "qa_pairs": []
}"""


class LLMEnricher:
    def __init__(self, config: ParserConfig, llm_manager: Any):
        self.config = config
        self.llm_manager = llm_manager
        self._text_client = None
        self._vision_client = None

    def _get_text_client(self, fresh: bool = False):
        if fresh:
            return self.llm_manager.get_client(self.config.llm_text_provider)
        if self._text_client is None:
            self._text_client = self.llm_manager.get_client(self.config.llm_text_provider)
        return self._text_client

    def _get_vision_client(self, fresh: bool = False):
        if fresh:
            return self.llm_manager.get_client(self.config.llm_vision_provider)
        if self._vision_client is None:
            self._vision_client = self.llm_manager.get_client(self.config.llm_vision_provider)
        return self._vision_client

    def _resolve_model(self, provider: str, model_override: Optional[str] = None) -> Optional[str]:
        if not self.llm_manager or not hasattr(self.llm_manager, "resolve_model"):
            return None
        try:
            return self.llm_manager.resolve_model(provider, model_override)
        except Exception:
            return None

    def _repair_json(self, raw_output: str, schema: str) -> Optional[dict]:
        if not raw_output:
            return None
        prompt = _pm.render("pdf_json_repair.txt", schema=schema, raw_output=raw_output)
        try:
            client = self._get_text_client(fresh=True)
            resp = client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("pdf_json_repair_system.txt")},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm_json_repair_max_tokens,
                temperature=0.0,
                timeout_seconds=self.config.llm_call_timeout_seconds,
                response_model=_AnyJSONObject,
            )
            parsed: Optional[_AnyJSONObject] = resp.get("parsed_object")
            if parsed is None:
                fixed = (resp.get("final_text") or "").strip()
                if not fixed:
                    return None
                parsed = _AnyJSONObject.model_validate_json(fixed)
            return parsed.model_dump()
        except Exception:
            return None

    def enrich_all(
        self,
        blocks: list[ContentBlock],
        global_context: str = "",
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> tuple[list[ContentBlock], EnrichmentMeta]:
        meta = EnrichmentMeta()
        doc_id = Path(global_context).name if global_context else "?"
        n_tables = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
        n_figures = sum(
            1
            for b in blocks
            if b.block_type == BlockType.FIGURE and b.figure_data and b.figure_data.image_path
        )
        table_indices: list[int] = []
        figure_indices: list[int] = []
        for idx, b in enumerate(blocks):
            if b.block_type == BlockType.TABLE:
                meta.table_count += 1
                if self.config.enrich_tables:
                    table_indices.append(idx)
                else:
                    b.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.SKIPPED, error_message="enrich_tables_disabled"
                    )
                    table_ix = len(table_indices) + meta.table_skipped + 1
                    logger.info("[%s] table %d/%d skip: enrich_tables_disabled", doc_id, table_ix, n_tables)
                    if progress_callback:
                        progress_callback(
                            "enrich_table",
                            {
                                "index": table_ix,
                                "total": n_tables,
                                "status": "skip",
                                "message": "enrich_tables_disabled",
                            },
                        )
                    blocks[idx] = b
                    meta.table_skipped += 1
            elif b.block_type == BlockType.FIGURE and b.figure_data and b.figure_data.image_path:
                meta.figure_count += 1
                if self.config.enrich_figures:
                    figure_indices.append(idx)
                else:
                    b.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.SKIPPED, error_message="enrich_figures_disabled"
                    )
                    blocks[idx] = b
                    meta.figure_skipped += 1
                    logger.info("[%s] figure skip: enrich_figures_disabled", doc_id)

        if table_indices:
            max_workers = max(1, int(self.config.llm_text_concurrency or 1))
            actual_workers = min(max_workers, len(table_indices))
            if actual_workers <= 1 or len(table_indices) <= 1:
                for table_ix, idx in enumerate(table_indices, 1):
                    b = self.enrich_table(
                        blocks[idx],
                        global_context,
                        use_fresh_client=False,
                        table_index=table_ix,
                        table_total=len(table_indices),
                        progress_callback=progress_callback,
                    )
                    blocks[idx] = b
                    if b.enrichment and b.enrichment.status == EnrichmentStatus.SUCCESS:
                        meta.table_success += 1
                    elif b.enrichment and b.enrichment.status == EnrichmentStatus.SKIPPED:
                        meta.table_skipped += 1
                    else:
                        meta.table_failed += 1
            else:
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(
                            self.enrich_table,
                            blocks[idx],
                            global_context,
                            True,
                            table_ix,
                            len(table_indices),
                            progress_callback,
                        ): idx
                        for table_ix, idx in enumerate(table_indices, 1)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            b = future.result()
                        except Exception as e:
                            b = blocks[idx]
                            b.enrichment = BlockEnrichment(
                                status=EnrichmentStatus.FAILED,
                                error_message=str(e),
                                provider=self.config.llm_text_provider,
                                model=self._resolve_model(
                                    self.config.llm_text_provider, self.config.llm_text_model
                                ),
                            )
                            b.needs_review = True
                            b.review_reasons.append("llm_table_enrich_failed")
                        blocks[idx] = b
                        if b.enrichment and b.enrichment.status == EnrichmentStatus.SUCCESS:
                            meta.table_success += 1
                        elif b.enrichment and b.enrichment.status == EnrichmentStatus.SKIPPED:
                            meta.table_skipped += 1
                        else:
                            meta.table_failed += 1

        if figure_indices:
            max_workers = max(1, int(self.config.llm_vision_concurrency or 1))
            actual_workers = min(max_workers, len(figure_indices))
            if actual_workers <= 1 or len(figure_indices) <= 1:
                for fig_ix, idx in enumerate(figure_indices, 1):
                    b = self.enrich_figure(
                        blocks[idx],
                        global_context,
                        use_fresh_client=False,
                        figure_index=fig_ix,
                        figure_total=len(figure_indices),
                        progress_callback=progress_callback,
                    )
                    blocks[idx] = b
                    if b.enrichment and b.enrichment.status == EnrichmentStatus.SUCCESS:
                        meta.figure_success += 1
                    elif b.enrichment and b.enrichment.status == EnrichmentStatus.SKIPPED:
                        meta.figure_skipped += 1
                    else:
                        meta.figure_failed += 1
            else:
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(
                            self.enrich_figure,
                            blocks[idx],
                            global_context,
                            True,
                            fig_ix,
                            len(figure_indices),
                            progress_callback,
                        ): idx
                        for fig_ix, idx in enumerate(figure_indices, 1)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            b = future.result()
                        except Exception as e:
                            b = blocks[idx]
                            b.enrichment = BlockEnrichment(
                                status=EnrichmentStatus.FAILED,
                                error_message=str(e),
                                provider=self.config.llm_vision_provider,
                                model=self._resolve_model(
                                    self.config.llm_vision_provider, self.config.llm_vision_model
                                ),
                            )
                            b.needs_review = True
                            b.review_reasons.append("llm_figure_enrich_failed")
                        blocks[idx] = b
                        if b.enrichment and b.enrichment.status == EnrichmentStatus.SUCCESS:
                            meta.figure_success += 1
                        elif b.enrichment and b.enrichment.status == EnrichmentStatus.SKIPPED:
                            meta.figure_skipped += 1
                        else:
                            meta.figure_failed += 1
        logger.info(
            "[%s] enrich_all done: tables total=%d success=%d failed=%d skipped=%d; figures total=%d success=%d failed=%d skipped=%d",
            doc_id,
            meta.table_count,
            meta.table_success,
            meta.table_failed,
            meta.table_skipped,
            meta.figure_count,
            meta.figure_success,
            meta.figure_failed,
            meta.figure_skipped,
        )
        return blocks, meta

    def enrich_table(
        self,
        block: ContentBlock,
        global_context: str = "",
        use_fresh_client: bool = False,
        table_index: Optional[int] = None,
        table_total: Optional[int] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> ContentBlock:
        doc_id = Path(global_context).name if global_context else "?"
        idx_label = f"{table_index or '?'}/{table_total or '?'}"
        if not block.table_data:
            block.enrichment = BlockEnrichment(status=EnrichmentStatus.SKIPPED, error_message="no_table_data")
            logger.info("[%s] table %s skip: no_table_data", doc_id, idx_label)
            if progress_callback:
                progress_callback("enrich_table", {"index": table_index, "total": table_total, "status": "skip", "message": "no_table_data"})
            return block
        td = block.table_data
        if not td.computed_stats:
            td.computed_stats = compute_table_stats(td.structured)
        prompt = _pm.render(
            "pdf_table_summary.txt",
            heading_path=" > ".join(block.heading_path) if block.heading_path else "(root)",
            table_title=td.title or "(no title)",
            table_markdown=td.markdown[:3000] if td.markdown else "",
            computed_stats=json.dumps(td.computed_stats, ensure_ascii=False, indent=2),
        )
        provider = self.config.llm_text_provider
        model_override = self.config.llm_text_model
        model_resolved = self._resolve_model(provider, model_override)
        logger.info(
            "[%s] table %s start provider=%s model=%s",
            doc_id,
            idx_label,
            provider,
            model_resolved,
        )
        if progress_callback:
            progress_callback(
                "enrich_table",
                {
                    "index": table_index,
                    "total": table_total,
                    "status": "start",
                    "message": f"{provider}/{model_resolved or ''}".strip("/"),
                },
            )
        prompt_hash = compute_hash(prompt)
        input_hash = compute_hash((td.markdown or "") + json.dumps(td.computed_stats or {}, ensure_ascii=False))
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                client = self._get_text_client(fresh=use_fresh_client)
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": "You are a senior data analyst. Output only the requested content."},
                        {"role": "user", "content": prompt},
                    ],
                    model=model_override,
                    max_tokens=self.config.llm_text_max_tokens,
                    temperature=self.config.llm_temperature,
                    timeout_seconds=self.config.llm_call_timeout_seconds,
                )
                text = (resp.get("final_text") or "").strip()
                if text:
                    provider_used = resp.get("provider") or provider
                    model_used = resp.get("model") or model_resolved
                    block.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.SUCCESS,
                        semantic_summary=text,
                        provider=provider_used,
                        model=model_used,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )
                    logger.info("[%s] table %s success", doc_id, idx_label)
                    if progress_callback:
                        progress_callback("enrich_table", {"index": table_index, "total": table_total, "status": "success"})
                    return block
            except Exception as e:
                if attempt == self.config.llm_max_retries:
                    block.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.FAILED,
                        error_message=str(e),
                        provider=provider,
                        model=model_resolved,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )
                    block.needs_review = True
                    block.review_reasons.append("llm_table_enrich_failed")
                    logger.info("[%s] table %s failed: %s", doc_id, idx_label, str(e)[:200])
                    if progress_callback:
                        progress_callback("enrich_table", {"index": table_index, "total": table_total, "status": "fail", "message": str(e)[:200]})
                    return block
        block.enrichment = block.enrichment or BlockEnrichment(
            status=EnrichmentStatus.FAILED,
            error_message="max_retries",
            provider=provider,
            model=model_resolved,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
        )
        logger.info("[%s] table %s failed: max_retries", doc_id, idx_label)
        if progress_callback:
            progress_callback("enrich_table", {"index": table_index, "total": table_total, "status": "fail", "message": "max_retries"})
        return block

    def enrich_figure(
        self,
        block: ContentBlock,
        global_context: str = "",
        use_fresh_client: bool = False,
        figure_index: Optional[int] = None,
        figure_total: Optional[int] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> ContentBlock:
        doc_id = Path(global_context).name if global_context else "?"
        idx_label = f"{figure_index or '?'}/{figure_total or '?'}"
        if not block.figure_data or not block.figure_data.image_path:
            block.enrichment = BlockEnrichment(status=EnrichmentStatus.SKIPPED, error_message="no_image")
            logger.info("[%s] figure %s skip: no_image", doc_id, idx_label)
            if progress_callback:
                progress_callback("enrich_figure", {"index": figure_index, "total": figure_total, "status": "skip", "message": "no_image"})
            return block
        fd = block.figure_data
        # 图片路径可能是相对于 output_dir 的
        img_full = Path(global_context) / fd.image_path if global_context else Path(fd.image_path)
        if not img_full.exists():
            # 尝试 assets 子路径
            img_full = Path(global_context) / "assets" / Path(fd.image_path).name if global_context else Path(fd.image_path)
        if not img_full.exists():
            block.enrichment = BlockEnrichment(status=EnrichmentStatus.FAILED, error_message="image_not_found")
            block.needs_review = True
            block.review_reasons.append("image_not_found")
            logger.info("[%s] figure %s failed: image_not_found", doc_id, idx_label)
            if progress_callback:
                progress_callback("enrich_figure", {"index": figure_index, "total": figure_total, "status": "fail", "message": "image_not_found"})
            return block

        with open(img_full, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt = _pm.render(
            "pdf_figure_interpret.txt",
            heading_path=" > ".join(block.heading_path) if block.heading_path else "(root)",
            caption=fd.caption or "(no caption)",
        )
        provider = self.config.llm_vision_provider
        model_override = self.config.llm_vision_model
        model_resolved = self._resolve_model(provider, model_override)
        logger.info(
            "[%s] figure %s start provider=%s model=%s",
            doc_id,
            idx_label,
            provider,
            model_resolved,
        )
        if progress_callback:
            progress_callback(
                "enrich_figure",
                {
                    "index": figure_index,
                    "total": figure_total,
                    "status": "start",
                    "message": f"{provider}/{model_resolved or ''}".strip("/"),
                },
            )
        prompt_hash = compute_hash(prompt)
        input_hash = compute_hash((fd.image_hash or "") + (fd.caption or ""))
        # Vision API 通常接受 content 为 list，包含 image_url 或 type=image_url
        user_content: list[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

        for attempt in range(self.config.llm_max_retries + 1):
            try:
                client = self._get_vision_client(fresh=use_fresh_client)
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": "You are a scientific visualization expert. Return ONLY valid JSON."},
                        {"role": "user", "content": user_content},
                    ],
                    model=model_override,
                    max_tokens=self.config.llm_vision_max_tokens,
                    temperature=self.config.llm_temperature,
                    timeout_seconds=self.config.llm_call_timeout_seconds,
                    response_model=_FigureEnrichmentResponse,
                )
                parsed: Optional[_FigureEnrichmentResponse] = resp.get("parsed_object")
                text = (resp.get("final_text") or "").strip()
                if parsed is None and text:
                    try:
                        parsed = _FigureEnrichmentResponse.model_validate_json(text)
                    except Exception:
                        parsed = None
                if parsed is None:
                    repaired = self._repair_json(text, FIGURE_JSON_SCHEMA)
                    if repaired:
                        parsed = _FigureEnrichmentResponse.model_validate(repaired)
                if parsed:
                    interp = FigureInterpretation(
                        figure_type=parsed.figure_type,
                        description=parsed.description,
                        qa_pairs=parsed.qa_pairs[:3],
                        key_findings=parsed.key_findings[:5],
                        evidence=parsed.evidence[:5],
                    )
                    provider_used = resp.get("provider") or provider
                    model_used = resp.get("model") or model_resolved
                    block.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.SUCCESS,
                        interpretation=interp,
                        provider=provider_used,
                        model=model_used,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )
                    logger.info("[%s] figure %s success", doc_id, idx_label)
                    if progress_callback:
                        progress_callback("enrich_figure", {"index": figure_index, "total": figure_total, "status": "success"})
                    return block
            except Exception as e:
                if attempt == self.config.llm_max_retries:
                    block.enrichment = BlockEnrichment(
                        status=EnrichmentStatus.FAILED,
                        error_message=str(e),
                        provider=provider,
                        model=model_resolved,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )
                    block.needs_review = True
                    block.review_reasons.append("llm_figure_enrich_failed")
                    logger.info("[%s] figure %s failed: %s", doc_id, idx_label, str(e)[:200])
                    if progress_callback:
                        progress_callback("enrich_figure", {"index": figure_index, "total": figure_total, "status": "fail", "message": str(e)[:200]})
                    return block
        block.enrichment = block.enrichment or BlockEnrichment(
            status=EnrichmentStatus.FAILED,
            error_message="max_retries",
            provider=provider,
            model=model_resolved,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
        )
        logger.info("[%s] figure %s failed: max_retries", doc_id, idx_label)
        if progress_callback:
            progress_callback("enrich_figure", {"index": figure_index, "total": figure_total, "status": "fail", "message": "max_retries"})
        return block


# ============================================================
# PDFProcessor
# ============================================================


def _serialize_obj(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_serialize_obj(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, dict, list, int, float, bool, type(None))):
        return {k: _serialize_obj(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_obj(x) for x in obj]
    return obj


def _deserialize_enriched(raw: dict) -> EnrichedDoc:
    def to_block(d: dict) -> ContentBlock:
        bt = BlockType(d.get("block_type", "text"))
        td = None
        if d.get("table_data"):
            tdd = d["table_data"]
            td = TableData(
                markdown=tdd.get("markdown", ""),
                structured=tdd.get("structured", []),
                row_count=tdd.get("row_count", 0),
                col_count=tdd.get("col_count", 0),
                title=tdd.get("title"),
                footnotes=tdd.get("footnotes", []),
                computed_stats=tdd.get("computed_stats"),
            )
        fd = None
        if d.get("figure_data"):
            fdd = d["figure_data"]
            fd = FigureData(
                image_path=fdd.get("image_path", ""),
                image_hash=fdd.get("image_hash", ""),
                width_px=fdd.get("width_px", 0),
                height_px=fdd.get("height_px", 0),
                caption=fdd.get("caption"),
                ocr_text=fdd.get("ocr_text"),
            )
        interp = None
        if d.get("enrichment") and d["enrichment"].get("interpretation"):
            it = d["enrichment"]["interpretation"]
            interp = FigureInterpretation(
                figure_type=it.get("figure_type", "unknown"),
                description=it.get("description", ""),
                qa_pairs=it.get("qa_pairs", [])[:3],
                key_findings=it.get("key_findings", [])[:5],
                evidence=it.get("evidence", [])[:5],
            )
        enrichment = None
        if d.get("enrichment"):
            status_val = d["enrichment"].get("status")
            try:
                status_enum = EnrichmentStatus(status_val) if status_val else EnrichmentStatus.SKIPPED
            except Exception:
                status_enum = EnrichmentStatus.SKIPPED
            enrichment = BlockEnrichment(
                status=status_enum,
                semantic_summary=d["enrichment"].get("semantic_summary"),
                interpretation=interp,
                error_message=d["enrichment"].get("error_message"),
                provider=d["enrichment"].get("provider"),
                model=d["enrichment"].get("model"),
                prompt_hash=d["enrichment"].get("prompt_hash"),
                input_hash=d["enrichment"].get("input_hash"),
            )
        return ContentBlock(
            block_id=d.get("block_id", ""),
            block_type=bt,
            page_index=d.get("page_index", 0),
            bbox=tuple(d.get("bbox", [0, 0, 0, 0])),
            reading_order=d.get("reading_order", -1),
            column_index=d.get("column_index"),
            heading_path=d.get("heading_path", []),
            text=d.get("text"),
            table_data=td,
            figure_data=fd,
            enrichment=enrichment,
            needs_review=d.get("needs_review", False),
            review_reasons=d.get("review_reasons", []),
        )

    src = raw.get("source", {})
    source = SourceMeta(
        filename=src.get("filename", ""),
        file_hash=src.get("file_hash"),
        num_pages=src.get("num_pages", 0),
    )
    pm = raw.get("parse_meta", {})
    parse_meta = ParseMeta(parser_version=pm.get("parser_version", "1.0.0"), parse_errors=pm.get("parse_errors", []))
    pages = [
        PageData(
            page_index=p.get("page_index", 0),
            width=p.get("width", 612),
            height=p.get("height", 792),
            layout_type=LayoutType(p.get("layout_type", "single_column")),
            column_boundaries=p.get("column_boundaries"),
        )
        for p in raw.get("pages", [])
    ]
    content_flow = [to_block(b) for b in raw.get("content_flow", [])]
    return EnrichedDoc(
        doc_id=raw.get("doc_id", ""),
        source=source,
        global_summary=raw.get("global_summary"),
        pages=pages,
        content_flow=content_flow,
        hierarchy=[],  # 简化反序列化
        parse_meta=parse_meta,
        enrichment_meta=None,
        claims=raw.get("claims", []),
        doc_metadata=raw.get("doc_metadata"),
    )


_DOI_RE = re.compile(
    r"(?:doi[:\s]*|(?:https?://)?(?:dx\.)?doi\.org/)"
    r"?(10\.\d{4,9}/[^\s,;)\]\"'<>]+[^\s,;)\]\"'<>.:])",
    re.IGNORECASE,
)

_SKIP_HEADING_LABELS = {
    "original article", "research article", "review", "letter",
    "communication", "open", "open access", "article", "brief report",
    "research paper", "full paper", "short communication",
    "abstract", "abstract:", "data note", "introduction",
    "keywords", "highlights", "graphical abstract", "contents",
}


def extract_doc_metadata(blocks: list[ContentBlock], scan_blocks: int = 30) -> dict:
    """
    从 content_flow 前 N 个 block 提取论文级元数据（DOI + 标题）。
    在 PDFProcessor.process() 完成解析后调用。
    """
    doi = None
    title = None

    for block in blocks[:scan_blocks]:
        text = block.text
        if not text or not isinstance(text, str):
            continue
        m = _DOI_RE.search(text)
        if m:
            doi = m.group(1).rstrip(".:")
            break

    # Pass 1: heading blocks
    for block in blocks[:15]:
        bt = block.block_type.value if isinstance(block.block_type, Enum) else str(block.block_type or "")
        bt = bt.lower()
        text = (block.text or "").strip()
        if bt != "heading" or not text:
            continue
        if text.lower() in _SKIP_HEADING_LABELS:
            continue
        if 10 <= len(text) <= 300:
            title = text
            break

    # Pass 2: fallback to text blocks on page 0
    if not title:
        for block in blocks[:15]:
            bt = block.block_type.value if isinstance(block.block_type, Enum) else str(block.block_type or "")
            bt = bt.lower()
            if bt not in ("text", ""):
                continue
            text = (block.text or "").strip()
            page = getattr(block, "page_index", 0) or 0
            if page != 0 or not text or not (15 <= len(text) <= 300):
                continue
            tl = text.lower()
            if tl in _SKIP_HEADING_LABELS:
                continue
            if tl.startswith(("http", "www.", "\u00a9", "copyright")):
                continue
            if re.search(r"\d{4}\s*(international|society|elsevier|springer|wiley|nature)", tl):
                continue
            digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
            if digit_ratio > 0.08:
                continue
            title = text
            break

    result: dict = {}
    if doi:
        result["doi"] = doi
    if title:
        result["title"] = title
    return result or None


class PDFProcessor:
    def __init__(
        self,
        config: Optional[ParserConfig] = None,
        llm_manager: Optional[Any] = None,
    ):
        self.config = config or ParserConfig()
        self.llm_manager = llm_manager

    def process(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        skip_enrichment: bool = False,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> EnrichedDoc:
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem
        out = output_dir or Path(self.config.output_dir) / doc_id
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Parse
        parser = DoclingParser(self.config)
        blocks, pages, raw_out = parser.parse(pdf_path)
        source = SourceMeta(filename=pdf_path.name, num_pages=len(pages))
        parse_meta = ParseMeta(parser_version=self.config.parser_version)

        # 2. Layout
        layout = LayoutAnalyzer(self.config)
        blocks, pages = layout.analyze(blocks, pages)

        # 3. Context
        linker = ContextLinker(self.config)
        blocks, hierarchy = linker.link_all(blocks, pages)

        # 4. Extract figures
        figure_blocks = [
            b for b in blocks
            if b.block_type == BlockType.FIGURE and b.figure_data and b.figure_data.caption
        ]
        extractor = FigureExtractor(self.config)
        extractor.extract_figures(pdf_path, figure_blocks, out)

        # 5. LLM enrich
        enrichment_meta = None
        if not skip_enrichment and self.llm_manager:
            enricher = LLMEnricher(self.config, self.llm_manager)
            blocks, enrichment_meta = enricher.enrich_all(blocks, str(out), progress_callback=progress_callback)

        # 6. Extract doc-level metadata (DOI, title)
        doc_meta = extract_doc_metadata(blocks)
        if doc_meta:
            logger.info("doc_metadata for %s: doi=%s title=%s",
                        doc_id, doc_meta.get("doi", "-"), (doc_meta.get("title") or "-")[:60])

        doc = EnrichedDoc(
            doc_id=doc_id,
            source=source,
            pages=pages,
            content_flow=blocks,
            hierarchy=hierarchy,
            parse_meta=parse_meta,
            enrichment_meta=enrichment_meta,
            doc_metadata=doc_meta,
        )

        # 6. Claim extraction（需要 LLM）
        if not skip_enrichment and self.llm_manager:
            try:
                from src.parser.claim_extractor import ClaimExtractor
                claim_extractor = ClaimExtractor()
                client = self.llm_manager.get_client()
                claims = claim_extractor.extract(doc, client)
                doc.claims = [c.to_dict() for c in claims]
            except Exception as e:
                logger.warning("Claim extraction failed for %s: %s", doc_id, e)

        self.save(doc, str(out))
        return doc

    def save(self, doc: EnrichedDoc, output_dir: str | Path) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = _serialize_obj(doc)
        json_path = out / "enriched.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(json_path)

    @staticmethod
    def load(json_path: str | Path) -> EnrichedDoc:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _deserialize_enriched(raw)
