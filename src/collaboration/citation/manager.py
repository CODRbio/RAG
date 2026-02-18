"""
引用管理：EvidencePack → Citation，cite_key 多格式生成，与 Canvas citation_pool 同步。

支持三种 cite_key 格式：
- numeric: 数字序号 [1], [2], [3]
- hash: 12位哈希 [a3f7b2c91e04]
- author_date: 作者年份 [Smith2023]，重复时加后缀 [Smith2023a]
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Literal, Optional, Set

from config.settings import settings
from src.collaboration.canvas.models import Citation
from src.retrieval.evidence import EvidenceChunk, EvidencePack


CiteKeyFormat = Literal["numeric", "hash", "author_date"]


class CiteKeyGenerator:
    """统一的引用键生成器，支持 numeric / hash / author_date 三种格式。"""

    def __init__(
        self,
        format: Optional[CiteKeyFormat] = None,
        existing_keys: Optional[Set[str]] = None,
        hash_length: Optional[int] = None,
        max_authors: Optional[int] = None,
    ):
        """
        初始化生成器。

        Args:
            format: 引用键格式，默认从配置读取
            existing_keys: 已存在的引用键集合（用于去重和编号）
            hash_length: hash 模式下的长度
            max_authors: author_date 模式下最多显示几个作者
        """
        self.format: CiteKeyFormat = format or settings.citation.key_format
        self.existing_keys: Set[str] = set(existing_keys) if existing_keys else set()
        self.hash_length = hash_length or settings.citation.hash_length
        self.max_authors = max_authors or settings.citation.author_date_max_authors
        self._numeric_counter = len(self.existing_keys) + 1

    def generate(self, citation: Citation) -> str:
        """根据配置的格式生成 cite_key。"""
        if self.format == "numeric":
            return self._numeric_key()
        elif self.format == "hash":
            return self._hash_key(citation)
        else:  # author_date
            return self._author_date_key(citation)

    def _numeric_key(self) -> str:
        """生成数字序号：1, 2, 3..."""
        key = str(self._numeric_counter)
        self._numeric_counter += 1
        self.existing_keys.add(key)
        return key

    def _hash_key(self, c: Citation) -> str:
        """生成哈希键：基于内容的 SHA256 前 N 位。"""
        raw = f"{c.title or ''}|{','.join(c.authors or [])}|{c.year or ''}|{c.doi or ''}|{c.url or ''}|{c.doc_id or ''}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[: self.hash_length]
        self.existing_keys.add(key)
        return key

    def _author_date_key(self, c: Citation) -> str:
        """生成作者年份键：Smith2023，重复时 Smith2023a, Smith2023b..."""
        base = self._extract_author_year(c)
        if base not in self.existing_keys:
            self.existing_keys.add(base)
            return base
        # 添加后缀 a-z
        for suffix in "abcdefghijklmnopqrstuvwxyz":
            key = f"{base}{suffix}"
            if key not in self.existing_keys:
                self.existing_keys.add(key)
                return key
        # 用尽26个字母，fallback 到 base_xxxx
        fallback = f"{base}_{hashlib.sha256(c.id.encode()).hexdigest()[:4]}"
        self.existing_keys.add(fallback)
        return fallback

    def _extract_author_year(self, c: Citation) -> str:
        """提取作者姓氏 + 年份，如 Smith2023 或 SmithJones2023。"""
        authors = c.authors or []
        year_part = str(c.year) if c.year else ""

        if not authors:
            # 无作者信息，使用标题首词
            if c.title:
                first_word = re.split(r"\s+", c.title.strip())[0]
                first_word = self._normalize_name(first_word)[:10]
                return f"{first_word}{year_part}" if first_word else f"Anon{year_part}"
            return f"Anon{year_part}"

        # 提取作者姓氏
        surnames = []
        for author in authors[: self.max_authors]:
            surname = self._extract_surname(author)
            if surname:
                surnames.append(surname)

        if len(authors) > self.max_authors:
            author_part = "".join(surnames) + "EtAl"
        else:
            author_part = "".join(surnames)

        return f"{author_part}{year_part}" if author_part else f"Anon{year_part}"

    def _extract_surname(self, author: str) -> str:
        """从作者全名提取姓氏（支持中英文）。"""
        author = author.strip()
        if not author:
            return ""

        # 检测是否为中文名（无空格且含汉字）
        if " " not in author and any("\u4e00" <= ch <= "\u9fff" for ch in author):
            # 中文名：取第一个字作为姓
            return self._normalize_name(author[0])

        # 西文名：常见格式 "First Last" 或 "Last, First"
        if "," in author:
            # "Smith, John" 格式
            parts = author.split(",")
            surname = parts[0].strip()
        else:
            # "John Smith" 格式
            parts = author.split()
            surname = parts[-1] if parts else ""

        return self._normalize_name(surname)

    def _normalize_name(self, name: str) -> str:
        """规范化名字：移除特殊字符，首字母大写。"""
        # 移除重音符号
        name = unicodedata.normalize("NFD", name)
        name = "".join(ch for ch in name if unicodedata.category(ch) != "Mn")
        # 只保留字母和数字
        name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", name)
        # 首字母大写
        return name.capitalize() if name else ""


# 模块级生成器（懒初始化）
_generator: Optional[CiteKeyGenerator] = None


def get_generator(
    format: Optional[CiteKeyFormat] = None,
    existing_keys: Optional[Set[str]] = None,
) -> CiteKeyGenerator:
    """获取或创建 cite_key 生成器。"""
    global _generator
    if _generator is None or format is not None or existing_keys is not None:
        _generator = CiteKeyGenerator(format=format, existing_keys=existing_keys)
    return _generator


def reset_generator():
    """重置生成器（测试或切换格式时使用）。"""
    global _generator
    _generator = None


# ============================================================
# 兼容原有接口
# ============================================================


def _make_cite_key(
    title: str,
    authors: List[str],
    year: int | None,
    doi: str | None,
    url: str | None,
    doc_id: str | None,
    format: Optional[CiteKeyFormat] = None,
) -> str:
    """生成 cite_key（兼容旧接口，推荐使用 CiteKeyGenerator）。"""
    # 创建临时 Citation 对象
    c = Citation(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        doc_id=doc_id,
    )
    gen = get_generator(format=format)
    return gen.generate(c)


def chunk_to_citation(
    chunk: EvidenceChunk,
    format: Optional[CiteKeyFormat] = None,
    generator: Optional[CiteKeyGenerator] = None,
) -> Citation:
    """将 EvidenceChunk 转换为 Citation。"""
    title = chunk.doc_title or chunk.chunk_id or ""
    authors = list(chunk.authors) if chunk.authors else []
    year = getattr(chunk, "year", None)
    doi = getattr(chunk, "doi", None) or None
    url = getattr(chunk, "url", None) or None
    doc_id = chunk.doc_id or None

    bbox = getattr(chunk, "bbox", None) or None
    page_num = getattr(chunk, "page_num", None) or None

    # 先创建不带 cite_key 的 Citation
    citation = Citation(
        id=chunk.chunk_id[:16] if chunk.chunk_id else "",
        title=title,
        authors=authors,
        year=year,
        doc_id=doc_id,
        url=url,
        doi=doi,
        cite_key="",
        bbox=bbox,
        page_num=page_num,
    )

    # 使用生成器生成 cite_key
    gen = generator or get_generator(format=format)
    citation.cite_key = gen.generate(citation)

    # 如果 id 为空，使用 cite_key
    if not citation.id:
        citation.id = citation.cite_key

    return citation


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    """去重：基于 cite_key 或 id。"""
    seen_keys: Set[str] = set()
    out: List[Citation] = []
    for c in citations:
        key = c.cite_key or c.id
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(c)
    return out


def merge_citations_by_document(citations: List[Citation]) -> List[Citation]:
    """
    按文档级别合并引文：同一文档/同一 URL 只保留一条。
    - 本地文献：按 doc_id 合并
    - Web 结果：按 URL 合并
    保留首次出现的条目（保持原有顺序）。
    """
    merged: Dict[str, Citation] = {}
    for c in citations:
        if c.url:
            key = (c.url or "").strip()
        else:
            key = (c.doc_id or c.cite_key or c.id or "").strip()
        if not key:
            key = c.cite_key or c.id
        if key not in merged:
            merged[key] = c
    return list(merged.values())


def resolve_response_citations(
    response_text: str,
    chunks: List[EvidenceChunk],
    format: Optional[CiteKeyFormat] = None,
    doc_key_to_cite_key: Optional[Dict[str, str]] = None,
    existing_cite_keys: Optional[Set[str]] = None,
    include_unreferenced_documents: bool = True,
) -> tuple[str, List[Citation], Dict[str, str]]:
    """
    对 LLM 回答做引文后处理：将 [ref_hash] 替换为正式 cite_key，并输出文档级引文列表。

    流程：
      1. 扫描 response_text 中的 [hex_hash] 模式
      2. hash → chunk → doc_group_key，按文档去重
      3. 按首次出现顺序为每个文档生成正式 cite_key
      4. 替换文本中所有 hash 为对应 cite_key
      5. 构建文档级 Citation 列表

    Args:
        response_text: LLM 原始回答（包含 [ref_hash] 引用）
        chunks:        本轮检索返回的所有 EvidenceChunk
        format:        cite_key 格式，默认读配置

    Returns:
        (resolved_text, citations, ref_map):
        - resolved_text: 替换后的回答文本
        - citations:     文档级 Citation 列表（按首次引用顺序）
        - ref_map:       ref_hash → cite_key 映射（供前端 / 调试）

    Notes:
        - 当传入 doc_key_to_cite_key / existing_cite_keys 时，会在原有映射基础上复用并增量生成，
          适用于跨阶段（write/refine/synthesize）保持稳定引用键。
        - include_unreferenced_documents=True 时，会把未在文本中出现但存在于 chunks 的文档也纳入 citations。
    """
    from collections import OrderedDict
    from src.retrieval.evidence import REF_HASH_LENGTH

    # ── 1. 建立 hash → chunk 查找表 ──
    hash_to_chunk: Dict[str, EvidenceChunk] = {}
    for c in chunks:
        hash_to_chunk[c.ref_hash] = c

    # ── 2. 扫描回答中出现的 hash ──
    pattern = re.compile(r"\[([0-9a-fA-F]{" + str(REF_HASH_LENGTH) + r"})\]")
    cited_hashes: List[str] = []
    seen_hashes: Set[str] = set()
    for m in pattern.finditer(response_text):
        h = m.group(1).lower()
        if h in hash_to_chunk and h not in seen_hashes:
            cited_hashes.append(h)
            seen_hashes.add(h)

    # 复用外部映射（可跨阶段保持稳定 cite_key）
    shared_doc_map = doc_key_to_cite_key if doc_key_to_cite_key is not None else {}
    shared_keys = existing_cite_keys if existing_cite_keys is not None else set()

    # ── 3. 按文档分组（保持首次引用顺序）──
    doc_groups: OrderedDict[str, List[EvidenceChunk]] = OrderedDict()
    for h in cited_hashes:
        chunk = hash_to_chunk[h]
        key = chunk.doc_group_key
        doc_groups.setdefault(key, []).append(chunk)

    # 未被引用的文档也追加到末尾（可选）
    if include_unreferenced_documents:
        for c in chunks:
            key = c.doc_group_key
            if key not in doc_groups:
                doc_groups.setdefault(key, []).append(c)

    # ── 4. 为每个文档生成 cite_key ──
    seed_keys = set(shared_keys) if shared_keys else set()
    if shared_doc_map:
        seed_keys.update(shared_doc_map.values())
    gen = CiteKeyGenerator(format=format, existing_keys=seed_keys)
    citations: List[Citation] = []
    for doc_key, group in doc_groups.items():
        if doc_key in shared_doc_map:
            cite_key = shared_doc_map[doc_key]
        else:
            # 先生成 cite_key，再写入共享映射
            temp = _pick_best_metadata(group)
            temp_citation = Citation(
                id=doc_key[:16],
                title=temp.doc_title or "",
                authors=list(temp.authors) if temp.authors else [],
                year=temp.year,
                doc_id=temp.doc_id,
                url=temp.url,
                doi=getattr(temp, "doi", None),
                cite_key="",
            )
            cite_key = gen.generate(temp_citation)
            shared_doc_map[doc_key] = cite_key
            shared_keys.add(cite_key)

        best = _pick_best_metadata(group)
        citation = Citation(
            id=doc_key[:16],
            title=best.doc_title or "",
            authors=list(best.authors) if best.authors else [],
            year=best.year,
            doc_id=best.doc_id,
            url=best.url,
            doi=getattr(best, "doi", None),
            cite_key=cite_key,
        )
        citations.append(citation)

    # ── 5. 构建 ref_hash → cite_key 映射 ──
    ref_map: Dict[str, str] = {}
    for h, chunk in hash_to_chunk.items():
        key = chunk.doc_group_key
        if key in shared_doc_map:
            ref_map[h] = shared_doc_map[key]

    # ── 6. 替换文本中的 hash ──
    def _replace_hash(m: re.Match) -> str:
        h = m.group(1).lower()
        cite_key = ref_map.get(h)
        if cite_key:
            return f"[{cite_key}]"
        return m.group(0)  # 未识别的 hash 保持原样

    resolved_text = pattern.sub(_replace_hash, response_text)

    return resolved_text, citations, ref_map


def _pick_best_metadata(chunks: List[EvidenceChunk]) -> EvidenceChunk:
    """从同一文档的多个 chunk 中选择元数据最完整的一个。"""
    best = chunks[0]
    best_score = 0
    for c in chunks:
        score = 0
        if c.doc_title:
            score += 2
        if c.authors:
            score += 2
        if c.year is not None:
            score += 1
        if getattr(c, "doi", None):
            score += 1
        if c.url:
            score += 1
        if score > best_score:
            best = c
            best_score = score
    return best


def sync_evidence_to_canvas(
    canvas_id: str,
    evidence_pack: EvidencePack,
    format: Optional[CiteKeyFormat] = None,
) -> List[str]:
    """
    将 EvidencePack 中的 chunk 转为 Citation，去重后合并写入 Canvas 引用池。

    Args:
        canvas_id: Canvas ID
        evidence_pack: 检索结果包
        format: 可选，覆盖配置中的 cite_key 格式

    Returns:
        本轮新增的 cite_key 列表
    """
    if not canvas_id or not evidence_pack or not evidence_pack.chunks:
        return []

    from src.collaboration.canvas.canvas_manager import get_canvas_store

    store = get_canvas_store()
    existing = store.get_citations(canvas_id)

    # 收集已有的 cite_key
    existing_keys = {c.cite_key or c.id for c in existing}

    # 创建生成器，传入已有 keys 以确保不重复
    gen = CiteKeyGenerator(format=format, existing_keys=existing_keys)

    # 转换新的 citations
    new_citations = _dedupe_citations(
        [chunk_to_citation(c, generator=gen) for c in evidence_pack.chunks]
    )
    if getattr(settings.citation, "merge_level", "document") == "document":
        new_citations = merge_citations_by_document(new_citations)

    # 合并
    by_key: Dict[str, Citation] = {c.cite_key or c.id: c for c in existing}
    for c in new_citations:
        by_key[c.cite_key or c.id] = c

    store.upsert_citations(canvas_id, list(by_key.values()))
    return [c.cite_key or c.id for c in new_citations]
