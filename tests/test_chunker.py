"""
chunker 单元测试：chunk_blocks / 句子切分 / 表格 / 图片 / claims 注入
"""

import pytest
from src.chunking.chunker import (
    Chunk,
    ChunkConfig,
    chunk_blocks,
    chunk_table_block,
    chunk_image_block,
    _sentence_tokenize,
    _generate_stable_id,
    _merge_metadata,
    _match_claims_to_chunk,
)


# ── 句子切分 ──

class TestSentenceTokenize:
    def test_basic_split(self):
        result = _sentence_tokenize("Hello world. How are you? Fine.")
        assert len(result) == 3

    def test_chinese_split(self):
        # 句子切分 regex 要求标点后有空格，中文句号后无空格时不拆分
        result = _sentence_tokenize("深海热泉是极端环境。 生物依靠化能合成。")
        assert len(result) == 2

    def test_empty_input(self):
        assert _sentence_tokenize("") == []
        assert _sentence_tokenize("   ") == []

    def test_single_sentence(self):
        result = _sentence_tokenize("No punctuation at end")
        assert result == ["No punctuation at end"]


# ── 稳定 ID 生成 ──

class TestGenerateStableId:
    def test_deterministic(self):
        id1 = _generate_stable_id("doc1", "Introduction", 0, "hello", 0)
        id2 = _generate_stable_id("doc1", "Introduction", 0, "hello", 0)
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _generate_stable_id("doc1", "Introduction", 0, "hello", 0)
        id2 = _generate_stable_id("doc2", "Introduction", 0, "hello", 0)
        assert id1 != id2

    def test_length_16(self):
        cid = _generate_stable_id("doc", "sec", 1, "prefix", 0)
        assert len(cid) == 16


# ── 元数据合并 ──

class TestMergeMetadata:
    def test_basic(self, sample_text_blocks):
        meta = _merge_metadata(sample_text_blocks[:2], "test_doc")
        assert meta["doc_id"] == "test_doc"
        assert meta["page_range"] == [0, 1]
        assert meta["section_path"] == "Introduction"

    def test_empty(self):
        meta = _merge_metadata([], "doc1")
        assert meta["doc_id"] == "doc1"
        assert meta["page_range"] == [0, 0]


# ── chunk_blocks 主逻辑 ──

class TestChunkBlocks:
    def test_basic_chunking(self, sample_text_blocks):
        chunks = chunk_blocks(sample_text_blocks, "test_doc")
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.text.strip()
            assert c.chunk_id

    def test_empty_blocks(self):
        chunks = chunk_blocks([], "empty_doc")
        assert chunks == []

    def test_blank_text_skipped(self):
        blocks = [{"block_type": "text", "heading_path": [], "text": "   ", "page_index": 0, "block_id": "b_empty"}]
        chunks = chunk_blocks(blocks, "doc")
        assert chunks == []

    def test_section_boundary_creates_new_chunk(self, sample_text_blocks):
        """不同 section 应导致 buffer 刷新"""
        cfg = ChunkConfig(target_chars=5000, min_chars=50, max_chars=10000)
        chunks = chunk_blocks(sample_text_blocks, "doc", config=cfg)
        sections = [c.meta.get("section_path") for c in chunks]
        # Introduction 和 Methods 应在不同 chunk
        assert "Methods" in sections

    def test_caption_footnote_skipped(self):
        blocks = [
            {"block_type": "caption", "heading_path": [], "text": "Figure 1 caption", "page_index": 0, "block_id": "c1"},
            {"block_type": "footnote", "heading_path": [], "text": "Footnote text", "page_index": 0, "block_id": "f1"},
        ]
        chunks = chunk_blocks(blocks, "doc")
        assert chunks == []

    def test_max_chars_split(self):
        """超长文本应被按句拆分"""
        long_text = ". ".join([f"Sentence number {i}" for i in range(100)]) + "."
        blocks = [{"block_type": "text", "heading_path": ["Long"], "text": long_text, "page_index": 0, "block_id": "long1"}]
        cfg = ChunkConfig(target_chars=200, min_chars=50, max_chars=300)
        chunks = chunk_blocks(blocks, "doc", config=cfg)
        assert len(chunks) > 1
        for c in chunks:
            assert c.content_type == "text"

    def test_mixed_block_types(self, sample_text_blocks, sample_table_block, sample_figure_block):
        all_blocks = sample_text_blocks + [sample_table_block, sample_figure_block]
        chunks = chunk_blocks(all_blocks, "doc")
        content_types = {c.content_type for c in chunks}
        assert "text" in content_types
        assert "table" in content_types


# ── 表格切块 ──

class TestChunkTableBlock:
    def test_basic_table(self, sample_table_block):
        chunks = chunk_table_block(sample_table_block, "doc", max_chars=5000)
        assert len(chunks) == 1
        assert chunks[0].content_type == "table"
        assert "Tube worm" in chunks[0].text

    def test_empty_table(self):
        block = {"block_type": "table", "heading_path": [], "text": "", "table_data": {}, "page_index": 0, "block_id": "t_e"}
        chunks = chunk_table_block(block, "doc", max_chars=5000)
        assert chunks == []

    def test_large_table_split(self):
        """大表格应被按行分片"""
        rows = "\n".join([f"| row{i} | val{i} |" for i in range(50)])
        md = "| Col1 | Col2 |\n|---|---|\n" + rows
        block = {"block_type": "table", "heading_path": [], "text": "", "table_data": {"markdown": md}, "page_index": 0, "block_id": "t_big"}
        chunks = chunk_table_block(block, "doc", max_chars=200, rows_per_chunk=5)
        assert len(chunks) > 1


# ── 图片切块 ──

class TestChunkImageBlock:
    def test_basic_figure(self, sample_figure_block):
        chunks = chunk_image_block(sample_figure_block, "doc")
        assert len(chunks) == 1
        assert chunks[0].content_type == "image_caption"
        assert "hydrothermal" in chunks[0].text

    def test_no_caption_skipped(self):
        block = {"block_type": "figure", "heading_path": [], "text": "", "figure_data": {}, "page_index": 0, "block_id": "f_nc"}
        chunks = chunk_image_block(block, "doc")
        assert chunks == []


# ── Claims 注入 ──

class TestClaimsInjection:
    def test_claims_matched(self, sample_text_blocks, sample_claims):
        chunks = chunk_blocks(sample_text_blocks, "test_doc", claims=sample_claims)
        claim_chunks = [c for c in chunks if c.meta.get("claims")]
        # 至少有一些 chunk 匹配到了 claims
        assert len(claim_chunks) > 0

    def test_no_claims(self, sample_text_blocks):
        chunks = chunk_blocks(sample_text_blocks, "doc", claims=None)
        for c in chunks:
            assert "claims" not in c.meta

    def test_match_claims_to_chunk_basic(self):
        chunk = Chunk(
            chunk_id="c1",
            text="test",
            content_type="text",
            meta={"source_uri": "doc#b001"},
        )
        claims = [
            {"claim_id": "cl1", "source_block_ids": ["b001"]},
            {"claim_id": "cl2", "source_block_ids": ["b999"]},
        ]
        matched = _match_claims_to_chunk(chunk, claims)
        assert matched == ["cl1"]

    def test_match_claims_empty(self):
        chunk = Chunk(chunk_id="c1", text="test", content_type="text", meta={"source_uri": "doc#b001"})
        assert _match_claims_to_chunk(chunk, []) == []
