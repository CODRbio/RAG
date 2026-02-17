"""
共享 Fixtures: Mock LLM / Embedder / Milvus 等。
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端"""
    client = MagicMock()

    def _chat(messages=None, model=None, max_tokens=None, **kwargs):
        return {"final_text": '{"intent": "chat", "confidence": 0.9, "params": {}}'}

    client.chat.side_effect = _chat
    return client


@pytest.fixture
def sample_text_blocks():
    """示例文本 blocks"""
    return [
        {
            "block_type": "text",
            "heading_path": ["Introduction"],
            "text": "Deep-sea hydrothermal vents are one of the most extreme environments on Earth. "
                    "These systems support unique chemosynthetic ecosystems. "
                    "Tube worms and mussels are among the dominant fauna.",
            "page_index": 0,
            "block_id": "b001",
        },
        {
            "block_type": "text",
            "heading_path": ["Introduction"],
            "text": "The discovery of hydrothermal vents in 1977 revolutionized our understanding of life. "
                    "Organisms at these vents derive energy from chemical reactions rather than sunlight.",
            "page_index": 1,
            "block_id": "b002",
        },
        {
            "block_type": "text",
            "heading_path": ["Methods"],
            "text": "Samples were collected using ROV at depths of 2500-3000m. "
                    "DNA extraction was performed using the DNeasy protocol. "
                    "Metagenomic sequencing was conducted on Illumina NovaSeq platform.",
            "page_index": 2,
            "block_id": "b003",
        },
    ]


@pytest.fixture
def sample_table_block():
    """示例表格 block"""
    return {
        "block_type": "table",
        "heading_path": ["Results", "Species Distribution"],
        "text": "",
        "table_data": {
            "markdown": "| Species | Count | Depth |\n|---|---|---|\n| Tube worm | 150 | 2600m |\n| Mussel | 230 | 2800m |",
        },
        "page_index": 3,
        "block_id": "t001",
    }


@pytest.fixture
def sample_figure_block():
    """示例图表 block"""
    return {
        "block_type": "figure",
        "heading_path": ["Results"],
        "text": "",
        "figure_data": {"caption": "Distribution of hydrothermal vent species at various depths"},
        "page_index": 4,
        "block_id": "f001",
    }


@pytest.fixture
def sample_claims():
    """示例 claims"""
    return [
        {
            "claim_id": "claim_001",
            "text": "Tube worms dominate at shallower vents",
            "evidence": "Survey data shows...",
            "source_block_ids": ["b001", "b002"],
        },
        {
            "claim_id": "claim_002",
            "text": "Chemosynthesis is the primary energy source",
            "evidence": "Isotopic analysis...",
            "source_block_ids": ["b002"],
        },
    ]
