"""
Full-text compressor: short passthrough, long compressed, LLM failure → truncate.
"""

import pytest
from unittest.mock import MagicMock

from src.retrieval.fulltext_compressor import (
    _word_count,
    compress_fulltext_hits_sync,
    WORD_THRESHOLD_DEFAULT,
    FALLBACK_TRUNCATE_CHARS,
)


class TestWordCount:
    def test_empty(self):
        assert _word_count("") == 0
        assert _word_count("   ") == 0

    def test_simple(self):
        assert _word_count("one two three") == 3
        assert _word_count("  one   two  ") == 2


class TestCompressShortPassthrough:
    def test_non_fulltext_unchanged(self):
        hits = [
            {"content": "x " * 400, "metadata": {"content_type": "snippet_sufficient"}},
        ]
        client = MagicMock()
        out = compress_fulltext_hits_sync(hits, "query", client, word_threshold=300)
        assert out is hits
        assert hits[0]["content"] == "x " * 400
        client.chat.assert_not_called()

    def test_short_fulltext_unchanged(self):
        short = "a b c " * 50  # 300 words
        hits = [{"content": short, "metadata": {"content_type": "full_text"}}]
        client = MagicMock()
        out = compress_fulltext_hits_sync(hits, "q", client, word_threshold=300)
        assert out is hits
        assert hits[0]["content"] == short
        client.chat.assert_not_called()


class TestCompressLongCallsLlm:
    def test_long_fulltext_compressed(self):
        long_text = "word " * 500
        hits = [{"content": long_text, "metadata": {"content_type": "full_text", "title": "T", "url": "https://u"}}]
        client = MagicMock()
        client.chat.return_value = {"final_text": "Short summary here."}
        out = compress_fulltext_hits_sync(
            hits, "query", client,
            word_threshold=300,
            max_output_words=400,
            max_concurrent=1,
        )
        assert out is hits
        assert hits[0]["content"] == "Short summary here."
        assert hits[0]["metadata"]["content_type"] == "full_text_compressed"
        # Compressor uses stripped content for original_full_text_chars
        assert hits[0]["metadata"]["original_full_text_chars"] == len(long_text.strip())
        client.chat.assert_called_once()


class TestCompressLlmFailureTruncates:
    def test_llm_raises_truncates(self):
        # Use enough chars so truncation kicks in (fallback_chars=2000)
        long_text = "x " * 1500  # 2999 chars
        hits = [{"content": long_text, "metadata": {"content_type": "full_text", "title": "", "url": ""}}]
        client = MagicMock()
        client.chat.side_effect = RuntimeError("api error")
        out = compress_fulltext_hits_sync(
            hits, "q", client,
            word_threshold=300,
            fallback_chars=FALLBACK_TRUNCATE_CHARS,
            max_concurrent=1,
        )
        assert out is hits
        content = hits[0]["content"]
        assert len(content) <= FALLBACK_TRUNCATE_CHARS + 3
        assert content.endswith("...")
        assert hits[0]["metadata"]["content_type"] == "full_text_compressed"
        assert hits[0]["metadata"]["original_full_text_chars"] == len(long_text.strip())
