"""
评测指标：检索/生成/引用相关的基础度量
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_CITE_RE = re.compile(r"\[([^\]]+)\]")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)


def _precision_recall_f1(pred_count: Counter, ref_count: Counter) -> Tuple[float, float, float]:
    if not pred_count and not ref_count:
        return 1.0, 1.0, 1.0
    if not pred_count or not ref_count:
        return 0.0, 0.0, 0.0
    common = pred_count & ref_count
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / max(1, sum(pred_count.values()))
    recall = num_same / max(1, sum(ref_count.values()))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def token_f1(pred: str, ref: str) -> Tuple[float, float, float]:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    return _precision_recall_f1(Counter(pred_tokens), Counter(ref_tokens))


def rouge_l_f1(pred: str, ref: str) -> Tuple[float, float, float]:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def extract_citations(text: str) -> List[str]:
    """
    从回答中抽取 [xxx] 形式的引用标记。
    支持 [A; B] / [A, B] / [A B]
    """
    out: List[str] = []
    for block in _CITE_RE.findall(text or ""):
        parts = re.split(r"[;,/|\s]+", block.strip())
        for p in parts:
            key = p.strip()
            if key:
                out.append(key)
    return out


def safe_mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> Tuple[float | None, float | None]:
    if not expected_ids:
        return None, None
    expected = set(expected_ids)
    topk = retrieved_ids[:k]
    hit = 1.0 if any(x in expected for x in topk) else 0.0
    recall = sum(1 for x in expected if x in topk) / max(1, len(expected))
    return recall, hit
