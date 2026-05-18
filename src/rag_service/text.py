from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for", "with", "on", "as", "by",
    "what", "which", "how", "when", "who", "must", "should", "does", "do", "can", "be", "it", "this",
    "吗", "的", "了", "是", "什么", "需要", "应该", "能不能", "哪些", "多少", "几", "个",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens: List[str] = []
    tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text))
    tokens.extend(re.findall(r"\d+(?:\.\d+)?", text))
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk)
    tokens.extend(["".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1))])
    return [t for t in tokens if t and t not in STOPWORDS]


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def term_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def keyword_hit_ratio(answer: str, expected_keywords: Iterable[str]) -> float:
    answer_l = answer.lower()
    keywords = list(expected_keywords)
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if str(kw).lower() in answer_l)
    return hits / len(keywords)
