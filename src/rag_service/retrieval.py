from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal

from .chunking import DocumentChunk
from .text import cosine_sparse, term_counter, tokenize


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    vector_score: float
    bm25_score: float
    rerank_score: float = 0.0


class Retriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.doc_term_counts: List[Counter] = [term_counter(c.text + " " + c.title) for c in chunks]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_term_counts]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.df = defaultdict(int)
        for counter in self.doc_term_counts:
            for term in counter:
                self.df[term] += 1
        self.n = len(chunks)
        self.idf = {term: math.log((self.n + 1) / (df + 0.5)) + 1.0 for term, df in self.df.items()}
        self.tfidf_docs = [self._tfidf(counter) for counter in self.doc_term_counts]

    def _tfidf(self, counter: Counter) -> Dict[str, float]:
        total = sum(counter.values()) or 1
        return {term: (count / total) * self.idf.get(term, 0.0) for term, count in counter.items()}

    def _bm25(self, query_terms: List[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        counter = self.doc_term_counts[idx]
        doc_len = self.doc_lengths[idx] or 1
        for term in query_terms:
            tf = counter.get(term, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = tf + k1 * (1 - b + b * doc_len / max(1e-9, self.avg_doc_len))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        max_v = max(values)
        min_v = min(values)
        if max_v == min_v:
            return [1.0 if max_v > 0 else 0.0 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def search(
        self,
        query: str,
        mode: Literal["vector_only", "hybrid"] = "hybrid",
        top_k: int = 5,
        hybrid_alpha: float = 0.62,
    ) -> List[RetrievedChunk]:
        q_counter = term_counter(query)
        q_tfidf = self._tfidf(q_counter)
        q_terms = list(q_counter.keys())

        raw_vector = [cosine_sparse(q_tfidf, doc_vec) for doc_vec in self.tfidf_docs]
        raw_bm25 = [self._bm25(q_terms, i) for i in range(self.n)]
        vector_scores = self._normalize(raw_vector)
        bm25_scores = self._normalize(raw_bm25)

        results: List[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            if mode == "vector_only":
                score = vector_scores[idx]
            else:
                score = hybrid_alpha * vector_scores[idx] + (1.0 - hybrid_alpha) * bm25_scores[idx]
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_scores[idx],
                    bm25_score=bm25_scores[idx],
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


class SimpleReranker:
    """Deterministic local reranker for case-study reproducibility.

    It rewards exact phrase hits, title/doc-id hits, and query-token coverage in candidate chunks.
    Replace this class with a cross-encoder or hosted reranker without changing API code.
    """

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        q_norm = query.lower().strip()
        q_terms = set(tokenize(query))
        rescored: List[RetrievedChunk] = []
        for result in candidates:
            text = (result.chunk.title + " " + result.chunk.text).lower()
            coverage = len(q_terms & set(tokenize(text))) / max(1, len(q_terms))
            phrase_bonus = 0.18 if q_norm and q_norm in text else 0.0
            title_bonus = 0.08 if q_terms & set(tokenize(result.chunk.title)) else 0.0
            result.rerank_score = 0.74 * result.score + 0.18 * coverage + phrase_bonus + title_bonus
            result.score = result.rerank_score
            rescored.append(result)
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]
