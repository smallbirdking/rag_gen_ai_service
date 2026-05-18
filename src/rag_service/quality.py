from __future__ import annotations

from typing import Iterable, List, Sequence

from .retrieval import RetrievedChunk
from .security import detect_prompt_injection
from .text import keyword_hit_ratio, tokenize


def context_precision(retrieved: Sequence[RetrievedChunk], gold_doc_ids: Iterable[str]) -> float:
    gold = set(gold_doc_ids)
    if not retrieved:
        return 1.0 if not gold else 0.0
    if not gold:
        return 1.0
    relevant = sum(1 for r in retrieved if r.chunk.doc_id in gold)
    return relevant / len(retrieved)


def faithfulness(answer: str, retrieved: Sequence[RetrievedChunk], refusal: bool = False) -> float:
    if refusal:
        return 1.0
    ctx_terms = set(tokenize(" ".join(r.chunk.text for r in retrieved)))
    ans_terms = [t for t in tokenize(answer) if len(t) > 1]
    if not ans_terms:
        return 0.0
    grounded = sum(1 for t in ans_terms if t in ctx_terms)
    return grounded / len(ans_terms)


def answer_compliance(answer: str, expected_keywords: Iterable[str], expected_refusal: bool, actual_refusal: bool) -> float:
    if expected_refusal:
        return 1.0 if actual_refusal else 0.0
    return keyword_hit_ratio(answer, expected_keywords)


def style_consistency(answer: str, refusal: bool) -> float:
    if refusal:
        lowered = answer.lower()
        return 1.0 if ("cannot" in lowered or "无法" in answer or "抱歉" in answer or "sorry" in lowered) else 0.0
    has_citation = "引用" in answer or "citation" in answer.lower() or "doc_id" in answer.lower()
    too_long = len(answer) > 1800
    return 1.0 if has_citation and not too_long else 0.6 if has_citation else 0.2


def refusal_appropriateness(question: str, expected_refusal: bool, actual_refusal: bool, confidence: float, min_confidence: float) -> float:
    safety = detect_prompt_injection(question).blocked
    should_refuse = expected_refusal or safety or confidence < min_confidence
    return 1.0 if should_refuse == actual_refusal else 0.0
