from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import TTLCache
from .chunking import load_chunks
from .config import AppConfig
from .generation import GroundedGenerator
from .observability import JsonLogger, MetricsStore
from .quality import answer_compliance, faithfulness, style_consistency
from .retrieval import RetrievedChunk, Retriever, SimpleReranker
from .schemas import AskRequest, AskResponse, Citation
from .security import detect_prompt_injection, redact_pii
from .text import normalize_text, stable_hash


@dataclass
class RuntimeComponents:
    config: AppConfig
    retriever: Retriever
    reranker: SimpleReranker
    generator: GroundedGenerator
    cache: TTLCache
    logger: JsonLogger
    metrics: MetricsStore


def build_runtime(config: AppConfig) -> RuntimeComponents:
    chunks = load_chunks(
        config.corpus_path,
        chunk_size=config.raw["corpus"].get("chunk_size_chars", 850),
        overlap=config.raw["corpus"].get("chunk_overlap_chars", 120),
    )
    llm_cfg = config.raw["llm"]
    return RuntimeComponents(
        config=config,
        retriever=Retriever(chunks),
        reranker=SimpleReranker(),
        generator=GroundedGenerator(
            provider=llm_cfg.get("provider", "mock"),
            model=llm_cfg.get("model", "gpt-5.4-mini"),
            temperature=float(llm_cfg.get("temperature", 0.1)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 600)),
        ),
        cache=TTLCache(ttl_seconds=int(config.raw["app"].get("cache_ttl_seconds", 600))),
        logger=JsonLogger(config.log_path),
        metrics=MetricsStore(),
    )


def _refusal_message(question: str, config: AppConfig) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in question):
        return config.raw["safety"].get("refusal_message_cn")
    return config.raw["safety"].get("refusal_message_en")


def _cache_key(request: AskRequest, mode: str, reranker_enabled: bool, model: str) -> str:
    history_text = "|".join(f"{m.role}:{m.content}" for m in request.history[-4:])
    payload = json.dumps(
        {
            "question": normalize_text(request.question),
            "history": normalize_text(history_text),
            "mode": mode,
            "reranker": reranker_enabled,
            "model": model,
            "top_k": request.top_k,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return stable_hash(payload)


def answer_question(runtime: RuntimeComponents, request: AskRequest, eval_context: Optional[Dict[str, Any]] = None) -> AskResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    cfg = runtime.config.raw
    retrieval_cfg = cfg["retrieval"]
    mode = request.retrieval_mode or retrieval_cfg.get("mode", "hybrid")
    top_k = request.top_k or int(retrieval_cfg.get("top_k", 5))
    reranker_enabled = retrieval_cfg.get("reranker_enabled", True) if request.reranker_enabled is None else request.reranker_enabled
    min_confidence = float(retrieval_cfg.get("min_confidence", 0.18))
    query = request.question
    history_suffix = " ".join(m.content for m in request.history[-2:] if m.role == "user")
    retrieval_query = f"{history_suffix} {query}".strip()

    token_usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    cache_hit = False
    refusal = False
    refusal_reason = None
    contexts: List[RetrievedChunk] = []
    confidence = 0.0
    prompt_injection = False
    pii_redacted_count = 0

    cache_key = _cache_key(request, mode, reranker_enabled, cfg["llm"].get("model", "mock"))
    cached = runtime.cache.get(cache_key)
    if cached:
        cache_hit = True
        latency_ms = int((time.perf_counter() - start) * 1000)
        cached_resp = AskResponse(**cached)
        cached_resp.cache_hit = True
        cached_resp.latency_ms = latency_ms
        runtime.metrics.add(latency_ms, cached_resp.token_usage, cached_resp.refusal, True)
        return cached_resp

    safety = detect_prompt_injection(query) if cfg["safety"].get("enable_prompt_injection_filter", True) else None
    prompt_injection = bool(safety and safety.blocked)
    if prompt_injection:
        refusal = True
        refusal_reason = safety.reason
        answer = _refusal_message(query, runtime.config)
    else:
        candidates = runtime.retriever.search(
            retrieval_query,
            mode=mode,
            top_k=max(top_k, int(retrieval_cfg.get("reranker_top_k", top_k))) if reranker_enabled else top_k,
            hybrid_alpha=float(retrieval_cfg.get("hybrid_alpha", 0.62)),
        )
        if reranker_enabled:
            contexts = runtime.reranker.rerank(query, candidates, top_k=min(top_k, int(retrieval_cfg.get("reranker_top_k", top_k))))
        else:
            contexts = candidates[:top_k]
        # Keep only the contexts that are strong enough to be used as grounding/citations.
        # This improves context precision and prevents weak zero-score chunks from diluting evidence.
        if contexts:
            top_score = contexts[0].score
            strong_contexts = [r for r in contexts if r.score >= max(min_confidence, top_score * 0.65)]
            contexts = strong_contexts or contexts[:1]
        confidence = contexts[0].score if contexts else 0.0
        if confidence < min_confidence:
            refusal = True
            refusal_reason = "low_retrieval_confidence"
            answer = _refusal_message(query, runtime.config)
        else:
            gen = runtime.generator.generate(query, contexts, _refusal_message(query, runtime.config))
            answer = gen.answer
            token_usage = {
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
                "cached_input_tokens": gen.cached_input_tokens,
            }

    if cfg["safety"].get("enable_pii_redaction", True):
        answer, pii_redacted_count = redact_pii(answer)

    citations = [
        Citation(doc_id=r.chunk.doc_id, chunk_id=r.chunk.chunk_id, title=r.chunk.title, score=round(r.score, 4))
        for r in contexts
    ]
    latency_ms = int((time.perf_counter() - start) * 1000)
    compliance_score = None
    if eval_context:
        compliance_score = answer_compliance(
            answer=answer,
            expected_keywords=eval_context.get("expected_keywords", []),
            expected_refusal=eval_context.get("expected_refusal", False),
            actual_refusal=refusal,
        )

    response = AskResponse(
        request_id=request_id,
        answer=answer,
        citations=citations,
        refusal=refusal,
        refusal_reason=refusal_reason,
        confidence=round(confidence, 4),
        latency_ms=latency_ms,
        token_usage=token_usage,
        cache_hit=cache_hit,
        diagnostics={
            "retrieval_mode": mode,
            "reranker_enabled": reranker_enabled,
            "prompt_injection_detected": prompt_injection,
            "pii_redacted_count": pii_redacted_count,
            "faithfulness_proxy": round(faithfulness(answer, contexts, refusal), 4),
            "style_consistency_proxy": round(style_consistency(answer, refusal), 4),
        },
    )
    runtime.cache.set(cache_key, response.model_dump())
    runtime.metrics.add(latency_ms, token_usage, refusal, cache_hit, compliance_score)
    runtime.logger.emit(
        {
            "request_id": request_id,
            "session_id": request.session_id,
            "normalized_query_hash": stable_hash(query),
            "retrieval_mode": mode,
            "reranker_enabled": reranker_enabled,
            "top_k": top_k,
            "retrieved_chunk_ids": [c.chunk_id for c in citations],
            "retrieval_scores": [c.score for c in citations],
            "confidence": response.confidence,
            "refusal": refusal,
            "refusal_reason": refusal_reason,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "pii_redacted_count": pii_redacted_count,
            "prompt_injection_detected": prompt_injection,
            "answer_compliance_score": compliance_score,
            "model": cfg["llm"].get("model"),
        }
    )
    return response
