---
doc_id: architecture_doc_cn_en
title: Architecture Document CN EN
language: bilingual
owner: Platform
version: 2026.03
---

# Architecture Document

## Request Flow
The API receives a multi-turn question with session_id and messages. It normalizes the latest user query, applies prompt-injection checks, retrieves candidate chunks, optionally reranks them, builds a grounded prompt, generates an answer, redacts PII, writes structured logs, updates metrics, and returns answer plus citations.

## 严格基于上下文回答
生成模块不得编造内部制度、接口参数或人员信息。若上下文不足，应拒答并说明需要补充哪类文档。回答需要列出引用的 doc_id 与 chunk_id，方便问题复现和审计。

## Observability
Every request should emit a JSON log record. Required fields include timestamp, request_id, session_id, normalized_query_hash, retrieval_mode, reranker_enabled, top_k, retrieved_chunk_ids, retrieval_scores, confidence, refusal_reason, cache_hit, latency_ms, token_usage, pii_redacted_count, prompt_injection_detected, answer_compliance_score, and model.

## Evolvability
The service separates ingestion, retrieval, reranking, generation, safety, cache, metrics, and API layers. This allows iterative optimization without changing public API contracts. Model version, retrieval mode, reranker switch, confidence threshold, and top_k are configuration items.
