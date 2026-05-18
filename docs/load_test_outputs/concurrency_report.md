# Concurrency Load Test Report

- Generated at UTC: `2026-05-18T05:36:22.970141+00:00`
- Source: `http://127.0.0.1:8010/ask`
- Concurrency: `5`
- Retrieval mode: `hybrid`
- Reranker enabled: `True`
- Target: p90 latency <= `10.0` seconds with no failed requests
- Result: `PASS`

| Metric | Value |
|---|---:|
| Requests | 25 |
| Successes | 25 |
| Failures | 0 |
| p50 latency ms | 2766.72 |
| p90 latency ms | 6987.43 |
| p95 latency ms | 6988.72 |
| Max latency ms | 7122.45 |
| Total wall time ms | 18020.14 |
| Throughput req/s | 1.39 |
| Cache hit rate | 0.0 |
| Refusal rate | 0.0 |

## Method

This run targeted a live FastAPI instance at `http://127.0.0.1:8010/ask` using `config/load_test_openai.yaml`, where `llm.provider=openai` and `cache_ttl_seconds=0`. That means the measured requests exercised the HTTP API, request validation, retrieval, optional reranking, refusal handling, PII redaction, metrics/logging, and real OpenAI answer generation without application-cache hits.

Detailed per-request rows are stored in `docs/load_test_outputs/concurrency_details.csv`.
