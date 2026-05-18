# Log Field Dictionary

All logs are emitted as one JSON object per line to `logs/app.jsonl`.

| Field | Type | Description | Example |
|---|---:|---|---|
| timestamp | string | UTC ISO-8601 time when record was emitted | `2026-05-18T02:30:00Z` |
| request_id | string | UUID for one API request | `b9c...` |
| session_id | string | Caller-provided session id for multi-turn diagnosis | `eval` |
| normalized_query_hash | string | SHA-256 prefix of normalized query; avoids raw PII in logs | `29be0d...` |
| retrieval_mode | string | `vector_only` or `hybrid` | `hybrid` |
| reranker_enabled | boolean | Whether reranker was applied from config/request | `true` |
| top_k | integer | Number of context chunks requested | `5` |
| retrieved_chunk_ids | array[string] | Final retrieved/reranked chunk ids | `technical_spec_cn::chunk_001` |
| retrieval_scores | array[number] | Final normalized retrieval/rerank scores | `[0.93, 0.52]` |
| confidence | number | Top context final score | `0.93` |
| refusal | boolean | Whether service refused to answer | `false` |
| refusal_reason | string/null | `prompt_injection_detected`, `low_retrieval_confidence`, etc. | `low_retrieval_confidence` |
| cache_hit | boolean | Whether response came from TTL cache | `false` |
| latency_ms | integer | End-to-end service latency | `38` |
| token_usage.input_tokens | integer | Approximate or provider-reported input tokens | `712` |
| token_usage.cached_input_tokens | integer | Cached input tokens where provider reports them | `0` |
| token_usage.output_tokens | integer | Approximate or provider-reported output tokens | `82` |
| pii_redacted_count | integer | Number of PII spans redacted in output | `1` |
| prompt_injection_detected | boolean | Prompt injection rule triggered | `false` |
| answer_compliance_score | number/null | Eval-time compliance score, omitted in normal traffic | `1.0` |
| model | string | Configured generator model | `gpt-5.4-mini` |

## Sample Log

```json
{"timestamp":"2026-05-18T02:30:00.000000+00:00","request_id":"b9c2f7a6-5cc4-49f0-8a4f-3f724d7965ba","session_id":"eval","normalized_query_hash":"e32a66e7fdd86c13","retrieval_mode":"hybrid","reranker_enabled":true,"top_k":5,"retrieved_chunk_ids":["technical_spec_cn::chunk_001","architecture_doc_cn_en::chunk_001"],"retrieval_scores":[0.9123,0.4881],"confidence":0.9123,"refusal":false,"refusal_reason":null,"cache_hit":false,"latency_ms":42,"token_usage":{"input_tokens":690,"output_tokens":86,"cached_input_tokens":0},"pii_redacted_count":0,"prompt_injection_detected":false,"answer_compliance_score":1.0,"model":"gpt-5.4-mini"}
```
