# RAG + Generative AI Service Case Study

This repository implements a configurable bilingual RAG QA + generative service for an internal knowledge base. It is built for the AIA case-study requirements: vector-only and hybrid retrieval, optional reranking, refusal handling, PII redaction, structured logs, caching, one-click evaluation, and quantitative reporting.

## 1. Requirement Mapping

| Case-study requirement | Implementation |
|---|---|
| Multi-turn RAG QA over internal knowledge base | `POST /ask` accepts `session_id`, `history`, and `question` |
| Bilingual CN/EN corpus | Sample CN/EN corpus under `data/corpus` with bilingual tokenization |
| Small portion of scanned PDFs | OCR text sample and ingestion fallback pattern in `data/corpus/scanned_pdf_ocr_sample.txt` |
| Retrieval controls | `config/app.yaml` supports `vector_only` and `hybrid`; request-level override supported |
| Reranker config switch | `retrieval.reranker_enabled` in config or request, no code change needed |
| Low-confidence/out-of-scope refusal | Confidence threshold + prompt-injection rules in `security.py` and `service.py` |
| PII handling | Output/log redaction in `security.py`; query logs store hashes instead of raw text |
| Structured logging/tracing | JSONL logs at `logs/app.jsonl`; dictionary in `docs/log_field_dictionary.md` |
| Caching | TTL application cache keyed by normalized query + retrieval/model settings |
| Minimal ops report | `/admin/report` and `MetricsStore.write_csv_report()` produce CSV metrics |
| Three retrieval comparisons | `scripts/evaluate.py` runs vector-only, hybrid, hybrid+rerank |
| Evaluation report | Generated to `docs/evaluation_report.md` and `docs/eval_outputs/*.csv` |
| Issue diagnosis | `docs/issue_diagnosis.md` contains evidence/fix/post-fix template |

## 2. Architecture

```text
Client
  -> FastAPI /ask
    -> safety check: prompt injection + request normalization
    -> retrieval: vector-only or hybrid BM25 + TF-IDF cosine
    -> optional reranker: local deterministic reranker over top-k
    -> confidence gate: refuse if top score < threshold
    -> grounded generator: mock local generator or OpenAI Responses API
    -> PII redaction
    -> structured JSON log + metrics + cache
    -> answer with citations
```

The implementation intentionally separates ingestion, chunking, retrieval, reranking, generation, safety, observability, and API layers so that retrieval strategy, model version, logging fields, and thresholds can evolve independently.

## 3. Quick Start

```bash
cd rag_gen_ai_service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python scripts/evaluate_mock.py  # local smoke test without API key
uvicorn rag_service.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"demo",
    "question":"RAG 服务需要支持哪两种检索模式？",
    "retrieval_mode":"hybrid",
    "reranker_enabled":true
  }'
```

Generate a minimal operations report:

```bash
curl -X POST 'http://localhost:8000/admin/report?path=docs/operations_report.csv'
```

Run the default 5-concurrent-request local load test without paid API calls:

```bash
python scripts/concurrency_load_test.py --concurrency 5 --requests 25
```

Load-test outputs are written under `docs/load_test_outputs/`. To test a live FastAPI instance instead, pass `--url http://localhost:8000/ask`.

## 4. Configuration

Main config: `config/app.yaml`.

Important options:

```yaml
retrieval:
  mode: hybrid          # vector_only | hybrid
  top_k: 5
  min_confidence: 0.18
  hybrid_alpha: 0.62
  reranker_enabled: true

llm:
  provider: openai      # final evaluation uses openai; mock is only for local smoke tests
  model: gpt-5.4-mini
```

Environment overrides:

```bash
export RAG_RETRIEVAL_MODE=hybrid
export RAG_RERANKER_ENABLED=true
export RAG_LLM_PROVIDER=openai
export RAG_MODEL=gpt-5.4-mini
```

Final evaluation and production-style runs use OpenAI. For a quick offline smoke test only, set `RAG_LLM_PROVIDER=mock` and run `python scripts/evaluate_mock.py`. To use OpenAI:

```bash
export OPENAI_API_KEY=sk-...
export RAG_LLM_PROVIDER=openai
uvicorn rag_service.main:app --host 0.0.0.0 --port 8000
```

The final evaluation script intentionally does not accept mock generation. It requires `OPENAI_API_KEY` so the submitted metrics come from real model calls.

## 5. Final Real Evaluation

The final evaluation now uses **real OpenAI API calls** for both answer generation and LLM-as-judge scoring. It will stop immediately if `OPENAI_API_KEY` is missing. This prevents accidentally submitting deterministic mock scores as final results.

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:PYTHONPATH="$PWD\src"
python scripts/evaluate.py --provider openai --model gpt-5.4-mini --evaluator-model gpt-5.4-nano
```

macOS / Linux:

```bash
export OPENAI_API_KEY='sk-...'
export PYTHONPATH=$PWD/src
python scripts/evaluate.py --provider openai --model gpt-5.4-mini --evaluator-model gpt-5.4-nano
```

For a low-cost paid smoke test, run only the first case from each retrieval configuration:

```bash
python scripts/evaluate.py --limit 1
```

Offline smoke test, not for submission:

```bash
python scripts/evaluate_mock.py
```

Mock outputs are written under `docs/mock_eval_outputs/` so they do not overwrite the real evaluation artifacts.

Outputs:

- `docs/evaluation_report.md`
- `docs/eval_outputs/eval_summary.csv`
- `docs/eval_outputs/eval_details.csv`
- `logs/app.jsonl`

Metrics implemented:

| Metric | Final method |
|---|---|
| Faithfulness | LLM-as-judge score from a separate evaluator model, using retrieved context and raw answer |
| Context Precision | Deterministic comparison between cited `doc_id` and gold `doc_id` |
| Answer Compliance | LLM-as-judge score against the question and expected concepts/refusal metadata |
| Style Consistency | LLM-as-judge score for concise, professional, cited style |
| Refusal Appropriateness | LLM-as-judge score plus expected refusal metadata |
| Latency | Real end-to-end API latency, p50 / p90 / p95 in milliseconds |
| Token usage | Provider-reported token usage when available |
| Cache hit rate | Application TTL cache hit count / total requests |

## 6. Security Controls

Implemented minimal controls:

1. Treat retrieved context as untrusted data.
2. Refuse prompt-injection patterns such as “ignore previous instructions” and hidden prompt extraction.
3. Refuse low-confidence or out-of-scope questions.
4. Redact emails, phone-like numbers, and Chinese ID-like patterns from generated output.
5. Avoid raw query text in structured logs; store a stable hash instead.

Production hardening suggestions:

- Add tenant/user authorization and document-level ACL filtering before retrieval.
- Use a stronger PII detector and classify logs by sensitivity.
- Add model-based groundedness evaluator for release gates.
- Store trace IDs in OpenTelemetry-compatible format.

## 7. Performance and Concurrency

The service is stateless except for in-memory cache/metrics and is safe for a small single-instance demo. The case-study target is p90 end-to-end latency below 10 seconds and at least 5 concurrent requests on one instance. For a real deployment, run:

```bash
uvicorn rag_service.main:app --host 0.0.0.0 --port 8000 --workers 2
```

For production, move cache and metrics to Redis/Prometheus and run a load test such as `hey` or `wrk` against `/ask`.

## 8. Cost Estimate

See `docs/cost_model.md`. The config defaults to `gpt-5.4-mini` for generation and `gpt-5.4-nano` for lower-cost evaluation. Retrieval/reranking are local and do not add per-token vendor cost in this implementation.

## 9. Repository Layout

```text
rag_gen_ai_service/
  config/app.yaml
  data/corpus/
  data/eval/eval_set.jsonl
  docs/
    cost_model.md
    evaluation_report.md
    issue_diagnosis.md
    log_field_dictionary.md
    operations_report.csv
  logs/app.jsonl
  scripts/evaluate.py
  scripts/run_demo.sh
  src/rag_service/
    cache.py
    chunking.py
    config.py
    generation.py
    main.py
    observability.py
    quality.py
    retrieval.py
    schemas.py
    security.py
    service.py
    text.py
  tests/
```

## 10. Notes for Interview / Submission

Suggested explanation:

- Start with hybrid+rerank as the default because internal corpora often mix natural-language questions, exact policy terms, and bilingual content.
- Keep vector-only as a fast/cheap baseline and for ablation comparison.
- Use refusal gating to protect against hallucination and out-of-scope questions.
- Use hashed query logs and PII redaction to reduce privacy risk while preserving reproducible diagnosis.
- Use config-driven model and retrieval switching to make optimization iterative rather than code-heavy.
