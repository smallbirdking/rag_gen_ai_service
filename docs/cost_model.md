# Token Cost Model

Pricing snapshot date: 2026-05-18. Update these numbers before final submission if vendor pricing changes.

Default model selection:

- Generation model: `gpt-5.4-mini`
- Evaluation model: `gpt-5.4-nano`
- Local retrieval/reranking: no per-token API cost in this implementation

## Model-Version Selection Rationale

The service uses a two-model strategy:

- `gpt-5.4-mini` for user-facing answer generation.
- `gpt-5.4-nano` for automated LLM-as-judge evaluation.

The core trade-off is that answer generation is customer-visible and must preserve bilingual accuracy, instruction following, citation discipline, and refusal behavior. Evaluation is internal, structured, and lower risk, so it can use a smaller model to reduce cost.

| Decision | Selected model | Quality trade-off | Cost trade-off | Latency trade-off | Rationale |
|---|---|---|---|---|---|
| User-facing RAG answer generation | `gpt-5.4-mini` | Stronger instruction following and bilingual answer quality than a nano-class model; enough for grounded QA without using a flagship model by default | Higher than `gpt-5.4-nano`, but materially lower than a flagship model | Real load test p90 stayed below the 10s target with 5 concurrent requests | Best default balance for internal QA where hallucination/refusal quality matters |
| LLM-as-judge scoring | `gpt-5.4-nano` | Lower quality margin than `mini`, but the task is constrained JSON scoring over retrieved context, answer, and expected metadata | Much cheaper than using `mini` for every judge call | Expected to be lower latency and lower variance than larger models | Keeps evaluation affordable because each eval case calls a judge in addition to the generator |
| Flagship model for generation | Not default | Highest expected quality | Highest expected cost | Higher or less predictable latency | Reserve for high-risk workflows or failed quality gates |
| `gpt-5.4-nano` for generation | Not default | Higher risk for bilingual nuance, citation discipline, and refusal consistency | Lowest generation cost | Lowest expected latency | Useful for local smoke tests or low-risk summarization, but not the default for submitted RAG QA |

## Quantitative Evidence

Current real evaluation results meet the case-study quality targets:

| Config | Faithfulness | Context precision | Answer compliance | Style consistency | Refusal appropriateness | p95 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vector-only | 1.0000 | 1.0000 | 1.0000 | 0.9625 | 1.0000 | 4508.0 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 0.9375 | 1.0000 | 3068.0 |
| hybrid+rerank | 1.0000 | 1.0000 | 1.0000 | 0.9125 | 1.0000 | 2769.0 |

The real OpenAI concurrency load test uses `gpt-5.4-mini`, disables the application cache, and sends 25 requests with 5 concurrent workers. The latest recorded run passed the target of p90 latency under 10 seconds:

| Metric | Value |
|---|---:|
| Requests | 25 |
| Failures | 0 |
| p50 latency ms | 2443.97 |
| p90 latency ms | 5598.26 |
| p95 latency ms | 6236.46 |
| Cache hit rate | 0.0 |

These results support `gpt-5.4-mini` as the default generation model: it met the quality thresholds while staying within the latency target on a single instance at 5-way concurrency.

## Operational Policy

Use this model policy during iteration:

- Keep `gpt-5.4-mini` as the default generator while faithfulness, answer compliance, refusal appropriateness, and p90 latency remain within target.
- Upgrade to a stronger model only if real evaluation shows repeated failures in groundedness, bilingual accuracy, refusal behavior, or executive-facing writing quality.
- Downgrade generation to `gpt-5.4-nano` only for low-risk tasks after a separate quality gate confirms it still meets thresholds.
- Keep the evaluator on `gpt-5.4-nano` unless judge rationales become unstable or disagree with manual review.
- Re-run `scripts/evaluate.py` and `scripts/concurrency_load_test.py --url ...` whenever changing `llm.model`, retrieval mode, reranker behavior, or prompts.

## Formula

For 1,000 calls:

```text
cost = 1000 * (
  avg_input_tokens / 1_000_000 * input_usd_per_1m_tokens
  + avg_cached_input_tokens / 1_000_000 * cached_input_usd_per_1m_tokens
  + avg_output_tokens / 1_000_000 * output_usd_per_1m_tokens
)
```

## Example estimate

Assume each call uses 1,100 input tokens, 0 cached input tokens, and 250 output tokens.

| Model | Input $/1M | Cached input $/1M | Output $/1M | Estimated cost / 1,000 calls |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 0.75 | 0.075 | 4.50 | USD 1.95 |
| gpt-5.4-nano | 0.20 | 0.02 | 1.25 | USD 0.51 |

## Trade-off

- `gpt-5.4-mini`: default for answer generation; better instruction following and bilingual generation quality.
- `gpt-5.4-nano`: default for automatic evaluation or low-risk summaries; lower cost, but expected quality margin is smaller.
- Prompt caching: useful when system/developer instructions and reusable context are stable. Cache hit metrics are tracked separately because provider-side caching and application TTL caching are different optimizations.
