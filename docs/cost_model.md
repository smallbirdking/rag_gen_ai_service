# Token Cost Model

Pricing snapshot date: 2026-05-18. Update these numbers before final submission if vendor pricing changes.

Default model selection:

- Generation model: `gpt-5.4-mini`
- Evaluation model: `gpt-5.4-nano`
- Local retrieval/reranking: no per-token API cost in this implementation

## Why `gpt-5.4-mini` for generation

`gpt-5.4-mini` is selected as the default production candidate because it is materially cheaper and lower-latency than flagship models while still being suitable for grounded internal QA and style-controlled generation. The service is designed so `llm.model` can be changed without code modifications.

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
