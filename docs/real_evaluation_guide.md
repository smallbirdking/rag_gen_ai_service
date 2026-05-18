# Real Evaluation Guide

The final evaluation path is `scripts/evaluate.py`. It is designed to produce real, submission-ready results.

## What changed

- `scripts/evaluate.py` requires `OPENAI_API_KEY`.
- It calls the live generator model for every evaluation question.
- It calls a separate live evaluator model as an LLM-as-judge.
- It records generator model, evaluator model, UTC timestamp, raw answer, citations, token usage, latency, judge rationale, and scores in CSV files.
- The old deterministic evaluation is preserved only as `scripts/evaluate_mock.py` for local smoke testing.

## Run on Windows PowerShell

```powershell
cd rag_gen_ai_service
.venv\Scripts\activate
$env:PYTHONPATH="$PWD\src"
$env:OPENAI_API_KEY="sk-..."
python scripts/evaluate.py --provider openai --model gpt-5.4-mini --evaluator-model gpt-5.4-nano
```

## Run on macOS / Linux

```bash
cd rag_gen_ai_service
source .venv/bin/activate
export PYTHONPATH=$PWD/src
export OPENAI_API_KEY='sk-...'
python scripts/evaluate.py --provider openai --model gpt-5.4-mini --evaluator-model gpt-5.4-nano
```

## Output files

- `docs/evaluation_report.md`: summary report
- `docs/eval_outputs/eval_summary.csv`: one row per retrieval configuration
- `docs/eval_outputs/eval_details.csv`: one row per question/configuration, including raw model answer and judge rationale
- `logs/app.jsonl`: structured service logs

## Cost control

A full run evaluates 8 questions across 3 retrieval configurations, so it makes 24 generator calls and 24 evaluator calls. Use this before a full run:

```bash
python scripts/evaluate.py --limit 1
```

This still uses real API calls but only one evaluation case per retrieval configuration.
