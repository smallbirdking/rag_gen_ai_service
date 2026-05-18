#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_service.config import deep_update, load_config, load_project_env  # noqa: E402
from rag_service.evaluation_llm import LLMJudge  # noqa: E402
from rag_service.quality import context_precision  # noqa: E402
from rag_service.schemas import AskRequest  # noqa: E402
from rag_service.service import answer_question, build_runtime  # noqa: E402


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows]

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        ordered = sorted(latencies)
        idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
        return float(ordered[idx])

    def mean(name: str) -> float:
        return round(statistics.mean(float(r[name]) for r in rows), 4) if rows else 0.0

    return {
        "cases": len(rows),
        "faithfulness": mean("faithfulness"),
        "context_precision": mean("context_precision"),
        "answer_compliance": mean("answer_compliance"),
        "style_consistency": mean("style_consistency"),
        "refusal_appropriateness": mean("refusal_appropriateness"),
        "refusal_rate": round(sum(1 for r in rows if r["refusal"]) / len(rows), 4) if rows else 0.0,
        "p50_latency_ms": pct(50),
        "p90_latency_ms": pct(90),
        "p95_latency_ms": pct(95),
        "total_input_tokens": sum(int(r["input_tokens"]) for r in rows),
        "total_output_tokens": sum(int(r["output_tokens"]) for r in rows),
        "judge_input_tokens": sum(int(r["judge_input_tokens"]) for r in rows),
        "judge_output_tokens": sum(int(r["judge_output_tokens"]) for r in rows),
        "cache_hit_rate": round(sum(1 for r in rows if r["cache_hit"]) / len(rows), 4) if rows else 0.0,
    }


def require_real_provider(provider: str) -> None:
    if provider != "openai":
        raise SystemExit("Final evaluation requires --provider openai. Use scripts/evaluate_mock.py only for local smoke tests.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required for real evaluation.\n"
            "PowerShell: $env:OPENAI_API_KEY='sk-...'\n"
            "macOS/Linux: export OPENAI_API_KEY='sk-...'"
        )
    if api_key in {"sk-...", "sk-"} or "..." in api_key:
        raise SystemExit(
            "OPENAI_API_KEY still looks like a placeholder. Replace it with a real key from your OpenAI project."
        )


def run_config(
    base_config_path: str,
    mode: str,
    reranker: bool,
    eval_rows: List[Dict[str, Any]],
    provider: str,
    model: str,
    evaluator_model: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = load_config(base_config_path)
    config.raw = deep_update(
        config.raw,
        {
            "retrieval": {"mode": mode, "reranker_enabled": reranker},
            "llm": {"provider": provider, "model": model, "evaluator_model": evaluator_model},
        },
    )
    runtime = build_runtime(config)
    judge = LLMJudge(provider=provider, model=evaluator_model, temperature=0.0)
    details = []
    total = len(eval_rows)
    for idx, case in enumerate(eval_rows, start=1):
        print(f"[{mode}{'+rerank' if reranker else ''}] case {idx}/{total}: {case['id']}", flush=True)
        req = AskRequest(question=case["question"], session_id="eval-real", retrieval_mode=mode, reranker_enabled=reranker)
        resp = answer_question(runtime, req, eval_context=case)
        retrieved = runtime.retriever.search(case["question"], mode=mode, top_k=5)
        cited_doc_ids = [c.doc_id for c in resp.citations]
        if resp.refusal:
            cp = 1.0
        else:
            cp = sum(1 for doc_id in cited_doc_ids if doc_id in set(case.get("gold_doc_ids", []))) / max(1, len(cited_doc_ids))

        judged_contexts = [r for r in retrieved if r.chunk.doc_id in set(cited_doc_ids)] or retrieved[:5]
        judge_result = judge.evaluate(
            question=case["question"],
            answer=resp.answer,
            contexts=judged_contexts,
            expected_keywords=case.get("expected_keywords", []),
            expected_refusal=bool(case.get("expected_refusal", False)),
            actual_refusal=resp.refusal,
        )
        row = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "id": case["id"],
            "question": case["question"],
            "mode": mode,
            "reranker_enabled": reranker,
            "generator_provider": provider,
            "generator_model": model,
            "evaluator_model": evaluator_model,
            "answer": resp.answer,
            "refusal": resp.refusal,
            "refusal_reason": resp.refusal_reason or "",
            "confidence": resp.confidence,
            "citations": ";".join(f"{c.doc_id}/{c.chunk_id}:{c.score}" for c in resp.citations),
            "context_precision": round(cp, 4),
            "faithfulness": round(judge_result.faithfulness, 4),
            "answer_compliance": round(judge_result.answer_compliance, 4),
            "style_consistency": round(judge_result.style_consistency, 4),
            "refusal_appropriateness": round(judge_result.refusal_appropriateness, 4),
            "judge_rationale": judge_result.rationale,
            "latency_ms": resp.latency_ms,
            "input_tokens": resp.token_usage.get("input_tokens", 0),
            "output_tokens": resp.token_usage.get("output_tokens", 0),
            "judge_input_tokens": judge_result.input_tokens,
            "judge_output_tokens": judge_result.output_tokens,
            "cache_hit": resp.cache_hit,
        }
        details.append(row)
    return details, summarize(details)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, summaries: List[Dict[str, Any]], model: str, evaluator_model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["config", "faithfulness", "context_precision", "answer_compliance", "style_consistency", "refusal_appropriateness", "p95_latency_ms", "refusal_rate"]
    lines = [
        "# Evaluation Report",
        "",
        "This report is generated by `scripts/evaluate.py` using real OpenAI API calls for both answer generation and LLM-as-judge evaluation.",
        "It is not based on deterministic mock answers. Re-run it with your own `OPENAI_API_KEY` before submission so the CSV files contain fresh, reproducible real-run outputs.",
        "",
        f"- Generator model: `{model}`",
        f"- Evaluator model: `{evaluator_model}`",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for s in summaries:
        lines.append("| " + " | ".join(str(s[h]) for h in headers) + " |")
    lines.extend(
        [
            "",
            "## Method",
            "For each evaluation case, the script calls the live generator model through the OpenAI Responses API, records the raw answer and token usage, then calls a separate evaluator model to score faithfulness, answer compliance, style consistency, and refusal appropriateness on a 0-1 rubric. Context precision remains deterministic because it compares cited document IDs against the gold document IDs in `data/eval/eval_set.jsonl`.",
            "",
            "## How to Reproduce",
            "```bash",
            "export OPENAI_API_KEY='sk-...'",
            "python scripts/evaluate.py --provider openai --model gpt-5.4-mini --evaluator-model gpt-5.4-nano",
            "```",
            "",
            "Detailed per-question real outputs are stored in `docs/eval_outputs/eval_details.csv`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Run final real-API RAG evaluation. Requires OPENAI_API_KEY.")
    parser.add_argument("--config", default=str(ROOT / "config/app.yaml"))
    parser.add_argument("--eval-set", default=str(ROOT / "data/eval/eval_set.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs/eval_outputs"))
    parser.add_argument("--provider", default=os.getenv("RAG_LLM_PROVIDER", "openai"), choices=["openai"])
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--evaluator-model", default=os.getenv("RAG_EVALUATOR_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--limit", type=int, default=0, help="Optional number of eval cases for a paid smoke run; 0 means all cases.")
    args = parser.parse_args()

    require_real_provider(args.provider)
    eval_rows = load_eval_set(Path(args.eval_set))
    if args.limit and args.limit > 0:
        eval_rows = eval_rows[: args.limit]

    matrix = [
        ("vector_only", False, "vector-only"),
        ("hybrid", False, "hybrid"),
        ("hybrid", True, "hybrid+rerank"),
    ]
    all_details: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    try:
        for mode, reranker, label in matrix:
            details, summary = run_config(args.config, mode, reranker, eval_rows, args.provider, args.model, args.evaluator_model)
            for row in details:
                row["config"] = label
            summary["config"] = label
            all_details.extend(details)
            summaries.append(summary)
    except Exception as exc:
        if exc.__class__.__name__ == "AuthenticationError":
            raise SystemExit(
                "OpenAI authentication failed: the API key currently visible to this process was rejected.\n"
                "Check whether your shell has an older OPENAI_API_KEY set; shell variables override .env in this project.\n"
                "macOS/Linux: run `unset OPENAI_API_KEY`, update `.env`, then run the evaluation again."
            ) from exc
        if exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}:
            raise SystemExit(
                "OpenAI request failed before a response was received. Check network/proxy access to api.openai.com.\n"
                "The client now uses OPENAI_TIMEOUT_SECONDS, defaulting to 30 seconds. Example: `OPENAI_TIMEOUT_SECONDS=10 python scripts/evaluate.py --limit 1 ...`"
            ) from exc
        raise

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "eval_details.csv", all_details)
    write_csv(out_dir / "eval_summary.csv", summaries)
    write_summary_markdown(ROOT / "docs/evaluation_report.md", summaries, args.model, args.evaluator_model)
    print(json.dumps({"summaries": summaries, "details_csv": str(out_dir / "eval_details.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
