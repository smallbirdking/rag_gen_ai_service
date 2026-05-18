#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_service.config import deep_update, load_config  # noqa: E402
from rag_service.schemas import AskRequest  # noqa: E402
from rag_service.service import answer_question, build_runtime  # noqa: E402


DEFAULT_QUESTIONS = [
    "How many annual leave days do full-time employees receive?",
    "远程办公最多每周几天？",
    "RAG 服务需要支持哪两种检索模式？",
    "What fields must be included in structured logs?",
    "员工的个人手机号和邮箱能不能原样写入日志？",
]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return float(ordered[idx])


def build_payload(idx: int, question: str, mode: str, reranker_enabled: bool) -> Dict[str, Any]:
    return {
        "session_id": f"load-test-{idx}",
        "question": question,
        "retrieval_mode": mode,
        "reranker_enabled": reranker_enabled,
    }


def run_in_process_request(runtime: Any, idx: int, question: str, mode: str, reranker_enabled: bool) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        request = AskRequest(**build_payload(idx, question, mode, reranker_enabled))
        response = answer_question(runtime, request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "request_index": idx,
            "ok": True,
            "status_code": 200,
            "latency_ms": round(elapsed_ms, 2),
            "service_latency_ms": response.latency_ms,
            "refusal": response.refusal,
            "cache_hit": response.cache_hit,
            "error": "",
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "request_index": idx,
            "ok": False,
            "status_code": 0,
            "latency_ms": round(elapsed_ms, 2),
            "service_latency_ms": 0,
            "refusal": False,
            "cache_hit": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_http_request(client: httpx.Client, url: str, idx: int, question: str, mode: str, reranker_enabled: bool) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(url, json=build_payload(idx, question, mode, reranker_enabled))
        elapsed_ms = (time.perf_counter() - started) * 1000
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "request_index": idx,
            "ok": response.is_success,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "service_latency_ms": data.get("latency_ms", 0),
            "refusal": data.get("refusal", False),
            "cache_hit": data.get("cache_hit", False),
            "error": "" if response.is_success else response.text[:500],
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "request_index": idx,
            "ok": False,
            "status_code": 0,
            "latency_ms": round(elapsed_ms, 2),
            "service_latency_ms": 0,
            "refusal": False,
            "cache_hit": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def summarize(rows: List[Dict[str, Any]], total_wall_ms: float, target_p90_seconds: float) -> Dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row["ok"]]
    failures = len([row for row in rows if not row["ok"]])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requests": len(rows),
        "successes": len(rows) - failures,
        "failures": failures,
        "p50_latency_ms": round(percentile(latencies, 50), 2),
        "p90_latency_ms": round(percentile(latencies, 90), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "total_wall_ms": round(total_wall_ms, 2),
        "throughput_rps": round((len(rows) / total_wall_ms) * 1000, 2) if total_wall_ms else 0.0,
        "cache_hit_rate": round(sum(1 for row in rows if row["cache_hit"]) / len(rows), 4) if rows else 0.0,
        "refusal_rate": round(sum(1 for row in rows if row["refusal"]) / len(rows), 4) if rows else 0.0,
        "target_p90_seconds": target_p90_seconds,
        "target_passed": failures == 0 and percentile(latencies, 90) <= target_p90_seconds * 1000,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Dict[str, Any], source: str, concurrency: int, mode: str, reranker_enabled: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if summary["target_passed"] else "FAIL"
    if source.startswith("http://") or source.startswith("https://"):
        method = (
            f"This run targeted a live FastAPI instance at `{source}` using the currently loaded service configuration. "
            "For the recorded real-API run, the service was started with `config/load_test_openai.yaml`, where "
            "`llm.provider=openai` and `cache_ttl_seconds=0`. That means the measured requests exercised the HTTP API, "
            "request validation, retrieval, optional reranking, refusal handling, PII redaction, metrics/logging, and real OpenAI answer generation without application-cache hits."
        )
    else:
        method = (
            "The default run uses the same service pipeline in-process with `llm.provider=mock` to avoid paid API calls and external network variance. "
            "It exercises request validation, retrieval, optional reranking, refusal handling, PII redaction, metrics, logging, and cache behavior under concurrent calls. "
            "Use `--url` to run the same request mix against a live FastAPI instance."
        )
    lines = [
        "# Concurrency Load Test Report",
        "",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Source: `{source}`",
        f"- Concurrency: `{concurrency}`",
        f"- Retrieval mode: `{mode}`",
        f"- Reranker enabled: `{reranker_enabled}`",
        f"- Target: p90 latency <= `{summary['target_p90_seconds']}` seconds with no failed requests",
        f"- Result: `{status}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Requests | {summary['requests']} |",
        f"| Successes | {summary['successes']} |",
        f"| Failures | {summary['failures']} |",
        f"| p50 latency ms | {summary['p50_latency_ms']} |",
        f"| p90 latency ms | {summary['p90_latency_ms']} |",
        f"| p95 latency ms | {summary['p95_latency_ms']} |",
        f"| Max latency ms | {summary['max_latency_ms']} |",
        f"| Total wall time ms | {summary['total_wall_ms']} |",
        f"| Throughput req/s | {summary['throughput_rps']} |",
        f"| Cache hit rate | {summary['cache_hit_rate']} |",
        f"| Refusal rate | {summary['refusal_rate']} |",
        "",
        "## Method",
        "",
        method,
        "",
        "Detailed per-request rows are stored in `docs/load_test_outputs/concurrency_details.csv`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small concurrent load test for the RAG service.")
    parser.add_argument("--config", default=str(ROOT / "config/app.yaml"))
    parser.add_argument("--url", default="", help="Optional live /ask endpoint URL. Omit for in-process mock load test.")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--mode", choices=["vector_only", "hybrid"], default="hybrid")
    parser.add_argument("--reranker-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-p90-seconds", type=float, default=10.0)
    parser.add_argument("--enable-cache", action="store_true", help="Keep cache enabled. Default disables cache for conservative latency measurement.")
    parser.add_argument("--out-dir", default=str(ROOT / "docs/load_test_outputs"))
    args = parser.parse_args()

    questions = [DEFAULT_QUESTIONS[i % len(DEFAULT_QUESTIONS)] for i in range(args.requests)]
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    if args.url:
        source = args.url
        with httpx.Client(timeout=args.target_p90_seconds + 5.0) as client:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(run_http_request, client, args.url, idx + 1, question, args.mode, args.reranker_enabled)
                    for idx, question in enumerate(questions)
                ]
                rows = [future.result() for future in as_completed(futures)]
    else:
        config = load_config(args.config)
        config.raw = deep_update(
            config.raw,
            {
                "app": {"cache_ttl_seconds": config.raw["app"].get("cache_ttl_seconds", 600) if args.enable_cache else 0},
                "retrieval": {"mode": args.mode, "reranker_enabled": args.reranker_enabled},
                "llm": {"provider": "mock"},
            },
        )
        runtime = build_runtime(config)
        source = "in-process mock runtime"
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(run_in_process_request, runtime, idx + 1, question, args.mode, args.reranker_enabled)
                for idx, question in enumerate(questions)
            ]
            rows = [future.result() for future in as_completed(futures)]

    total_wall_ms = (time.perf_counter() - started) * 1000
    rows.sort(key=lambda row: int(row["request_index"]))
    summary = summarize(rows, total_wall_ms, args.target_p90_seconds)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "concurrency_details.csv", rows)
    write_report(out_dir / "concurrency_report.md", summary, source, args.concurrency, args.mode, args.reranker_enabled)
    print(json.dumps({"summary": summary, "details_csv": str(out_dir / "concurrency_details.csv")}, ensure_ascii=False, indent=2))

    if not summary["target_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
