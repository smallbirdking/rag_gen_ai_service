from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List


class JsonLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, record: Dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class MetricsStore:
    latencies_ms: List[int] = field(default_factory=list)
    token_inputs: int = 0
    token_outputs: int = 0
    token_cached_inputs: int = 0
    total_requests: int = 0
    refusals: int = 0
    cache_hits: int = 0
    compliance_scores: List[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def add(self, latency_ms: int, token_usage: Dict[str, int], refusal: bool, cache_hit: bool, compliance: float | None = None) -> None:
        with self._lock:
            self.total_requests += 1
            self.latencies_ms.append(latency_ms)
            self.token_inputs += int(token_usage.get("input_tokens", 0))
            self.token_outputs += int(token_usage.get("output_tokens", 0))
            self.token_cached_inputs += int(token_usage.get("cached_input_tokens", 0))
            self.refusals += int(refusal)
            self.cache_hits += int(cache_hit)
            if compliance is not None:
                self.compliance_scores.append(compliance)

    @staticmethod
    def percentile(values: List[int], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = min(len(sorted_values) - 1, max(0, int(round((p / 100) * (len(sorted_values) - 1)))))
        return float(sorted_values[k])

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests": self.total_requests,
                "p50_latency_ms": self.percentile(self.latencies_ms, 50),
                "p90_latency_ms": self.percentile(self.latencies_ms, 90),
                "p95_latency_ms": self.percentile(self.latencies_ms, 95),
                "avg_latency_ms": statistics.mean(self.latencies_ms) if self.latencies_ms else 0,
                "input_tokens": self.token_inputs,
                "cached_input_tokens": self.token_cached_inputs,
                "output_tokens": self.token_outputs,
                "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests else 0.0,
                "refusal_rate": self.refusals / self.total_requests if self.total_requests else 0.0,
                "answer_compliance_rate": statistics.mean(self.compliance_scores) if self.compliance_scores else 0.0,
            }

    def write_csv_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
