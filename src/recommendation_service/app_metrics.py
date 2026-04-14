"""
App metrics for the recommendation service: HTTP, cache hit/miss, recommendation
duration percentiles, algorithm candidates, impressions, and NDCG — written to
/metrics-data/app_metrics.json for the metrics sidecar.
"""
import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional


_DURATION_WINDOW = 500  # rolling samples for percentile computation


class MetricsStore:
    """Thread-safe store for HTTP and recommendation metrics, written to JSON for the metrics sidecar."""

    def __init__(
        self,
        metrics_file: Optional[str] = None,
        write_interval: Optional[float] = None,
    ) -> None:
        self._metrics_file = metrics_file or os.getenv("METRICS_FILE", "/metrics-data/app_metrics.json")
        self._write_interval = write_interval or float(os.getenv("METRICS_WRITE_INTERVAL", "15.0"))
        self._lock = threading.Lock()
        self._http_requests: Dict[str, int] = defaultdict(int)
        self._http_errors: Dict[str, int] = defaultdict(int)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._impressions: int = 0
        # Rolling window of pipeline durations (cache-miss path only)
        self._duration_samples: deque = deque(maxlen=_DURATION_WINDOW)
        # Cumulative candidate counts per algorithm
        self._candidates_by_algorithm: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _status_class(status_code: int) -> str:
        if status_code < 400:
            return "2xx"
        if status_code < 500:
            return "4xx"
        return "5xx"

    @staticmethod
    def _percentile(sorted_samples: List[float], p: float) -> float:
        """Return the p-th percentile (0–100) from a pre-sorted list."""
        if not sorted_samples:
            return 0.0
        idx = (p / 100.0) * (len(sorted_samples) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_samples) - 1)
        return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (idx - lo)

    def record_request(self, method: str, path: str, status_code: int) -> None:
        key = f"{method}|{path}|{self._status_class(status_code)}"
        with self._lock:
            self._http_requests[key] += 1

    def record_error(self, method: str, path: str) -> None:
        key = f"{method}|{path}"
        with self._lock:
            self._http_errors[key] += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    def record_impression(self) -> None:
        """Increment every time a recommendations response is served (cache hit or miss)."""
        with self._lock:
            self._impressions += 1

    def record_recommendation_duration_seconds(self, seconds: float) -> None:
        """Record pipeline duration (cache-miss path). Added to rolling window for percentiles."""
        with self._lock:
            self._duration_samples.append(seconds)

    def record_algorithm_candidates(self, algorithm: str, count: int) -> None:
        """Accumulate candidate counts per algorithm (cbr, als, graph, session, popularity, linucb)."""
        with self._lock:
            self._candidates_by_algorithm[algorithm] += count

    def _build_payload(self) -> dict:
        with self._lock:
            sorted_samples = sorted(self._duration_samples)
            return {
                "http_requests_total": dict(self._http_requests),
                "http_errors_total": dict(self._http_errors),
                "recommendation_cache_hits_total": self._cache_hits,
                "recommendation_cache_misses_total": self._cache_misses,
                "recommendation_impressions_total": self._impressions,
                # Keep last-value for backward compat with existing Grafana panels
                "recommendation_duration_seconds": sorted_samples[-1] if sorted_samples else 0.0,
                # Percentiles
                "recommendation_duration_p50_seconds": self._percentile(sorted_samples, 50),
                "recommendation_duration_p95_seconds": self._percentile(sorted_samples, 95),
                "recommendation_duration_p99_seconds": self._percentile(sorted_samples, 99),
                # Algorithm candidates (cumulative)
                "candidates_by_algorithm": dict(self._candidates_by_algorithm),
            }

    def _write_loop(self) -> None:
        while True:
            time.sleep(self._write_interval)
            parent = os.path.dirname(self._metrics_file)
            if not os.path.isdir(parent):
                continue
            try:
                payload = self._build_payload()
                with open(self._metrics_file, "w") as f:
                    json.dump(payload, f, indent=0)
            except Exception:
                pass

    def start_metrics_writer(self) -> None:
        t = threading.Thread(target=self._write_loop, daemon=True)
        t.start()


_store = MetricsStore()

record_request = _store.record_request
record_error = _store.record_error
record_cache_hit = _store.record_cache_hit
record_cache_miss = _store.record_cache_miss
record_impression = _store.record_impression
record_recommendation_duration_seconds = _store.record_recommendation_duration_seconds
record_algorithm_candidates = _store.record_algorithm_candidates
start_metrics_writer = _store.start_metrics_writer
