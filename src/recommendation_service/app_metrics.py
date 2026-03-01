"""
Write /metrics-data/app_metrics.json for the metrics sidecar. No Prometheus dependency.
Supports HTTP metrics and recommendation cache/duration metrics.
"""
import json
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional


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
        self._last_recommendation_duration_seconds: float = 0.0

    @staticmethod
    def _status_class(status_code: int) -> str:
        if status_code < 400:
            return "2xx"
        if status_code < 500:
            return "4xx"
        return "5xx"

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

    def record_recommendation_duration_seconds(self, seconds: float) -> None:
        """Record time taken to generate recommendation results (on cache miss)."""
        with self._lock:
            self._last_recommendation_duration_seconds = seconds

    def _build_payload(self) -> dict:
        with self._lock:
            return {
                "http_requests_total": dict(self._http_requests),
                "http_errors_total": dict(self._http_errors),
                "recommendation_cache_hits_total": self._cache_hits,
                "recommendation_cache_misses_total": self._cache_misses,
                "recommendation_duration_seconds": self._last_recommendation_duration_seconds,
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
record_recommendation_duration_seconds = _store.record_recommendation_duration_seconds
start_metrics_writer = _store.start_metrics_writer
