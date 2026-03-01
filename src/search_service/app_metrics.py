"""
Write /metrics-data/app_metrics.json for the metrics sidecar. No Prometheus dependency.
"""
import json
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional


class MetricsStore:
    """Thread-safe store for HTTP metrics, written to JSON for the metrics sidecar."""

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

    def _build_payload(self) -> dict:
        with self._lock:
            return {
                "http_requests_total": dict(self._http_requests),
                "http_errors_total": dict(self._http_errors),
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
start_metrics_writer = _store.start_metrics_writer
