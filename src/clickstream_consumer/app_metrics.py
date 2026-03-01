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
    """Thread-safe store for event metrics, written to JSON for the metrics sidecar."""

    def __init__(
        self,
        metrics_file: Optional[str] = None,
        write_interval: Optional[float] = None,
    ) -> None:
        self._metrics_file = metrics_file or os.getenv("METRICS_FILE", "/metrics-data/app_metrics.json")
        self._write_interval = write_interval or float(os.getenv("METRICS_WRITE_INTERVAL", "15.0"))
        self._lock = threading.Lock()
        self._events_processed: Dict[str, int] = defaultdict(int)

    def record_event(self, event_type: str) -> None:
        with self._lock:
            self._events_processed[event_type] += 1

    def _build_payload(self) -> dict:
        with self._lock:
            return {"events_processed_total": dict(self._events_processed)}

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

record_event = _store.record_event
start_metrics_writer = _store.start_metrics_writer
