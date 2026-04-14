"""
App metrics for the clickstream consumer: event-type counts, per-position engagement
counts, and NDCG@10 — written to /metrics-data/app_metrics.json for the metrics sidecar.
"""
import json
import math
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional


_NDCG_K = 10


def _compute_ndcg_at_k(engagement_by_position: Dict[int, int], k: int = _NDCG_K) -> float:
    """
    Compute NDCG@k from cumulative engagement counts per rank position (1-indexed).
    DCG  = sum(eng[i] / log2(i + 1))  for i in 1..k  (i=1 → log2(2)=1)
    IDCG = DCG with counts sorted descending (ideal ordering)
    """
    dcg = sum(
        engagement_by_position.get(i, 0) / math.log2(i + 1)
        for i in range(1, k + 1)
    )
    counts_sorted = sorted(engagement_by_position.values(), reverse=True)[:k]
    idcg = sum(c / math.log2(i + 2) for i, c in enumerate(counts_sorted))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


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
        # Engagement by rank position 1..N (populated when position map is available in Redis)
        self._engagement_by_position: Dict[int, int] = defaultdict(int)

    def record_event(self, event_type: str) -> None:
        with self._lock:
            self._events_processed[event_type] += 1

    def record_engagement_at_position(self, position: int) -> None:
        """Record that the book at `position` (1-indexed) in recommendations was engaged with."""
        with self._lock:
            self._engagement_by_position[position] += 1

    def _build_payload(self) -> dict:
        with self._lock:
            engagement_copy = dict(self._engagement_by_position)
        ndcg = _compute_ndcg_at_k(engagement_copy, _NDCG_K)
        with self._lock:
            return {
                "events_processed_total": dict(self._events_processed),
                "engagement_by_position": {str(k): v for k, v in engagement_copy.items()},
                "recommendation_ndcg_at_10": ndcg,
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

record_event = _store.record_event
record_engagement_at_position = _store.record_engagement_at_position
start_metrics_writer = _store.start_metrics_writer
