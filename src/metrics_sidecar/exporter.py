"""
Metrics sidecar: runs alongside each app pod and exposes /metrics in Prometheus format.
Reads app-provided metrics from a shared volume JSON file (/metrics-data/app_metrics.json)
and serves them as Prometheus gauges. No Prometheus or metrics logic in the main app—
the app only writes JSON; this sidecar handles exposition for scraping.
"""
import json
import os
import logging
import time
from prometheus_client import start_http_server, REGISTRY, Gauge, PROCESS_COLLECTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics-sidecar")

METRICS_FILE = os.environ.get("METRICS_FILE", "/metrics-data/app_metrics.json")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
SERVICE_NAME = os.environ.get("SERVICE_NAME", "app")
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL", "15.0"))

# --- HTTP metrics ---
app_http_requests_total = Gauge(
    "app_http_requests_total",
    "Total HTTP requests (from app metrics file)",
    ["method", "path", "status_class"],
    registry=REGISTRY,
)
app_http_errors_total = Gauge(
    "app_http_errors_total",
    "Total HTTP errors (from app metrics file)",
    ["method", "path"],
    registry=REGISTRY,
)

# --- Clickstream events ---
app_events_processed_total = Gauge(
    "app_events_processed_total",
    "Total events processed by type (clickstream consumer)",
    ["event_type"],
    registry=REGISTRY,
)

# --- Recommendation cache ---
recommendation_cache_hits_total = Gauge(
    "recommendation_cache_hits_total",
    "Total recommendation cache hits (combined endpoint)",
    registry=REGISTRY,
)
recommendation_cache_misses_total = Gauge(
    "recommendation_cache_misses_total",
    "Total recommendation cache misses (combined endpoint)",
    registry=REGISTRY,
)
recommendation_impressions_total = Gauge(
    "recommendation_impressions_total",
    "Total recommendation responses served (cache hit + miss)",
    registry=REGISTRY,
)

# --- Latency (last value + percentiles from rolling window) ---
recommendation_duration_seconds = Gauge(
    "recommendation_duration_seconds",
    "Last pipeline duration on cache-miss (seconds)",
    registry=REGISTRY,
)
recommendation_duration_p50_seconds = Gauge(
    "recommendation_duration_p50_seconds",
    "P50 pipeline duration over rolling 500-sample window (seconds)",
    registry=REGISTRY,
)
recommendation_duration_p95_seconds = Gauge(
    "recommendation_duration_p95_seconds",
    "P95 pipeline duration over rolling 500-sample window (seconds)",
    registry=REGISTRY,
)
recommendation_duration_p99_seconds = Gauge(
    "recommendation_duration_p99_seconds",
    "P99 pipeline duration over rolling 500-sample window (seconds)",
    registry=REGISTRY,
)

# --- Algorithm candidates ---
recommendation_candidates_total = Gauge(
    "recommendation_candidates_total",
    "Cumulative candidate books retrieved per algorithm",
    ["algorithm"],
    registry=REGISTRY,
)

# --- Engagement & NDCG (from clickstream consumer) ---
recommendation_engagement_by_position = Gauge(
    "recommendation_engagement_by_position",
    "Cumulative engagement events at each recommendation rank position",
    ["position"],
    registry=REGISTRY,
)
recommendation_ndcg_at_10 = Gauge(
    "recommendation_ndcg_at_10",
    "NDCG@10 computed from per-position engagement counts",
    registry=REGISTRY,
)

# --- Sidecar health ---
sidecar_up = Gauge("metrics_sidecar_up", "Sidecar is running", ["service"], registry=REGISTRY)


def read_app_metrics():
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in %s: %s", METRICS_FILE, e)
        return None
    except Exception as e:
        logger.warning("Failed to read %s: %s", METRICS_FILE, e)
        return None


def apply_metrics(data: dict):
    if not data:
        return

    # HTTP requests
    if isinstance(data.get("http_requests_total"), dict):
        for label_str, v in data["http_requests_total"].items():
            parts = label_str.split("|")
            method, path, status_class = (
                (parts[0], parts[1], parts[2]) if len(parts) >= 3
                else ("GET", "unknown", "2xx")
            )
            try:
                app_http_requests_total.labels(method=method, path=path, status_class=status_class).set(float(v))
            except Exception:
                pass

    # HTTP errors
    if isinstance(data.get("http_errors_total"), dict):
        for label_str, v in data["http_errors_total"].items():
            parts = label_str.split("|")
            method, path = (parts[0], parts[1]) if len(parts) >= 2 else ("GET", "unknown")
            try:
                app_http_errors_total.labels(method=method, path=path).set(float(v))
            except Exception:
                pass

    # Clickstream events
    if isinstance(data.get("events_processed_total"), dict):
        for ev_type, v in data["events_processed_total"].items():
            try:
                app_events_processed_total.labels(event_type=str(ev_type)).set(float(v))
            except Exception:
                pass

    # Cache
    _set_gauge(recommendation_cache_hits_total, data.get("recommendation_cache_hits_total"))
    _set_gauge(recommendation_cache_misses_total, data.get("recommendation_cache_misses_total"))
    _set_gauge(recommendation_impressions_total, data.get("recommendation_impressions_total"))

    # Latency
    _set_gauge(recommendation_duration_seconds, data.get("recommendation_duration_seconds"))
    _set_gauge(recommendation_duration_p50_seconds, data.get("recommendation_duration_p50_seconds"))
    _set_gauge(recommendation_duration_p95_seconds, data.get("recommendation_duration_p95_seconds"))
    _set_gauge(recommendation_duration_p99_seconds, data.get("recommendation_duration_p99_seconds"))

    # Algorithm candidates
    if isinstance(data.get("candidates_by_algorithm"), dict):
        for algo, count in data["candidates_by_algorithm"].items():
            try:
                recommendation_candidates_total.labels(algorithm=str(algo)).set(float(count))
            except Exception:
                pass

    # Engagement by position
    if isinstance(data.get("engagement_by_position"), dict):
        for pos_str, count in data["engagement_by_position"].items():
            try:
                recommendation_engagement_by_position.labels(position=str(pos_str)).set(float(count))
            except Exception:
                pass

    # NDCG@10
    _set_gauge(recommendation_ndcg_at_10, data.get("recommendation_ndcg_at_10"))


def _set_gauge(gauge: Gauge, value) -> None:
    if value is None:
        return
    try:
        gauge.set(float(value))
    except (TypeError, ValueError):
        pass


def main():
    try:
        REGISTRY.register(PROCESS_COLLECTOR)
    except Exception:
        pass
    logger.info("Starting metrics sidecar on port %s (service=%s)", METRICS_PORT, SERVICE_NAME)
    sidecar_up.labels(service=SERVICE_NAME).set(1)
    start_http_server(METRICS_PORT)
    while True:
        data = read_app_metrics()
        if data:
            apply_metrics(data)
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    main()
