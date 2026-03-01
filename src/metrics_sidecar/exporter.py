"""
Metrics sidecar: exposes /metrics in Prometheus format.
Reads app-provided metrics from shared volume (JSON). No metrics logic in main app.
"""
import json
import os
import logging
import time
from prometheus_client import start_http_server, REGISTRY, Gauge, Counter, PROCESS_COLLECTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics-sidecar")

METRICS_FILE = os.environ.get("METRICS_FILE", "/metrics-data/app_metrics.json")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
SERVICE_NAME = os.environ.get("SERVICE_NAME", "app")
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL", "15.0"))

# App metrics from file - all Gauges so app can write absolute values
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
app_events_processed_total = Gauge(
    "app_events_processed_total",
    "Total events processed (from app metrics file)",
    ["event_type"],
    registry=REGISTRY,
)
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
recommendation_duration_seconds = Gauge(
    "recommendation_duration_seconds",
    "Time taken to generate recommendation results (last cache-miss request)",
    registry=REGISTRY,
)
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
    # HTTP: expect {"http_requests_total": {"method|path|status_class": value}} or nested
    if "http_requests_total" in data and isinstance(data["http_requests_total"], dict):
        for label_str, v in data["http_requests_total"].items():
            parts = label_str.split("|")
            if len(parts) >= 3:
                method, path, status_class = parts[0], parts[1], parts[2]
            else:
                method, path, status_class = "GET", "unknown", "2xx"
            try:
                app_http_requests_total.labels(method=method, path=path, status_class=status_class).set(float(v))
            except Exception:
                pass
    if "http_errors_total" in data and isinstance(data["http_errors_total"], dict):
        for label_str, v in data["http_errors_total"].items():
            parts = label_str.split("|")
            method, path = (parts[0], parts[1]) if len(parts) >= 2 else ("GET", "unknown")
            try:
                app_http_errors_total.labels(method=method, path=path).set(float(v))
            except Exception:
                pass

    # Events
    if "events_processed_total" in data and isinstance(data["events_processed_total"], dict):
        for ev_type, v in data["events_processed_total"].items():
            try:
                app_events_processed_total.labels(event_type=str(ev_type)).set(float(v))
            except Exception:
                pass
    # Recommendation cache
    if "recommendation_cache_hits_total" in data:
        try:
            recommendation_cache_hits_total.set(float(data["recommendation_cache_hits_total"]))
        except (TypeError, ValueError):
            pass
    if "recommendation_cache_misses_total" in data:
        try:
            recommendation_cache_misses_total.set(float(data["recommendation_cache_misses_total"]))
        except (TypeError, ValueError):
            pass
    if "recommendation_duration_seconds" in data:
        try:
            recommendation_duration_seconds.set(float(data["recommendation_duration_seconds"]))
        except (TypeError, ValueError):
            pass


def main():
    # Expose process and platform metrics for the sidecar (optional cluster/node view per pod)
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
