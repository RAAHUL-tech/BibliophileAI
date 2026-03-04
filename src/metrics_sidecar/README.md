# Metrics Sidecar

Sidecar container that exposes Prometheus-format metrics on port 9090. It reads a JSON file written by the main application and serves those values as Prometheus gauges so the main app has no direct Prometheus dependency.

## How it works

- **Shared volume**: The pod mounts an `emptyDir` at `/metrics-data`. The main container writes `/metrics-data/app_metrics.json`; the sidecar reads it periodically (e.g. every 15s).
- **Metrics**: The sidecar defines gauges for HTTP requests/errors, recommendation cache hits/misses and duration, and events processed. It maps keys from the JSON (e.g. `http_requests_total`, `recommendation_cache_hits_total`) to Prometheus metrics.
- **Exposition**: `exporter.py` starts an HTTP server on `METRICS_PORT` (default 9090) and serves `/metrics` for Prometheus scraping.
- **ServiceMonitor**: Kubernetes Service exposes port name `metrics` (9090) so the Prometheus Operator discovers and scrapes the sidecar.

## Implementation in this project

- **exporter.py**: Uses `prometheus_client` (Gauge); reads JSON, parses label keys (e.g. `method|path|status_class`), sets gauge values; optional process collector.
- **Environment**: `METRICS_FILE`, `METRICS_PORT`, `SERVICE_NAME`, `SCRAPE_INTERVAL`.
- **Dockerfile**: Single stage, runs `exporter.py`.
- **Usage**: Included in each app deployment (user-auth, search, recommendation, clickstream-consumer) as a second container with the same volume mount.

## JSON shape (written by main app)

- `http_requests_total`: `{ "METHOD|PATH|STATUS_CLASS": count }`
- `http_errors_total`: `{ "METHOD|PATH": count }`
- `events_processed_total`: `{ "event_type": count }` (clickstream consumer)
- `recommendation_cache_hits_total`, `recommendation_cache_misses_total`, `recommendation_duration_seconds` (recommendation service only)
