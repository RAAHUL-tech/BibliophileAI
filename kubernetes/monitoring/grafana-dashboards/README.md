# Grafana dashboards

Predefined Grafana dashboard JSON for the BibliophileAI monitoring stack. Loaded into Grafana via ConfigMap so they appear automatically after install.

## Contents

| File | Description |
|------|-------------|
| **bibliophile-app-metrics.json** | Dashboard "BibliophileAI – Cluster, Node & App metrics": sidecar up, total HTTP requests, recommendation cache hits/misses and duration, cluster (running/failed pods, replicas per deployment, pods by namespace), node CPU %, HTTP status codes per service, events processed. |

## How it is used

- **install-monitoring.sh** creates a ConfigMap from `bibliophile-app-metrics.json` and labels it with `grafana_dashboard=1`. The kube-prometheus-stack Grafana is configured to load dashboards from ConfigMaps with this label in the monitoring namespace.
- **Datasource**: Panels assume a Prometheus datasource with UID `prometheus` (the default Prometheus instance created by the stack).
- **Metrics**: App panels use metrics exposed by the metrics sidecar and scraped via ServiceMonitors: `metrics_sidecar_up`, `app_http_requests_total`, `recommendation_cache_hits_total`, `recommendation_cache_misses_total`, `recommendation_duration_seconds`, `app_events_processed_total`. Cluster panels use `kube_pod_status_phase`, `kube_deployment_status_replicas_available` (kube-state-metrics) and `node_cpu_seconds_total` (node-exporter).

## Editing

Edit the JSON (panel queries, titles, layout). Re-apply the ConfigMap and refresh Grafana (or re-run `install-monitoring.sh`). If the Prometheus datasource UID differs in your install, update the `datasource.uid` in each panel target.
