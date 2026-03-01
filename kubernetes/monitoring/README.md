# BibliophileAI Monitoring (Prometheus + Grafana)

Monitoring uses the **Prometheus Operator** (kube-prometheus-stack) for cluster- and node-level metrics, and a **metrics sidecar** in each app pod to expose `/metrics` without adding metrics code to the main application.

## Components

- **Prometheus** – scrapes ServiceMonitors (app sidecars + node-exporter + kube-state-metrics), stores metrics.
- **Grafana** – dashboards for cluster, nodes, and BibliophileAI app metrics (CTR, NDCG, training, etc.).
- **Metrics sidecar** – one container per app pod: reads optional `/metrics-data/app_metrics.json` and exposes Prometheus metrics on port 9090.

## Quick start

1. **Build and push the metrics-sidecar image** (once):
   ```bash
   docker build -t rahulkrish28/metrics-sidecar:latest src/metrics_sidecar
   docker push rahulkrish28/metrics-sidecar:latest
   ```

2. **Install the monitoring stack**:
   ```bash
   cd kubernetes/monitoring
   ./install-monitoring.sh
   # Or build sidecar and install: ./install-monitoring.sh --build
   ```

3. **Apply ServiceMonitors and deployments** (so Prometheus scrapes app metrics):
   ```bash
   # install-monitoring.sh now applies ServiceMonitors automatically
   kubectl apply -f kubernetes/user-auth-deployment.yaml
   kubectl apply -f kubernetes/search-deployment.yaml
   kubectl apply -f kubernetes/recommendation-deployment.yaml
   kubectl apply -f kubernetes/consumer-deployment.yaml
   ```
   Services must have labels matching the ServiceMonitors (e.g. `app: recommendation`) so Prometheus discovers the metrics endpoints.

4. **Access Grafana** (default admin/admin):
   ```bash
   kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
   ```
   Open http://localhost:3000. Use **Explore** with datasource **Prometheus** to run queries.

5. **Access Prometheus**:
   ```bash
   kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
   ```
   Open http://localhost:9090.

## Dashboards

- **Built-in** (from kube-prometheus-stack): Node Exporter, Kubernetes / cluster, Prometheus, etc.
- **BibliophileAI – App metrics**: CTR, NDCG@k, training metrics per algorithm, events processed. Loaded from the `grafana-dashboard-bibliophile-app` ConfigMap.

## Exposing app metrics (optional, no code in main app)

Each pod has a shared volume at `/metrics-data`. The main container can write a JSON file there; the sidecar reads it and exposes Prometheus metrics. **No Prometheus or metrics logic in the main app** – only writing a JSON file.

**File:** `/metrics-data/app_metrics.json` (inside the pod)

**Example shape:**

```json
{
  "http_requests_total": { "GET|/api/v1/books|2xx": 1000 },
  "http_errors_total": { "GET|/api/v1/books": 5 },
  "events_processed_total": { "page_turn": 10000, "bookmark_add": 500 }
}
```

- **http_requests_total** / **http_errors_total**: keys like `"METHOD|PATH|STATUS_CLASS"` or `"METHOD|PATH"` for errors.
- **ctr**: click-through rate per category (0–1).
- **ndcg_at_k**: NDCG per category and k (e.g. `"5"`, `"10"`).
- **training**: per-algorithm metrics (e.g. ndcg, loss) and optional **duration_seconds**, **phase**.
- **events_processed_total**: counts per event type (e.g. clickstream consumer).

The sidecar polls this file every 15s and exports the values as Prometheus gauges. If the file is missing, only `metrics_sidecar_up` and process metrics are exposed.

## ServiceMonitors

| ServiceMonitor            | Service               | Port 9090 (metrics) |
|---------------------------|-----------------------|----------------------|
| user-auth-service         | user-auth-service     | sidecar              |
| search-service            | search-service        | sidecar              |
| recommendation-service    | recommendation-service| sidecar              |
| clickstream-consumer      | clickstream-consumer  | sidecar              |
| model-training-service    | (when deployed)       | sidecar              |

Prometheus is configured to use all ServiceMonitors in all namespaces (`serviceMonitorSelector: {}`). **ServiceMonitors are applied by `install-monitoring.sh`.** Each app Service must have a matching label (e.g. `app: recommendation`) so the ServiceMonitor can select it. If app metrics show "No data" in Grafana, verify: (1) ServiceMonitors are applied: `kubectl get servicemonitor -n default`; (2) Services have the right labels; (3) App pods are running with the metrics sidecar; (4) In Prometheus → Status → Targets, the app endpoints show as "up".

## Cluster and node metrics

- **Node metrics**: node-exporter DaemonSet (CPU, memory, disk, network per node).
- **Cluster metrics**: kube-state-metrics (pods, deployments, statefulsets, etc.).

Both are included in kube-prometheus-stack and appear in the default Kubernetes and Node dashboards in Grafana.
