# ServiceMonitors

Prometheus Operator ServiceMonitor resources that tell Prometheus how to discover and scrape the **metrics sidecar** in each application pod. Each app Service must have a matching label so the ServiceMonitor selects it.

## Role

- **ServiceMonitor** selects a **Service** by label (e.g. `app: recommendation`). Prometheus discovers the Service’s endpoints (pods) and scrapes the configured port and path.
- **Port**: All use the port named `metrics` (9090) on the Service, which targets the sidecar container in each pod.
- **Path**: `/metrics` (Prometheus exposition format).
- **Namespace**: These ServiceMonitors are in `default`; the apps also run in `default`. Prometheus (in `monitoring`) is configured with `serviceMonitorNamespaceSelector: {}` so it picks up ServiceMonitors in all namespaces.

## Files

| File | Service | Selector |
|------|---------|----------|
| **user-auth-servicemonitor.yaml** | user-auth-service | `app: user-auth` |
| **search-servicemonitor.yaml** | search-service | `app: search` |
| **recommendation-servicemonitor.yaml** | recommendation-service | `app: recommendation` |
| **clickstream-consumer-servicemonitor.yaml** | clickstream-consumer | `app: clickstream-consumer` |

## Requirements

- The corresponding Kubernetes **Service** must have `metadata.labels` matching the selector (e.g. `app: recommendation`). The deployments in this repo set these labels on the Service.
- The **Deployment** must include the metrics sidecar container and a port named `metrics` (9090) on the Service.
- **install-monitoring.sh** runs `kubectl apply -f servicemonitors/` so these are applied when the monitoring stack is installed.

## Verifying

- `kubectl get servicemonitor -n default`
- In Prometheus UI (Status → Targets), look for targets such as `recommendation-service.default.svc:9090` and confirm they are "up".
