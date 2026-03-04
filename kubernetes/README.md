# Kubernetes

Kubernetes manifests and scripts for running BibliophileAI in a cluster (e.g. Kind, minikube, or cloud). This folder covers ingress, Redis, app deployments, Airflow, and monitoring.

## Contents

| Resource | Description |
|----------|-------------|
| **ingress.yaml** | NGINX Ingress: routes `/api/v1/user`, `/api/v1/recommend`, `/api/v1/search` to user-auth, recommendation, and search services; CORS and long timeouts for recommend. |
| **install-ingress.sh** | Installs ingress-nginx via Helm, waits for controller and admission webhook, applies `ingress.yaml` (with retries). |
| **redis.yaml** | Redis for app cache (recommendation cache, sessions): PVC, Deployment, Service. Used as `REDIS_URL`. |
| **redis-feast.yaml** | Separate Redis for Feast online store: Deployment, Service. Used as `FEAST_REDIS_URL`. |
| **user-auth-deployment.yaml** | Service + Deployment for user service (port 8000, metrics sidecar). |
| **search-deployment.yaml** | Service + Deployment for search service (port 8002, metrics sidecar). |
| **recommendation-deployment.yaml** | Service + Deployment for recommendation service (port 8001, metrics sidecar). |
| **consumer-deployment.yaml** | Service + Deployment for clickstream consumer (port 8002, metrics sidecar). |
| **airflow-values.yaml** | Helm values overrides for Airflow (scheduler, workers, DAGs, executor, resources). |
| **install-airflow.sh** | Builds Airflow image with DAGs, pushes to registry, installs Airflow via Helm. |
| **monitoring/** | Prometheus stack, Grafana, ServiceMonitors, dashboards; see [monitoring/README.md](monitoring/README.md). |

## Typical workflow

1. Create cluster (e.g. `kind create cluster --config kind-cluster-config.yaml`).
2. Apply secrets (e.g. `kubectl apply -f secrets.yaml` after filling values).
3. Deploy Redis: `kubectl apply -f redis.yaml`, `kubectl apply -f redis-feast.yaml`.
4. Install ingress: `./install-ingress.sh`.
5. Deploy apps: `kubectl apply -f user-auth-deployment.yaml` (and search, recommendation, consumer).
6. Optional: install Airflow (`./install-airflow.sh`) and monitoring (`./monitoring/install-monitoring.sh`).
7. Expose: `kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80`; frontend calls `http://localhost:8080`.

## kind-cluster-config.yaml

Kind cluster definition: control-plane + worker node(s), optional registry mirror and node labels (e.g. `has-cpu=true` for scheduling).
