#!/usr/bin/env bash
# Build Airflow image with DAGs, push to registry, and install Airflow via Helm.
# Expose API on localhost:8082 with: kubectl port-forward svc/airflow-api-server 8082:8080 --namespace default
# (Ingress uses 8080, so Airflow uses 8082 to avoid conflict.)
#
# Uninstall: helm uninstall airflow && kubectl delete pvc -l app.kubernetes.io/instance=airflow

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AIRFLOW_DIR="$REPO_ROOT/airflow"
KUBE_CONTEXT="${KUBE_CONTEXT:-kind-bibliophileai-cluster}"

echo "Adding Helm repo apache-airflow..."
helm repo add apache-airflow https://airflow.apache.org
helm repo update

echo "Building Docker image rahulkrish28/airflow-with-dags:latest from $AIRFLOW_DIR..."
docker build -t rahulkrish28/airflow-with-dags:latest "$AIRFLOW_DIR"

echo "Pushing image to registry..."
docker push rahulkrish28/airflow-with-dags:latest

echo "Installing Airflow via Helm (context: $KUBE_CONTEXT)..."
helm upgrade --install airflow apache-airflow/airflow \
  --kube-context "$KUBE_CONTEXT" \
  -f "$SCRIPT_DIR/airflow-values.yaml"

echo "Waiting for Airflow API server to be ready..."
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/component=api-server,app.kubernetes.io/instance=airflow \
  --timeout=300s \
  --context "$KUBE_CONTEXT" || true

echo ""
echo "Done. To expose the Airflow API on localhost:8082, run:"
echo "  kubectl port-forward svc/airflow-api-server 8082:8080 --namespace default"
echo ""
echo "Then open the Airflow UI (if enabled) or use the API at http://localhost:8082"
echo ""
echo "To uninstall Airflow and remove PVCs:"
echo "  helm uninstall airflow"
echo "  kubectl delete pvc -l app.kubernetes.io/instance=airflow"
