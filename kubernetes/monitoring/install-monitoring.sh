#!/usr/bin/env bash
# Install Prometheus Operator stack (Prometheus, Grafana, node-exporter, kube-state-metrics)
# and optionally build/push the metrics-sidecar image.
# Usage:
#   ./install-monitoring.sh              # install stack only
#   ./install-monitoring.sh --build      # build and push metrics-sidecar image then install
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MONITORING_NS="monitoring"
KUBE_CONTEXT="${KUBE_CONTEXT:-}"

echo "Adding Helm repo prometheus-community..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update


echo "Creating namespace $MONITORING_NS if not exists..."
kubectl create namespace "$MONITORING_NS" --dry-run=client -o yaml | kubectl apply -f -

echo "Installing kube-prometheus-stack..."
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n "$MONITORING_NS" \
  -f "$SCRIPT_DIR/prometheus-stack-values.yaml" \
  --create-namespace \
  $([[ -n "$KUBE_CONTEXT" ]] && echo "--kube-context $KUBE_CONTEXT")

echo "Waiting for Prometheus and Grafana to be ready..."
# Prometheus is a StatefulSet in recent kube-prometheus-stack (not a Deployment)
kubectl wait --for=condition=ready statefulset/kube-prometheus-stack-prometheus -n "$MONITORING_NS" --timeout=300s 2>/dev/null || \
  kubectl wait --for=condition=available deployment/kube-prometheus-stack-prometheus -n "$MONITORING_NS" --timeout=300s 2>/dev/null || true
kubectl wait --for=condition=available deployment/kube-prometheus-stack-grafana -n "$MONITORING_NS" --timeout=300s || true

echo "Creating Grafana dashboard ConfigMap (BibliophileAI app metrics)..."
kubectl create configmap grafana-dashboard-bibliophile-app \
  --from-file=bibliophile-app-metrics.json="$SCRIPT_DIR/grafana-dashboards/bibliophile-app-metrics.json" \
  -n "$MONITORING_NS" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl label configmap grafana-dashboard-bibliophile-app -n "$MONITORING_NS" grafana_dashboard=1 --overwrite

echo "Applying ServiceMonitors (so Prometheus scrapes app metrics from sidecars)..."
kubectl apply -f "$SCRIPT_DIR/servicemonitors/"

echo ""
echo "Monitoring stack is installed."
echo "Access Grafana (default admin/admin):"
echo "  kubectl port-forward -n $MONITORING_NS svc/kube-prometheus-stack-grafana 3000:80"
echo "  Open http://localhost:3000"
echo ""
echo "Access Prometheus:"
echo "  kubectl port-forward -n $MONITORING_NS svc/kube-prometheus-stack-prometheus 9090:9090"
echo "  Open http://localhost:9090"
