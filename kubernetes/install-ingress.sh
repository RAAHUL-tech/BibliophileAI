#!/usr/bin/env bash
# Install NGINX Ingress controller via Helm and apply BibliophileAI Ingress.
# Expose on localhost:8080 with: kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
# (Port 80 requires sudo on macOS/Linux; 8080 works without.)

set -e

echo "Adding Helm repo ingress-nginx..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

echo "Creating namespace ingress-nginx if not present..."
kubectl create namespace ingress-nginx --dry-run=client -o yaml | kubectl apply -f -

echo "Installing ingress-nginx (NGINX Ingress Controller)..."
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.type=ClusterIP

echo "Waiting for ingress controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

echo "Applying BibliophileAI Ingress (routes /api/v1/user and /api/v1/recommend)..."
kubectl apply -f "$(dirname "$0")/ingress.yaml"

echo ""
echo "Done. To expose the Ingress on localhost:8080, run:"
echo "  kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80"
echo ""
echo "Then access:"
echo "  User API:        http://localhost:8080/api/v1/user/..."
echo "  Recommend API:   http://localhost:8080/api/v1/recommend/..."
echo "  Search API:   http://localhost:8080/api/v1/search/..."
