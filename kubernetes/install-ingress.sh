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

# Admission webhook may not be ready immediately after the controller pod is ready
echo "Applying BibliophileAI Ingress (routes /api/v1/user, /api/v1/recommend, /api/v1/search)..."
INGRESS_FILE="$(dirname "$0")/ingress.yaml"
for i in $(seq 1 10); do
  if kubectl apply -f "$INGRESS_FILE"; then
    break
  fi
  if [[ $i -eq 10 ]]; then
    echo "Failed to apply Ingress after 10 retries. Run: kubectl apply -f $INGRESS_FILE" >&2
    exit 1
  fi
  echo "  Retry $i/10 in 5s (admission webhook may still be starting)..."
  sleep 5
done

echo ""
echo "Done. To expose the Ingress on localhost:8080, run:"
echo "  kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80"
echo ""
echo "Then access:"
echo "  User API:        http://localhost:8080/api/v1/user/..."
echo "  Recommend API:   http://localhost:8080/api/v1/recommend/..."
echo "  Search API:   http://localhost:8080/api/v1/search/..."
