# BibliophileAI – Setup Guide

Step-by-step guide to run the full stack locally: cluster, secrets, services, ingress, frontend, and optionally Airflow and monitoring.

---

## 1. Prerequisites

| Tool | Purpose | Check |
|------|---------|-------|
| **Docker** | Build and run containers | `docker --version` |
| **kubectl** | Kubernetes CLI | `kubectl version --client` |
| **Kind** | Local Kubernetes cluster | `kind version` |
| **Helm 3** | Install charts (Ingress, Airflow, Prometheus) | `helm version` |
| **Node.js 18+** | Frontend (Vite + React) | `node -v` |
| **Python 3.9+** | Data import scripts | `python3 --version` |

---

## 2. External Services

Create accounts and note credentials for each service — you'll add them to Kubernetes Secrets.

| Service | What you need |
|---------|--------------|
| **Supabase** | Project URL + service_role key |
| **MongoDB Atlas** | Connection string (`mongodb+srv://...`) |
| **Neo4j Aura** | URI, username, password |
| **Pinecone** | API key; create two indices: `user-preferences-index`, `book-metadata-index` |
| **AWS** | Access Key ID + Secret Access Key; create an S3 bucket and an SQS FIFO queue |
| **Google Cloud** | OAuth 2.0 Web client ID (for "Login with Google") |

Use the same AWS region (e.g. `us-east-2`) for S3 and SQS.

---

## 3. Clone and create the cluster

```bash
git clone https://github.com/RAAHUL-tech/BibliophileAI.git
cd BibliophileAI

kind create cluster --config kind-cluster-config.yaml
kubectl cluster-info
```

---

## 4. Kubernetes Secrets

All services read credentials from a Secret named `secret` in the `default` namespace.

**Do not commit real credentials.** Copy `kubernetes/secrets.yaml` to `kubernetes/my-secrets.yaml`, fill in base64-encoded values, and apply it.

Encode a value:
```bash
echo -n "your-value" | base64
```

Secret keys the services expect:

| Key | Used by |
|-----|--------|
| `SECRET_KEY` | JWT signing (32+ chars) |
| `SUPABASE_URL` | All services + importers |
| `SUPABASE_KEY` | All services + importers |
| `GOOGLE_CLIENT_ID` | User service (OAuth) |
| `PINECONE_API_KEY` | Search, Recommendation, User |
| `AWS_ACCESS_KEY_ID` | All S3/SQS and training |
| `AWS_SECRET_ACCESS_KEY` | All S3/SQS and training |
| `AWS_DEFAULT_REGION` | Consumer, training |
| `SQS_QUEUE_URL` | User (producer) + Consumer |
| `MONGO_URI` | Recommendation, Consumer, Training |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | User, Recommendation, Graph training |
| `S3_URI` | Recommendation, Training |
| `REDIS_URL` | Recommendation + Popularity training |
| `FEAST_REDIS_URL` | Recommendation (Feast online store) |
| `ALS_S3_PREFIX`, `GRAPH_S3_PREFIX`, `SASREC_S3_PREFIX`, `POPULARITY_S3_PREFIX`, `LTR_S3_PREFIX`, `LINUCB_S3_PREFIX` | Training + Recommendation |
| `RECOMMENDATION_SERVICE_URL` | User service (internal refresh) |
| `INTERNAL_API_KEY` | Recommendation internal endpoints |

Secret manifest structure:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: secret
  namespace: default
type: Opaque
data:
  SECRET_KEY: <base64>
  SUPABASE_URL: <base64>
  SUPABASE_KEY: <base64>
  GOOGLE_CLIENT_ID: <base64>
  PINECONE_API_KEY: <base64>
  AWS_ACCESS_KEY_ID: <base64>
  AWS_SECRET_ACCESS_KEY: <base64>
  AWS_DEFAULT_REGION: <base64>
  SQS_QUEUE_URL: <base64>
  MONGO_URI: <base64>
  NEO4J_URI: <base64>
  NEO4J_USER: <base64>
  NEO4J_PASSWORD: <base64>
  S3_URI: <base64>
  REDIS_URL: <base64>
  FEAST_REDIS_URL: <base64>
  ALS_S3_PREFIX: <base64>
  GRAPH_S3_PREFIX: <base64>
  SASREC_S3_PREFIX: <base64>
  POPULARITY_S3_PREFIX: <base64>
  LTR_S3_PREFIX: <base64>
  LINUCB_S3_PREFIX: <base64>
  RECOMMENDATION_SERVICE_URL: <base64>
  INTERNAL_API_KEY: <base64>
```

```bash
kubectl apply -f kubernetes/my-secrets.yaml
```

---

## 5. Deploy Redis

```bash
kubectl apply -f kubernetes/redis.yaml          # app cache
kubectl apply -f kubernetes/redis-feast.yaml    # Feast online store
kubectl get pods -l app=redis
kubectl get pods -l app=feast-redis
```

Wait until both pods are `Running`.

---

## 6. Build and load application images

For Kind, images must be loaded into the cluster after building.

```bash
# Metrics sidecar (used by every app pod)
docker build -t rahulkrish28/metrics-sidecar:latest src/metrics_sidecar
kind load docker-image rahulkrish28/metrics-sidecar:latest

# Services
docker build -t rahulkrish28/user-service:latest src/user_service
kind load docker-image rahulkrish28/user-service:latest

docker build -t rahulkrish28/search-service:latest src/search_service
kind load docker-image rahulkrish28/search-service:latest

docker build -t rahulkrish28/recommendation-service:latest src/recommendation_service
kind load docker-image rahulkrish28/recommendation-service:latest

docker build -t rahulkrish28/clickstream-consumer:latest src/clickstream_consumer
kind load docker-image rahulkrish28/clickstream-consumer:latest
```

For a remote cluster, push to a registry instead of `kind load docker-image`.

---

## 7. Install Ingress and deploy services

```bash
cd kubernetes
./install-ingress.sh    # installs NGINX Ingress controller + BibliophileAI routes
cd ..

kubectl apply -f kubernetes/user-auth-deployment.yaml
kubectl apply -f kubernetes/search-deployment.yaml
kubectl apply -f kubernetes/recommendation-deployment.yaml
kubectl apply -f kubernetes/consumer-deployment.yaml

kubectl get pods    # wait until all are Running and Ready
```

If a pod fails: `kubectl describe pod <name>` and `kubectl logs <pod> -c <container>`.

---

## 8. Expose the API

```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
```

Leave this running. The API is now available at `http://localhost:8080`.

---

## 9. Frontend

```bash
cd frontend/bibliophile-ai-frontend
npm install
```

Optionally set your Google OAuth client ID:
```bash
# .env.local
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
```

```bash
npm run dev    # http://localhost:5173
```

Open http://localhost:5173 and register a user. Complete onboarding (genres, authors, demographics). The first recommendation load may take 1–2 minutes on cache miss; subsequent loads are fast from Redis.

---

## 10. (Optional) Monitoring

```bash
cd kubernetes/monitoring
./install-monitoring.sh
```

- **Grafana:** `kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80` → http://localhost:3000 (default: `admin` / `admin`)
- **Prometheus:** `kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090` → http://localhost:9090

---

## 11. (Optional) Airflow and training jobs

### Install Airflow

```bash
cd kubernetes
./install-airflow.sh
```

Then in the Airflow UI (Admin → Variables), add the same credentials used in the Kubernetes Secret:
`MONGO_URI`, `S3_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `NEO4J_*`, `REDIS_URL`, and the model S3 prefixes.

### Build and load training images

```bash
docker build -t rahulkrish28/als-train:latest src/model_training_service/als_train
docker build -t rahulkrish28/graph-train:latest src/model_training_service/graph_train
docker build -t rahulkrish28/ltr-train:latest src/model_training_service/ltr_train
docker build -t rahulkrish28/popularity-train:latest src/model_training_service/popularity_train
docker build -t rahulkrish28/sasrec-train:latest src/model_training_service/sasrec_train

kind load docker-image rahulkrish28/als-train:latest
kind load docker-image rahulkrish28/graph-train:latest
kind load docker-image rahulkrish28/ltr-train:latest
kind load docker-image rahulkrish28/popularity-train:latest
kind load docker-image rahulkrish28/sasrec-train:latest
```

Expose the Airflow UI:
```bash
kubectl port-forward svc/airflow-api-server 8082:8080 --namespace default
```

Open http://localhost:8082, unpause the DAGs, and trigger runs.

---

## 12. (Optional) Populate book data

If Supabase and Pinecone/Neo4j are empty, seed them with the import scripts.

```bash
# Set credentials in your shell or a .env file
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="..."
export PINECONE_API_KEY="..."
export NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_S3_BUCKET="your-bucket"

cd book_data_importer
pip install -r requirements.txt

python import_books.py          # Gutendex → Supabase
python scrape_descriptions.py   # optional: add book descriptions
python book_embedding.py        # Supabase books → Pinecone
python graph_book_importer.py   # books/authors/genres → Neo4j
python upload_epubs_to_s3.py    # EPUBs → S3
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pod `ImagePullBackOff` | Run `kind load docker-image <image>:<tag>` after building; or push to a registry for remote clusters |
| `504` on `/api/v1/recommend/combined` | First-run pipeline can take 1–2 min; Ingress is already set to 300s — wait it out |
| Frontend "No data" / network errors | Ensure port-forward on 8080 is active; check CORS and that backend pods are healthy |
| Grafana "No data" for app metrics | Check ServiceMonitors are applied and services have correct labels; verify the metrics sidecar is running |
| `boto3 NoRegionError` | Set `AWS_DEFAULT_REGION` in consumer and producer environments |

---

## Quick-start checklist

1. Prerequisites + external service accounts
2. `git clone` → `kind create cluster` → `kubectl apply -f kubernetes/my-secrets.yaml`
3. `kubectl apply -f kubernetes/redis.yaml` and `redis-feast.yaml`
4. Build + `kind load` all app images; `./kubernetes/install-ingress.sh`; apply the four deployment YAMLs
5. `kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80`
6. `cd frontend/bibliophile-ai-frontend && npm install && npm run dev` → http://localhost:5173
7. (Optional) Monitoring, Airflow, training images, book import scripts
