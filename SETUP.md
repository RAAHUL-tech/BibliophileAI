# BibliophileAI — Setup Guide

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Docker | Latest | `docker --version` |
| kubectl | Any | `kubectl version --client` |
| Kind | Any | `kind version` |
| Helm | 3+ | `helm version` |
| Node.js | 18+ | `node -v` |
| Python | 3.9+ | `python3 --version` |

---

## External Services

Create accounts and note credentials before starting. You'll encode them as Kubernetes Secrets.

| Service | What you need |
|---------|--------------|
| [Supabase](https://supabase.com) | Project URL + service_role key |
| [MongoDB Atlas](https://www.mongodb.com/atlas) | Connection string (`mongodb+srv://...`) |
| [Neo4j Aura](https://neo4j.com/cloud/aura) | URI, username, password |
| [Pinecone](https://pinecone.io) | API key · create two indices: `user-preferences-index`, `book-metadata-index` |
| [AWS](https://aws.amazon.com) | Access Key ID + Secret · S3 bucket · SQS FIFO queue (same region for both) |
| [Google Cloud](https://console.cloud.google.com) | OAuth 2.0 Web client ID |

---

## 1 · Clone and create the cluster

```bash
git clone https://github.com/RAAHUL-tech/BibliophileAI.git
cd BibliophileAI

kind create cluster --config kind-cluster-config.yaml
kubectl cluster-info
```

---

## 2 · Secrets

All services read credentials from a Secret named `secret` in the `default` namespace.

Encode each value:
```bash
echo -n "your-value" | base64
```

Copy `kubernetes/secrets.yaml`, fill in the base64 values, and apply:

```bash
kubectl apply -f kubernetes/my-secrets.yaml
```

**Required keys:**

| Key | Used by |
|-----|--------|
| `SECRET_KEY` | JWT signing (32+ chars) |
| `SUPABASE_URL`, `SUPABASE_KEY` | All services |
| `GOOGLE_CLIENT_ID` | User service (OAuth) |
| `PINECONE_API_KEY` | Search, Recommendation, User |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | All S3/SQS |
| `SQS_QUEUE_URL` | User service + Consumer |
| `MONGO_URI` | Recommendation, Consumer, Training |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | User, Recommendation, Graph training |
| `S3_URI` | Recommendation, Training |
| `REDIS_URL` | Recommendation, Consumer, Popularity training |
| `FEAST_REDIS_URL` | Recommendation (Feast online store) |
| `ALS_S3_PREFIX`, `GRAPH_S3_PREFIX`, `SASREC_S3_PREFIX`, `POPULARITY_S3_PREFIX`, `LTR_S3_PREFIX`, `LINUCB_S3_PREFIX` | Training + Recommendation |
| `RECOMMENDATION_SERVICE_URL` | User service (internal refresh) |
| `INTERNAL_API_KEY` | Recommendation internal endpoints |

---

## 3 · Deploy Redis

```bash
kubectl apply -f kubernetes/redis.yaml          # app cache
kubectl apply -f kubernetes/redis-feast.yaml    # Feast online store

kubectl get pods -l app=redis
kubectl get pods -l app=feast-redis
```

Wait until both are `Running`.

---

## 4 · Build and load images

Services that need **root** build context (Dockerfile copies from multiple repo dirs):
```bash
docker build -f src/recommendation_service/Dockerfile -t rahulkrish28/recommendation-service:latest .
docker build -f src/model_training_service/ltr_train/Dockerfile -t rahulkrish28/ltr-train:latest .
```

Services that build from their own directory:
```bash
docker build -t rahulkrish28/user-service:latest          src/user_service
docker build -t rahulkrish28/search-service:latest        src/search_service
docker build -t rahulkrish28/clickstream-consumer:latest  src/clickstream_consumer
docker build -t rahulkrish28/metrics-sidecar:latest       src/metrics_sidecar
```

Load into Kind (append `--name <cluster-name>` if your cluster isn't named `kind`):
```bash
for img in recommendation-service user-service search-service clickstream-consumer metrics-sidecar; do
  kind load docker-image rahulkrish28/$img:latest --name bibliophileai-cluster
done
```

---

## 5 · Ingress and application services

```bash
cd kubernetes && ./install-ingress.sh && cd ..

kubectl apply -f kubernetes/user-auth-deployment.yaml
kubectl apply -f kubernetes/search-deployment.yaml
kubectl apply -f kubernetes/recommendation-deployment.yaml
kubectl apply -f kubernetes/consumer-deployment.yaml

kubectl get pods    # wait until all Running and Ready
```

Apply the HPA for the recommendation service:
```bash
kubectl apply -f kubernetes/recommendation-hpa.yaml
```

---

## 6 · Expose the API

```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
```

Leave this running. The API is now reachable at `http://localhost:8080`.

---

## 7 · Frontend

```bash
cd frontend/bibliophile-ai-frontend
npm install
```

Optional — Google OAuth:
```bash
echo "VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com" > .env.local
```

```bash
npm run dev    # http://localhost:5173
```

Register a user, complete onboarding. The first recommendation load triggers the full pipeline (1–2 min on cache miss); all subsequent loads are fast from Redis.

---

## 8 · Monitoring (optional)

```bash
cd kubernetes/monitoring && ./install-monitoring.sh && cd ../..

# Apply custom alerting rules
kubectl apply -f kubernetes/monitoring/alerting-rules.yaml

# Grafana
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
# → http://localhost:3000  (admin / admin)

# Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
# → http://localhost:9090/alerts
```

The Grafana dashboard is provisioned automatically via ConfigMap. It includes P50/P95/P99 latency, cache hit rate, NDCG@10, engagement rates, algorithm candidate counts, and cluster health.

---

## 9 · Airflow and model training (optional)

```bash
cd kubernetes && ./install-airflow.sh && cd ..
```

Build and load training images:
```bash
docker build -t rahulkrish28/als-train:latest        src/model_training_service/als_train
docker build -t rahulkrish28/graph-train:latest      src/model_training_service/graph_train
docker build -t rahulkrish28/sasrec-train:latest     src/model_training_service/sasrec_train
docker build -t rahulkrish28/popularity-train:latest src/model_training_service/popularity_train
# ltr-train was already built with root context in step 4

for img in als-train graph-train sasrec-train popularity-train ltr-train; do
  kind load docker-image rahulkrish28/$img:latest --name bibliophileai-cluster
done
```

Expose the Airflow UI and set Variables (same credentials as Kubernetes Secrets):
```bash
kubectl port-forward svc/airflow-api-server 8082:8080
# → http://localhost:8082  — Admin → Variables → add MONGO_URI, S3_URI, AWS credentials, model prefixes
```

Unpause the 5 DAGs and trigger a run. Each training job saves its model to S3 and — for LTR — writes offline eval metrics (`eval_metrics_latest.json`) alongside the model.

---

## 10 · Populate book data (optional)

Required if Supabase / Pinecone / Neo4j are empty.

```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="..."
export PINECONE_API_KEY="..."
export NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_S3_BUCKET="your-bucket"

cd book_data_importer && pip install -r requirements.txt

python import_books.py          # Gutendex API → Supabase
python scrape_descriptions.py   # optional: add descriptions
python book_embedding.py        # Supabase books → Pinecone
python graph_book_importer.py   # books/authors/genres → Neo4j
python upload_epubs_to_s3.py    # EPUBs → S3
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImagePullBackOff` | Run `kind load docker-image <image>:latest --name <cluster>` after building |
| `504` on `/recommend/combined` | First pipeline run takes 1–2 min — Ingress is already set to 300s; wait it out |
| Frontend "No data" / CORS errors | Confirm port-forward on `8080` is active and healthy |
| Grafana "No data" for app metrics | Verify ServiceMonitors are applied and metrics sidecar container is running |
| Custom alerts not in Prometheus | Run `helm upgrade kube-prometheus-stack ...` after adding `ruleSelectorNilUsesHelmValues: false` to `prometheus-stack-values.yaml` |
| Consumer Redis warning at startup | Expected on first boot if Redis isn't ready yet — connection retries automatically on next event |

---

## Quick-start checklist

```
□ External service accounts + credentials ready
□ kind create cluster --config kind-cluster-config.yaml
□ kubectl apply -f kubernetes/my-secrets.yaml
□ kubectl apply -f kubernetes/redis.yaml && redis-feast.yaml
□ Build + kind load all service images
□ ./kubernetes/install-ingress.sh
□ kubectl apply -f kubernetes/user-auth-deployment.yaml (+ search, recommendation, consumer)
□ kubectl apply -f kubernetes/recommendation-hpa.yaml
□ kubectl port-forward ingress-nginx ... 8080:80
□ npm install && npm run dev  →  http://localhost:5173
□ (optional) monitoring, Airflow, training images, book import
```
