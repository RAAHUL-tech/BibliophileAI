# BibliophileAI – Setup Guide

This guide walks a **new user** through setting up the full BibliophileAI stack on their machine: cluster, databases, secrets, services, ingress, optional Airflow and monitoring, frontend, and data import.

---

## 1. Prerequisites

Install the following on your computer.

### Required

| Tool | Purpose | Install / Check |
|------|--------|-----------------|
| **Docker** | Build and run containers | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Engine + Compose). Run `docker --version`. |
| **kubectl** | Kubernetes CLI | [Install kubectl](https://kubernetes.io/docs/tasks/tools/). Run `kubectl version --client`. |
| **Kind** | Local Kubernetes cluster | `go install sigs.k8s.io/kind@latest` or see [kind docs](https://kind.sigs.k8s.io/docs/user/quick-start/#installation). Run `kind version`. |
| **Helm 3** | Install charts (Ingress, Airflow, Prometheus stack) | [Install Helm](https://helm.sh/docs/intro/install/). Run `helm version`. |
| **Node.js 18+** | Frontend (Vite, React) | [Node.js](https://nodejs.org/). Run `node -v` and `npm -v`. |
| **Python 3.9+** | Data import scripts, Feast, local dev (optional) | Pre-installed on macOS/Linux or [python.org](https://www.python.org/). Run `python3 --version`. |

### Optional (for running training jobs and Feast locally)

- **Ray** – if you run training jobs on your machine instead of in Kubernetes.
- **Git** – to clone the repo.

---

## 2. External services and accounts

BibliophileAI uses several managed services. Create accounts and note credentials (you will put them into Kubernetes Secrets and/or `.env`).

| Service | Purpose |
|--------|---------|
| **Supabase** | PostgreSQL (users, books, preferences, bookmarks, reviews). Create project → get **Project URL** and **service_role key**. |
| **MongoDB Atlas** | Click-stream events (and session data for SASRec). Create cluster → get **connection string** (e.g. `mongodb+srv://...`). |
| **Neo4j Aura** (or self-hosted) | Social graph (users, books, authors, follows, read/rated). Get **URI**, **user**, **password**. |
| **Pinecone** | Vector index for user preferences and book embeddings. Create index (e.g. `user-preferences-index`, `book-metadata-index`) → **API key**. |
| **AWS** | S3 (models, Feast, EPUBs), SQS (clickstream queue). Create IAM user → **Access Key ID**, **Secret Access Key**. Create S3 bucket(s) and SQS FIFO queue. |
| **Google Cloud** | OAuth for “Login with Google”. Create OAuth 2.0 Client ID (Web application) → **Client ID**. |

Use the same AWS region (e.g. `us-east-2`) for S3 and SQS; the clickstream consumer expects `AWS_REGION` or `AWS_DEFAULT_REGION`.

---

## 3. Clone the repository

```bash
git clone https://github.com/RAAHUL-tech/BibliophileAI.git
cd BibliophileAI
```

---

## 4. Kubernetes cluster (Kind)

Create a local cluster using the project’s Kind config (control-plane + workers):

```bash
kind create cluster --config kind-cluster-config.yaml
kubectl cluster-info
kubectl get nodes
```

If you use a different cluster name, set `KUBE_CONTEXT` when running Helm/scripts (e.g. `KUBE_CONTEXT=kind-your-name`).

---

## 5. Kubernetes Secrets

Application and training pods read credentials from a Kubernetes Secret named `secret` in the `default` namespace.

### 5.1 Create your own secret (recommended)

**Do not commit real credentials.** Create a local file (e.g. `kubernetes/my-secrets.yaml`) from the template below and fill in your values. All values under `data` must be **base64-encoded**.

Encode on the command line:

```bash
echo -n "your-secret-value" | base64
```

Example (macOS/Linux):

```bash
# Example (replace with your real values)
echo -n "my-jwt-secret-key" | base64
echo -n "https://xxxx.supabase.co" | base64
echo -n "eyJhbGc..." | base64
# ... etc.
```

### 5.2 Secret keys the apps expect

Use these keys in your Secret (same as in `kubernetes/secrets.yaml`). Omit keys you don’t use (e.g. some training prefixes) if a component isn’t deployed.

| Key | Used by | Description |
|-----|--------|-------------|
| `SECRET_KEY` | User / Recommendation | JWT signing key (e.g. 32+ character string). |
| `SUPABASE_URL` | User, Search, Recommendation, Importers | Supabase project URL. |
| `SUPABASE_KEY` | User, Search, Recommendation, Importers | Supabase `service_role` key. |
| `GOOGLE_CLIENT_ID` | User (OAuth) | Google OAuth 2.0 Web client ID. |
| `PINECONE_API_KEY` | Search, Recommendation, User embeddings | Pinecone API key. |
| `AWS_ACCESS_KEY_ID` | All S3/SQS and training | AWS access key. |
| `AWS_SECRET_ACCESS_KEY` | All S3/SQS and training | AWS secret key. |
| `AWS_DEFAULT_REGION` or `AWS_REGION` | Consumer, training | e.g. `us-east-2`. |
| `SQS_QUEUE_URL` | User (producer), Consumer | Full SQS FIFO queue URL. |
| `MONGO_URI` | Recommendation, Consumer, Training | MongoDB connection string. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | User, Recommendation, Graph training | Neo4j connection and auth. |
| `S3_URI` | Recommendation, Training | S3 bucket (e.g. `my-bucket` or `s3://my-bucket`). |
| `REDIS_URL` | Recommendation, Popularity training | Redis URL (in-cluster: `redis://redis:6379/0`). |
| `FEAST_REDIS_URL` | Recommendation (Feast online store) | Redis for Feast (in-cluster: `redis://feast-redis:6379`). |
| `ALS_S3_PREFIX`, `GRAPH_S3_PREFIX`, `SASREC_S3_PREFIX`, `POPULARITY_S3_PREFIX`, `LTR_S3_PREFIX`, `LINUCB_S3_PREFIX` | Training + Recommendation | S3 key prefixes for each model/artifact. |
| `RECOMMENDATION_SERVICE_URL` | User (internal refresh) | e.g. `http://recommendation-service:8001`. |
| `INTERNAL_API_KEY` | Recommendation (internal endpoints) | Shared key for internal refresh calls. |

Reference structure (all values in `data` must be base64):

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

Apply the secret:

```bash
kubectl apply -f kubernetes/my-secrets.yaml
# Or, if you use the repo template after filling: kubectl apply -f kubernetes/secrets.yaml
```

---

## 6. Deploy infrastructure in the cluster

### 6.1 Redis (app cache and sessions)

```bash
kubectl apply -f kubernetes/redis.yaml
kubectl get pods -l app=redis
```

### 6.2 Redis for Feast (online feature store)

```bash
kubectl apply -f kubernetes/redis-feast.yaml
kubectl get pods -l app=feast-redis
```

Wait until Redis and Feast Redis pods are `Running`.

---

## 7. Build and load application Docker images

The services run as containers. For **Kind**, load images into the cluster after building (no registry required for local dev). Replace the image names if you use your own registry.

Build and load (run from repo root):

```bash
# Metrics sidecar (used by every app pod)
docker build -t rahulkrish28/metrics-sidecar:latest src/metrics_sidecar
kind load docker-image rahulkrish28/metrics-sidecar:latest

# User service
docker build -t rahulkrish28/user-service:latest src/user_service
kind load docker-image rahulkrish28/user-service:latest

# Search service
docker build -t rahulkrish28/search-service:latest src/search_service
kind load docker-image rahulkrish28/search-service:latest

# Recommendation service
docker build -t rahulkrish28/recommendation-service:latest src/recommendation_service
kind load docker-image rahulkrish28/recommendation-service:latest

# Clickstream consumer
docker build -t rahulkrish28/clickstream-consumer:latest src/clickstream_consumer
kind load docker-image rahulkrish28/clickstream-consumer:latest
```

If your deployment YAMLs use different image names/tags, adjust the `-t` and `kind load docker-image` accordingly. For a **remote cluster**, push to a registry and ensure the cluster can pull:

```bash
docker push rahulkrish28/metrics-sidecar:latest
docker push rahulkrish28/user-service:latest
# ... etc.
```

---

## 8. Install Ingress and apply routes

```bash
cd kubernetes
./install-ingress.sh
```

This installs the NGINX Ingress controller and applies the BibliophileAI Ingress (routes for `/api/v1/user`, `/api/v1/recommend`, `/api/v1/search`). If the webhook is not ready, the script retries applying the Ingress.

---

## 9. Deploy application services

Deploy all four app services (each includes the metrics sidecar):

```bash
kubectl apply -f kubernetes/user-auth-deployment.yaml
kubectl apply -f kubernetes/search-deployment.yaml
kubectl apply -f kubernetes/recommendation-deployment.yaml
kubectl apply -f kubernetes/consumer-deployment.yaml
```

Check pods and services:

```bash
kubectl get pods
kubectl get svc
```

Wait until pods are `Running` and `Ready`. If any pod fails, check `kubectl describe pod <name>` and `kubectl logs <pod> -c <container>`.

---

## 10. Expose the API on your machine

Port-forward the Ingress controller so the API is available at `http://localhost:8080` (frontend expects this by default):

```bash
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80
```

Leave this terminal running. In another terminal, you can run the frontend and optional steps below.

---

## 11. Frontend setup and run

### 11.1 Install dependencies and configure Google OAuth

```bash
cd frontend/bibliophile-ai-frontend
npm install
```

Optional: create a `.env` (or `.env.local`) for Google Login:

```bash
# .env or .env.local (do not commit if it contains secrets)
VITE_GOOGLE_CLIENT_ID=your-google-oauth-client-id.apps.googleusercontent.com
```

If you skip this, “Login with Google” will not work until you set the client ID.

### 11.2 Start the dev server

```bash
npm run dev
```

Vite runs at **http://localhost:5173**. The app calls **http://localhost:8080** for all APIs (ensure the port-forward from step 10 is active).

### 11.3 Verify

- Open http://localhost:5173.
- Register a new user or log in (password or Google if configured).
- Complete onboarding (genres, authors, demographics) if prompted.
- You should see the homepage with recommendation categories and search. First-time recommendations may take longer (cache miss); subsequent loads should be faster.

---

## 12. Optional: Monitoring (Prometheus + Grafana)

To scrape app and cluster metrics and view the BibliophileAI dashboard:

```bash
cd kubernetes/monitoring
./install-monitoring.sh
```

This installs the Prometheus stack and applies the ServiceMonitors. Then:

- **Grafana**: `kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80` → http://localhost:3000 (default login often `admin` / `admin`).
- **Prometheus**: `kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090` → http://localhost:9090.

See [kubernetes/monitoring/README.md](kubernetes/monitoring/README.md) for dashboard and metric details.

---

## 13. Optional: Airflow and training jobs

Airflow runs the periodic training DAGs (ALS, graph, LTR, popularity, SASRec) as Kubernetes pods.

### 13.1 Build and install Airflow

From repo root (script uses `KUBE_CONTEXT`; default in script is `kind-bibliophileai-cluster`; if your Kind cluster has another name, set it):

```bash
cd kubernetes
./install-airflow.sh
```

This builds the Airflow image (with DAGs), pushes it to the registry you use, and installs Airflow via Helm. If you use Kind without a registry, you may need to load the Airflow image into Kind and adjust the Helm values so the scheduler/workers use the loaded image.

### 13.2 Configure Airflow Variables

Training pods need the same credentials as the apps. In the Airflow UI, set **Variables** (e.g. Admin → Variables) such as:

- `MONGO_URI`
- `S3_URI`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `ALS_S3_PREFIX`, `GRAPH_S3_PREFIX`, `SASREC_S3_PREFIX`, `POPULARITY_S3_PREFIX`, `LTR_S3_PREFIX`
- `REDIS_URL`
- (and any others your DAGs or `airflow-values.yaml` reference)

### 13.3 Build and load (or push) training images

Build the training images and, for Kind, load them:

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

Expose the Airflow UI (port 8082 to avoid conflict with Ingress 8080):

```bash
kubectl port-forward svc/airflow-api-server 8082:8080 --namespace default
```

Open http://localhost:8082, unpause the DAGs, and trigger runs as needed.

---

## 14. Optional: Populate books and data

If Supabase and Pinecone/Neo4j are empty, run the book data importers (with the same credentials configured in your environment or in the cluster).

### 14.1 Environment for importers

Set the same Supabase, Pinecone, Neo4j, and AWS variables (e.g. in a `.env` file or export in the shell). Example:

```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
export PINECONE_API_KEY="your-pinecone-key"
export NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_S3_BUCKET="your-bucket"
```

### 14.2 Run import scripts (order suggested)

```bash
cd book_data_importer
pip install -r requirements.txt

# 1) Import books from Gutendex into Supabase
python import_books.py

# 2) Optional: scrape descriptions and update Supabase
python scrape_descriptions.py

# 3) Upsert book embeddings to Pinecone
python book_embedding.py

# 4) Create Neo4j graph (books, authors, genres)
python graph_book_importer.py

# 5) Upload EPUBs to S3 for the reader
python upload_epubs_to_s3.py
```

See [book_data_importer/README.md](book_data_importer/README.md) for details.

---

## 15. Troubleshooting

- **Pods not starting**: `kubectl describe pod <name>` and `kubectl logs <pod> -c <container>`. Check image name/tag and that the `secret` exists and keys match what the deployment expects.
- **504 on /api/v1/recommend/combined**: Ingress proxy timeout; the recommendation pipeline can take 1–2 minutes on first run. The project’s Ingress is already set to 300s; if you changed it, restore a higher proxy read timeout.
- **Frontend “No data” or network errors**: Ensure the Ingress port-forward is running on 8080 and the frontend is calling `http://localhost:8080`. Check CORS and that the backend pods are healthy.
- **Grafana “No data” for app metrics**: Ensure ServiceMonitors are applied, Services have the correct labels (e.g. `app: recommendation`), and app pods include the metrics sidecar. See [kubernetes/monitoring/servicemonitors/README.md](kubernetes/monitoring/servicemonitors/README.md).
- **SQS / boto3 “NoRegionError”**: Set `AWS_REGION` or `AWS_DEFAULT_REGION` in the consumer (and any producer) environment.
- **Kind: ImagePullBackOff**: For local Kind, use `kind load docker-image <image>:<tag>` after building. For a remote registry, ensure the cluster can pull and image names in YAML match.

---

## 16. Quick reference: minimal run order

1. Prerequisites + external accounts.
2. Clone repo → create Kind cluster → create and apply Secret.
3. `kubectl apply -f kubernetes/redis.yaml` and `kubernetes/redis-feast.yaml`.
4. Build and load (or push) app images + metrics-sidecar; `./kubernetes/install-ingress.sh`; apply the four deployment YAMLs.
5. `kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 8080:80`.
6. Frontend: `cd frontend/bibliophile-ai-frontend && npm install && npm run dev` → http://localhost:5173.
7. Optional: monitoring, Airflow, training images, book import scripts as above.

For per-component details, see the README files in each folder (e.g. `src/user_service/README.md`, `kubernetes/README.md`, `kubernetes/monitoring/README.md`).
