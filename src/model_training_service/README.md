# Model Training Service

Offline training jobs for the recommendation system. Each subfolder is a standalone runnable (often used inside Kubernetes pods triggered by Airflow). All jobs read from MongoDB (click_stream) and/or Neo4j and write models or artifacts to S3 and optionally Redis.

## Subfolders and algorithms

| Subfolder | Algorithm | Input | Output | Used by |
|-----------|-----------|--------|--------|---------|
| **als_train** | Alternating Least Squares (implicit CF) | MongoDB events → user-item matrix | User/book factor Parquet on S3 | `collaborative_filtering.py` |
| **graph_train** | Node2Vec on user–book–author–genre graph | Neo4j graph | Node embeddings (Parquet) on S3 | `graph_recommendation.py` |
| **ltr_train** | XGBoost LambdaRank (LTR) | MongoDB relevance labels + Feast/S3 features | XGBoost model on S3 | `ltr_ranking.py` |
| **popularity_train** | Time-decayed popularity | MongoDB events | Redis trending keys + PyTorch model on S3 | `popularity_recommendation.py` |
| **sasrec_train** | SASRec (transformer session model) | MongoDB sessions | PyTorch checkpoint on S3 | `sasrec_inference.py` |

## Shared pattern

- **Ray**: Used (where applicable) for distributed load/train (e.g. ALS, graph, LTR, popularity, SASRec).
- **Entrypoint**: Each subfolder has an `entrypoint.sh` that starts a Ray head, runs the Python trainer, then stops Ray.
- **Environment**: `MONGO_URI`, `S3_URI`, `*_S3_PREFIX`, AWS keys; Neo4j for graph; Redis for popularity; Feast repo/bucket for LTR.
- **Airflow**: DAGs in `airflow/airflow-dags/` trigger Kubernetes pods that run the corresponding image and entrypoint.

See each subfolder’s README for algorithm details and implementation.
