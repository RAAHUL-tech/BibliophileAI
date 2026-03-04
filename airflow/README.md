# Airflow

Apache Airflow is used to **schedule and run** the recommendation model training jobs as Kubernetes pods. DAGs are packaged into a custom Docker image and deployed via Helm; each DAG triggers a one-off pod that runs the corresponding training script (ALS, graph, LTR, popularity, SASRec).

## Functionality

- **DAGs**: Each DAG defines a schedule and a single task (or task group) that runs a training job in a pod.
  - **als_cf_training** (`als_train_dag.py`): Daily; runs `als-train` image, entrypoint runs ALS training and uploads user/book factors to S3.
  - **graph_analytics_training** (`graph_train_dag.py`): Daily; runs `graph-train` image for Node2Vec on Neo4j graph, uploads embeddings to S3.
  - **ltr_training** (`ltr_train_dag.py`): Daily; runs `ltr-train` image for XGBoost LambdaRank, uploads model to S3.
  - **popularity_training** (`popularity_dag.py`): Hourly; runs `popularity-train` image to update Redis trending keys and S3 model.
  - **sasrec_training** (`sasrec_train_dag.py`): Daily; runs `sasrec-train` image for SASRec transformer, uploads checkpoint to S3.
- **KubernetesPodOperator**: Each task uses `KubernetesPodOperator` with the training image, namespace `default`, resource limits, and env vars from Airflow Variables (e.g. `MONGO_URI`, `S3_URI`, `ALS_S3_PREFIX`, AWS keys). Pods are deleted after completion (`is_delete_operator_pod=True`).
- **Variables**: Secrets and config (MongoDB, S3, Neo4j, Redis, AWS, etc.) are stored in Airflow Variables and passed into the pods so the training code can connect to the same data stores as the rest of the system.

## Implementation in this project

- **airflow-dags/**: Python files defining the DAGs; loaded by Airflow at startup from the image or a mounted volume.
- **Dockerfile**: Builds an image that includes the DAGs (e.g. copied into `/opt/airflow/dags`) and any dependencies; image is pushed to a registry (e.g. `rahulkrish28/airflow-with-dags:latest`).
- **Kubernetes**: Airflow is installed via Helm using `kubernetes/airflow-values.yaml`; `install-airflow.sh` builds the image, pushes it, and runs `helm upgrade --install` with that values file. The executor is typically KubernetesExecutor or LocalExecutor with KubernetesPodOperator.
- **Training images**: Each training job has its own image (e.g. `rahulkrish28/als-train:latest`); the DAGs reference these images and the corresponding entrypoint (e.g. `/app/entrypoint.sh`).

## Running

1. Install Airflow: from repo root, `./kubernetes/install-airflow.sh` (builds DAG image, installs Helm release).
2. Create Airflow Variables for `MONGO_URI`, `S3_URI`, `*_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, Neo4j, Redis, etc.
3. Unpause the DAGs in the Airflow UI; they will run on schedule and launch the training pods in the cluster.
