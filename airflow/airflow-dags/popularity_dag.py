from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from kubernetes.client import models as k8s

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="popularity_training",
    default_args=default_args,
    schedule="@hourly",  # Every hour
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "popularity"],
) as dag:

    popularity_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/popularity-train:latest",  
        cmds=["/app/entrypoint.sh"],
        name="popularity-trainer",
        task_id="popularity_training_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        # Resource limits to ensure sufficient memory
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "1Gi", "cpu": "500m"},
            limits={"memory": "4Gi", "cpu": "2"},
        ),
        env_vars={
            "MONGO_URI": Variable.get("MONGO_URI", default_var=""),
            "REDIS_URL": Variable.get("REDIS_URL", default_var="redis://redis:6379/0"),
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "POPULARITY_S3_PREFIX": Variable.get("POPULARITY_S3_PREFIX", default_var="Popularity_Train"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
            "RAY_ADDRESS": "local",
        },
    )
