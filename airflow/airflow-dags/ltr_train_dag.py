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
    dag_id="ltr_training",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "learning-to-rank"],
) as dag:

    ltr_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/ltr-train:latest",
        cmds=["/app/entrypoint.sh"],
        name="ltr-trainer",
        task_id="ltr_training_task",
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
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "LTR_S3_PREFIX": Variable.get("LTR_S3_PREFIX", default_var="LTR_Train"),
            "FEAST_REPO_PATH": "/app/feature_repo",
            "LTR_EVENT_DAYS": Variable.get("LTR_EVENT_DAYS", default_var="90"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
        },
    )
