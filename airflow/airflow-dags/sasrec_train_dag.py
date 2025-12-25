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
    dag_id="sasrec_training",
    default_args=default_args,
    #schedule="*/30 * * * *",  # every 30 minutes
    schedule="@daily",  
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "sasrec"],
) as dag:

    sasrec_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/sasrec-train:latest",
        cmds=["/app/entrypoint.sh"],
        name="sasrec-trainer",
        task_id="sasrec_train_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        # Schedule only on dedicated SASRec node (if it exists)
        # TODO: Uncomment after recreating cluster with dedicated node
        # If node doesn't exist, the pod won't schedule - comment out to use any node
        # node_selector={"workload": "sasrec-training"},
        # Resource limits to ensure sufficient memory
        # Start with minimal requests - increase if needed after confirming it works
        # If pod still doesn't schedule, comment out container_resources entirely
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "1Gi", "cpu": "500m"},
            limits={"memory": "6Gi", "cpu": "4"},
        ),
        env_vars={
            "MONGO_URI": Variable.get("MONGO_URI", default_var=""),
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "SASREC_S3_PREFIX": Variable.get("SASREC_S3_PREFIX", default_var="SASRec_Train"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
            "RAY_ADDRESS": "local",
        },
    )
