from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="als_cf_training",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "ALS"]
) as dag:

    als_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/als-train:latest",
        cmds=["/app/entrypoint.sh"], 
        name="als-trainer",
        task_id="als_cf_train_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        env_vars={
            "MONGO_URI": Variable.get("MONGO_URI", default_var=""),
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "ALS_S3_PREFIX": Variable.get("ALS_S3_PREFIX", default_var="ALS_Train"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
            "RAY_ADDRESS": "local"
        },
    )
