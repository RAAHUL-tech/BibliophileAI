from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=15)
}

with DAG(
    dag_id="als_cf_training",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["recommendation", "ALS"]
) as dag:

    als_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/als-job:latest",
        cmds=["python", "als_train.py"],
        name="als-trainer",
        task_id="als_cf_train_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        env_vars={
            "MONGO_URI": Variable.get("MONGO_URI", default_var="replace_this_with_your_dev_uri"),
        },
    )
