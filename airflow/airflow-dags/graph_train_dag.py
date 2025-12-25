from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="graph_analytics_training",
    default_args=default_args,
    schedule="@daily",  
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "graph"],
) as dag:

    graph_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/graph-train:latest",  
        cmds=["/app/entrypoint.sh"],
        name="graph-trainer",
        task_id="graph_analytics_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        env_vars={
            "NEO4J_URI": Variable.get("NEO4J_URI", default_var=""),
            "NEO4J_USER": Variable.get("NEO4J_USER", default_var="neo4j"),
            "NEO4J_PASSWORD": Variable.get("NEO4J_PASSWORD", default_var=""),
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "GRAPH_S3_PREFIX": Variable.get("GRAPH_S3_PREFIX", default_var="Graph_Train"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
            "RAY_ADDRESS": "local",
        },
    )
