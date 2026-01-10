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
    dag_id="linucb_training",
    default_args=default_args,
    schedule="@daily",  
    start_date=datetime.utcnow() - timedelta(days=1),
    catchup=False,
    tags=["recommendation", "linucb"],
) as dag:

    linucb_job = KubernetesPodOperator(
        namespace="default",
        image="rahulkrish28/linucb-train:latest",  
        cmds=["/app/entrypoint.sh"],
        name="linucb-trainer",
        task_id="linucb_training_task",
        is_delete_operator_pod=True,
        in_cluster=True,
        get_logs=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "1Gi", "cpu": "500m"},
            limits={"memory": "4Gi", "cpu": "2"},    
        ),
        env_vars={
            "MONGO_URI": Variable.get("MONGO_URI", default_var=""),
            "REDIS_URL": Variable.get("REDIS_URL", default_var="redis://redis:6379/0"),
            "S3_URI": Variable.get("S3_URI", default_var=""),
            "PINECONE_API_KEY": Variable.get("PINECONE_API_KEY", default_var=""),
            "NEO4J_URI": Variable.get("NEO4J_URI", default_var=""),
            "NEO4J_USER": Variable.get("NEO4J_USER", default_var="neo4j"),
            "NEO4J_PASSWORD": Variable.get("NEO4J_PASSWORD", default_var=""),
            "LINUCB_S3_PREFIX": Variable.get("LINUCB_S3_PREFIX", default_var="LinUCB_Train"),
            "GRAPH_S3_PREFIX": Variable.get("GRAPH_S3_PREFIX", default_var="Graph_Train"),
            "ALS_S3_PREFIX": Variable.get("ALS_S3_PREFIX", default_var="ALS_Train"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID", default_var=""),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY", default_var=""),
        },
    )
