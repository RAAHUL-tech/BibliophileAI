FROM apache/airflow:3.0.2

COPY airflow-dags /opt/airflow/dags

RUN pip install --upgrade pip
RUN pip install "apache-airflow-providers-cncf-kubernetes>=10.0.0"
