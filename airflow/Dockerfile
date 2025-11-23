FROM apache/airflow:3.0.2

# Upgrade pip first
RUN pip install --upgrade pip

# Install required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy DAGs into Airflow
COPY airflow-dags /opt/airflow/dags
