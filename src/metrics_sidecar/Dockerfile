FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY exporter.py .

EXPOSE 9090
ENV METRICS_PORT=9090
CMD ["python", "-u", "exporter.py"]
