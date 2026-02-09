# Build from repo root only (context must include feast_feature_store and src/):
#   cd /path/to/BibliophileAI
#   docker build -f src/model_training_service/ltr_train/Dockerfile -t rahulkrish28/ltr-train:latest .
FROM python:3.10-slim

WORKDIR /app

COPY src/model_training_service/ltr_train/requirements.txt .
COPY src/model_training_service/ltr_train/entrypoint.sh .

RUN chmod +x entrypoint.sh \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/model_training_service/ltr_train/ltr_train.py .
COPY feast_feature_store/feature_repo /app/feature_repo

ENTRYPOINT ["./entrypoint.sh"]
