# Build from repo root so feature_repo is in context:
#   docker build -f src/recommendation_service/Dockerfile -t recommendation-service .
FROM python:3.9-slim
WORKDIR /app

COPY src/recommendation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/recommendation_service/ .

# Feast feature repo (for FEAST_REPO_PATH, materialize_incremental, get_online_features)
COPY feast_feature_store/feature_repo /app/feature_repo
ENV FEAST_REPO_PATH=/app/feature_repo

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
