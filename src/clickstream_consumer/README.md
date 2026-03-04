# Clickstream Consumer

Service that consumes user events from an AWS SQS queue and persists them to MongoDB. It also records event counts for the metrics sidecar so Prometheus can scrape “events processed” metrics.

## Functionality

- **SQS consumer**: Long-polls the FIFO queue (`SQS_QUEUE_URL`); receives messages (JSON body with `user_id`, `event_type`, `item_id`, `metadata`, etc.).
- **Persistence**: Each message is written to MongoDB (e.g. `click_stream.events` collection) with an added `received_at` timestamp.
- **Metrics**: Event counts per `event_type` are written to `/metrics-data/app_metrics.json` via `app_metrics.record_event()` so the metrics sidecar exposes `app_events_processed_total` by event type.
- **Decoupling**: The user service (and any other producer) sends events to SQS; this consumer is the single writer to MongoDB, enabling async processing and backpressure at the queue.

## Implementation in this project

- **consumer.py**: Boto3 SQS client (with `AWS_REGION` / `AWS_DEFAULT_REGION`), MongoDB client with retry and Server API; loop: `receive_message` → parse → `insert_one` → `record_event` → `delete_message`. Starts metrics writer thread (`start_metrics_writer`).
- **app_metrics.py**: `MetricsStore` with `record_event(event_type)`; writes `events_processed_total` dict to the shared JSON file.
- **Dockerfile**: Port 8002 (same as search for K8s Service); optional metrics port 9090 via sidecar.
- **Kubernetes**: `consumer-deployment.yaml` (Service + Deployment with metrics sidecar); env: `SQS_QUEUE_URL`, `AWS_REGION`, `MONGO_URI`.
