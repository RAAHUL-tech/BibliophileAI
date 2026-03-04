# User Service

FastAPI service for user authentication, registration, profile, preferences, bookmarks, reviews, and sessions. It also produces clickstream events to SQS for the consumer pipeline.

## Functionality

- **Authentication**: Password-based login (JWT) and Google OAuth; JWT creation/validation via `auth.py`.
- **User store**: Supabase for users, preferences, books, bookmarks, reviews, sessions (`supabase_client.py`).
- **Social graph**: Neo4j for follows, read/rated/bookmarked books, preferences (`neo4j_client.py`).
- **User embeddings**: Pinecone user-preferences index for content-based recommendation; preferences text is embedded and upserted (`user_embeddings.py`).
- **Clickstream**: Events (e.g. page_turn, bookmark) sent to AWS SQS FIFO via `produce.py` for the clickstream consumer.
- **Metrics**: HTTP request/error counts written to `/metrics-data/app_metrics.json` for the metrics sidecar (`app_metrics.py`).

## API (high level)

- `POST /api/v1/user/register`, `POST /login` (form), Google callback
- Profile: get/update profile, preferences (genres, authors, pincode, age)
- Bookmark: add/remove, list bookmarks
- Reviews: submit, list, average rating per book
- Sessions: create, end, get current session (for clickstream)
- Books: by IDs, popular authors; EPUB download URL template (S3)

## Implementation in this project

- **main.py**: Routes, middleware (CORS, metrics), dependency `get_current_user` (JWT + Supabase lookup).
- **schemas.py**: Pydantic models for request/response.
- **Dockerfile**: Multi-stage build; port 8000.
- **Kubernetes**: `user-auth-deployment.yaml` (Service + Deployment with metrics sidecar, env from Secret).

## Environment

`SECRET_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `PINECONE_API_KEY`, `SQS_QUEUE_URL`, `GOOGLE_CLIENT_ID`, etc.
