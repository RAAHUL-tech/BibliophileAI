# Search Service

FastAPI service that provides semantic book search: query text is embedded via Pinecone, similar books are retrieved from the book index, and full metadata is loaded from Supabase.

## Algorithm

- **Embedding**: Query is sent to Pinecone’s embed API (e.g. `llama-text-embed-v2`) with `input_type: "query"` to get a vector.
- **Vector search**: The book-metadata index is queried with this vector (e.g. top-k similarity); returns book IDs and scores.
- **Enrichment**: Book IDs are resolved to full book records from Supabase (title, authors, categories, language, thumbnail, etc.); Pinecone is not used for display metadata.

## Implementation in this project

- **main.py**: Single search endpoint (e.g. `POST /api/v1/search`) that:
  1. Embeds the request query with Pinecone embed API (correct `parameters` and `inputs`).
  2. Queries the book index with the returned vector.
  3. Fetches book details from Supabase by ID.
  4. Returns a list of books with metadata.
- **app_metrics.py**: Writes HTTP request/error counts to `/metrics-data/app_metrics.json` for the metrics sidecar.
- **Dockerfile**: Port 8002.
- **Kubernetes**: `search-deployment.yaml` (Service + Deployment with metrics sidecar).

## Environment

`SUPABASE_URL`, `SUPABASE_KEY`, `PINECONE_API_KEY`, `BOOKS_FETCH_TIMEOUT` (optional). Book index name is fixed (e.g. `book-metadata-index`) in code.
