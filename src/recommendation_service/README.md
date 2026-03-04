# Recommendation Service

FastAPI service that serves **combined recommendations** in a Netflix-style layout: categories such as Content-Based, Collaborative Filtering, Social (graph), Session-Based (SASRec), Trending Now, For You (LinUCB), and Top Picks (LTR). Results are cached in Redis; on cache miss a full pipeline runs (or content-based + trending for new users).

## Algorithms and how they are implemented

| Algorithm | Role | Implementation in this project |
|-----------|------|--------------------------------|
| **Content-based** | User preference vector vs book vectors | `content_based_recommendation.py`: user vector from Pinecone user index → query book index → top-k book IDs/scores. |
| **Collaborative filtering (ALS)** | User–item matrix factorization | `collaborative_filtering.py`: user/book factor matrices loaded from S3 (trained by `als_train`), inner product for scores. |
| **Graph / social** | Personalized PageRank on user–book–author–genre graph | `graph_recommendation.py`: Neo4j subgraph around user, weighted edges, PageRank with social boost from friends’ ratings; embeddings from S3 (from `graph_train`). |
| **Session-based (SASRec)** | Transformer over item sequence | `sasrec_inference.py`: session from MongoDB, trained SASRec model from S3 (from `sasrec_train`), next-item style scoring. |
| **Popularity / trending** | Time-decayed event counts | `popularity_recommendation.py`: Redis keys per window (7d/30d/90d), optional S3 PyTorch fallback (from `popularity_train`). |
| **LinUCB** | Contextual bandit for ranking | `linucb_inference.py` + `linucb_helper.py`: per-user θ, A, b in Redis; feature x from content/ALS/graph + user/book metadata; score = θ′x + α√(x′A⁻¹x); reward on logout from MongoDB. |
| **LTR (Learning-to-Rank)** | XGBoost LambdaRank over 29 features | `ltr_ranking.py`: features from Feast (or in-memory from feature_service), model from S3 (from `ltr_train`); `ltr_postprocess.py`: diversity (author/genre), MMR, language rule for Top Picks. |
| **Feast** | Feature store for LTR | `feature_engineering/feature_service.py`: builds 29-feature vectors from `combined_scores_map` and session context, writes to Feast offline store (S3), materializes to Redis for online LTR. |

## Request flow

1. **Cache hit**: Return cached categories from Redis; inject “Trending Now” from global Redis.
2. **Cache miss (new user)**: Only content-based + trending; result cached.
3. **Cache miss (existing user)**: Run all algorithms above, merge candidates, LinUCB rank, optional Feast + LTR for “Top Picks”, fetch books from Supabase, build categories; cache (excluding Trending Now).

## Internal refresh

Background pods (or internal calls) can trigger `POST /internal/recommend/refresh-*` to update a single category in the user’s cache (e.g. after preference change).

## Files

- **main.py**: Combined endpoint, cache get/set, pipeline orchestration, internal refresh routes.
- **app_metrics.py**: HTTP, cache hit/miss, recommendation duration → `/metrics-data/app_metrics.json`.
- **Dockerfile** / **kubernetes**: `recommendation-deployment.yaml`, port 8001, metrics sidecar; env: Redis, Supabase, Pinecone, Neo4j, MongoDB, S3, Feast Redis, etc.
