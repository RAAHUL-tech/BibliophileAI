# LTR (Learning-to-Rank) Training

Trains an XGBoost LambdaRank model to rank (user, book) candidates by engagement. Uses relevance labels derived from click-stream events and 29 features from the Feast feature store (or S3 parquet). The trained model is used by the recommendation service to produce the “Top Picks” category.

## Algorithm

- **Objective**: Learning-to-rank with NDCG; XGBoost’s `rank:ndcg` objective.
- **Labels**: Per (user_id, book_id) from MongoDB events over a time window (e.g. 90 days). Relevance 0–5: e.g. 5 = review with rating≥4, 4 = read/complete, 3 = many page_turn, 2 = some page_turn, 1 = bookmark/click, 0 = ignore.
- **Features**: 29 features aligned with Feast `user_book_features`: retrieval scores (6), genre/author/language match, ratings, social (friend reads, author following, etc.), session (position, genre drift, time since action, device, session length), popularity (global rank, trending), diversity. Features are read from Feast offline store or from S3 parquet written by the recommendation service.
- **Training**: Entity dataframe (user_id, book_id, user_book, relevance, timestamp); join to feature table; train XGBoost rank model; save to S3.

## Implementation in this project

- **ltr_train.py**:
  1. **Labels**: Ray task aggregates MongoDB `click_stream.events` into (user_id, book_id, user_book, relevance, timestamp) with the 0–5 relevance rules.
  2. **Features**: Load feature parquet from S3 (or Feast) for the same (user_book) keys; merge labels + features.
  3. **Train**: XGBoost train with `objective='rank:ndcg'`; save model (e.g. JSON or binary) to S3 under `LTR_S3_PREFIX`.
  4. **Upload**: Model path and prefix configurable via env.
- **entrypoint.sh**: Starts Ray head, runs `python ltr_train.py`, stops Ray.
- **Recommendation service**: `ltr_ranking.py` loads the model from S3; `feature_service` and Feast provide the 29 features at request time; `ltr_postprocess.py` applies diversity and business rules to the LTR output.

## Environment

`MONGO_URI`, `S3_URI`, `LTR_S3_PREFIX`, `FEAST_REPO_PATH`, `FEAST_S3_BUCKET`, `LTR_EVENT_DAYS`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAY_ADDRESS`.
