# ALS (Alternating Least Squares) Training

Trains a collaborative filtering model using implicit feedback (clicks, reads, reviews, bookmarks). Produces user and item (book) factor matrices that the recommendation service uses to score books per user via inner product.

## Algorithm

- **Model**: Alternating Least Squares (ALS) for matrix factorization. We approximate the user–item interaction matrix as \( X \approx U V^T \), where \( U \) (user factors) and \( V \) (item factors) are learned.
- **Implicit feedback**: No explicit ratings required. Events are weighted (e.g. review=3, read=2, page_turn/bookmark=1); these weights are summed per (user, book) and optionally scaled by a confidence factor (e.g. `ALS_ALPHA`).
- **Library**: `implicit.als.AlternatingLeastSquares`; factors, regularization, and iterations are configurable via env.

## Implementation in this project

- **als_train.py**:
  1. **Load**: Ray task reads from MongoDB `click_stream.events`; keeps rows with `item_id`; filters by event type (read, page_turn, review, bookmark_add).
  2. **Weights**: Maps event type to weight (e.g. 3 for review, 2 for read, 1 for others); groups by (user_id, book_id) and sums weights.
  3. **Matrix**: Builds a sparse user–item matrix (rows=users, cols=books), optionally multiplied by `ALS_ALPHA`.
  4. **Train**: Fits ALS; writes user_factors and book_factors to Parquet.
  5. **Upload**: Uploads `user_factors.parquet` and `book_factors.parquet` to S3 under `ALS_S3_PREFIX`.
- **entrypoint.sh**: Starts Ray head, runs `python als_train.py`, stops Ray.
- **Recommendation service**: `collaborative_filtering.py` downloads these Parquet files from S3 and computes scores as user_factor · book_factor for top-k books.

## Environment

`MONGO_URI`, `S3_URI`, `ALS_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAY_ADDRESS` (optional), `ALS_ALPHA`, `ALS_FACTORS`, `ALS_REG`, `ALS_ITER`.
