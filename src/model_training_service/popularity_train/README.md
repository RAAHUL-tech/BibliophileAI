# Popularity / Trending Training

Computes time-decayed popularity scores per book from click-stream events and writes “trending” lists to Redis. Optionally trains a small PyTorch model and uploads it to S3 as a fallback when Redis is empty.

## Algorithm

- **Time decay**: Events (read, page_turn, review, bookmark_add) are weighted (e.g. complete=10, review=7, read=3×progress, etc.). Each event contributes to a book’s score with exponential decay: weight × exp(-λ × age_in_days), where λ = ln(2) / half_life.
- **Windows**: Multiple windows (e.g. 7d, 30d, 90d) with different half-lives and weights; combined (e.g. 0.5×7d + 0.3×30d + 0.2×90d) for a single trending score.
- **Smoothing**: Optional Bayesian-style smoothing (e.g. prior count M) so new books can appear.
- **Output**: Sorted list of (book_id, score) per window stored in Redis (e.g. `popularity:trending:7d`); global mean/std for normalization; optional PyTorch model on S3 for the recommendation service fallback.

## Implementation in this project

- **popularity_train.py**:
  1. **Events**: Ray worker reads from MongoDB `click_stream.events` within each window; computes per-book weighted decayed score; optionally applies smoothing.
  2. **Redis**: Writes sorted sets or list keys (e.g. `popularity:trending:{window}`) and global stats (mean, std) for normalization.
  3. **PyTorch**: Optionally trains a small model (e.g. that takes book id or features and outputs score) and saves to S3 under `POPULARITY_S3_PREFIX` for fallback when Redis is empty.
  4. **Upload**: Model file (e.g. `popularity_latest.pt`) to S3.
- **entrypoint.sh**: Starts Ray head, runs `python popularity_train.py`, stops Ray.
- **Recommendation service**: `popularity_recommendation.py` reads from Redis first; if missing, loads PyTorch model from S3 and computes trending list.

## Environment

`MONGO_URI`, `REDIS_URL`, `S3_URI`, `POPULARITY_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAY_ADDRESS`. Optional: half-life and window weight config.
