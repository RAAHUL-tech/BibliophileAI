import os
import math
import logging
import ray
from datetime import datetime, timedelta
from typing import Dict, Any, Iterable, Tuple
from pymongo import MongoClient
import redis
import boto3
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(address=os.getenv("RAY_ADDRESS", "local"))

# Config from env
MONGO_URI = os.environ["MONGO_URI"]
REDIS_URL = os.environ["REDIS_URL"]
S3_URI = os.environ["S3_URI"]
POPULARITY_S3_PREFIX = os.environ.get("POPULARITY_S3_PREFIX", "Popularity_Train")
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}
# Mongo + Redis clients
mongo_client = MongoClient(MONGO_URI)
events_col = mongo_client["click_stream"]["events"]
books_col = mongo_client["metadata"]["books"]
redis_client = redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Configs
HALF_LIVES = {"7d": 7.0, "30d": 30.0, "90d": 90.0}
WINDOW_WEIGHTS = {"7d": 0.5, "30d": 0.3, "90d": 0.2}
WINDOW_DAYS = {"7d": 7, "30d": 30, "90d": 90}
SMOOTHING_M = 10.0

POPULARITY_KEY_PATTERN = "popularity:trending:{window}"

@ray.remote
class PopularityWorker:
    def __init__(self, mongo_uri: str):
        self.mongo_client = MongoClient(mongo_uri)
        self.events_col = self.mongo_client["click_stream"]["events"]
    
    def event_weight(self, event: Dict[str, Any]) -> float:
        etype = event.get("event_type")
        meta = event.get("metadata") or {}
        if etype == "complete": return 10.0
        if etype == "rating": return 5.0 * (float(meta.get("rating", 0.0)) / 5.0)
        if etype == "review": return 7.0
        if etype == "read": return 3.0 * min(float(meta.get("progress", 0.0)) / 100.0, 1.0)
        if etype == "bookmark": return 2.5
        if etype in ("click", "view"): return 1.0
        return 0.0
    
    def compute_window(self, window_days: int, half_life_days: float) -> Tuple[Dict[str, float], Dict[str, int]]:
        now = datetime.utcnow()
        start_time = now - timedelta(days=window_days)
        lam = math.log(2.0) / half_life_days
        
        cursor = self.events_col.find(
            {"item_id": {"$ne": None}, "event_type": {"$in": list(VALID_EVENTS)}, "timestamp": {"$gte": start_time}},
            projection={"item_id": 1, "event_type": 1, "metadata": 1, "timestamp": 1}
        )
        
        scores, counts = {}, {}
        for ev in cursor:
            bid = str(ev["item_id"])
            ts = ev["timestamp"]
            w = self.event_weight(ev)
            if w <= 0: continue
            
            dt_days = (now - ts).total_seconds() / 86400.0
            decay = math.exp(-lam * dt_days)
            contrib = w * decay
            
            scores[bid] = scores.get(bid, 0.0) + contrib
            counts[bid] = counts.get(bid, 0) + 1
        
        return scores, counts

def main():
    logger.info("Starting Popularity Training with Ray...")
    
    # Create parallel workers
    workers = [PopularityWorker.remote(MONGO_URI) for _ in range(3)]
    
    # Compute all windows in parallel
    futures = []
    for label, half_life in HALF_LIVES.items():
        window_days = WINDOW_DAYS[label]
        worker = workers[len(futures) % len(workers)]
        future = worker.compute_window.remote(window_days, half_life)
        futures.append((label, future))
    
    # Wait for all windows
    results = ray.get(futures)
    per_window_scores = {label: result[0] for label, result in results}
    per_window_counts = {label: result[1] for label, result in results}
    
    # Multi-window aggregation
    all_books = set().union(*[s.keys() for s in per_window_scores.values()])
    multi_scores, interaction_counts = {}, {}
    
    for bid in all_books:
        total, total_count = 0.0, 0
        for label, alpha in WINDOW_WEIGHTS.items():
            s = per_window_scores[label].get(bid, 0.0)
            c = per_window_counts[label].get(bid, 0)
            total += alpha * s
            total_count += c
        multi_scores[bid] = total
        interaction_counts[bid] = total_count
    
    # Normalization + smoothing
    #genres = fetch_book_genres(all_books)
    smoothed = normalize_and_smooth(multi_scores, interaction_counts)
    
    # Store to Redis
    store_popularity_in_redis("multi", smoothed)
    for label in HALF_LIVES.keys():
        store_popularity_in_redis(label, per_window_scores[label])
    
    # Save to S3
    save_to_s3(smoothed, interaction_counts)
    
    logger.info(f"Completed: {len(smoothed)} books computed and stored")
    ray.shutdown()

def fetch_book_genres(book_ids: Iterable[str]) -> Dict[str, str]:
    cur = books_col.find(
        {"id": {"$in": [str(bid) for bid in book_ids]}},
        projection={"id": 1, "genre": 1}
    )
    return {str(doc["id"]): doc.get("genre", "unknown") for doc in cur}

def normalize_and_smooth(raw_scores: Dict[str, float], interaction_counts: Dict[str, int]) -> Dict[str, float]:
    genres = fetch_book_genres(raw_scores.keys())
    
    by_genre = {}
    for bid, score in raw_scores.items():
        g = genres.get(bid, "unknown")
        by_genre.setdefault(g, []).append(score)
    
    genre_stats = {}
    for g, vals in by_genre.items():
        if not vals: continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
        sigma = math.sqrt(var) if var > 0 else 1.0
        genre_stats[g] = (mu, sigma)
        redis_client.hset(POPULARITY_GENRE_STATS_KEY, f"{g}:mean", mu)
        redis_client.hset(POPULARITY_GENRE_STATS_KEY, f"{g}:std", sigma)
    
    all_vals = list(raw_scores.values())
    mu_global = sum(all_vals) / len(all_vals) if all_vals else 0.0
    redis_client.set(POPULARITY_GLOBAL_MEAN_KEY, mu_global)
    
    smoothed = {}
    for bid, raw in raw_scores.items():
        g = genres.get(bid, "unknown")
        mu_g, sigma_g = genre_stats.get(g, (mu_global, 1.0))
        z = (raw - mu_g) / sigma_g
        n_b = interaction_counts.get(bid, 0)
        smooth = (n_b * z + SMOOTHING_M * mu_global) / (n_b + SMOOTHING_M)
        smoothed[bid] = smooth
    
    return smoothed

def store_popularity_in_redis(window_label: str, scores: Dict[str, float]):
    key = POPULARITY_KEY_PATTERN.format(window=window_label)
    pipe = redis_client.pipeline()
    pipe.delete(key)
    if scores:
        pipe.zadd(key, {bid: score for bid, score in scores.items()})
    pipe.execute()
    redis_client.expire(key, 3600)

def save_to_s3(smoothed_scores: Dict[str, float], counts: Dict[str, int]):
    def _parse_s3_uri(uri: str):
        if not uri.startswith("s3://"):
            uri = "s3://" + uri.lstrip("/")
        bucket_key = uri[5:]
        bucket, key = bucket_key.split("/", 1)
        return bucket, key
    
    model_uri = f"{S3_URI.rstrip('/')}/{POPULARITY_S3_PREFIX}/popularity_latest.pt"
    bucket, key = _parse_s3_uri(model_uri)
    
    checkpoint = {
        "smoothed_scores": smoothed_scores,
        "interaction_counts": counts,
        "timestamp": datetime.utcnow().isoformat(),
        "half_lives": HALF_LIVES,
        "window_weights": WINDOW_WEIGHTS
    }
    
    import torch
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)
    logger.info(f"Saved popularity model to s3://{bucket}/{key}")

if __name__ == "__main__":
    main()
