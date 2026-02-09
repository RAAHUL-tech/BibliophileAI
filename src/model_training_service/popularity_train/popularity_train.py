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
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
POPULARITY_GLOBAL_STD_KEY = "popularity:global_std"
POPULARITY_GLOBAL_MEAN_KEY = "popularity:global_mean"

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
            {"item_id": {"$ne": None}, "event_type": {"$in": list(VALID_EVENTS)}, "received_at": {"$gte": start_time}},
            projection={"item_id": 1, "event_type": 1, "metadata": 1, "received_at": 1}
        )
        
        scores, counts = {}, {}
        for ev in cursor:
            bid = str(ev["item_id"])
            ts = ev["received_at"]
            w = self.event_weight(ev)
            if w <= 0: continue
            
            dt_days = (now - ts).total_seconds() / 86400.0
            decay = math.exp(-lam * dt_days)
            contrib = w * decay
            
            scores[bid] = scores.get(bid, 0.0) + contrib
            counts[bid] = counts.get(bid, 0) + 1
        
        return scores, counts

def main():
    # Initialize Ray (with ignore_reinit_error since entrypoint.sh also starts Ray)
    ray.init(address=os.getenv("RAY_ADDRESS", "local"))
    logger.info("Starting Popularity Training with Ray...")
    
    # Create parallel workers
    workers = [PopularityWorker.remote(MONGO_URI) for _ in range(3)]
    
    # Compute all windows in parallel
    futures = []
    labels = []
    for label, half_life in HALF_LIVES.items():
        window_days = WINDOW_DAYS[label]
        worker = workers[len(futures) % len(workers)]
        future = worker.compute_window.remote(window_days, half_life)
        futures.append(future)
        labels.append(label)
    
    # Wait for all windows
    results = ray.get(futures)
    per_window_scores = {label: result[0] for label, result in zip(labels, results)}
    per_window_counts = {label: result[1] for label, result in zip(labels, results)}
    
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
    logging.info("Training workflow completed & uploaded to S3 successfully!")
    # ---- Stop Ray cleanly to prevent Airflow duplicate task run ----
    ray.shutdown()
    logging.info("Ray cluster shut down. Exiting container.")

def normalize_and_smooth(raw_scores: Dict[str, float], interaction_counts: Dict[str, int]) -> Dict[str, float]:
    """Global normalization + interaction-weighted smoothing"""
    
    # Global statistics
    all_vals = list(raw_scores.values())
    if not all_vals:
        return {}
    mu_global = sum(all_vals) / len(all_vals)
    var = sum((v - mu_global) ** 2 for v in all_vals) / max(len(all_vals) - 1, 1)
    sigma_global = math.sqrt(var) if var > 0 else 1.0   
    # Cache global stats in Redis
    redis_client.set(POPULARITY_GLOBAL_MEAN_KEY, mu_global)
    redis_client.set(POPULARITY_GLOBAL_STD_KEY, sigma_global)    
    # Z-score normalization + interaction-weighted smoothing
    smoothed = {}
    for bid, raw in raw_scores.items():
        # Global Z-score
        z = (raw - mu_global) / sigma_global
        
        # Interaction-weighted smoothing (more interactions = less smoothing)
        n_b = interaction_counts.get(bid, 0)
        smooth = (n_b * z + SMOOTHING_M * 0.0) / (n_b + SMOOTHING_M)  # Smooth toward 0
        
        smoothed[bid] = smooth
    return smoothed

def store_popularity_in_redis(window_label: str, scores: Dict[str, float]):
    """
    Write global popularity scores to Redis. Key: popularity:trending:{window}.
    Recommendation service reads this key for all users (no per-user popularity storage).
    """
    key = POPULARITY_KEY_PATTERN.format(window=window_label)
    pipe = redis_client.pipeline()
    pipe.delete(key)
    if scores:
        pipe.zadd(key, {bid: score for bid, score in scores.items()})
    pipe.execute()
    redis_client.expire(key, 365 * 24 * 3600)

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
    
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)
    logger.info(f"Saved popularity model to s3://{bucket}/{key}")

if __name__ == "__main__":
    main()
