import os
import ray
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from pymongo import MongoClient
import redis
import boto3
import io
from datetime import datetime
import logging
from helper import get_content_similarity, get_als_score, get_graph_pagerank

logger = logging.getLogger(__name__)

# Config (pure values only - serializable)
MONGO_URI = os.environ["MONGO_URI"]
REDIS_URL = os.environ["REDIS_URL"]
S3_URI = os.environ["S3_URI"]
LINUCB_S3_PREFIX = os.environ.get("LINUCB_S3_PREFIX", "LinUCB_Train")
D = 6  # Context dimension
LAMBDA = 1.0
GAMMA = 0.99

VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}

# Initialize Ray FIRST
ray.init(address=os.getenv("RAY_ADDRESS", "local"))


def get_redis_client():
    """Lazy Redis client per call"""
    return redis.from_url(REDIS_URL)

def get_s3_client():
    """Lazy S3 client per call"""
    return boto3.client("s3", 
                       aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                       aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

@ray.remote(num_cpus=0.25, max_concurrency=4) 
class LinUCBWorker:
    """SERIALIZABLE worker - no external connections"""
    
    def __init__(self, book_id: str, redis_url: str):  
        self.book_id = book_id
        self.redis_url = redis_url  
        self.A = LAMBDA * np.eye(D)
        self.c = np.zeros(D)
    
    def update(self, events: List[Tuple[np.ndarray, float, float]]):
        """Pure math - no external calls"""
        for x, r, decay in events:
            self.A += decay * np.outer(x, x)
            self.c += decay * r * x
        
        try:
            A_inv = np.linalg.inv(self.A)
            return {self.book_id: (A_inv, self.c)}
        except np.linalg.LinAlgError:
            return {}

def build_context_features(book_id: str) -> np.ndarray:
    redis_client = get_redis_client()  # Fresh per call
    
    # Inline mock features (replace with your helpers later)
    pop_multi = redis_client.zscore("popularity:trending:multi", book_id) or 0.0
    pop_7d = redis_client.zscore("popularity:trending:7d", book_id) or 0.0
    pop_30d = redis_client.zscore("popularity:trending:30d", book_id) or 0.0
    
    content_sim = get_content_similarity(book_id)
    als_score = get_als_score(book_id)
    graph_pr = get_graph_pagerank(book_id)
    
    context = np.array([
        min(pop_multi / 10000.0, 1.0),
        min(pop_7d / 5000.0, 1.0),
        min(pop_30d / 2000.0, 1.0),
        content_sim,
        als_score,
        graph_pr
    ])
    return context

def fetch_clickstream_events(days: int = 30) -> pd.DataFrame:
    """Fetch events (unchanged)"""
    mongo_client = MongoClient(MONGO_URI)
    col = mongo_client["click_stream"]["events"]
    
    end_date = datetime.utcnow()
    start_date = end_date - pd.Timedelta(days=days)
    
    pipeline = [
        {"$match": {
            "received_at": {"$gte": start_date, "$lte": end_date},
            "event_type": {"$in": list(VALID_EVENTS)},
            "item_id": {"$exists": True, "$ne": None}
        }},
        {"$sort": {"received_at": 1}},
        {"$group": {
            "_id": "$item_id",
            "events": {"$push": {"event_type": "$event_type", "timestamp": "$received_at"}}
        }}
    ]
    
    events = list(col.aggregate(pipeline))
    mongo_client.close()
    
    processed = []
    for event_group in events:
        book_id = event_group['_id']
        max_reward = max({
            'read': 1.0, 'page_turn': 0.5, 'review': 0.8, 'bookmark_add': 0.1
        }.get(e['event_type'], 0.0) for e in event_group['events'])
        processed.append({'book_id': book_id, 'reward': max_reward})
    
    df = pd.DataFrame(processed)
    logger.info(f"Processed {len(df)} books")
    return df

@ray.remote
def process_book_events(book_id: str, events_df: pd.DataFrame, redis_url: str) -> Dict:
    """Worker task - passes Redis URL as string"""
    worker = LinUCBWorker.remote(book_id, redis_url)
    
    book_events = events_df[events_df['book_id'] == book_id]
    if len(book_events) == 0 or book_events['reward'].iloc[0] == 0.0:
        return {}
    
    training_events = []
    for _, row in book_events.iterrows():
        x = build_context_features(book_id)
        print(f"Processing book {book_id} with context features {x}")
        r = row['reward']
        decay = 1.0  # Simplified
        training_events.append((x, r, decay))
    
    model = ray.get(worker.update.remote(training_events))
    return model

def distributed_train_linucb(events_df: pd.DataFrame) -> Dict:
    """Ray distributed training - passes config only"""
    top_books = events_df[events_df['reward'] > 0]['book_id'].unique()[:5000]  
    logger.info(f"Training {len(top_books)} books")
    
    futures = [
        process_book_events.remote(book_id, events_df, REDIS_URL) 
        for book_id in top_books
    ]
    results = ray.get(futures)
    
    models = {}
    for result in results:
        models.update(result)
    
    return models

def save_to_s3(models: Dict):
    """Save models"""
    s3_client = get_s3_client()
    bucket = S3_URI.replace("s3://", "").split("/")[0]
    key = f"{LINUCB_S3_PREFIX}/linucb_latest.pt"
    
    top_models = dict(list(models.items())[:1000])
    checkpoint = {
        "models": {k: (v[0].tolist(), v[1].tolist()) for k, v in top_models.items()},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    buf.seek(0)
    s3_client.upload_fileobj(buf, bucket, key)
    logger.info(f"Saved {len(top_models)} models")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Fetching clickstream events...")
    events = fetch_clickstream_events(days=90)
    
    print("Distributed LinUCB training...")
    models = distributed_train_linucb(events)
    
    print("Saving to S3...")
    save_to_s3(models)
    
    print(f"Ray LinUCB training complete: {len(models)} models")
    ray.shutdown()
    logging.info("Ray cluster shut down. Exiting container.")
