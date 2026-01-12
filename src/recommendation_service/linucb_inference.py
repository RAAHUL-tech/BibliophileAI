import os
import torch
import numpy as np
import boto3
from io import BytesIO
from typing import List, Tuple, Dict, Any
from pymongo import MongoClient
from datetime import datetime, timedelta
import redis
from linucb_helper import get_content_similarity, get_als_score, get_graph_pagerank

VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}


class LinUCBServe:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        self.mongo_client = MongoClient(os.environ["MONGO_URI"])
        self.redis_client = redis.from_url(os.environ["REDIS_URL"])
        self.models = self._load_models()
        self.alpha = 1.2
    
    def _load_models(self) -> dict:
        """Load trained LinUCB models from S3"""
        bucket = os.environ["S3_URI"].replace("s3://", "").split("/")[0]
        key = f"{os.environ['LINUCB_S3_PREFIX']}/linucb_latest.pt"
        
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            checkpoint = torch.load(BytesIO(obj["Body"].read()))
            
            numpy_models = {}
            for book_id, (A_inv, b) in checkpoint["models"].items():
                numpy_models[book_id] = (np.array(A_inv), np.array(b))
            
            print(f"Loaded {len(numpy_models)} LinUCB models")
            return numpy_models
        except Exception as e:
            print(f"Model load failed: {e}")
            return {}
    
    def get_recently_read_books(self, user_id: str, days: int = 30) -> List[str]:
        """Get user's recently read books from MongoDB clickstream"""
        col = self.mongo_client["click_stream"]["events"]
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        pipeline = [
            {"$match": {
                "user_id": user_id,
                "event_type": {"$in": list(VALID_EVENTS)},
                "received_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": "$item_id",
                "last_read": {"$max": "$received_at"}
            }},
            {"$sort": {"last_read": -1}},
            {"$limit": 50},
            {"$project": {"_id": 1}}
        ]
        
        books = [doc["_id"] for doc in col.aggregate(pipeline)]
        print(f"Found {len(books)} recent reads for user {user_id}")
        return books

    def build_context_features(self, book_id: str) -> np.ndarray:
        # Inline mock features (replace with your helpers later)
        pop_multi = self.redis_client.zscore("popularity:trending:multi", book_id) or 0.0
        pop_7d = self.redis_client.zscore("popularity:trending:7d", book_id) or 0.0
        pop_30d = self.redis_client.zscore("popularity:trending:30d", book_id) or 0.0
        
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
    
    def score_candidate_books(self, book_ids: List[str], user_id: str = None) -> List[Tuple[str, float]]:
        """
        Score ALL candidate books with LinUCB + recent read bonus
        Returns: [(book_id, linucb_score), ...] sorted descending
        """
        scores = []
        recent_reads = set(self.get_recently_read_books(user_id)) if user_id else set()
        
        for book_id in book_ids:
            # 1. D=6 context features
            context = self.build_context_features(book_id)
            
            # 2. LinUCB score
            linucb_score = self._linucb_score(book_id, context)
            
            # 3. Recent read bonus (+20% if user read recently)
            read_bonus = 0.2 if book_id in recent_reads else 0.0
            
            # 4. Final blended score
            final_score = linucb_score * (1.0 + read_bonus)
            scores.append((book_id, float(final_score), context.tolist()))
        
        # Sort by final score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:200]  # Top 200 max
    
    def _linucb_score(self, book_id: str, context: np.ndarray) -> float:
        """LinUCB confidence-weighted score"""
        if book_id not in self.models:
            return 0.5  # Baseline
        
        A_inv, b = self.models[book_id]
        theta = A_inv @ b
        
        # Exploration bonus
        cb = self.alpha * np.sqrt(context @ A_inv @ context.T)
        mean_score = context @ theta
        
        return float(mean_score + cb)
    
    def get_linucb_ranked(self, book_ids: List[str], user_id: str = None) -> Tuple[List[str], List[float]]:
        """Full ranking with metadata"""
        scored = self.score_candidate_books(book_ids, user_id)
        
        book_ids = []
        scores = []
        for book_id, score, context in scored:
            book_ids.append(book_id)
            scores.append(score)
        
        print(f"LinUCB ranked {len(book_ids)} books for user {user_id} with scores {scores}")
        return book_ids, scores

# Global singleton
linucb_ranker = LinUCBServe()
