# src/recommendation_service/popularity_recommendation.py
import os
import redis
import boto3
import torch
import io
from typing import List, Dict, Tuple
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from botocore.exceptions import ClientError

# Redis client
redis_client = redis.from_url(os.environ["REDIS_URL"])
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)

S3_URI = os.environ.get("S3_URI")
POPULARITY_S3_PREFIX = os.environ.get("POPULARITY_S3_PREFIX", "Popularity_Train")

POPULARITY_WINDOWS = ["multi", "7d", "30d", "90d"]
POPULARITY_KEY_PATTERN = "popularity:trending:{window}"

def get_s3_popularity_fallback(window: str, top_k: int) -> Tuple[List[str], List[float]]:
    """Fallback to S3 PyTorch popularity model if Redis is empty"""
    try:
        model_uri = f"{S3_URI.rstrip('/')}/{POPULARITY_S3_PREFIX}/popularity_latest.pt"
        
        # Parse S3 URI
        if not model_uri.startswith("s3://"):
            model_uri = "s3://" + model_uri.lstrip("/")
        bucket_key = model_uri[5:]
        bucket, key = bucket_key.split("/", 1)
        
        # Download PyTorch checkpoint from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(response['Body'].read())
        buf.seek(0)
        
        # Load PyTorch checkpoint
        checkpoint = torch.load(buf, map_location='cpu')
        smoothed_scores: Dict[str, float] = checkpoint["smoothed_scores"]
        
        # Sort by score and take top_k
        sorted_items = sorted(smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k]
        
        book_ids = [book_id for book_id, _ in top_items]
        scores = [float(score) for _, score in top_items]
        print(f"S3 fallback loaded {len(book_ids)} books for window {window} with scores {scores}")
        return book_ids, scores
        
    except (ClientError, FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"S3 fallback failed for {window}: {e}")
        return [], []

async def get_popularity_recommendations(
    user_id: str,
    window: str = "multi",
    top_k: int = 50
) -> Tuple[List[str], List[float]]:
    """
    Get top trending books for user (Redis primary + S3 PyTorch fallback).
    Returns: (book_ids: List[str], scores: List[float])
    """
    if window not in POPULARITY_WINDOWS:
        raise ValueError(f"Invalid window. Choose: {POPULARITY_WINDOWS}")
    
    key = POPULARITY_KEY_PATTERN.format(window=window)
    
    # Try Redis first (primary source)
    try:
        if redis_client.exists(key) and redis_client.zcard(key) > 0:
            books_scores = redis_client.zrevrange(key, 0, top_k - 1, withscores=True)
            if books_scores:
                book_ids = [book_id_bytes.decode("utf-8") for book_id_bytes, _ in books_scores]
                scores = [float(score) for _, score in books_scores]
                print(f"Redis hit: {len(book_ids)} books for window {window} with scores {scores}")
                return book_ids, scores
    except Exception as e:
        print(f"Redis failed: {e}")
    
    # Fallback to S3 if Redis unavailable or empty
    print(f"Falling back to S3 for window {window}")
    return get_s3_popularity_fallback(window, top_k)
