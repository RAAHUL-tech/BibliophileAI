import os
from typing import List, Tuple
import pandas as pd
import numpy as np
import asyncio
import boto3
from io import BytesIO

S3_BUCKET = os.getenv("S3_URI")
ALS_PREFIX = os.getenv("ALS_S3_PREFIX")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def _download_parquet(bucket: str, s3_key: str):
    s3_obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return pd.read_parquet(BytesIO(s3_obj["Body"].read()))

async def recommend_als_books(user_id: str, top_k: int = 50) -> Tuple[List[str], List[float]]:
    loop = asyncio.get_running_loop()
    user_factors, book_factors = await asyncio.gather(
        loop.run_in_executor(None, _download_parquet, S3_BUCKET, f"{ALS_PREFIX}/user_factors.parquet"),
        loop.run_in_executor(None, _download_parquet, S3_BUCKET, f"{ALS_PREFIX}/book_factors.parquet"),
    )

    user_row = user_factors[user_factors["user_id"] == user_id]
    if user_row.empty:
        return [], []

    user_vec = user_row.drop(columns=["user_id"]).to_numpy().flatten()
    book_vecs = book_factors.drop(columns=["book_id"]).to_numpy()
    book_ids = book_factors["book_id"].values

    # Compute relevance scores and select top_k
    scores = book_vecs @ user_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommended_books = book_ids[top_indices]
    recommended_scores = scores[top_indices]
    print(f"ALS Recommendations for user {user_id}: {recommended_books} with scores {recommended_scores}")
    return recommended_books.tolist(), recommended_scores.tolist()
