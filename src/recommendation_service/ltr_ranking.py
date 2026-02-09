"""
Stage 4: Learning-to-Rank (LTR) with XGBoost LambdaRank.
Ranks candidates by predicted engagement using a model trained with rank:ndcg objective.
Expects feature matrix from Feast (29 features); returns sorted (book_ids, scores).
"""
import os
import tempfile
import logging
from typing import Tuple, List, Optional
import numpy as np
import boto3
from io import BytesIO
import xgboost as xgb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 29 features in same order as feature_service / Feast user_book_features view
LTR_FEATURE_NAMES = [
    "retrieval_0", "retrieval_1", "retrieval_2", "retrieval_3", "retrieval_4", "retrieval_5",
    "genre_match_count", "author_match", "language_match",
    "avg_rating", "user_rating", "avg_rating_diff", "rating_count", "user_pref_strength",
    "friend_reads_count", "friend_avg_rating", "author_following", "mutual_likes", "social_proximity",
    "session_position", "session_genre_drift", "time_since_last_action",
    "is_mobile", "is_desktop", "is_tablet", "session_length",
    "global_pop_rank", "trending_score", "intra_list_diversity",
]

# Optional: strip Feast prefix from column names
FEAST_PREFIX = "user_book_features:"


def _load_model():
    """Load XGBoost LTR model from S3 (trained with rank:ndcg). Returns None if unavailable."""
    s3_uri = os.getenv("S3_URI", "").rstrip("/")
    if not s3_uri or "s3://" not in s3_uri:
        bucket = s3_uri or ""
    else:
        bucket = s3_uri.replace("s3://", "").split("/")[0]
    prefix = os.getenv("LTR_S3_PREFIX", "LTR_Train")
    key = f"{prefix}/ltr_xgb_latest.json"
    if not bucket:
        logger.debug("LTR: S3_URI not set")
        return None
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            os.write(fd, body)
            os.close(fd)
            fd = None
            bst = xgb.Booster()
            bst.load_model(path)
            logger.info("LTR: loaded XGBoost model from S3")
            return bst
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
    except Exception as e:
        logger.warning(f"LTR: model load failed: {e}")
        return None


# Lazy-loaded singleton
_ltr_model = None


def _get_model():
    global _ltr_model
    if _ltr_model is None:
        _ltr_model = _load_model()
    return _ltr_model


def _normalize_feature_df(df):
    """Ensure DataFrame has columns matching LTR_FEATURE_NAMES (strip Feast prefix if present)."""
    df = df.copy()
    rename = {}
    for c in df.columns:
        if c.startswith(FEAST_PREFIX):
            rename[c] = c[len(FEAST_PREFIX):]
    if rename:
        df = df.rename(columns=rename)
    return df


def _extract_book_ids_from_user_book(user_id: str, df) -> List[str]:
    """Get book_id from user_book column (format user_id_book_id)."""
    if "user_book" not in df.columns:
        return []
    book_ids = []
    for v in df["user_book"]:
        s = str(v)
        if "_" in s and s.startswith(user_id):
            book_ids.append(s[len(user_id) + 1:].strip())
        elif "_" in s:
            book_ids.append(s.split("_", 1)[1].strip())
        else:
            book_ids.append(s)
    return book_ids


def rank_candidates(
    user_id: str,
    feature_df,
    top_k: int = 100,
) -> Tuple[List[str], List[float]]:
    """
    Rank candidates using XGBoost LTR model (LambdaRank/NDCG).
    feature_df: DataFrame from Feast get_online_features (must include user_book + 29 features).
    Returns (book_ids, scores) sorted by score descending.
    """
    if feature_df is None or feature_df.empty:
        return [], []
    bst = _get_model()
    if bst is None:
        return [], []
    df = _normalize_feature_df(feature_df)
    missing = [f for f in LTR_FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.warning(f"LTR: missing features {missing[:5]}...")
        return [], []
    X = df[LTR_FEATURE_NAMES].fillna(0).astype(np.float32)
    book_ids = _extract_book_ids_from_user_book(user_id, df)
    if len(book_ids) != len(X):
        logger.warning("LTR: book_id count != feature row count")
        return [], []
    try:
        dmat = xgb.DMatrix(X, feature_names=LTR_FEATURE_NAMES)
        scores = bst.predict(dmat)
    except Exception as e:
        logger.warning(f"LTR predict failed: {e}")
        return [], []
    idx = np.argsort(-np.asarray(scores))
    top_idx = idx[:top_k]
    out_ids = [book_ids[i] for i in top_idx]
    out_scores = [float(scores[i]) for i in top_idx]
    print(f"LTR ranking for user {user_id}: {out_ids} with scores {out_scores} with top_k {top_k}")
    return out_ids, out_scores
