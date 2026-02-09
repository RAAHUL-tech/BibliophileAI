import os
import sys
import io
import logging
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from pymongo import MongoClient
import boto3
from feast import FeatureStore
import xgboost as xgb
import ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI")
S3_URI = os.environ.get("S3_URI", "").rstrip("/")
LTR_S3_PREFIX = os.environ.get("LTR_S3_PREFIX", "LTR_Train")
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "")
FEAST_S3_BUCKET = os.environ.get("FEAST_S3_BUCKET", "bibliophile-ai-feast")
EVENT_DAYS = int(os.environ.get("LTR_EVENT_DAYS", "90"))

# Relevance: 5=completed+rated>=4, 4=completed, 3=read>=50%, 2=read>=10%, 1=click, 0=ignore
VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}

FEATURE_COLS = [
    "retrieval_0", "retrieval_1", "retrieval_2", "retrieval_3", "retrieval_4", "retrieval_5",
    "genre_match_count", "author_match", "language_match",
    "avg_rating", "user_rating", "avg_rating_diff", "rating_count", "user_pref_strength",
    "friend_reads_count", "friend_avg_rating", "author_following", "mutual_likes", "social_proximity",
    "session_position", "session_genre_drift", "time_since_last_action",
    "is_mobile", "is_desktop", "is_tablet", "session_length",
    "global_pop_rank", "trending_score", "intra_list_diversity",
]


@ray.remote
def load_labels() -> pd.DataFrame:
    """
    Aggregate (user_id, item_id) -> relevance 0-5 from click_stream.events.
    Returns DataFrame with columns: user_id, book_id, user_book, relevance, timestamp.
    """
    client = MongoClient(MONGO_URI)
    col = client["click_stream"]["events"]
    end = datetime.utcnow()
    start = end - timedelta(days=EVENT_DAYS)
    pipeline = [
        {"$match": {
            "event_type": {"$in": list(VALID_EVENTS)},
            "received_at": {"$gte": start, "$lte": end},
        }},
        {"$group": {
            "_id": {"user_id": "$user_id", "item_id": "$item_id"},
            "events": {"$push": {"type": "$event_type", "at": "$received_at", "meta": "$metadata"}},
            "last_at": {"$max": "$received_at"},
        }},
    ]
    rows = []
    for doc in col.aggregate(pipeline):
        uid = doc["_id"]["user_id"]
        bid = doc["_id"]["item_id"]
        events = doc.get("events", [])
        last_at = doc.get("last_at", end)
        rel = 0
        for e in events:
            t = e.get("type")
            meta = e.get("meta") or {}
            if t == "review":
                rating = meta.get("rating") or 0
                if rating >= 4:
                    rel = max(rel, 5)
                else:
                    rel = max(rel, 4)
            elif t == "read":
                rel = max(rel, 4)
            elif t == "page_turn":
                count = sum(1 for x in events if (x.get("type")) == "page_turn")
                rel = max(rel, 3 if count >= 5 else 2)
            elif t == "bookmark_add":
                rel = max(rel, 1)
        rows.append({
            "user_id": uid,
            "book_id": str(bid),
            "user_book": f"{uid}_{bid}",
            "relevance": rel,
            "timestamp": last_at,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@ray.remote
def load_features_direct_s3(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Read feature parquet directly from S3; returns merged feature DataFrame for entity_df."""
    bucket = FEAST_S3_BUCKET
    prefix = "features/"
    needed = ["user_book", "user_id", "book_id", "query_id"] + FEATURE_COLS
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        paginator = s3.get_paginator("list_objects_v2")
        frames = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith("user_book_features.parquet"):
                    continue
                buf = io.BytesIO()
                s3.download_fileobj(bucket, key, buf)
                buf.seek(0)
                df = pd.read_parquet(buf)
                if df.empty:
                    continue
                if "timestamp" in df.columns:
                    df = df.dropna(subset=["timestamp"])
                if df.empty:
                    continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates(subset=["user_book"], keep="last")
        want = [c for c in needed if c in combined.columns]
        if len(want) < len(needed):
            missing = set(needed) - set(combined.columns)
            logging.warning("Direct S3 parquet missing columns: %s", missing)
        out = entity_df[["user_book"]].merge(combined, on="user_book", how="inner")
        logging.info("Direct S3 read: %d parquet file(s), %d rows after merge", len(frames), len(out))
        return out
    except Exception as e:
        logging.warning("Direct S3 feature read failed: %s", e)
        return pd.DataFrame()


@ray.remote
def load_features_feast(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Load user-book features from Feast offline store (S3 Parquet)."""
    if not os.path.isdir(FEAST_REPO_PATH):
        return pd.DataFrame()
    entity_df = entity_df[["user_book", "timestamp"]].drop_duplicates()
    if not pd.api.types.is_datetime64_any_dtype(entity_df["timestamp"]):
        entity_df["timestamp"] = pd.to_datetime(entity_df["timestamp"], utc=True)
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    feature_names = [
        "user_book_features:query_id", "user_book_features:user_id", "user_book_features:book_id",
        "user_book_features:retrieval_0", "user_book_features:retrieval_1", "user_book_features:retrieval_2",
        "user_book_features:retrieval_3", "user_book_features:retrieval_4", "user_book_features:retrieval_5",
        "user_book_features:genre_match_count", "user_book_features:author_match", "user_book_features:language_match",
        "user_book_features:avg_rating", "user_book_features:user_rating", "user_book_features:avg_rating_diff",
        "user_book_features:rating_count", "user_book_features:user_pref_strength",
        "user_book_features:friend_reads_count", "user_book_features:friend_avg_rating",
        "user_book_features:author_following", "user_book_features:mutual_likes", "user_book_features:social_proximity",
        "user_book_features:session_position", "user_book_features:session_genre_drift",
        "user_book_features:time_since_last_action", "user_book_features:is_mobile", "user_book_features:is_desktop",
        "user_book_features:is_tablet", "user_book_features:session_length", "user_book_features:global_pop_rank",
        "user_book_features:trending_score", "user_book_features:intra_list_diversity",
    ]
    try:
        df = store.get_historical_features(entity_df=entity_df, features=feature_names).to_df()
        return df
    except Exception as e:
        logging.warning("Feast get_historical_features failed: %s", e)
        return pd.DataFrame()


@ray.remote
def train_and_upload_s3(train_df: pd.DataFrame) -> None:
    """Train XGBoost rank:ndcg and upload model to S3."""
    prefix = LTR_S3_PREFIX
    bucket = S3_URI.replace("s3://", "").split("/")[0] if S3_URI.startswith("s3://") else S3_URI.split("/")[0]
    if not bucket or train_df is None or train_df.empty:
        logging.warning("No training data or S3_URI")
        return
    missing = [f for f in FEATURE_COLS if f not in train_df.columns]
    if missing:
        logging.warning("Missing features: %s", missing[:5])
        return
    train_df = train_df.sort_values("user_id").reset_index(drop=True)
    uids = train_df["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(uids)
    n = max(1, int(0.8 * len(uids)))
    train_uids, val_uids = set(uids[:n]), set(uids[n:])
    tr = train_df[train_df["user_id"].isin(train_uids)]
    va = train_df[train_df["user_id"].isin(val_uids)]
    if va.empty:
        va = tr
    X_tr = tr[FEATURE_COLS].fillna(0).astype(np.float32)
    y_tr = tr["relevance"].astype(np.int32)
    grp_tr = tr.groupby("user_id", sort=False).size().values.astype(np.uint32)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
    dtrain.set_group(grp_tr)
    X_va = va[FEATURE_COLS].fillna(0).astype(np.float32)
    y_va = va["relevance"].astype(np.int32)
    grp_va = va.groupby("user_id", sort=False).size().values.astype(np.uint32)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURE_COLS)
    dval.set_group(grp_va)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "lambda": 1.0,
        "tree_method": "hist",
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    key = f"{prefix}/ltr_xgb_latest.json"
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        os.close(fd)
        bst.save_model(tmp_path)
        with open(tmp_path, "rb") as f:
            body = f.read()
        client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logging.info("Saved LTR model to s3://%s/%s", bucket, key)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    ray.init(address=os.getenv("RAY_ADDRESS", "local"))

    logger.info("Building relevance labels from MongoDB (Ray)...")
    labels_df = ray.get(load_labels.remote())
    if labels_df.empty or len(labels_df) < 1:
        logger.warning("Insufficient label data")
        ray.shutdown()
        return

    entity_df = labels_df[["user_book", "timestamp"]].copy()
    entity_df["timestamp"] = pd.to_datetime(entity_df["timestamp"], utc=True)

    logger.info("Loading historical features from S3 (Ray)...")
    features_df = ray.get(load_features_direct_s3.remote(entity_df))
    if features_df.empty:
        logger.info("Direct S3 empty; falling back to Feast offline store (Ray)...")
        features_df = ray.get(load_features_feast.remote(entity_df))
    if features_df.empty:
        logger.warning(
            "No features from direct S3 or Feast; ensure parquet files exist under s3://%s/features/",
            FEAST_S3_BUCKET,
        )
        ray.shutdown()
        return

    rename = {c: c.replace("user_book_features:", "") for c in features_df.columns if c.startswith("user_book_features:")}
    features_df = features_df.rename(columns=rename)
    features_df = features_df.dropna()
    if "user_book" not in features_df.columns:
        logger.warning("Feast output missing user_book column")
        ray.shutdown()
        return

    train_df = features_df.merge(
        labels_df[["user_book", "user_id", "relevance"]],
        on="user_book",
        how="inner",
        suffixes=("", "_label"),
    )
    if "user_id_label" in train_df.columns:
        train_df = train_df.drop(columns=["user_id_label"])
    if "user_id" not in train_df.columns:
        train_df["user_id"] = train_df["user_book"].str.split("_", n=1).str[0]
    if train_df.empty or "relevance" not in train_df.columns:
        logger.warning("No merged training data")
        ray.shutdown()
        return

    logger.info("Training XGBoost LTR and uploading to S3 (Ray)...")
    ray.get(train_and_upload_s3.remote(train_df))
    logger.info("LTR training done.")

    ray.shutdown()
    logger.info("Ray cluster shut down. Exiting.")


if __name__ == "__main__":
    main()
