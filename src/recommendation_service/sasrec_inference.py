import os
import io
import time
import logging
from typing import List, Set, Tuple

import torch
import boto3
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from sasrec_model import SASRec

# ---------------- Mongo client with Stable API ----------------

def get_mongo_client() -> MongoClient:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is not set")

    for i in range(5):
        try:
            client = MongoClient(
                mongo_uri,
                server_api=ServerApi("1"),
                serverSelectionTimeoutMS=5000,
            )
            client.admin.command("ping")
            logging.info("Connected to MongoDB Atlas cluster!")
            return client
        except Exception as e:
            logging.error(f"MongoDB connection failed (attempt {i + 1}): {e}")
            time.sleep(5)

    raise RuntimeError("Failed to connect to MongoDB after 5 attempts")


mongo = get_mongo_client()
events_col = mongo["click_stream"]["events"]

VALID_EVENTS: Set[str] = {"read", "page_turn", "review", "bookmark_add"}
MAX_LEN = 100
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_item2id = None
_id2item = None

# ---------------- S3 / model loading ----------------

S3_URI = os.getenv("S3_URI", "")
SASREC_PREFIX = os.getenv("SASREC_S3_PREFIX", "SASRec_Train")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def _parse_s3_uri(uri: str):
    """
    Accepts either 's3://bucket/path' or 'bucket/path' and
    returns (bucket, key).
    """
    if not uri.startswith("s3://"):
        uri = "s3://" + uri.lstrip("/")

    bucket_key = uri[5:]
    bucket, key = bucket_key.split("/", 1)
    return bucket, key


def _load_model_if_needed():
    global _model, _item2id, _id2item, MAX_LEN
    if _model is not None:
        return

    model_uri = f"{S3_URI.rstrip('/')}/{SASREC_PREFIX.strip('/')}/sasrec_latest.pt"
    bucket, key = _parse_s3_uri(model_uri)
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)

    ckpt = torch.load(buf, map_location=_device)
    _item2id = ckpt["item2id"]
    _id2item = ckpt["id2item"]
    MAX_LEN = ckpt.get("max_len", 100)

    num_items = len(_item2id)
    _model = SASRec(num_items=num_items, max_len=MAX_LEN).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    logging.info("SASRec model loaded with %d items, max_len=%d", num_items, MAX_LEN)

# ---------------- helpers ----------------

def _get_user_items(user_id: str, history_len: int = 20) -> List[str]:
    """
    Return up to `history_len` most recent item_ids for this user
    across all sessions, restricted to VALID_EVENTS, oldest → newest.
    """
    cursor = events_col.find(
        {
            "user_id": user_id,
            "item_id": {"$ne": None},
            "event_type": {"$in": list(VALID_EVENTS)},
        },
        sort=[("received_at", -1)],  # newest first
        limit=history_len,
    )
    items = [doc["item_id"] for doc in cursor if doc.get("item_id")]
    return list(reversed(items))  


# ---------------- main recommendation API ----------------

def recommend_for_session(user_id: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
    _load_model_if_needed()

    items = _get_user_items(user_id)
    print(f"[SASRec] user_id={user_id}, items={items}")
    if len(items) < 2:
        return []

    seq_ids = [_item2id[i] for i in items if i in _item2id]
    if len(seq_ids) < 2:
        return []

    seq_ids = seq_ids[-MAX_LEN:]
    pad_len = MAX_LEN - len(seq_ids)
    input_seq = [0] * pad_len + seq_ids

    input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model.predict_next(input_tensor)[0]
        logits[0] = -1e9
        for idx in seq_ids:
            logits[idx] = -1e9

        k = min(top_k, logits.size(0) - 1)
        if k <= 0:
            return []
        topk = torch.topk(logits, k=k)
        indices = topk.indices.tolist()
        logits = topk.values  # shape: [K]
        min_v = logits.min()
        max_v = logits.max()
        if max_v > min_v:
            scores = (logits - min_v) / (max_v - min_v)
        else:
            scores = torch.zeros_like(logits)
        scores = scores.tolist()
        recs = [_id2item[i] for i in indices if i in _id2item]
        print(f"4. [SASRec] recs for user={user_id}: {recs} with scores {scores}")
        return recs, scores
