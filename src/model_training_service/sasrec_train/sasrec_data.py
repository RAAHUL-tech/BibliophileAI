import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import torch
from torch.utils.data import Dataset


MAX_LEN = 100
# keep only these event types
VALID_EVENTS = {"read", "page_turn", "review", "bookmark_add"}
logging.basicConfig(level=logging.INFO)

def get_mongo_client() -> MongoClient:
    """Create Mongo client with retry and Stable API."""
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


def load_sessions_for_training(
    min_len: int = 2,
    days_back: int = None,
    valid_events: set[str] = VALID_EVENTS,
) -> List[Dict[str, Any]]:
    """
    Build per-session sequences of item_ids from recent events.
    Keeps only specified event types and sessions within `days_back` days.
    """
    # Get days_back from environment variable or use default
    if days_back is None:
        days_back = int(os.getenv("SASREC_DAYS_BACK", "30"))
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    logging.info(f"Loading sessions from MongoDB (last {days_back} days, cutoff: {cutoff})")
    
    # First, check total event count for debugging
    total_events = events_col.count_documents({})
    logging.info(f"Total events in collection: {total_events}")
    
    if total_events == 0:
        logging.warning("No events found in MongoDB collection 'click_stream.events'")
        return []
    
    # Check events matching criteria
    matching_events = events_col.count_documents({
        "item_id": {"$ne": None},
        "event_type": {"$in": list(valid_events)},
        "received_at": {"$gte": cutoff},
    })
    logging.info(f"Events matching criteria (last {days_back} days): {matching_events}")

    pipeline = [
        {
            "$match": {
                "item_id": {"$ne": None},
                "event_type": {"$in": list(valid_events)},
                "received_at": {"$gte": cutoff},
            }
        },
        {"$sort": {"session_id": 1, "timestamp": 1}},
        {
            "$group": {
                "_id": "$session_id",
                "user_id": {"$first": "$user_id"},
                "last_timestamp": {"$max": "$received_at"},
                "events": {
                    "$push": {
                        "item_id": "$item_id",
                        "event_type": "$event_type",
                        "timestamp": "$timestamp",
                    }
                },
            }
        },
        {"$match": {"last_timestamp": {"$gte": cutoff}}},
    ]

    sessions: List[Dict[str, Any]] = []
    session_count = 0
    for doc in events_col.aggregate(pipeline, allowDiskUse=True):
        session_count += 1
        items = [e["item_id"] for e in doc["events"] if e.get("item_id")]
        if len(items) >= min_len:
            sessions.append(
                {"session_id": doc["_id"], "user_id": doc["user_id"], "items": items}
            )
    
    logging.info(f"Found {session_count} sessions, {len(sessions)} with >= {min_len} items")
    return sessions


def build_vocab(sessions: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    all_items = set()
    for s in sessions:
        all_items.update(s["items"])
    item2id = {item: i + 1 for i, item in enumerate(sorted(all_items))}
    id2item = {i: item for item, i in item2id.items()}
    return item2id, id2item


class SASRecDataset(Dataset):
    def __init__(self, sessions, item2id, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.item2id = item2id
        self.sequences = []

        for s in sessions:
            seq = [item2id[x] for x in s["items"] if x in item2id]
            if len(seq) < 2:
                continue
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = seq[-self.max_len:]

        input_seq = seq[:-1]
        target_seq = seq[1:]

        pad_len = self.max_len - len(input_seq)
        if pad_len < 0:
            input_seq = input_seq[-self.max_len:]
            target_seq = target_seq[-self.max_len:]
            pad_len = 0

        input_padded = [0] * pad_len + input_seq
        target_padded = [0] * pad_len + target_seq

        return torch.LongTensor(input_padded), torch.LongTensor(target_padded)
