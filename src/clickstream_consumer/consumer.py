"""
Clickstream consumer: long-poll SQS for user events, persist to MongoDB, and record
event counts and per-position engagement for the metrics sidecar. Runs alongside the
main app in the consumer pod.
"""
import os
import boto3
import json
from datetime import datetime
from typing import Optional
import time
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from app_metrics import record_event, record_engagement_at_position, start_metrics_writer

# Setup logging
logging.basicConfig(level=logging.INFO)

# SQS (region required by boto3; use AWS_REGION or AWS_DEFAULT_REGION)
_aws_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
try:
    sqs = boto3.client("sqs", region_name=_aws_region)
    QUEUE_URL = os.environ["SQS_QUEUE_URL"]
except KeyError:
    raise RuntimeError("SQS_QUEUE_URL not set in environment")

# MongoDB with retry using ServerApi
def get_mongo_client():
    mongo_uri = os.environ["MONGO_URI"]
    for i in range(5):
        try:
            client = MongoClient(mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            logging.info("Connected to MongoDB Atlas cluster!")
            return client
        except Exception as e:
            logging.error(f"MongoDB connection failed (attempt {i+1}): {e}")
            time.sleep(5)
    raise RuntimeError("Failed to connect to MongoDB after 5 attempts")

mongo = get_mongo_client()
collection = mongo['click_stream']['events']

# Redis: optional — used to look up recommendation positions for NDCG tracking.
# Fails silently if REDIS_URL is not set or Redis is unreachable.
_redis_client = None
try:
    import redis as _redis_lib
    _redis_url = os.environ.get("REDIS_URL")
    if _redis_url:
        _redis_client = _redis_lib.from_url(_redis_url, socket_connect_timeout=2, socket_timeout=2)
        _redis_client.ping()
        logging.info("Connected to Redis for position tracking")
except Exception as _redis_err:
    logging.warning(f"Redis not available for position tracking (non-fatal): {_redis_err}")
    _redis_client = None

# Event types that indicate the user engaged with a recommended book
_ENGAGEMENT_EVENT_TYPES = {"read", "page_turn", "bookmark_add", "review"}


def _lookup_recommendation_position(user_id: str, book_id: str) -> Optional[int]:
    """
    Return the rank (1-indexed) of `book_id` in the user's last recommendation batch,
    or None if the position map is not in Redis or Redis is unavailable.
    """
    if _redis_client is None or not user_id or not book_id:
        return None
    try:
        raw = _redis_client.get(f"positions:{user_id}")
        if not raw:
            return None
        position_map = json.loads(raw)
        return position_map.get(str(book_id))
    except Exception:
        return None


def process_msg(msg):
    try:
        data = json.loads(msg['Body'])
        data["received_at"] = datetime.utcnow()
        collection.insert_one(data)

        event_type = data.get("event_type") or data.get("event") or "unknown"
        record_event(event_type)

        # For engagement events, attribute to recommendation position for NDCG
        if event_type in _ENGAGEMENT_EVENT_TYPES:
            user_id = str(data.get("user_id") or "")
            book_id = str(data.get("item_id") or "")
            position = _lookup_recommendation_position(user_id, book_id)
            if position is not None:
                record_engagement_at_position(int(position))

        logging.info(f"Inserted to mongo: {event_type} for user {data.get('user_id')}")
    except Exception as e:
        logging.error(f"Failed to process message: {e}")


def consume():
    """Poll the queue once using long polling"""
    try:
        res = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=20  # enables long polling
        )
        messages = res.get("Messages", [])
        for msg in messages:
            try:
                process_msg(msg)
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except Exception as e:
                logging.error(f"Error processing message: {e}")
    except Exception as e:
        logging.error(f"Error receiving messages: {e}")

def run_consumer():
    start_metrics_writer()
    logging.info("Starting SQS consumer with long polling...")
    while True:
        consume()
        time.sleep(10)  # small delay between polls (not tight loop)


if __name__ == "__main__":
    run_consumer()
