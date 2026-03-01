import os
import boto3
import json
from datetime import datetime
import time
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from app_metrics import record_event, start_metrics_writer

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
            # Ping to check the connection
            client.admin.command('ping')
            logging.info("Connected to MongoDB Atlas cluster!")
            return client
        except Exception as e:
            logging.error(f"MongoDB connection failed (attempt {i+1}): {e}")
            time.sleep(5)
    raise RuntimeError("Failed to connect to MongoDB after 5 attempts")

mongo = get_mongo_client()
collection = mongo['click_stream']['events']

def process_msg(msg):
    try:
        data = json.loads(msg['Body'])
        data["received_at"] = datetime.utcnow()
        collection.insert_one(data)
        event_type = data.get("event_type") or data.get("event") or "unknown"
        record_event(event_type)
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
