import os
import boto3
import json
from datetime import datetime
import time
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# Setup logging
logging.basicConfig(level=logging.INFO)

# SQS
try:
    sqs = boto3.client("sqs")
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
        logging.info(f"Inserted to mongo: {data.get('event')} for user {data.get('user_id')}")
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
    logging.info("Starting SQS consumer with long polling...")
    while True:
        consume()
        time.sleep(10)  # small delay between polls (not tight loop)


if __name__ == "__main__":
    run_consumer()
