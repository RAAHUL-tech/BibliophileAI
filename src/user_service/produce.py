"""
Clickstream producer: sends user events (e.g. page_turn, bookmark) to an AWS SQS FIFO
queue for the clickstream consumer to persist and process.
"""
import os
import boto3
import json

sqs = boto3.client("sqs")
QUEUE_URL = os.environ["SQS_QUEUE_URL"]

def send_click_event(event: dict, group_id: str):
    resp = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(event),
        MessageGroupId=group_id
    )
    return resp["MessageId"]
