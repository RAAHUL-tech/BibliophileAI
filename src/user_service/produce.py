# Clickstream producer using AWS SQS FIFO queue
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
