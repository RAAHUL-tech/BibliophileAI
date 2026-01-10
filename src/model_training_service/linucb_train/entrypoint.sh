#!/bin/bash
set -e

echo "Starting Ray head node for popularity training..."
ray start --head --port=6379 --disable-usage-stats

# Wait for Ray to fully start
sleep 5

echo "Running LinUCB Training job..."
python linucb_train.py

echo "Shutting down Ray..."
ray stop
