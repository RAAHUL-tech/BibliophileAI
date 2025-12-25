#!/bin/bash
set -e

echo "Starting Ray head node..."
ray start --head --port=6379 --disable-usage-stats

sleep 5

echo "Running SASRec training job..."
python sasrec_train.py

echo "Shutting down Ray..."
ray stop
