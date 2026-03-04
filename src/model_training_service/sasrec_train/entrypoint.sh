#!/bin/bash
# Entrypoint for SASRec training container: start Ray head, run sasrec_train.py, then stop Ray.
set -e

echo "Starting Ray head node..."
ray start --head --port=6379 --disable-usage-stats

sleep 5

echo "Running SASRec training job..."
python sasrec_train.py

echo "Shutting down Ray..."
ray stop
