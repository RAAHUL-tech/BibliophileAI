#!/bin/bash
set -e

echo "Starting Ray head node..."
ray start --head --port=6379 --disable-usage-stats

# Wait for Ray to fully start
sleep 5

echo "Running LTR (Learning-to-Rank) training job..."
python ltr_train.py

echo "Shutting down Ray..."
ray stop

echo "LTR training finished."
