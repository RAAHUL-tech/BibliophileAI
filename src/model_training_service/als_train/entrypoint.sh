#!/bin/bash
set -e

echo "Starting Ray head node..."
ray start --head --port=6379 --disable-usage-stats &

# Wait for Ray to fully start
sleep 5

echo "Running ALS Training job..."
python als_train.py

echo "Shutting down Ray..."
ray stop
