#!/bin/bash
set -e
echo "Running LTR (Learning-to-Rank) training..."
python ltr_train.py
echo "LTR training finished."
