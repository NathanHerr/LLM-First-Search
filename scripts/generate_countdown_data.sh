#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ../../data/countdown

# Generate Countdown data
echo "Generating Countdown data..."

# Generate 1000 samples with list length 8 and max target 100
# Split into 80% training and 20% validation
python src/generators/countdown_generate.py \
    --seed 42 \
    --data_dir ../../data/countdown \
    --list_length 8 \
    --max_target 100 \
    --num_samples 1000 \
    --val_ratio 0.2 \
    --check_duplicates

echo "Countdown data generation complete!" 