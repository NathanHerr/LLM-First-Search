#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ../../data/sudoku

# Generate Sudoku puzzles for different difficulties
echo "Generating Sudoku puzzles..."

# Generate 100 puzzles for each difficulty level (easy, medium, hard, expert)
# Using 3x3 grid (standard Sudoku)
python src/generators/sudoku_generator.py \
    --difficulty easy \
    --width 3 \
    --height 3 \
    --number 100 \
    --show-solutions 0 \
    --output ../../data/sudoku/ \
    --save 1

python src/generators/sudoku_generator.py \
    --difficulty medium \
    --width 3 \
    --height 3 \
    --number 100 \
    --show-solutions 0 \
    --output ../../data/sudoku/ \
    --save 1

python src/generators/sudoku_generator.py \
    --difficulty hard \
    --width 3 \
    --height 3 \
    --number 100 \
    --show-solutions 0 \
    --output ../../data/sudoku/ \
    --save 1

python src/generators/sudoku_generator.py \
    --difficulty expert \
    --width 3 \
    --height 3 \
    --number 100 \
    --show-solutions 0 \
    --output ../../data/sudoku/ \
    --save 1

echo "Sudoku data generation complete!" 