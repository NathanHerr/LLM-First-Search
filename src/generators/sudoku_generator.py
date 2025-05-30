#!/usr/bin/env python3
"""
Sudoku Generator Script

This script generates Sudoku puzzles with varying difficulty levels using the py-sudoku library.
It can generate and save multiple puzzles, and allows customization of difficulty.
It can also plot the distribution of puzzle difficulties.
"""

import argparse
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sudoku import Sudoku


def check_solution_uniqueness(sudoku, width, height):
    """
    Check if a Sudoku puzzle has a unique solution using a more robust approach.
    
    Args:
        sudoku (Sudoku): The Sudoku puzzle to check
        width (int): Width of each box in the Sudoku grid
        height (int): Height of each box in the Sudoku grid
        
    Returns:
        bool: True if the puzzle has a unique solution, False otherwise
    """
    # First, find any solution
    first_solution = sudoku.solve()
    if not first_solution:
        return False  # No solution exists
        
    # Create a copy of the board to work with
    board = [row[:] for row in sudoku.board]
    
    # Find all blank cells
    blanks = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] is None:
                blanks.append((i, j))
    
    # Try to find a different solution
    def find_alternative_solution(board, blanks, index=0):
        if index == len(blanks):
            # Found a complete solution
            return board
            
        i, j = blanks[index]
        possible_values = set(range(1, len(board) + 1))
        
        # Remove values already in row, column, and box
        for x in range(len(board)):
            if board[i][x] is not None:
                possible_values.discard(board[i][x])  # row
            if board[x][j] is not None:
                possible_values.discard(board[x][j])  # column
            
        # Remove values in the same box
        box_i, box_j = (i // height) * height, (j // width) * width
        for x in range(height):
            for y in range(width):
                if board[box_i + x][box_j + y] is not None:
                    possible_values.discard(board[box_i + x][box_j + y])
        
        # Try each possible value
        for value in possible_values:
            board[i][j] = value
            solution = find_alternative_solution(board, blanks, index + 1)
            if solution:
                # Check if this solution is different from the first one
                is_different = False
                for x in range(len(board)):
                    for y in range(len(board[0])):
                        if solution[x][y] != first_solution.board[x][y]:
                            is_different = True
                            break
                    if is_different:
                        break
                if is_different:
                    return solution
            board[i][j] = None
            
        return None
    
    # Try to find an alternative solution
    alternative_solution = find_alternative_solution(board, blanks)
    return alternative_solution is None


def generate_complete_board(width=3, height=None):
    """
    Generate a complete, valid Sudoku board using the Sudoku package's built-in solver.
    
    Args:
        width (int): Width of each box in the Sudoku grid
        height (int): Height of each box in the Sudoku grid
        
    Returns:
        list: A complete, valid Sudoku board
    """
    if height is None:
        height = width
    
    # Create a Sudoku puzzle and solve it to get a complete board
    sudoku = Sudoku(width, height)
    solution = sudoku.solve()
    return solution.board


def remove_numbers(board, difficulty_value, width, height):
    """
    Remove numbers from a complete board while maintaining a unique solution.
    
    Args:
        board (list): Complete Sudoku board
        difficulty_value (float): Target difficulty (0.0-1.0)
        width (int): Width of each box in the Sudoku grid
        height (int): Height of each box in the Sudoku grid
        
    Returns:
        list: Board with some numbers removed
    """
    size = len(board)
    cells = [(i, j) for i in range(size) for j in range(size)]
    random.shuffle(cells)
    
    # Calculate how many cells to remove based on difficulty
    cells_to_remove = int(difficulty_value * size * size)
    
    # Create a copy of the board to work with
    puzzle = [row[:] for row in board]
    
    # Try to remove numbers while maintaining uniqueness
    removed = 0
    for i, j in cells:
        if removed >= cells_to_remove:
            break
            
        # Store the value we're trying to remove
        temp = puzzle[i][j]
        puzzle[i][j] = None
        
        # Check if the puzzle still has a unique solution
        sudoku = Sudoku(width, height, board=puzzle)
        if check_solution_uniqueness(sudoku, width, height):
            removed += 1
        else:
            # If removing this number creates multiple solutions, put it back
            puzzle[i][j] = temp
    
    return puzzle


def generate_sudoku(difficulty="medium", width=3, height=None):
    """
    Generate a Sudoku puzzle with the specified difficulty and dimensions.
    Ensures the puzzle has a unique solution.
    
    Args:
        difficulty (str): Difficulty level: "easy", "medium", "hard", or "expert"
        width (int): Width of each box in the Sudoku grid (default: 3 for standard 9x9 puzzle)
        height (int): Height of each box in the Sudoku grid (default: same as width)
        
    Returns:
        Sudoku: A generated Sudoku puzzle with appropriate difficulty and dimensions and unique solution
    """
    # Map difficulty levels to py-sudoku difficulty values (0.0-1.0)
    difficulty_map = {
        "easy": 0.3,
        "medium": 0.5,
        "hard": 0.7,
        "expert": 0.9
    }

    # Get the difficulty value (default to medium if not found)
    difficulty_value = difficulty_map.get(difficulty.lower(), 0.5)
    
    # Use width for height if height is not specified
    if height is None:
        height = width
    
    start_time = time.time()
    
    # Generate a complete board
    complete_board = generate_complete_board(width, height)
    
    # Remove numbers to create the puzzle
    puzzle_board = remove_numbers(complete_board, difficulty_value, width, height)
    
    # Create the Sudoku object
    sudoku = Sudoku(width, height, board=puzzle_board)
    
    generation_time = time.time() - start_time
    grid_width = width * height
    grid_height = height * width
    
    print(f"Generated a {difficulty} {grid_width}x{grid_height} puzzle with unique solution in {generation_time:.2f} seconds")
    return sudoku


def save_sudoku_to_file(sudoku, filename, include_solution=False):
    """
    Save a Sudoku puzzle to a text file.
    
    Args:
        sudoku (Sudoku): The Sudoku puzzle to save
        filename (str): The filename to save to
        include_solution (bool): Whether to include the solution in the file
    """
    with open(filename, 'w') as f:
        # Write the puzzle
        f.write("SUDOKU PUZZLE:\n")
        
        # Convert board to string format
        board_str = ""
        for row in sudoku.board:
            for cell in row:
                board_str += str(cell) if cell != 0 else "."
        
        f.write(board_str + "\n\n")
        
        # Write a more readable version
        f.write("READABLE FORMAT:\n")
        for row in sudoku.board:
            f.write(" ".join(str(cell) if cell != 0 else "." for cell in row) + "\n")
        
        # Write solution if requested
        if include_solution:
            solution = sudoku.solve()
            f.write("\nSOLUTION:\n")
            for row in solution.board:
                f.write(" ".join(str(cell) for cell in row) + "\n")


def save_sudokus_to_pickle(sudokus, solutions, difficulty_name, width, height, output_dir):
    """
    Save all generated Sudoku puzzles and their solutions to a pickle file.
    
    Args:
        sudokus (list): List of Sudoku puzzles
        solutions (list): List of corresponding solutions
        difficulty_name (str): Name of the difficulty level used for generating puzzles
        output_dir (str): Directory to save the pickle file
        width (int): Width of each box in the Sudoku grid
        height (int): Height of each box in the Sudoku grid
        
    Returns:
        str: Path to the saved pickle file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data dictionary with puzzles and solutions
    data = {
        'difficulty_name': difficulty_name,
        'puzzles': sudokus,
        'solutions': solutions,
        'count': len(sudokus),
        'box_width': width,
        'box_height': height,
        'grid_width': width * height,
        'grid_height': height * width
    }
    
    # Create filename with timestamp
    filename = f"sudoku_diff_{difficulty_name}_w_{width}_h_{height}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    return filepath


def display_sudoku(sudoku, show_solution=False):
    """
    Display a Sudoku puzzle to the console.
    
    Args:
        sudoku (Sudoku): The Sudoku puzzle to display
        show_solution (bool): Whether to also show the solution
    """
    print("\nGenerated Sudoku:")
    sudoku.show()
    
    if show_solution:
        solution = sudoku.solve()
        print("\nSolution:")
        solution.show()


def main():
    """Main function to parse arguments and generate Sudoku puzzles."""
    parser = argparse.ArgumentParser(description="Generate Sudoku puzzles with varying difficulty levels.")
    parser.add_argument('-d', '--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard', 'expert'],
                        help='Difficulty level of the Sudoku puzzle')
    parser.add_argument('-w', '--width', type=int, default=2,
                        help='Width of each box in the Sudoku grid (default: 3 for standard 9x9 puzzle)')
    parser.add_argument('--height', type=int, default=2,
                        help='Height of each box in the Sudoku grid (default: same as width)')
    parser.add_argument('-n', '--number', type=int, default=60,
                        help='Number of puzzles to generate')
    parser.add_argument('--show-solutions', type=int, default=0, choices=[0, 1],
                        help='Display solutions for puzzles (0=off, 1=on)')
    parser.add_argument('-o', '--output', type=str, default='../../data/sudoku/unique/',
                        help='Output directory for saving puzzles and plots')
    parser.add_argument('-s', '--save', type=int, default=1, choices=[0, 1],
                        help='Save puzzles to a pickle file (0=off, 1=on)')
    
    args = parser.parse_args()
    
    # Set height to width if not specified
    height = args.height if args.height is not None else args.width
    
    # Create output directory if needed
    output_dir = args.output

    for difficulty in ['easy', 'medium', 'hard', 'expert']:
        # Initialize lists to store puzzles and solutions
        puzzles = []
        solutions = []

        # Generate the requested number of puzzles
        for i in range(args.number):
            print(f"Generating {difficulty} Sudoku puzzle #{i+1}...")
            # Generate the puzzle with appropriate difficulty and dimensions
            sudoku = generate_sudoku(difficulty, width=args.width, height=height)

            # Generate the solution
            solution = sudoku.solve()

            # Store the puzzle and solution
            puzzles.append(sudoku)
            solutions.append(solution)
            print(f"Puzzle #{i+1} generated")

            # Display the puzzle and solution if requested
            if args.show_solutions == 1:
                display_sudoku(sudoku, show_solution=True)

        # Save all puzzles and solutions to a pickle file if requested
        if args.save == 1:
            pickle_path = save_sudokus_to_pickle(puzzles, solutions, difficulty, args.width, height, output_dir)
            print(f"Saved {len(puzzles)} puzzles with solutions to {pickle_path}")

    print("\nDone!")


if __name__ == "__main__":
    main() 