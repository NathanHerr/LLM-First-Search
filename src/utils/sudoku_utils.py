"""
Sudoku Utilities Module

This module provides utility functions for working with Sudoku puzzles, separated from the
SudokuNode class to keep the code more modular.
"""

from copy import deepcopy

import numpy as np

from src.utils.common_utils import update_metrics_log, query_agent


def get_empty_cells(board):
    """
    Find all empty cells in the Sudoku board.

    Args:
        board (numpy.ndarray): Array representing the Sudoku board

    Returns:
        list: List of (row, col) tuples for empty cells
    """
    empty_cells = []
    grid_size = board.shape[0]
    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] == 0:
                empty_cells.append((row, col))
    return empty_cells


def determine_box_dimensions(grid_size):
    """
    Determine the box dimensions for a given grid size.

    Args:
        grid_size (int): The size of the grid (e.g., 9 for a 9x9 puzzle)

    Returns:
        tuple: (box_width, box_height)
    """
    # Try to detect if it's a perfect square
    sqrt_size = int(np.sqrt(grid_size))
    if sqrt_size * sqrt_size == grid_size:
        # Perfect square - use same width and height
        return sqrt_size, sqrt_size

    # Not a perfect square - try to find suitable factors
    factors = []
    for i in range(2, int(np.sqrt(grid_size)) + 1):
        if grid_size % i == 0:
            factors.append(i)

    if len(factors) > 0:
        box_width = factors[-1]  # Use the largest factor
        box_height = grid_size // box_width
        return box_width, box_height

    # No factors found, use default 3x3 or best guess
    print(f"Warning: Could not determine box dimensions for grid size {grid_size}")
    return 3, 3


def is_board_solved(board):
    """
    Check if the Sudoku puzzle is solved correctly.

    Args:
        board (numpy.ndarray): Current board state

    Returns:
        bool: True if the puzzle is completely filled and valid
    """
    grid_size = board.shape[0]
    box_width, box_height = determine_box_dimensions(grid_size)

    # Check if there are any empty cells
    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] == 0:
                return False

    # Check rows
    for row in range(grid_size):
        if set(board[row]) != set(range(1, grid_size + 1)):
            return False

    # Check columns
    for col in range(grid_size):
        if set(board[:, col]) != set(range(1, grid_size + 1)):
            return False

    # Check boxes
    for box_row in range(0, grid_size, box_height):
        for box_col in range(0, grid_size, box_width):
            box = [board[r][c] for r in range(box_row, box_row + box_height)
                   for c in range(box_col, box_col + box_width)]
            if set(box) != set(range(1, grid_size + 1)):
                return False

    return True


def update_board(board, row, col, value=-1, mark_as_placeholder=False):
    """
    Update the board with a new value at the specified position.

    Args:
        board (numpy.ndarray): Current board state
        row (int): Row index
        col (int): Column index
        value (int): Value to place (-1 for placeholder)
        mark_as_placeholder (bool): If True, mark the cell with negative value to indicate it's a placeholder

    Returns:
        numpy.ndarray: Updated board
    """
    new_board = deepcopy(board)
    if new_board[row][col] == 0:
        # If mark_as_placeholder is True, store the value as a negative number
        if mark_as_placeholder:
            # Use -1 to represent placeholder in the array
            new_board[row][col] = -1
        else:
            if value == -1:
                raise ValueError("Value cannot be -1")
            new_board[row][col] = value
    return new_board


def board_to_string(board):
    """
    Return a string representation of the Sudoku board.

    Args:
        board (numpy.ndarray): Current board state

    Returns:
        str: Formatted Sudoku board
    """
    grid_size = board.shape[0]
    box_width, box_height = determine_box_dimensions(grid_size)

    # Calculate the width needed for each cell (for larger puzzles)
    cell_width = max(2, len(str(grid_size)) + 1)

    # Calculate the width of separator line
    sep_width = (cell_width * grid_size) + (grid_size // box_width)

    result = ""
    for i in range(grid_size):
        if i % box_height == 0 and i > 0:
            result += "-" * sep_width + "\n"

        for j in range(grid_size):
            if j % box_width == 0 and j > 0:
                result += "| "

            cell = board[i][j]
            if cell == 0:
                result += "." + " " * (cell_width - 1)
            elif cell == -1:  # -1 is a placeholder
                # Display placeholder with special formatting (e.g., [X])
                placeholder_str = "[X]"
                result += placeholder_str + " " * max(0, cell_width - len(placeholder_str))
            elif cell < 0:  # Negative values are placeholders
                # Display placeholders with special formatting (e.g., [5] for placeholder 5)
                placeholder_str = f"[{abs(cell)}]"
                result += placeholder_str + " " * max(0, cell_width - len(placeholder_str))
            else:
                result += str(cell) + " " * (cell_width - len(str(cell)))

        result += "\n"

    return result


def from_py_sudoku(py_sudoku):
    """
    Create a board from a py-sudoku Sudoku object.

    Args:
        py_sudoku (sudoku.Sudoku): Sudoku puzzle

    Returns:
        numpy.ndarray: Array representing the Sudoku board
    """
    # Get dimensions from py_sudoku
    grid_size = len(py_sudoku.board)

    # Convert py-sudoku to a numpy array
    # Replace None values with 0 for empty cells
    board = np.zeros((grid_size, grid_size), dtype=int)

    for i in range(grid_size):
        for j in range(grid_size):
            if py_sudoku.board[i][j] is not None:
                board[i][j] = py_sudoku.board[i][j]

    return board


def get_valid_numbers_for_cell(board, row, col):
    """
    Calculate all valid numbers that can be placed in a specific cell.

    Args:
        board (numpy.ndarray): Current board state
        row (int): Row index of the cell
        col (int): Column index of the cell

    Returns:
        list: List of valid numbers (1 to grid_size) that can be placed in this cell
    """
    if board[row][col] != 0 and board[row][col] != -1:
        return []  # Cell is already filled

    grid_size = board.shape[0]
    box_width, box_height = determine_box_dimensions(grid_size)
    valid_numbers = []

    for num in range(1, grid_size + 1):
        # Check row
        if num in board[row]:
            continue

        # Check column
        if num in board[:, col]:
            continue

        # Check box
        box_row, box_col = box_height * (row // box_height), box_width * (col // box_width)
        box_values = [board[r][c] for r in range(box_row, box_row + box_height)
                      for c in range(box_col, box_col + box_width)]
        if num in box_values:
            continue

        valid_numbers.append(num)

    return valid_numbers


def is_valid_move(board, row, col, num):
    """
    Check if placing a number in a specific cell is valid.

    Args:
        board (numpy.ndarray): Current board state
        row (int): Row index of the cell
        col (int): Column index of the cell
        num (int): Number to place (1 to grid_size)

    Returns:
        bool: True if the move is valid
    """
    grid_size = board.shape[0]
    box_width, box_height = determine_box_dimensions(grid_size)

    # Check if the cell is already filled
    if board[row][col] != 0:
        return False

    # Check row
    for x in range(grid_size):
        cell_value = board[row][x]
        # Consider both regular values and placeholders (negative values)
        if cell_value == num:
            return False

    # Check column
    for x in range(grid_size):
        cell_value = board[x][col]
        # Consider both regular values and placeholders (negative values)
        if cell_value == num:
            return False

    # Check box
    box_row, box_col = box_height * (row // box_height), box_width * (col // box_width)
    for r in range(box_row, box_row + box_height):
        for c in range(box_col, box_col + box_width):
            cell_value = board[r][c]
            # Consider both regular values and placeholders (negative values)
            if cell_value == num:
                return False

    return True


def calculate_move_accuracy(correct_moves, total_moves):
    """
    Calculate the accuracy of moves made so far.

    Args:
        correct_moves (int): Number of correct moves made
        total_moves (int): Total number of moves made

    Returns:
        float: Move accuracy percentage (0-100)
    """
    if total_moves == 0:
        return 0
    return (correct_moves / total_moves) * 100


def calculate_board_accuracy_metrics(initial_board, current_board, solution_board):
    """
    Calculate various board accuracy metrics by comparing the current board to the solution.

    Args:
        initial_board (numpy.ndarray): The initial board state
        current_board (numpy.ndarray): The current board state
        solution_board (numpy.ndarray): The solution board

    Returns:
        dict: Dictionary containing accuracy metrics:
            - correct_cells: Number of cells filled correctly
            - filled_cells: Number of initially empty cells that are now filled
            - total_empty_cells: Total number of initially empty cells
            - board_accuracy: Percentage of filled cells that are correct
            - completion_percentage: Percentage of empty cells that are filled
    """
    # Initialize counters
    correct_cells = 0
    filled_cells = 0
    total_empty_cells = 0

    # Count how many cells were initially empty
    for i in range(initial_board.shape[0]):
        for j in range(initial_board.shape[1]):
            if initial_board[i][j] == 0:  # Initially empty cell
                total_empty_cells += 1
                # Check if the cell was filled
                if current_board[i][j] != 0:
                    filled_cells += 1
                    # Check if it was filled correctly
                    if current_board[i][j] == solution_board[i][j]:
                        correct_cells += 1

    # Calculate board accuracy (percentage of filled cells that are correct)
    board_accuracy = 0
    if filled_cells > 0:
        board_accuracy = (correct_cells / filled_cells) * 100

    # Calculate completion percentage (percentage of empty cells that are filled)
    completion_percentage = 0
    if total_empty_cells > 0:
        completion_percentage = (filled_cells / total_empty_cells) * 100

    return {
        "correct_cells": correct_cells,
        "filled_cells": filled_cells,
        "total_empty_cells": total_empty_cells,
        "board_accuracy": board_accuracy,
        "completion_percentage": completion_percentage
    }


def get_incorrect_cells(initial_board, current_board, solution_board):
    """
    Get a list of cells that have been filled incorrectly.

    Args:
        initial_board (numpy.ndarray): The initial board state
        current_board (numpy.ndarray): The current board state
        solution_board (numpy.ndarray): The solution board

    Returns:
        list: List of (row, col) tuples for incorrectly filled cells
    """
    incorrect_cells = []
    grid_size = current_board.shape[0]

    for row in range(grid_size):
        for col in range(grid_size):
            if initial_board[row][col] == 0 and current_board[row][col] != 0:
                # This is a cell filled by the agent
                if current_board[row][col] != solution_board[row][col]:
                    incorrect_cells.append((row, col, current_board[row][col], solution_board[row][col]))

    return incorrect_cells


def update_sudoku_metrics_log(metrics_log, step, explorer, move_accuracy_data, board_accuracy_data, token_usage=None):
    """Update metrics log with Sudoku-specific metrics.

    Args:
        metrics_log: The metrics log to update
        step: The current step number
        explorer: The Explorer object tracking the game tree
        move_accuracy_data: Dictionary with move accuracy metrics
        board_accuracy_data: Dictionary with board accuracy metrics
        token_usage: List of token usage entries (optional)

    Returns:
        dict: The newly added metrics entry
    """

    # First get standard metrics
    metrics_entry = update_metrics_log(metrics_log, step, explorer, token_usage)

    # Then add Sudoku-specific metrics
    metrics_entry.update(move_accuracy_data)
    metrics_entry.update(board_accuracy_data)

    # Replace the last entry with our enhanced version
    metrics_log[-1] = metrics_entry
    return metrics_entry


def get_total_empty_cells(board):
    """
    Calculate the total number of empty cells in a board.

    Args:
        board (numpy.ndarray): The board state

    Returns:
        int: Number of empty cells (cells with value 0)
    """
    return len(get_empty_cells(board))


def calculate_sudoku_accuracy_metrics(board, current_board, solution_board, correct_moves, total_moves):
    """Calculate both move accuracy and board accuracy metrics for Sudoku.

    Args:
        board: Initial board state
        current_board: Current board state
        solution_board: Solution board
        correct_moves: Count of correct moves so far
        total_moves: Total moves attempted

    Returns:
        tuple: (move_accuracy_data, board_accuracy_data)
    """
    # Calculate move accuracy
    move_accuracy = calculate_move_accuracy(correct_moves, total_moves)
    move_accuracy_data = {
        "move_accuracy": move_accuracy,
        "correct_moves": correct_moves,
        "total_moves": total_moves
    }

    # Calculate board accuracy metrics
    board_accuracy_data = calculate_board_accuracy_metrics(
        board, current_board, solution_board
    )

    return move_accuracy_data, board_accuracy_data


def find_best_board_state(explorer, initial_board, solution_board):
    """Find the best board state in the explorer's tree based on the average of board accuracy and completion percentage.

    Args:
        explorer (Explorer): The explorer object containing the game tree
        initial_board (numpy.ndarray): Initial board state
        solution_board (numpy.ndarray): Solution board

    Returns:
        tuple: (best_board, best_score, best_metrics) or (None, 0, None) if no valid board found
    """
    best_node = None
    best_score = 0
    best_metrics = None

    def traverse_node(node):
        if node is None:
            return

        # Calculate metrics for current node's board
        metrics = calculate_board_accuracy_metrics(initial_board, node.game_node.board, solution_board)
        board_accuracy = metrics["board_accuracy"]
        completion_percentage = metrics["completion_percentage"]

        # Calculate score as average of board accuracy and completion percentage
        score = (board_accuracy + completion_percentage) / 2

        # Update if this is the best score so far
        nonlocal best_node, best_score, best_metrics
        if score > best_score:
            best_node = node
            best_score = score
            best_metrics = metrics

        # Recursively traverse only expanded children
        for child in node.children:
            if child.expanded:  # Only traverse expanded children
                traverse_node(child)

    # Start traversal from root
    traverse_node(explorer.explorer_root)

    if best_node is None:
        return None, 0, None

    return best_node.game_node.board, best_score, best_metrics


def evaluate_child_moves(agent, explorer, current_node, token_usage):
    """Evaluate all possible moves from the current node using the agent and store the values. Sudoku Specific"""
    # Skip if no children
    if not current_node.children:
        return token_usage

    # Get the current board state for the request
    current_board_str = board_to_string(current_node.game_node.board)

    # Collect all possible moves from the children
    moves_list = {}
    move_index = 0
    move_details = {}  # Map to track details of each move

    for child in current_node.children:
        if hasattr(child.game_node, 'last_move') and child.game_node.last_move:
            row, col, number = child.game_node.last_move
            move_desc = f"Place {number} at cell ({row}, {col})"
            moves_list[move_index] = move_desc
            # Track the details of this move
            move_details[move_index] = {
                'row': row,
                'col': col,
                'number': number
            }
            move_index += 1

    # Skip if no valid moves
    if not moves_list:
        return token_usage

    # Ask the agent to evaluate the possible moves
    res, usage = query_agent(
        agent,
        query_type="child_moves",
        current_board=current_board_str,
        moves_list=moves_list
    )
    token_usage.append(usage)

    # Check if the response contains the move values
    if res["resp"] is None:
        return token_usage

    # Get the move values (which have already been validated by the agent)
    move_values = res["resp"]

    # Set the value of each child node
    for move_idx, value in move_values.items():
        try:
            move_info = move_details[int(move_idx)]
            # Find the child node that matches this move
            for child in current_node.children:
                if (hasattr(child.game_node, 'last_move') and
                        child.game_node.last_move == (move_info['row'], move_info['col'], move_info['number'])):
                    child.value = value
                    # Set adjusted_value equal to value initially
                    child.adjusted_value = value
                    break
        except (ValueError, TypeError):
            # Skip invalid keys - validation should have handled most cases already
            continue

    return token_usage


def evaluate_sudoku_node_value(agent, node, token_usage):
    """Evaluate a Sudoku node's value using the agent and store it in the node."""
    current_board_str = board_to_string(node.game_node.board)
    res, usage = query_agent(
        agent,
        query_type="value",
        current_board=current_board_str,
        empty_cells=node.get_possible_actions()
    )

    token_usage.append(usage)
    node.value = res["resp"]
    return node.value, token_usage


def get_sudoku_data_path(data_dir, sudoku_difficulty, width, height=None):
    """
    Get the path to the input data file for Sudoku.

    Args:
        data_dir (str): Directory for data
        sudoku_difficulty (str): Difficulty level
        width (int): Width of each box in the Sudoku grid
        height (int, optional): Height of each box in the Sudoku grid (default: same as width)

    Returns:
        str: Path to the input data file
    """
    if height is None:
        height = width
    return f"{data_dir}/sudoku_diff_{sudoku_difficulty}_w_{width}_h_{height}.pkl"
