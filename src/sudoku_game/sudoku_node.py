#!/usr/bin/env python3
"""
SudokuNode Module

This module defines the SudokuNode class, which represents a node in the MCTS search tree
for solving Sudoku puzzles. Each node represents a complete board state with information
about untried numbers for each empty cell.
"""

from copy import deepcopy

from src.base_game.base_node import BaseNode
from src.utils.sudoku_utils import (
    board_to_string,
    get_empty_cells,
    get_valid_numbers_for_cell,
    is_valid_move,
    update_board,
    is_board_solved,
    determine_box_dimensions
)


class SudokuNode(BaseNode):
    """
    A node in the MCTS search tree for Sudoku puzzles.
    
    This class represents a complete Sudoku board state, with information about
    all empty cells and their untried numbers.
    
    This class implements the game node interface required by PathNode in game_tree.py:
    - get_legal_next_nodes(): Returns possible next moves
    - is_game_over(): Checks if the game is over
    - game_result(): Returns the outcome (win/loss)
    - get_action_description(): Describes the action that led to this node
    - get_state_description(): Describes the state at this node
    """
    
    def __init__(self, board, parent=None, last_move=None, expand_on_init=False):
        """
        Initialize a SudokuNode.
        
        Args:
            board (numpy.ndarray): Array representing the current Sudoku board state
            parent (SudokuNode, optional): The parent node in the search tree
            last_move (tuple, optional): Tuple of (row, col, num) representing the last move made
        """
        super().__init__()
        self.board = deepcopy(board)  # Store a copy of the current board state
        self.grid_size = board.shape[0]  # The grid size (e.g., 9 for a 9x9 puzzle)
        
        # Determine box dimensions
        self.box_width, self.box_height = determine_box_dimensions(self.grid_size)
        
        # Dictionary where keys are (row, col) tuples of empty cells
        # and values are lists of untried numbers for each cell
        self.untried_numbers = {}
        if expand_on_init:
            self.expand()

        self.parent = parent  # Reference to the parent node
        self.last_move = last_move  # The move that led to this state
        
        # For MCTS compatibility
        self.visits = 0
        self.value = 0
        self.heuristic = self._calculate_heuristic()
    
    @property
    def possible_numbers(self):
        """
        Get the possible numbers for the cell that was filled in this node.
        
        Returns:
            list: List of possible numbers from the parent's untried_numbers for the last filled cell
        """
        if self.parent and self.last_move:
            # Extract the cell coordinates from last_move (row, col, num)
            row, col, _ = self.last_move
            # Return the possible numbers from parent's untried_numbers
            return self.parent.untried_numbers.get((row, col), [])
        return []
    
    def _initialize_untried_numbers(self):
        """
        Initialize the dictionary of untried numbers for each empty cell.
        Only include numbers that are actually valid for each cell based on Sudoku rules.
        """
        empty_cells = get_empty_cells(self.board)
        
        for row, col in empty_cells:
            # Use the utility function to get valid numbers efficiently
            valid_numbers = get_valid_numbers_for_cell(self.board, row, col)
            
            # Store the list of valid numbers as untried numbers for this cell
            self.untried_numbers[(row, col)] = valid_numbers
    
    def _calculate_heuristic(self):
        """
        Calculate a heuristic value for this node.
        
        Lower values are better. The heuristic is based on:
        - Number of empty cells
        - Average number of untried options per cell
        
        Returns:
            float: Heuristic value
        """
        empty_cells = get_empty_cells(self.board)
        if not empty_cells:
            return 0  # Best possible heuristic - no empty cells
        
        # If untried_numbers hasn't been initialized yet, use len(empty_cells) as a rough estimate
        if not self.untried_numbers:
            return len(empty_cells)
            
        # Average number of untried options per cell
        total_options = sum(len(options) for options in self.untried_numbers.values())
        avg_options = total_options / len(self.untried_numbers)
        
        # Weight by the number of empty cells
        return avg_options * len(self.untried_numbers)
    
    def is_valid_move(self, row, col, num):
        """
        Check if placing a number in a specific cell is valid.
        
        Args:
            row (int): Row index
            col (int): Column index
            num (int): Number to place (1 to grid_size)
            
        Returns:
            bool: True if the move is valid
        """
        # Handle -1 as a special placeholder case
        if num == -1:
            # For -1, we just check if the cell is empty
            return self.board[row][col] == 0
        
        return is_valid_move(self.board, row, col, num)
    
    def make_move(self, row, col, num, expand_on_init=False):
        """
        Make a move by placing a number in a specific cell.
        
        Args:
            row (int): Row index
            col (int): Column index
            num (int): Number to place
            expand_on_init (bool): Set if node expanded on init
            
        Returns:
            SudokuNode: New node with the updated board state

        """
        # Check if the move is valid
        is_valid = self.is_valid_move(row, col, num)
        # print(f"Checking move ({row}, {col}, {num}): is_valid = {is_valid}")
        
        if not is_valid:
            return None
            
        # Create a new board with the number placed and marked as a placeholder
        mark_as_placeholder = num == -1
        new_board = update_board(self.board, row, col, num, mark_as_placeholder=mark_as_placeholder)
        
        # Create a new node with the updated board
        # possible_numbers will be automatically set in __init__ from parent's untried_numbers
        new_node = SudokuNode(new_board, parent=self, last_move=(row, col, num), expand_on_init=expand_on_init)
        
        return new_node
    
    def expand(self):
        """
        Expand this node by initializing untried_numbers if not already initialized.
        """
        if not self.untried_numbers:
            self._initialize_untried_numbers()
    
    def get_legal_next_nodes(self):
        """
        Generate all legal next moves from this node.
        
        Returns:
            list: List of (heuristic, node) tuples for possible next moves across all empty cells
        """
        next_possible_nodes = []
        
        # If the game is over, there are no legal next nodes
        if self.is_game_over():
            return []
        
        # Make sure untried_numbers is initialized
        if not self.untried_numbers:
            self.expand()
            
        # If no empty cells left, return empty list
        if not self.untried_numbers:
            return []
        
        # Generate nodes for all empty cells
        for cell, valid_numbers in self.untried_numbers.items():
            row, col = cell
            # Use the first valid number for this cell as a placeholder
            # This matches Countdown's pattern of creating one node per possible action
            if valid_numbers:  # Make sure there's at least one valid number
                new_node = self.make_move(row, col, -1)  # Use -1 as placeholder
                if new_node:
                    next_possible_nodes.append((new_node.heuristic, new_node))
        
        return next_possible_nodes
    
    def select_next_node(self, next_node):
        """
        Return the new state after making the move.
        
        Args:
            next_node (SudokuNode): The next node to move to
            
        Returns:
            SudokuNode: The next node
        """
        return next_node
    
    def is_game_over(self):
        """
        Check if the game is over based on empty cells and untried numbers.
        
        Returns:
            bool: True if the game is over
        """
        # Check if there are no untried numbers attribute or all number lists are empty
        empty_cells = get_empty_cells(self.board)

        if (not hasattr(self, 'untried_numbers') or 
            all(len(nums) == 0 for nums in self.untried_numbers.values())) or not empty_cells:
            return True
        return False
    
    def game_result(self):
        """
        Calculate the result of the game.
        
        Returns:
            float: 1.0 if the puzzle is solved correctly, 0.0 otherwise
        """
        # Check if the board is completely filled and valid
        if is_board_solved(self.board):
            return 1.0
            
        return 0.0
    
    def get_action_description(self):
        """
        Return a description of the action that led to this node.
        
        Returns:
            str: Description of the action
        """
        if self.last_move:
            row, col, num = self.last_move
            if num == -1:
                return f"Consider cell ({row}, {col}) where the following numbers {self.possible_numbers} have not been tried"
            return f"Place {num} at ({row}, {col})"
        else:
            return "Initial state"
            
    def get_state_description(self):
        """
        Return a description of the state at this node.
        
        Returns:
            str: Description of the state
        """
        board_str = board_to_string(self.board)
        return f"Board State:\n{board_str}"
    
    def __str__(self):
        """
        Return a string representation of this node.
        
        Returns:
            str: String representation
        """
        num_empty = len(self.untried_numbers)
        if self.last_move:
            row, col, num = self.last_move
            last_move_str = f"Last move: {num} at ({row}, {col})"
        else:
            last_move_str = "Initial state"
            
        return f"SudokuNode with {num_empty} empty cells. {last_move_str}"
    
    def __lt__(self, other):
        """
        Compare nodes based on their heuristic values for priority queue operations.
        
        Args:
            other (SudokuNode): The node to compare with
            
        Returns:
            bool: True if this node's heuristic is less than the other's
        """
        # Sort by heuristic (fewer is better)
        return self.heuristic < other.heuristic
    
    @classmethod
    def create_initial_node(cls, board):
        """
        Create an initial SudokuNode from a board.
        
        Args:
            board (numpy.ndarray): Initial board state
            
        Returns:
            SudokuNode: Initial node
        """
        return cls(board)
