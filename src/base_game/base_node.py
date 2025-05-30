#!/usr/bin/env python3
"""
BaseNode Abstract Base Class

This module defines the BaseNode abstract base class that provides a common interface
for game-specific node implementations used in search algorithms like MCTS, BFS, etc.

All game node implementations should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod


class BaseNode(ABC):
    """
    Abstract base class for game nodes in search trees.
    
    This class defines the interface that all game-specific node implementations
    must follow to be compatible with search algorithms like MCTS, BFS, etc.
    
    Subclasses must implement all abstract methods to define game-specific behavior.
    """
    
    @abstractmethod
    def get_legal_next_nodes(self):
        """
        Generate all legal next moves/states from this node.
        
        Returns:
            list: List of (heuristic, node) tuples for possible next moves.
                  The heuristic can be None if not applicable.
        """
        pass
    
    @abstractmethod
    def select_next_node(self, next_node):
        """
        Return the new state after selecting a move.
        
        Args:
            next_node: The node representing the next state
            
        Returns:
            BaseNode: The next node/state
        """
        pass
    
    @abstractmethod
    def is_game_over(self):
        """
        Check if the game/puzzle is over.
        
        Returns:
            bool: True if the game is over (solved, no more moves, etc.)
        """
        pass
    
    @abstractmethod
    def game_result(self):
        """
        Calculate the result/outcome of the game.
        
        Returns:
            float: Numeric value representing the game outcome.
                   Higher values typically indicate better outcomes.
                   Common convention: 1.0 for win/success, 0.0 or negative for loss/failure.
        """
        pass
    
    @abstractmethod
    def get_action_description(self):
        """
        Return a description of the action that led to this node.
        
        Returns:
            str: Human-readable description of the action/move that created this state.
                 Should return empty string or appropriate message for initial/root states.
        """
        pass
    
    @abstractmethod
    def get_state_description(self):
        """
        Return a description of the current state at this node.
        
        Returns:
            str: Human-readable description of the current game state.
        """
        pass
    
    @abstractmethod
    def __lt__(self, other):
        """
        Compare nodes for ordering (typically used in priority queues).
        
        Args:
            other (BaseNode): The node to compare with
            
        Returns:
            bool: True if this node should be ordered before the other node
        """
        pass
    
    def __str__(self):
        """
        Default string representation using state description.
        Subclasses can override this for custom string representation.
        
        Returns:
            str: String representation of the node
        """
        return f"{self.__class__.__name__}: {self.get_state_description()}" 