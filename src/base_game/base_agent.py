#!/usr/bin/env python3
"""
Base Agent Abstract Class

This module defines the BaseAgent abstract base class that provides a common interface
for game-specific agent implementations used for solving puzzles via LLM interactions.

All game agent implementations should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod


class BaseInstructions(ABC):
    """
    Abstract base class for game instruction classes.
    
    This class enforces that all game-specific instruction implementations
    contain the required instruction templates, user requests, response formats,
    and common utilities needed for LLM interactions.
    
    Subclasses must implement all abstract methods and properties.
    """
    
    def __init__(self):
        """Initialize the instructions class and verify required attributes."""
        super().__init__()
        self._verify_required_attributes()
    
    def _verify_required_attributes(self):
        """Verify that all required instruction attributes are present."""
        required_attributes = [
            # System instructions for each query type
            'system_instruction_prior',
            'system_instruction_value', 
            'system_instruction_explore',
            'system_instruction_child_values',
            
            # User request templates for each query type
            'user_request_prior',
            'user_request_value',
            'user_request_explore',
            'user_request_child_values'
        ]
    
        
        missing_attributes = []
        
        # Check standard required attributes
        for attr in required_attributes:
            if not hasattr(self, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise AttributeError(
                f"{self.__class__.__name__} is missing required instruction attributes: "
                f"{missing_attributes}"
            )
    
    @abstractmethod
    def get_system_instruction(self, query_type, **kwargs):
        """
        Get the system instruction for the specified query type.
        
        Args:
            query_type (str): The type of query ('prior', 'value', 'explore', etc.)
            **kwargs: Additional game-specific parameters
            
        Returns:
            str: The system instruction for the specified query type
        """
        pass
    
    @abstractmethod
    def generate_request(self, query_type, **kwargs):
        """
        Generate a user request for the specified query type.
        
        Args:
            query_type (str): The type of query ('prior', 'value', 'explore', etc.)
            **kwargs: Parameters for the request (e.g., current_board, action_list)
            
        Returns:
            str: The formatted request string
        """
        pass


class BaseAgent(ABC):
    """
    Abstract base class for game agents that interact with LLMs.
    
    This class defines the interface that all game-specific agent implementations
    must follow to be compatible with puzzle solving algorithms.
    
    Subclasses must implement all abstract methods to define game-specific behavior.
    """
    
    @abstractmethod
    def __init__(self, model, **kwargs):
        """
        Initialize the agent.
        
        Args:
            model: LLM client instance
            **kwargs: Additional arguments specific to the game/implementation
        """
        pass
    
    @abstractmethod
    def reset(self, query_type):
        """
        Reset the agent state for a new query.
        
        Args:
            query_type (str): The type of query ('prior', 'value', 'explore', etc.)
        """
        pass
    
    @abstractmethod
    def ask(self, query_type, **kwargs):
        """
        Ask the LLM a question and return the response.
        
        Args:
            query_type (str): The type of query ('prior', 'value', 'explore', etc.)
            **kwargs: Additional arguments specific to the query type
            
        Returns:
            dict: Response containing 'resp', 'full_response', and 'token_usage'
        """
        pass
    
    @abstractmethod
    def update_message_history(self, new_message):
        """
        Update the conversation history with a new message.
        
        Args:
            new_message (dict): Message dictionary with 'role' and 'content'
        """
        pass
    
    @abstractmethod
    def _get_api_params(self, messages, response_format=None):
        """
        Helper method to generate API parameters based on reasoning mode.
        
        Args:
            messages (list): List of message objects to send to the API
            response_format (dict, optional): Format specification for the response
            
        Returns:
            dict: Parameters to use for the API call
        """
        pass
    
    @abstractmethod
    def _handle_retry(self, message_content, **kwargs):
        """
        Helper method to handle retry attempts for boxed JSON responses.
        
        Args:
            message_content (str): The retry message content
            **kwargs: Additional arguments (e.g., is_correction for SudokuAgent)
            
        Returns:
            tuple: (json_list, full_response, token_usage)
        """
        pass
    
    @abstractmethod
    def _ask(self, query_type, attempt_num, **kwargs):
        """
        Internal implementation of asking the LLM.
        
        Args:
            query_type (str): The type of query
            attempt_num (int): Current attempt number
            **kwargs: Additional arguments for the query
            
        Returns:
            dict: Response containing 'resp', 'full_response', and 'token_usage'
        """
        pass 