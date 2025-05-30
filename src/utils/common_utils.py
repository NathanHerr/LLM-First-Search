"""
Core utility functions used across all games and components.
"""
import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math

import json5
import openai
from openai import AzureOpenAI

from dotenv import load_dotenv


@dataclass
class GameConfig:
    """Configuration for all game-related settings."""
    # Model settings
    model_name: str = "gpt-4o"
    model_type: str = "openai"
    temperature: float = 0.0
    max_tokens: int = 16384
    timeout: int = 300
    reasoning: int = 0

    # Experiment settings
    seed: int = 0
    num_iterations: int = 1

    # Countdown settings
    countdown_batch_num: int = 0
    countdown_batch_size: int = 30
    countdown_split: str = "val"
    countdown_difficulty: str = "3"

    # Sudoku settings
    sudoku_size: int = 9
    sudoku_width: int = 3
    sudoku_height: Optional[int] = None  # None means use width
    sudoku_num_puzzles: int = 60
    sudoku_difficulty: str = "easy"


def calculate_token_stats(token_usage: List[Tuple[str, Dict[str, int]]]) -> Dict[str, int]:
    """
    Calculate token usage statistics from a token_usage list.
    
    Args:
        token_usage (list): List of tuples (query_type, token_usage_dict)
        
    Returns:
        dict: Dictionary with total_tokens and api_call_count
    """
    total_tokens = sum(usage.get("total_tokens", 0) for _, usage in token_usage)
    api_call_count = len(token_usage)

    return {
        "total_tokens": total_tokens,
        "api_call_count": api_call_count
    }


def calculate_and_check_token_usage(token_usage: List[Tuple[str, Dict[str, int]]],
                                    max_token_usage: Optional[int],
                                    step_type: Optional[str] = None) -> Tuple[int, bool]:
    """Calculate total token usage and check if it exceeds the maximum limit.
    
    Args:
        token_usage: List of (query_type, usage_dict) tuples
        max_token_usage: Maximum allowed token usage or None
        step_type: Optional string describing the current step (for logging)
        
    Returns:
        tuple: (total_tokens, exceeded_limit)
    """
    # Calculate total token usage
    token_stats = calculate_token_stats(token_usage)
    total_tokens = token_stats["total_tokens"]

    if max_token_usage is None or max_token_usage <= 0:
        return total_tokens, False

    # Check if we've exceeded the limit
    if total_tokens >= max_token_usage:
        step_info = f" after {step_type}" if step_type else ""
        print(f"Reached maximum token usage{step_info}: {total_tokens}")
        return total_tokens, True

    return total_tokens, False


def query_agent(agent, query_type: str, **kwargs) -> Tuple[Dict[str, Any], Tuple[str, Dict[str, int]]]:
    """Query the agent and track token usage consistently.
    
    Args:
        agent: The agent to query
        query_type: Type of query ("action", "best", etc.)
        **kwargs: Query-specific arguments
        
    Returns:
        tuple: (response, token_usage_entry)
    """
    res = agent.ask(query_type=query_type, **kwargs)
    token_usage_entry = (query_type, res["token_usage"])
    return res, token_usage_entry


def initialize_metrics_tracking(explorer) -> List[Dict[str, Any]]:
    """Initialize the metrics tracking log consistently for both games.
    
    Args:
        explorer: The Explorer object tracking the game tree
        
    Returns:
        list: Initial metrics log with the starting state
    """
    metrics_log = []
    metrics_log.append({
        "step": 0,
        "tree_size": explorer.tree_size(),
        "expanded_nodes": explorer.expanded_nodes_count(),
        "unexpanded_nodes": len(explorer.get_unexpanded_nodes()),
        "total_tokens": 0,  # Initialize token usage
        "api_call_count": 0,  # Initialize token usage
        "backtrack_count": 0  # Initialize backtrack count
    })
    return metrics_log


def update_metrics_log(metrics_log: List[Dict[str, Any]],
                       step: int,
                       explorer,
                       token_usage: List[Tuple[str, Dict[str, int]]],
                       backtrack_count: int = 0) -> Dict[str, Any]:
    """Update metrics log with current state.
    
    Args:
        metrics_log: The metrics log to update
        step: The current step number
        explorer: The Explorer object tracking the game tree
        token_usage: List of token usage entries
        backtrack_count: Current backtrack count (default 0)
        
    Returns:
        dict: The newly added metrics entry
    """
    token_stats = calculate_token_stats(token_usage)
    metrics_entry = {
        "step": step,
        "tree_size": explorer.tree_size(),
        "expanded_nodes": explorer.expanded_nodes_count(),
        "unexpanded_nodes": len(explorer.get_unexpanded_nodes()),
        "total_tokens": token_stats["total_tokens"],
        "api_call_count": token_stats["api_call_count"],
        "backtrack_count": backtrack_count  # Add backtrack count to metrics
    }
    metrics_log.append(metrics_entry)
    return metrics_entry


def load_game_state(output_path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load existing game state from pickle file"""
    try:
        with open(output_path, 'rb') as f:
            all_outputs = pickle.load(f)
            processed_games = len(all_outputs)
    except (FileNotFoundError, EOFError):
        all_outputs = []
        processed_games = 0
    return all_outputs, processed_games


def save_game_state(output_path: str, all_outputs: List[Dict[str, Any]], game_idx: int,
                    outputs: Dict[str, Any]) -> None:
    """Save game state to pickle file"""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if game_idx < len(all_outputs):
        all_outputs[game_idx] = outputs
    else:
        all_outputs.append(outputs)

    with open(output_path, 'wb') as f:
        pickle.dump(all_outputs, f)


def get_game_output_path(args, search_method: str, game_type: str) -> str:
    """Get the output path for a game based on its type"""
    if game_type == "countdown":
        return get_countdown_output_path(
            args.output_dir,
            search_method,
            args.batch_num,
            args.batch_size,
            args.split,
            args.countdown_difficulty,
            args.model_name,
            args.temperature,
            args.max_token_usage,
            getattr(args, 'c_value', None)  # Use getattr to safely get c_value if it exists
        )
    elif game_type == "sudoku":
        return get_sudoku_output_path(
            args.output_dir,
            search_method,
            args.batch_num,
            args.batch_size,
            args.sudoku_size,
            args.sudoku_difficulty,
            args.model_name,
            args.temperature,
            args.sudoku_width,
            args.sudoku_height,
            args.max_token_usage,
            getattr(args, 'c_value', None)  # Use getattr to safely get c_value if it exists
        )
    else:
        raise ValueError(f"Unknown game type: {game_type}")


def create_openai_client(model_type, timeout=300, is_azure=False):
    """
    Create an OpenAI client with the specified settings.
    
    Args:
        model_type (str): Type of model (openai, anthropic, nvidia, etc.)
        timeout (int): Timeout for API calls in seconds
        is_azure (bool): Whether to use Azure OpenAI endpoint
        
    Returns:
        client: An API client for the specified model type
    """
    if model_type.lower() == "nvidia":
        return openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            timeout=timeout
        )
    elif model_type.lower() == "openai":
        if is_azure:
            return AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                timeout=timeout
            )
        else:
            return openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=timeout
            )
    else:
        raise ValueError(f"Model type {model_type} is not supported yet.")


def check_json_list(json_list):
    """Parse a list of JSON strings and return the first valid one."""
    for jl in json_list:
        try:
            resp_dict = json5.loads(jl)
            return resp_dict
        except Exception as e:
            continue
    return None


def create_sorted_nodes_by_value(nodes, use_adjusted_value=True):
    """Create a list of nodes sorted by their value (ascending).

    Args:
        nodes: List of nodes to sort
        use_adjusted_value: If True, use adjusted_value for sorting when available
    """
    # Filter out nodes without values
    nodes_with_values = [node for node in nodes if node.value is not None]

    # Sort by value (ascending)
    if use_adjusted_value:
        # Use adjusted value when available, fall back to regular value otherwise
        return sorted(nodes_with_values,
                      key=lambda node: node.adjusted_value if node.adjusted_value is not None else node.value)
    else:
        # Use regular value for sorting
        return sorted(nodes_with_values, key=lambda node: node.value)


def get_standard_parser():
    """
    Create a standardized argument parser with parameters for both Countdown and Sudoku games.

    Returns:
        argparse.ArgumentParser: A parser with arguments for both games
    """
    DEFAULT_CONFIG = GameConfig()

    parser = argparse.ArgumentParser(description="Language model reasoning for games")

    # Common arguments for all configurations
    parser.add_argument('--model_name', type=str, default=DEFAULT_CONFIG.model_name,
                        help='Name of the model to use')
    parser.add_argument('--model_type', type=str, default=DEFAULT_CONFIG.model_type,
                        help='Type of model to use (openai, anthropic, etc.)')
    parser.add_argument('--is_azure', type=int, choices=[0, 1], default=0,
                        help='Whether to use Azure OpenAI endpoint (0=no, 1=yes)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG.temperature,
                        help='Temperature for sampling')
    parser.add_argument('--timeout', type=int, default=DEFAULT_CONFIG.timeout,
                        help='Timeout for API calls in seconds')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_CONFIG.max_tokens,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--reasoning', type=int, default=DEFAULT_CONFIG.reasoning,
                        help='Enable reasoning mode (0=disabled, 1=enabled) for API calls (uses reasoning_effort with OpenAI)')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG.seed,
                        help='Random seed')
    parser.add_argument('--num_its', type=int, default=DEFAULT_CONFIG.num_iterations,
                        help='Number of times to run each game/puzzle (for statistical significance)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory for input data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output data')

    # Game selection
    parser.add_argument('--game_type', type=str, choices=['countdown', 'sudoku'], required=True,
                        help='Type of game to play (countdown or sudoku)')

    # Countdown-specific arguments
    parser.add_argument('--split', type=str, default=DEFAULT_CONFIG.countdown_split,
                        help='Data split to use (test, train, etc.) - for Countdown')
    parser.add_argument('--countdown_difficulty', type=str, default=DEFAULT_CONFIG.countdown_difficulty,
                        help='Difficulty level for Countdown puzzles')
    parser.add_argument('--batch_num', type=int, default=DEFAULT_CONFIG.countdown_batch_num,
                        help='Batch number - for Countdown')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG.countdown_batch_size,
                        help='Batch size - for Countdown')
    parser.add_argument('--search_method', type=str, required=True,
                        choices=['tot_bfs', 'mcts', 'dfs', 'lfs', 'bestfs'],
                        help='Search method - for Countdown')

    # Sudoku-specific arguments
    parser.add_argument('--sudoku_size', type=int, default=DEFAULT_CONFIG.sudoku_size,
                        help='Size of the Sudoku grid (e.g., 9 for 9x9)')
    parser.add_argument('--sudoku_width', type=int, default=DEFAULT_CONFIG.sudoku_width,
                        help='Width of the Sudoku box')
    parser.add_argument('--sudoku_height', type=int, default=DEFAULT_CONFIG.sudoku_height,
                        help='Height of the Sudoku box (can be different from width for rectangular grids)')
    parser.add_argument('--sudoku_difficulty', type=str, default=DEFAULT_CONFIG.sudoku_difficulty,
                        help='Difficulty level for Sudoku generation')
    parser.add_argument('--num_puzzles', type=int, default=DEFAULT_CONFIG.sudoku_num_puzzles,
                        help='Number of Sudoku puzzles to solve')

    # Search-specific arguments
    parser.add_argument("--max_token_usage", type=int, default=None,
                        help="Maximum total token usage to allow before stopping")

    return parser


def get_sudoku_output_path(output_dir, search_method, batch_num, batch_size, sudoku_size, difficulty, model_name,
                           temperature, width=None, height=None, max_token_usage=None, c_value=None):
    """
    Get the path to the output file for Sudoku results.

    Args:
        output_dir (str): Directory for output data
        search_method (str): Search method used
        batch_num (int): Batch number
        batch_size (int): Batch size
        sudoku_size (int): Size of the Sudoku grid
        difficulty (str): Difficulty level
        model_name (str): Name of the model used
        temperature (float): Temperature setting
        width (int, optional): Width of each box in the Sudoku grid
        height (int, optional): Height of each box in the Sudoku grid
        max_token_usage (int, optional): Maximum tokens for model responses and token usage limit
        c_value (float, optional): The exploration constant for MCTS

    Returns:
        str: Path to the output file
    """
    # Replace any slashes in model name with underscores
    model_name_safe = model_name.replace('/', '_')

    # Determine width and height if not provided
    if width is None:
        width = int(math.sqrt(sudoku_size))
    if height is None:
        height = width

    # Use the same path pattern for all experiment types
    base_path = f"{output_dir}/batch_{batch_num}_size_{batch_size}_sudoku_w_{width}_h_{height}_{difficulty}_responses_{model_name_safe}_{temperature}"

    # Use max_tokens for the token usage limit (mtu = max token usage)
    if search_method == "mcts":
        # Convert c_value to integer by multiplying by 10 to avoid decimal points in filenames
        c_value_int = int(c_value * 10) if c_value is not None else 5  # Default to 0.5 if not specified
        return f"{base_path}_llm_mcts_mtu_{max_token_usage}_c{c_value_int}.pkl"
    elif search_method == "lfs":
        return f"{base_path}_lfs_mtu_{max_token_usage}.pkl"
    elif search_method == "bestfs":
        return f"{base_path}_bestfs_mtu_{max_token_usage}.pkl"
    elif search_method == "tot_bfs":
        return f"{base_path}_tot_bfs_mtu_{max_token_usage}.pkl"
    else:
        # Fallback for other methods
        return f"{base_path}_{search_method}_mtu_{max_token_usage}.pkl"


def get_countdown_output_path(output_dir, search_method, batch_num, batch_size, split, countdown_difficulty, model_name,
                              temperature, max_token_usage=None, c_value=None):
    """
    Get the path to the output file for Countdown results.

    Args:
        output_dir (str): Directory for output data
        search_method (str): Search method used
        batch_num (int): Batch number
        batch_size (int): Batch size
        split (str): Data split
        countdown_difficulty (str): Difficulty level for Countdown puzzles
        model_name (str): Name of the model used
        temperature (float): Temperature setting
        max_token_usage (int, optional): Maximum tokens for model responses and token usage limit
        c_value (float, optional): The exploration constant for MCTS

    Returns:
        str: Path to the output file
    """
    # Replace any slashes in model name with underscores
    model_name_safe = model_name.replace('/', '_')

    # Use the same path pattern for all experiment types
    base_path = f"{output_dir}/batch_{batch_num}_size_{batch_size}_{split}_{countdown_difficulty}_responses_{model_name_safe}_{temperature}"

    # Use max_tokens for the token usage limit (mtu = max token usage)
    if search_method == "mcts":
        # Convert c_value to integer by multiplying by 10 to avoid decimal points in filenames
        c_value_int = int(c_value * 10) if c_value is not None else 5  # Default to 0.5 if not specified
        return f"{base_path}_llm_mcts_mtu_{max_token_usage}_c{c_value_int}.pkl"
    elif search_method == "lfs":
        return f"{base_path}_lfs_mtu_{max_token_usage}.pkl"
    elif search_method == "bestfs":
        return f"{base_path}_bestfs_mtu_{max_token_usage}.pkl"
    elif search_method == "tot_bfs":
        return f"{base_path}_tot_bfs_mtu_{max_token_usage}.pkl"
    else:
        # Fallback for other methods
        return f"{base_path}_{search_method}_mtu_{max_token_usage}.pkl"
