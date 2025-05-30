import json
import pickle
import random
import time

import numpy as np
from tqdm import tqdm

import src.utils.sudoku_utils as sudoku_utils
# Countdown imports
from src.countdown_game.countdown_agent import CountdownAgent
from src.countdown_game.countdown_node import CountdownNode
# Sudoku imports
from src.sudoku_game.sudoku_agent import SudokuAgent
from src.sudoku_game.sudoku_node import SudokuNode
# Import utility functions
from src.utils.common_utils import (
    calculate_and_check_token_usage,
    initialize_metrics_tracking,
    update_metrics_log,
    calculate_token_stats,
    save_game_state,
    load_game_state,
    get_game_output_path, create_openai_client, get_standard_parser,
    create_sorted_nodes_by_value,
)
from src.utils.countdown_utils import mult_heuristic, evaluate_child_node_values, \
    get_countdown_data_path
# Import common tree components
from src.utils.tree_utils import PathNode, Explorer
from src.utils.sudoku_utils import (
    update_sudoku_metrics_log,
    calculate_sudoku_accuracy_metrics,
    find_best_board_state, evaluate_child_moves, get_sudoku_data_path
)
from dotenv import load_dotenv

"""
Tree of Thought BFS Implementation - Works with both Countdown and Sudoku

This file implements a breadth-first search approach to solving puzzles where
the agent filters and prioritizes paths to explore.
Support for both Countdown and Sudoku puzzle types.
"""

load_dotenv()

def parse_args():
    parser = get_standard_parser()
    # Add ToT-BFS specific arguments
    parser.add_argument('--top_n', type=int, default=3,
                      help='Number of top nodes to consider in ToT-BFS')
    return parser.parse_args()

def run_countdown(args, model):
    """Run the Countdown puzzle solver using ToT-BFS approach"""
    
    # Create the agent
    agent = CountdownAgent(
        model=model,
        model_name=args.model_name,
        model_type=args.model_type,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
    )
    
    # Set up the data path for input data
    data_path = get_countdown_data_path(args.data_dir, args.split, args.countdown_difficulty)

    # Load all games data
    with open(data_path, 'r') as file:
        all_games = json.load(file)

    # Select the batch based on batch_num and batch_size
    batch = all_games[args.batch_num*args.batch_size:(args.batch_num+1)*args.batch_size]
    
    # Get output path using common utility
    output_path = get_game_output_path(args, "tot_bfs", "countdown")
    
    # Load existing outputs using common utility
    all_outputs, processed_games = load_game_state(output_path)
        
    # Process each game in batch and collect responses
    for game_idx, game in enumerate(tqdm(batch, desc=f"Batch {args.batch_num}")):
        # Only skip if we have all iterations for this game
        if game_idx < len(all_outputs) and len(all_outputs[game_idx]) == args.num_its:
            # Skip games that have completed all iterations
            continue
            
        target, nums = game['target'], game['nums']
        
        # Initialize outputs list for this game or load existing iterations
        if game_idx < len(all_outputs):
            # We're resuming from a partially completed game
            outputs = all_outputs[game_idx]
            completed_iterations = len(outputs)
            print(f"Resuming game {game_idx} from iteration {completed_iterations}")
        else:
            # Starting a new game
            outputs = []
            completed_iterations = 0
            
        # Run only the remaining iterations
        for iter_idx in tqdm(range(completed_iterations, args.num_its), desc=f"Game {game_idx}"):
            root_node = PathNode(CountdownNode(0, None, nums, [], mult_heuristic, target))
            root_node.expand()
            explorer = Explorer(root_node)
            token_usage = []
            start = time.time()
            best_node = root_node
            atts = 0
            backtrack_count = 0
            
            # Initialize metrics tracking
            metrics_log = initialize_metrics_tracking(explorer)
            
            # Evaluate root node's children
            token_usage = evaluate_child_node_values(agent, explorer, root_node, token_usage)
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}")
            
            # Create initial list of nodes to explore from root's children
            nodes_to_explore = []
            for child in root_node.children:
                if child.value is not None:
                    nodes_to_explore.append(child)
            
            # Main ToT-BFS loop
            while True:
                # Check token usage limit
                total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage)
                if limit_exceeded:
                    print(f"Reached maximum token usage: {total_tokens}")
                    break
                
                if not nodes_to_explore:
                    print("No more nodes to explore!")
                    break
                
                print(f"Current nodes to explore: {len(nodes_to_explore)}")
                
                # Sort nodes by their values and take top N (highest values last in ascending order)
                sorted_nodes = create_sorted_nodes_by_value(nodes_to_explore)
                selected_nodes = sorted_nodes[-args.top_n:]  # Take top N nodes
                best_node = selected_nodes[-1]  # Best node is the last one (highest value)
                
                print(f"Selected {len(selected_nodes)} nodes to expand")
                
                # Check if best node is terminal
                if best_node.game_node.is_game_over():
                    print(f"Found terminal node with value: {best_node.game_node.game_result()}")
                    break
                
                # Clear the list of nodes to explore
                nodes_to_explore = []
                
                # Expand selected nodes and evaluate their children
                for node in selected_nodes:
                    if not node.expanded:
                        node.expand()
                        # Evaluate the children of this node
                        token_usage = evaluate_child_node_values(agent, explorer, node, token_usage)
                        # Check token usage after each evaluation
                        _, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage)
                        if limit_exceeded:
                            print(f"Reached maximum token usage during child evaluation")
                            break
                    # Add all children with values to the nodes_to_explore list
                    for child in node.children:
                        if child.value is not None:
                            nodes_to_explore.append(child)
                
                atts += 1
                
                # Update metrics after expansion
                metrics_entry = update_metrics_log(metrics_log, atts, explorer, token_usage, backtrack_count)
                metrics_entry["selected_node_value"] = best_node.value
                if best_node.adjusted_value is not None:
                    metrics_entry["selected_node_adjusted_value"] = best_node.adjusted_value
                metrics_entry["nodes_to_explore"] = len(nodes_to_explore)
                
                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
                
                if best_node.game_node.game_result() > 0:
                    print("We won!")
                    break
            
            end = time.time()
            
            # Calculate token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            api_call_count = token_stats["api_call_count"]
            
            print(f"Final State: Total Tokens: {total_tokens}, API Calls: {api_call_count}")
            print(f"Best node value: {best_node.game_node.game_result()}")
            
            outputs.append({
                "target": target,
                "available_numbers": nums,
                "final_node": best_node,
                "explorer": explorer,
                "won": best_node.game_node.game_result() > 0,
                "attempts": atts,
                "token_usage": token_usage,
                "time_taken": end - start,
                "metrics_log": metrics_log,
                "total_tokens": total_tokens,
                "api_call_count": api_call_count,
                "backtrack_count": backtrack_count,
                "iteration": iter_idx,
            })
            
            # Save progress after each iteration
            save_game_state(output_path, all_outputs, game_idx, outputs)
            print(f"Game {game_idx}, Iteration {iter_idx} saved to {output_path}")
            
        # Game is complete, ensure it's properly saved
        save_game_state(output_path, all_outputs, game_idx, outputs)
        print(f"Game {game_idx} complete - all {len(outputs)} iterations saved.")

def run_sudoku(args, model):
    """Run the Sudoku puzzle solver using ToT-BFS approach"""
    
    # Create the agent
    agent = SudokuAgent(
        model=model,
        model_name=args.model_name,
        model_type=args.model_type,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        box_width=args.sudoku_width,
        box_height=args.sudoku_height if args.sudoku_height else args.sudoku_width,
        reasoning=args.reasoning,
    )
    
    # Set up the data path for input data
    data_path = get_sudoku_data_path(
        args.data_dir, 
        args.sudoku_difficulty, 
        args.sudoku_width,
        args.sudoku_height
    )

    # Load all puzzles data
    try:
        with open(data_path, 'rb') as file:
            all_puzzles = pickle.load(file)
            puzzles = all_puzzles['puzzles']
            solutions = all_puzzles['solutions']
            print(f"Loaded {len(puzzles)} pre-generated Sudoku puzzles")
    except (FileNotFoundError, EOFError) as e:
        print(f"Error loading puzzles: {e}")
        print(f"Please generate puzzles first using sudoku_generator.py and save them to {data_path}")
        return

    # Select the batch based on batch_num and batch_size
    batch = puzzles[args.batch_num*args.batch_size:min((args.batch_num+1)*args.batch_size, len(puzzles))]
    batch_solutions = solutions[args.batch_num*args.batch_size:min((args.batch_num+1)*args.batch_size, len(solutions))]
    
    # Get output path using common utility
    output_path = get_game_output_path(args, "tot_bfs", "sudoku")
    
    # Load existing outputs using common utility
    all_outputs, processed_puzzles = load_game_state(output_path)
        
    # Process each puzzle in batch and collect responses
    for puzzle_idx, puzzle in enumerate(tqdm(batch, desc=f"Batch {args.batch_num}")):
        # Only skip if we have all iterations for this puzzle
        if puzzle_idx < len(all_outputs) and len(all_outputs[puzzle_idx]) == args.num_its:
            # Skip puzzles that have completed all iterations
            continue
            
        # Get the solution for this puzzle
        solution = batch_solutions[puzzle_idx]
        solution_board = sudoku_utils.from_py_sudoku(solution)
        
        # Initialize outputs list for this puzzle or load existing iterations
        if puzzle_idx < len(all_outputs):
            # We're resuming from a partially completed puzzle
            outputs = all_outputs[puzzle_idx]
            completed_iterations = len(outputs)
            print(f"Resuming puzzle {puzzle_idx} from iteration {completed_iterations}")
        else:
            # Starting a new puzzle
            outputs = []
            completed_iterations = 0
            
        # Run only the remaining iterations
        for iter_idx in tqdm(range(completed_iterations, args.num_its), desc=f"Puzzle {puzzle_idx}"):
            # Convert the loaded puzzle to our internal format
            board = sudoku_utils.from_py_sudoku(puzzle)
            
            print("Initial Sudoku Board:")
            print(sudoku_utils.board_to_string(board))
            
            # Create initial node
            root_sudoku_node = SudokuNode.create_initial_node(board)
            root_node = PathNode(root_sudoku_node)
            root_node.expand()
            explorer = Explorer(root_node)
            
            # Initial setup
            start = time.time()
            token_usage = []
            best_node = root_node
            atts = 0

            # Initialize metrics tracking
            metrics_log = initialize_metrics_tracking(explorer)
            
            # Track correct moves for calculating accuracy
            correct_moves = 0
            total_moves = 0

            # Track backtracking
            backtrack_count = 0

            # Evaluate root node's children
            token_usage = evaluate_child_moves(agent, explorer, root_node, token_usage)
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}")

            # Create initial list of nodes to explore from root's children
            nodes_to_explore = []
            for child in root_node.children:
                if child.value is not None:
                    nodes_to_explore.append(child)

            # Main ToT-BFS loop
            while True:
                # Check token usage limit
                total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage)
                if limit_exceeded:
                    print(f"Reached maximum token usage: {total_tokens}")
                    break
                
                if not nodes_to_explore:
                    print("No more nodes to explore!")
                    break
                
                print(f"Current nodes to explore: {len(nodes_to_explore)}")
                
                # Sort nodes by their values and take top N (highest values last in ascending order)
                sorted_nodes = create_sorted_nodes_by_value(nodes_to_explore)
                selected_nodes = sorted_nodes[-args.top_n:]  # Take top N nodes
                best_node = selected_nodes[-1]  # Best node is the last one (highest value)
                
                print(f"Selected {len(selected_nodes)} nodes to expand")
                
                # Check if best node's move is correct for move accuracy tracking
                if hasattr(best_node.game_node, 'last_move') and best_node.game_node.last_move:
                    row, col, number = best_node.game_node.last_move
                    if solution_board[row][col] == number:
                        correct_moves += 1
                    total_moves += 1


                
                # Expand selected nodes and evaluate their children
                nodes_to_explore = []
                for node in selected_nodes:
                    if not node.expanded:
                        node.expand()
                        # Evaluate the children of this node
                        token_usage = evaluate_child_moves(agent, explorer, node, token_usage)
                        # Check token usage after each evaluation
                        _, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage)
                        if limit_exceeded:
                            print(f"Reached maximum token usage during child evaluation")
                            break
                    for child in node.children:
                        if child.value is not None:
                            nodes_to_explore.append(child)

                atts += 1
                
                # Calculate and update metrics
                move_accuracy_data, board_accuracy_data = calculate_sudoku_accuracy_metrics(
                    board, 
                    best_node.game_node.board, 
                    solution_board, 
                    correct_moves, 
                    total_moves
                )

                print(f"Board Accuracy: {board_accuracy_data}")

                # Update metrics log
                metrics_entry = update_sudoku_metrics_log(
                    metrics_log=metrics_log, 
                    step=atts, 
                    explorer=explorer, 
                    move_accuracy_data=move_accuracy_data,
                    board_accuracy_data=board_accuracy_data,
                    token_usage=token_usage
                )
                metrics_entry["selected_node_value"] = best_node.value
                if best_node.adjusted_value is not None:
                    metrics_entry["selected_node_adjusted_value"] = best_node.adjusted_value
                metrics_entry["nodes_to_explore"] = len(nodes_to_explore)
                metrics_entry["backtrack_count"] = backtrack_count
                metrics_entry["expanded_nodes"] = explorer.expanded_nodes_count()
                
                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")

                # Check if we've solved the puzzle with best_node
                if sudoku_utils.is_board_solved(best_node.game_node.board):
                    print("We won!")
                    break

                # Check if best node is terminal
                if best_node.game_node.is_game_over():
                    print(f"Found terminal node with value: {best_node.game_node.game_result()}")
                    break
            
            end = time.time()
            
            # Calculate final token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            api_call_count = token_stats["api_call_count"]
            
            # Get final board state
            final_board = best_node.game_node.board
            is_solved = sudoku_utils.is_board_solved(final_board)
            
            print("Final board state:")
            print(sudoku_utils.board_to_string(final_board))
            print(f"Puzzle solved: {is_solved}")
            
            # Calculate final metrics
            move_accuracy = sudoku_utils.calculate_move_accuracy(correct_moves, total_moves)
            print(f"Move accuracy: {move_accuracy:.2f}% ({correct_moves}/{total_moves} correct moves)")
            
            # Get board accuracy metrics
            board_accuracy_metrics = sudoku_utils.calculate_board_accuracy_metrics(
                board, final_board, solution_board
            )
            
            print(f"Final State: Total Tokens: {total_tokens}, API Calls: {api_call_count}")
            print(f"Board accuracy: {board_accuracy_metrics['board_accuracy']:.2f}%")
            print(f"Board completion: {board_accuracy_metrics['completion_percentage']:.2f}%")
            
            # Find the best board state from the entire tree
            best_board, best_score, best_metrics = find_best_board_state(explorer, board, solution_board)
            if best_board is not None:
                print(f"Best board found with score: {best_score:.2f}")
                print(f"Board accuracy: {best_metrics['board_accuracy']:.2f}%")
                print(f"Completion percentage: {best_metrics['completion_percentage']:.2f}%")
            
            outputs.append({
                "initial_board": board.copy(),
                "final_board": final_board.copy(),
                "solution_board": solution_board.copy(),
                "best_board": best_board.copy() if best_board is not None else None,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "final_node": best_node,
                "explorer": explorer,
                "won": best_node.game_node.game_result() > 0,
                "attempts": atts,
                "token_usage": token_usage,
                "time_taken": end - start,
                "metrics_log": metrics_log,
                "move_accuracy": move_accuracy,
                "correct_moves": correct_moves,
                "total_moves": total_moves,
                "board_accuracy": board_accuracy_metrics["board_accuracy"],
                "correct_cells": board_accuracy_metrics["correct_cells"],
                "filled_cells": board_accuracy_metrics["filled_cells"],
                "total_empty_cells": board_accuracy_metrics["total_empty_cells"],
                "completion_percentage": board_accuracy_metrics["completion_percentage"],
                "total_tokens": total_tokens,
                "api_call_count": api_call_count,
                "backtrack_count": backtrack_count,
                "expanded_nodes": explorer.expanded_nodes_count(),
                "iteration": iter_idx,
            })
            
            # Save progress after each iteration
            save_game_state(output_path, all_outputs, puzzle_idx, outputs)
            print(f"Puzzle {puzzle_idx}, Iteration {iter_idx} saved to {output_path}")
            
        # Puzzle is complete, ensure it's properly saved
        save_game_state(output_path, all_outputs, puzzle_idx, outputs)
        print(f"Puzzle {puzzle_idx} complete - all {len(outputs)} iterations saved.")

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the OpenAI client
    model = create_openai_client(args.model_type, args.timeout, args.is_azure)
    
    # Run the appropriate game type
    if args.game_type == 'countdown':
        run_countdown(args, model)
    elif args.game_type == 'sudoku':
        run_sudoku(args, model)
    else:
        raise ValueError(f"Unknown game type: {args.game_type}")