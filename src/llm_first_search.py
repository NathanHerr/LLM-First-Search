import random
import time
from copy import copy

from tqdm import tqdm

import src.utils.sudoku_utils as sudoku_utils
# Countdown imports
from src.countdown_game.countdown_agent import CountdownAgent
from src.countdown_game.countdown_node import CountdownNode
# Sudoku imports
from src.sudoku_game.sudoku_agent import SudokuAgent
from src.sudoku_game.sudoku_node import SudokuNode
# Import utility functions
from src.utils.common_utils import *
from src.utils.countdown_utils import mult_heuristic, evaluate_child_node_values, \
    evaluate_countdown_node_value, get_countdown_data_path
# Import shared components
# Import common tree components
from src.utils.tree_utils import PathNode, Explorer
from src.utils.sudoku_utils import *
from src.utils.common_utils import create_sorted_nodes_by_value, get_standard_parser
from src.utils.sudoku_utils import evaluate_child_moves, evaluate_sudoku_node_value, get_sudoku_data_path

"""
LLM Explorer V3 Implementation - Works with both Countdown and Sudoku

This file implements an advanced exploratory approach to solving puzzles where
the agent decides whether to explore new paths or continue with the current one.
Unlike V2, when nodes are expanded, we ask the LLM to evaluate their value 
and use these values to prioritize which paths to explore.
Support for both Countdown and Sudoku puzzle types.
"""

load_dotenv()

def parse_args():
    # Get the standard parser that now includes arguments for both games
    parser = get_standard_parser()
    return parser.parse_args()

def run_countdown(args, model):
    """Run the Countdown puzzle solver using LLM First Search approach"""
    
    # Create the agent - directly use CountdownAgent
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
        import json
        all_games = json.load(file)

    # Select the batch based on batch_num and batch_size
    batch = all_games[args.batch_num*args.batch_size:(args.batch_num+1)*args.batch_size]
    
    # Get output path using common utility
    output_path = get_game_output_path(args, "lfs", "countdown")
    
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

            # Initialize tracking variables
            atts = 0
            token_usage = []
            start = time.time()
            
            # Initialize metrics tracking with value tracking
            metrics_log = initialize_metrics_tracking(explorer)
            # Add estimated value to the first log entry
            if metrics_log and len(metrics_log) > 0:
                metrics_log[0]["estimated_value"] = 0  # Initialize with default value
            
            # Track backtracking - in ExplorerV3 we backtrack when switching nodes 
            backtrack_count = 0
            
            # Evaluate the root node's value
            node_value, token_usage = evaluate_countdown_node_value(agent, explorer, root_node, token_usage)
            
            # Expand the root node if not already expanded
            if not root_node.expanded:
                root_node.expand()
            
            # Evaluate the values of the root's children directly
            if root_node.children:
                token_usage = evaluate_child_node_values(agent, explorer, root_node, token_usage)
            
            # Initialize the current node to the root node
            current_node = root_node
            
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
            
            # Continue until we find a solution or run out of paths
            while True:
                # Decide whether to explore or continue based on the queue and agent
                should_explore = current_node.game_node.is_game_over()
                
                # Only ask about exploration if there are frontier nodes to explore
                if not should_explore:
                    if explorer.frontier_nodes:
                        # Get the current state sequence for the explore decision
                        current_sequence = explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(current_node))
                        
                        # Ask the agent if it wants to explore a new path or continue with the current one
                        res, usage = query_agent(
                            agent,
                            query_type="explore", 
                            current_sequence=current_sequence, 
                            action_list=current_node.get_possible_actions()
                        )
                        token_usage.append(usage)
                        
                        # Check if we've exceeded the token usage limit after the explore call
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "explore query")
                        if limit_exceeded:
                            break
                        
                        should_explore = res["resp"]
                    else:
                        print("No frontier nodes available to explore, continuing with current path")
                else:
                    if not explorer.frontier_nodes:
                        print("No more paths to explore!")
                        break
                
                # Process exploration decision
                # Use nodes that have unexpanded children and choose from already valued nodes
                if should_explore and explorer.frontier_nodes:
                    # Increment backtrack count when we explore a new path
                    backtrack_count += 1
                    
                    # Evaluate any nodes with unexpanded children that don't have values yet
                    for node in explorer.frontier_nodes:
                        # Evaluate this node if it doesn't have a value yet
                        if node.value is None:
                            node_value, token_usage = evaluate_countdown_node_value(agent, explorer, node, token_usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "value query")
                        if limit_exceeded:
                            break
                    
                    # Create a sorted list of nodes by value
                    sorted_nodes = create_sorted_nodes_by_value(explorer.frontier_nodes, use_adjusted_value=False)
                    
                    if sorted_nodes:
                        # Select the highest valued node (last in the sorted list)
                        current_node = sorted_nodes[-1]
                        print(f"Agent decided to explore a new path: {current_node}")
                        
                        # Make sure it's expanded
                        if not current_node.expanded:
                            current_node.expand()
                            
                            # Directly evaluate the values of the children nodes
                            token_usage = evaluate_child_node_values(agent, explorer, current_node, token_usage)
                            
                            # Check if we've exceeded the token usage limit
                            total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "child_values query")
                            if limit_exceeded:
                                break
                    else:
                        # We don't have any valued nodes to explore
                        print("No valued nodes to explore, continuing with current path")
                else:
                    # Agent chose to continue with the current path
                    print("Agent decided to continue with the current path")
                    # No additional incrementation for continuing on the same path
                    
                # Process the current node path
                # If the current node is a terminal node, we need to try a different path
                if not current_node.game_node.is_game_over():
                    # Instead of asking for the next action, directly select the child with the highest value
                    current_sequence = explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(current_node))
                    possible_next_nodes = copy(current_node.children)
                    
                    # Find the child with the highest value
                    best_child = None
                    best_value = -1
                    best_index = -1
                    
                    for i, child in enumerate(possible_next_nodes):
                        # If a child doesn't have a value (shouldn't happen), use 0
                        child_value = child.value if child.value is not None else 0
                        if child_value > best_value:
                            best_value = child_value
                            best_child = child
                            best_index = i
                    
                    if best_child is None:
                        # This should never happen if we've evaluated all children correctly
                        print("Warning: No child found with a value. This shouldn't happen.")
                        # Fallback: Make an action query as before
                        res, usage = query_agent(agent, query_type="action", current_sequence=current_sequence, action_list=current_node.get_possible_actions())
                        token_usage.append(usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "action query")
                        if limit_exceeded:
                            break
                        
                        selected_action = res["resp"]
                        selected_node = possible_next_nodes.pop(selected_action)
                    else:
                        # Use the child with the highest value
                        print(f"Selecting child {best_child} with value {best_value}")
                        selected_node = best_child
                    
                    # Expand the selected node and evaluate its children
                    if not selected_node.expanded:
                        selected_node.expand()
                    else:
                        print(f"This shouldn't happen")
                    
                    # Directly evaluate the values of the children nodes if there are any
                    if selected_node.children:
                        token_usage = evaluate_child_node_values(agent, explorer, selected_node, token_usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "child_values query")
                        if limit_exceeded:
                            break
                    
                    # Move to the selected node
                    current_node = selected_node
                    
                    # Update metrics with estimated value
                    atts += 1
                    metrics_entry = update_metrics_log(metrics_log, atts, explorer, token_usage, backtrack_count)
                    metrics_entry["selected_node_value"] = current_node.value
                    if current_node.adjusted_value is not None:
                        metrics_entry["selected_node_adjusted_value"] = current_node.adjusted_value
                    metrics_entry["frontier_size"] = len(explorer.frontier_nodes)
                    
                    # Record whether we used direct selection or action query
                    if best_child is not None:
                        metrics_entry["selection_method"] = "direct_value"
                    else:
                        metrics_entry["selection_method"] = "action_query"

                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
                
                # Check if we've won or if we've run out of paths
                if current_node.game_node.is_game_over():
                    if current_node.game_node.game_result() > 0:
                        print("We won!")
                        break
                    else:
                        print("That didnt work, lets try another path!")
                
                # Check if we've run out of paths to explore  
                if not explorer.frontier_nodes:
                    print("No more paths to explore!")
                    break
            end = time.time()
            
            # Calculate token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            api_call_count = token_stats["api_call_count"]
            
            print(f"Final State: Total Backtracks: {backtrack_count}, Total Tokens: {total_tokens}")
            
            outputs.append({
                "target": target,
                "available_numbers": nums,
                "final_node": current_node,
                "explorer": explorer,
                "won": current_node.game_node.game_result() > 0,
                "attempts": atts,
                "token_usage": token_usage,
                "time_taken": end - start,
                "metrics_log": metrics_log,
                "explore_enabled": True,
                "total_tokens": total_tokens,
                "api_call_count": api_call_count,
                "backtrack_count": backtrack_count,  # Add backtrack count to output
                "iteration": iter_idx,  # Add iteration number for tracking
            })
            
            # Save progress after each iteration using common utility
            save_game_state(output_path, all_outputs, game_idx, outputs)
            print(f"Game {game_idx}, Iteration {iter_idx} saved to {output_path}")
                
        # Game is complete, ensure it's properly saved
        save_game_state(output_path, all_outputs, game_idx, outputs)
        print(f"Game {game_idx} complete - all {len(outputs)} iterations saved.")

def run_sudoku(args, model):
    """Run the Sudoku puzzle solver using explorer v3 approach"""
    
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
    output_path = get_game_output_path(args, "lfs", "sudoku")
    
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
            
            # Calculate total empty cells once at the beginning
            total_empty_cells = sudoku_utils.get_total_empty_cells(board)
            
            # Create initial node with our SudokuNode
            root_sudoku_node = SudokuNode.create_initial_node(board)
            print(f"Created initial node with {total_empty_cells} empty cells")
            
            # Create the path node and explorer
            root_node = PathNode(root_sudoku_node)
            root_node.expand()
            explorer = Explorer(root_node)
            
            # Initial setup
            start = time.time()
            token_usage = []
            atts = 0
            current_node = root_node
            
            # Initialize metrics tracking
            metrics_log = initialize_metrics_tracking(explorer)
            # Add estimated value to the first log entry
            if metrics_log and len(metrics_log) > 0:
                metrics_log[0]["estimated_value"] = 0  # Initialize with zero
            
            # Track correct moves for calculating accuracy
            correct_moves = 0
            total_moves = 0
            
            # Track backtracking
            backtrack_count = 0
            
            # Evaluate the root node's value
            node_value, token_usage = evaluate_sudoku_node_value(agent, root_node, token_usage)
            
            # Evaluate the values of the root's children directly
            if root_node.children:
                token_usage = evaluate_child_moves(agent, explorer, root_node, token_usage)
            
            # Initialize the current node to the root node
            current_node = root_node
            
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
            
            # Continue until we find a solution or run out of paths
            while True:
                # Decide whether to explore or continue based on the queue and agent
                should_explore = current_node.game_node.is_game_over()
                
                # Only ask about exploration if there are frontier nodes to explore
                if not should_explore:
                    if explorer.frontier_nodes:
                        # Get the current board state for the explore decision
                        current_board_str = sudoku_utils.board_to_string(current_node.game_node.board)
                        
                        # Ask the agent if it wants to explore a new path or continue with the current one
                        res, usage = query_agent(
                            agent,
                            query_type="explore", 
                            current_board=current_board_str,
                            empty_cells=current_node.get_possible_actions()
                        )
                        token_usage.append(usage)
                        
                        # Check if we've exceeded the token usage limit after the explore call
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "explore query")
                        if limit_exceeded:
                            break
                        
                        should_explore = res["resp"]
                    else:
                        print("No frontier nodes available to explore, continuing with current path")
                else:
                    if not explorer.frontier_nodes:
                        print("No more paths to explore!")
                        break
                
                # Process exploration decision
                # Use nodes that have unexpanded children and choose from already valued nodes
                if should_explore and explorer.frontier_nodes:
                    # Increment backtrack count when we explore a new path
                    backtrack_count += 1
                    
                    # Evaluate any nodes with unexpanded children that don't have values yet
                    for node in explorer.frontier_nodes:
                        # Evaluate this node if it doesn't have a value yet
                        if node.value is None:
                            node_value, token_usage = evaluate_sudoku_node_value(agent, node, token_usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "value query")
                        if limit_exceeded:
                            break
                    
                    # Create a sorted list of nodes by value
                    sorted_nodes = create_sorted_nodes_by_value(explorer.frontier_nodes, use_adjusted_value=False)
                    
                    if sorted_nodes:
                        # Select the highest valued node (last in the sorted list)
                        current_node = sorted_nodes[-1]
                        print(f"Agent decided to explore a new path: {current_node}")
                        if hasattr(current_node.game_node, 'last_move') and current_node.game_node.last_move:
                            row, col, number = current_node.game_node.last_move
                            if solution_board[row][col] == number:
                                correct_moves += 1
                            total_moves += 1
                        # Make sure it's expanded
                        if not current_node.expanded:
                            current_node.expand()
                            
                            # Directly evaluate the values of the children nodes
                            token_usage = evaluate_child_moves(agent, explorer, current_node, token_usage)
                            
                            # Check if we've exceeded the token usage limit
                            total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "child_moves query")
                            if limit_exceeded:
                                break
                    else:
                        # We don't have any valued nodes to explore
                        print("No valued nodes to explore, continuing with current path")
                else:
                    # Agent chose to continue with the current path
                    print("Agent decided to continue with the current path")
                    # No additional incrementation for continuing on the same path
                    
                # Process the current node path
                # If the current node is a terminal node, we need to try a different path
                if not current_node.game_node.is_game_over():
                    # Instead of asking for the next action, directly select the child with the highest value
                    possible_next_nodes = copy(current_node.children)
                    
                    # Find the child with the highest value
                    best_child = None
                    best_value = -1
                    best_index = -1
                    
                    for i, child in enumerate(possible_next_nodes):
                        # If a child doesn't have a value (shouldn't happen), use 0
                        child_value = child.value if child.value is not None else 0
                        if child_value > best_value:
                            best_value = child_value
                            best_child = child
                            best_index = i
                    
                    if best_child is None:
                        # This should never happen if we've evaluated all children correctly
                        print("Warning: No child found with a value. This shouldn't happen.")
                        # Fallback: Make an action query as before
                        current_board_str = sudoku_utils.board_to_string(current_node.game_node.board)
                        res, usage = query_agent(
                            agent,
                            query_type="action",
                            current_board=current_board_str,
                            empty_cells=current_node.get_possible_actions()
                        )
                        token_usage.append(usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "action query")
                        if limit_exceeded:
                            break
                        
                        selected_action = res["resp"]
                        selected_node = possible_next_nodes.pop(selected_action)
                    else:
                        # Use the child with the highest value
                        print(f"Selecting child {best_child} with value {best_value}")
                        selected_node = best_child

                    if hasattr(selected_node.game_node, 'last_move') and selected_node.game_node.last_move:
                        row, col, number = selected_node.game_node.last_move
                        if solution_board[row][col] == number:
                            correct_moves += 1
                        total_moves += 1

                    # Expand the selected node and evaluate its children
                    if not selected_node.expanded:
                        selected_node.expand()
                    else:
                        print(f"This shouldn't happen")
                    
                    # Directly evaluate the values of the children nodes if there are any
                    if selected_node.children:
                        token_usage = evaluate_child_moves(agent, explorer, selected_node, token_usage)
                        
                        # Check if we've exceeded the token usage limit
                        total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "child_moves query")
                        if limit_exceeded:
                            break
                    
                    # Move to the selected node
                    current_node = selected_node

                    # Update metrics with estimated value
                    atts += 1
                    
                    # Calculate and update metrics
                    move_accuracy_data = {
                        "move_accuracy": sudoku_utils.calculate_move_accuracy(correct_moves, total_moves),
                        "correct_moves": correct_moves,
                        "total_moves": total_moves
                    }
                    
                    # Calculate board accuracy metrics
                    board_accuracy_data = sudoku_utils.calculate_board_accuracy_metrics(
                        board, current_node.game_node.board, solution_board
                    )
                    print(f"Board Accuracy: {board_accuracy_data}")

                    # Update metrics log with Sudoku-specific metrics
                    metrics_entry = update_sudoku_metrics_log(
                        metrics_log=metrics_log, 
                        step=atts, 
                        explorer=explorer, 
                        move_accuracy_data=move_accuracy_data,
                        board_accuracy_data=board_accuracy_data,
                        token_usage=token_usage
                    )
                    
                    # Add V3-specific metrics
                    metrics_entry["selected_node_value"] = current_node.value
                    if current_node.adjusted_value is not None:
                        metrics_entry["selected_node_adjusted_value"] = current_node.adjusted_value
                    metrics_entry["frontier_size"] = len(explorer.frontier_nodes)
                    
                    # Record whether we used direct selection or action query
                    if best_child is not None:
                        metrics_entry["selection_method"] = "direct_value"
                    else:
                        metrics_entry["selection_method"] = "action_query"

                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
                
                # Check if we've won or if we've run out of paths
                if current_node.game_node.is_game_over():
                    if current_node.game_node.game_result() > 0:
                        print("We won!")
                        break
                    else:
                        print("That didnt work, lets try another path!")
                
                # Check if we've run out of paths to explore  
                if not explorer.frontier_nodes:
                    print("No more paths to explore!")
                    break
            end = time.time()
            
            # Check if the puzzle is solved
            final_board = current_node.game_node.board
            is_solved = sudoku_utils.is_board_solved(final_board)
            
            print("Final board state:")
            print(sudoku_utils.board_to_string(final_board))
            print(f"Puzzle solved: {is_solved}")
            
            # Calculate move accuracy
            move_accuracy = sudoku_utils.calculate_move_accuracy(correct_moves, total_moves)
            print(f"Move accuracy: {move_accuracy:.2f}% ({correct_moves}/{total_moves} correct moves)")
            
            # Get board accuracy metrics
            move_accuracy_data, board_accuracy_data  = calculate_sudoku_accuracy_metrics(
                board, final_board, solution_board, correct_moves, total_moves
            )
            
            # Calculate token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Final State: Total Backtracks: {backtrack_count}, Total Tokens: {total_tokens}")
            api_call_count = token_stats["api_call_count"]
            
            # Extract metrics for easier access
            correct_cells = board_accuracy_data["correct_cells"]
            filled_cells = board_accuracy_data["filled_cells"]
            total_empty_cells = board_accuracy_data["total_empty_cells"]
            board_accuracy = board_accuracy_data["board_accuracy"]
            completion_percentage = board_accuracy_data["completion_percentage"]
            
            print(f"Empty cells: {total_empty_cells}")
            print(f"Board completion: {completion_percentage:.2f}% ({filled_cells}/{total_empty_cells} empty cells filled)")
            print(f"Board accuracy: {board_accuracy:.2f}% ({correct_cells}/{filled_cells} filled cells correct)")
            
            # Print incorrect cells for debugging
            if filled_cells > correct_cells:
                incorrect_cells = sudoku_utils.get_incorrect_cells(board, final_board, solution_board)
                print("Incorrect cells (row, col, agent's value, correct value):")
                for row, col, agent_value, correct_value in incorrect_cells:
                    print(f"  ({row}, {col}): {agent_value} should be {correct_value}")
            
            # Find the best board state from the entire tree
            best_board, best_score, best_metrics = find_best_board_state(explorer, board, solution_board)
            if best_board is not None:
                print(f"Best board found with score: {best_score:.2f}")
                print(f"Board accuracy: {best_metrics['board_accuracy']:.2f}%")
                print(f"Completion percentage: {best_metrics['completion_percentage']:.2f}%")
            
            # Collect results
            outputs.append({
                "initial_board": board.copy(),
                "final_board": final_board.copy(),
                "solution_board": solution_board.copy(),
                "best_board": best_board.copy() if best_board is not None else None,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "final_node": current_node,
                "explorer": explorer,
                "won": current_node.game_node.game_result() > 0,
                "attempts": atts,
                "token_usage": token_usage,
                "time_taken": end - start,
                "metrics_log": metrics_log,
                "move_accuracy": move_accuracy,
                "correct_moves": correct_moves,
                "total_moves": total_moves,
                "board_accuracy": board_accuracy,
                "correct_cells": correct_cells,
                "filled_cells": filled_cells,
                "total_empty_cells": total_empty_cells,
                "completion_percentage": completion_percentage,
                "backtrack_count": backtrack_count,
                "total_tokens": total_tokens,
                "api_call_count": api_call_count,
                "iteration": iter_idx,  # Add iteration number for tracking
            })
            
            # Save progress after each iteration using common utility
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
