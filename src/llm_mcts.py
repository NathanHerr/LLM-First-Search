import json
import random
import time
from collections import deque

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
from src.utils.common_utils import get_standard_parser
from src.utils.countdown_utils import mult_heuristic, get_countdown_data_path
from src.utils.sudoku_utils import *
from src.utils.sudoku_utils import get_sudoku_data_path
# Import common tree components
from src.utils.tree_utils import PathNode, Explorer

load_dotenv()

class MonteCarloTreeSearchNode(PathNode):
    def __init__(self, game_node, parent=None):
        super().__init__(game_node, parent)
        self.action_priors = None
        self._reward = []
        self._number_of_visits = 0

    def q(self):
        return sum(self._reward)

    def n(self):
        return self._number_of_visits

    def expand(self):
        # Override PathNode's expand method to incorporate MCTS specifics
        self.expanded = True
        
        # Expand the game node first if it has an expand method
        if hasattr(self.game_node, 'expand'):
            self.game_node.expand()
        
        # Check if this is a SudokuNode and use special expansion if it is
        if hasattr(self.game_node, 'board') and hasattr(self.game_node, 'untried_numbers'):
            self.expand_for_sudoku()
        else:
            # Use common interface for other game types (e.g., Countdown)
            possible_next_nodes = deque(self.game_node.get_legal_next_nodes())
            while possible_next_nodes:
                _, next_node = possible_next_nodes.popleft()
                child_node = MonteCarloTreeSearchNode(next_node, parent=self)
                self.children.append(child_node)

    def expand_for_sudoku(self):
        """
        Special expansion method for Sudoku that creates a node for each possible valid number
        for each empty cell, rather than just one node per empty cell.
        """
        # Make sure untried_numbers is initialized for the game node
        if not self.game_node.untried_numbers:
            self.game_node.expand()
            
        # If no empty cells left, return
        if not self.game_node.untried_numbers:
            return
            
        # Generate nodes for all valid numbers for all empty cells
        for cell, valid_numbers in self.game_node.untried_numbers.items():
            row, col = cell
            # Create a node for each valid number for this cell
            for num in valid_numbers:
                if self.game_node.is_valid_move(row, col, num):
                    new_node = self.game_node.make_move(row, col, num, True)
                    if new_node:
                        child_node = MonteCarloTreeSearchNode(new_node, parent=self)
                        self.children.append(child_node)

    def is_terminal_node(self):
        return self.game_node.is_game_over()

    def rollout(self):
        current_rollout_node = self.game_node
        while not current_rollout_node.is_game_over():
            possible_moves = current_rollout_node.get_legal_next_nodes()
            next_rollout = self.rollout_policy(possible_moves)
            current_rollout_node = next_rollout
        return current_rollout_node.game_result()

    def rollout_policy(self, possible_moves):
        possible_moves.sort()  # Sort by heuristic value
        pruned_nodes = possible_moves[1:]  # Nodes that will be pruned
        return possible_moves[:1][0][-1]

    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self._reward.append(reward)
        if self.parent:
            self.parent.backpropagate(reward)

    def best_child(self, c_value=0.5):
        total_children_visits = sum([c.n() for c in self.children])
        total_parent_visits = self.visits_to_root(self.parent) + 1
        c_param = np.log((total_children_visits + 19652 + 1) / 19652) + c_value
        choices_weights = [(c.q() / (c.n() + 1)) + c_param * p * (np.sqrt(total_parent_visits) / (1 + c.n())) for p, c
                           in zip(self.action_priors, self.children)]
        return self.children[np.argmax(choices_weights)]

    def visits_to_root(self, parent):
        # Base case: If the node is None, return 0 (since we can't add anything more)
        if parent is None:
            return 0
        # Recursively call sum_to_root on the parent_node
        parent_sum = self.visits_to_root(parent.parent)
        # Add the current node's 'n' to the sum of the parent nodes
        return self.n() + parent_sum

def set_action_priors_countdown(current_node, agent, explorer, token_usage):
    if current_node.action_priors is None:
        agent_resp = agent.ask(
            current_sequence=explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(current_node)),
            action_list=current_node.get_possible_actions(),
            query_type="prior"
        )
        # Get the operation scores dictionary from the response
        operation_scores_dict = agent_resp['resp']
        
        # Convert dictionary response to a list in the correct order for the children
        # We need to ensure the scores are in the same order as the children nodes
        all_child_priors = []
        
        # Get possible actions to match with children
        possible_actions = current_node.get_possible_actions()
        
        # Iterate through each child and find its corresponding score
        for i, child in enumerate(current_node.children):
            # Try to get the score from the dictionary using the index as string key
            score = operation_scores_dict.get(str(i), None)
            
            # If not found by index, try to use the description if available
            if score is None and hasattr(child.game_node, 'get_action_description'):
                action_desc = child.game_node.get_action_description()
                # Look for this description in the possible actions
                for j, action in possible_actions.items():
                    if action == action_desc:
                        score = operation_scores_dict.get(str(j), None)
                        break
            
            # If still not found, use a default value
            if score is None:
                score = 1.0 / len(current_node.children)
                
            all_child_priors.append(score)
        
        # Normalize the priors to ensure they sum to 1.0
        total = sum(all_child_priors)
        if total > 0:
            all_child_priors = [p / total for p in all_child_priors]
        else:
            all_child_priors = [1.0 / len(all_child_priors)] * len(all_child_priors)
            
        current_node.action_priors = all_child_priors
        token_usage.append(("prior", agent_resp['token_usage']))
    return token_usage

def set_action_priors_sudoku(current_node, agent, token_usage):
    """Set action priors for each possible move in a Sudoku node."""
    if current_node.action_priors is None:
        # Get the current board state
        current_board_str = sudoku_utils.board_to_string(current_node.game_node.board)
        
        # Collect all possible moves from the current node's untried numbers
        moves_list = {}
        move_index = 0
        move_details = {}  # Map to track details of each move
        
        # Get all possible moves from untried numbers
        if hasattr(current_node.game_node, 'untried_numbers'):
            for (row, col), numbers in current_node.game_node.untried_numbers.items():
                for number in numbers:
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
            # If no moves, set equal priors for all children
            if current_node.children:
                equal_prior = 1.0 / len(current_node.children)
                current_node.action_priors = [equal_prior] * len(current_node.children)
            return token_usage
        
        # Get prior values for each move from the agent
        agent_resp = agent.ask(
            query_type="prior",
            current_board=current_board_str,
            action_list=moves_list
        )
        token_usage.append(("prior", agent_resp['token_usage']))
        
        # Get prior probabilities for each move
        move_priors = agent_resp['resp']
        
        # Now map these move-level priors to all the children nodes
        all_child_priors = []
        
        # Create a mapping of (row, col, number) to their prior probabilities
        move_to_prior = {}
        for move_idx, prior_value in move_priors.items():
            move_info = move_details[int(move_idx)]
            move_key = (move_info['row'], move_info['col'], move_info['number'])
            move_to_prior[move_key] = prior_value
        
        # Assign priors to each child based on its move
        for child in current_node.children:
            child_move = None
            if hasattr(child.game_node, 'last_move') and child.game_node.last_move:
                row, col, number = child.game_node.last_move
                child_move = (row, col, number)
            
            # If we have a mapping for this move, use its prior
            if child_move and child_move in move_to_prior:
                all_child_priors.append(move_to_prior[child_move])
            else:
                # Fallback if we can't find the move in our mapping
                all_child_priors.append(1.0 / len(current_node.children))
        
        # Normalize the priors if needed
        if all_child_priors:
            total = sum(all_child_priors)
            if total > 0:
                all_child_priors = [p / total for p in all_child_priors]
            else:
                all_child_priors = [1.0 / len(all_child_priors)] * len(all_child_priors)
        
        current_node.action_priors = all_child_priors
    
    return token_usage

def tree_policy_puct(agent, explorer, token_usage, game_type, c_value=0.5):
    current_node = explorer.explorer_root
    while not current_node.is_terminal_node():
        if not current_node.expanded:
            current_node.expand()
            return current_node, token_usage
        else:
            if game_type == 'countdown':
                token_usage = set_action_priors_countdown(current_node, agent, explorer, token_usage)
            elif game_type == 'sudoku':
                token_usage = set_action_priors_sudoku(current_node, agent, token_usage)
            current_node = current_node.best_child(c_value)
    
    # Expand the terminal node before returning it
    if not current_node.expanded:
        current_node.expand()
    return current_node, token_usage

def update_mcts_metrics_log(metrics_log, atts, explorer, token_usage, backtrack_count, estimated_value=None):
    """Update metrics log with MCTS-specific data"""
    metrics_entry = update_metrics_log(metrics_log, atts, explorer, token_usage, backtrack_count)
    if estimated_value is not None:
        metrics_entry["estimated_value"] = estimated_value
    return metrics_entry

def parse_args():
    parser = get_standard_parser()
    # Add MCTS-specific arguments
    parser.add_argument('--c_value', type=float, default=0.5,
                      help='The exploration constant (c) for MCTS UCB calculation')
    return parser.parse_args()

def run_countdown(args, model):
    """Run the Countdown puzzle solver using MCTS approach"""
    
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
    output_path = get_game_output_path(args, "mcts", "countdown")
    
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
            root_node = MonteCarloTreeSearchNode(game_node=CountdownNode(0, None, nums, [], mult_heuristic, target))
            root_node.expand()
            explorer = Explorer(root_node)
            token_usage = []
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            backtrack_count = 0
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
            current_node = root_node
            atts = 0
            start = time.time()
            
            # Initialize metrics tracking
            metrics_log = initialize_metrics_tracking(explorer)
            # Add estimated value to the first log entry
            if metrics_log and len(metrics_log) > 0:
                metrics_log[0]["estimated_value"] = 0  # Initialize with zero
            
            # Track backtracking - in MCTS we backtrack during selection phase
            
            # Main MCTS Loop
            while True:
                # MCTS step 1: Selection - use PUCT to select a node
                current_node, token_usage = tree_policy_puct(agent, explorer, token_usage, "countdown", args.c_value)
                # Check if we've reached the maximum token usage
                total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage)
                if limit_exceeded:
                    break
                
                if current_node.is_terminal_node():
                    if current_node.game_node.game_result() > 0:
                        current_node.backpropagate(1)
                        backtrack_count += 1
                        break
                    else:
                        current_node.backpropagate(-1)
                else:
                    # Use query_agent utility function for value queries
                    res, usage = query_agent(
                        agent,
                        query_type="value",
                        current_sequence=explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(current_node)),
                        action_list=current_node.get_possible_actions()
                    )
                    token_usage.append(usage)
                    
                    # Check if we've exceeded the token usage limit
                    total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "value query")
                    if limit_exceeded:
                        break
                    
                    # Backpropagate the value
                    current_node.backpropagate(res["resp"])

                atts += 1
                backtrack_count += 1
                # Update metrics with MCTS-specific data
                metrics_entry = update_mcts_metrics_log(
                    metrics_log, 
                    atts, 
                    explorer, 
                    token_usage, 
                    backtrack_count,
                )
                
                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
                
                if current_node.game_node.game_result() > 0:
                    print("We won!")
                    break

            # Calculate token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            api_call_count = token_stats["api_call_count"]

            end = time.time()
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
    """Run the Sudoku puzzle solver using MCTS approach"""
    
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
    output_path = get_game_output_path(args, "mcts", "sudoku")
    
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
            root_node = MonteCarloTreeSearchNode(root_sudoku_node)
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
            
            # Track backtracking - in MCTS we backtrack during selection phase
            backtrack_count = 0
            
            # For storing board state for display
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Initial State: Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
            
            while True:
                current_node, token_usage = tree_policy_puct(agent, explorer, token_usage, 'sudoku', args.c_value)
                estimated_value = None
                # Check if we've exceeded the token usage limit after prior calls
                total_tokens, limit_exceeded = calculate_and_check_token_usage(token_usage, args.max_token_usage, "PUCT calls")
                if limit_exceeded:
                    break
                
                if current_node.is_terminal_node():
                    if current_node.game_node.game_result() > 0:
                        current_node.backpropagate(1)
                        # Increment backtrack count during selection phase
                        backtrack_count += 1
                        break
                    else:
                        current_node.backpropagate(-1)
                else:
                    # Get the current board state
                    current_board_str = sudoku_utils.board_to_string(current_node.game_node.board)
                    action_list = current_node.get_possible_actions()
                    
                    # Ask agent for value estimation
                    full_resp = agent.ask(
                        query_type="value",
                        current_board=current_board_str,
                        action_list=action_list
                    )
                    
                    # Track token usage
                    token_usage.append(("value", full_resp["token_usage"]))
                    
                    # Check if we've exceeded the token usage limit
                    if args.max_token_usage is not None:
                        total_tokens = sum(usage.get("total_tokens", 0) for _, usage in token_usage)
                        if total_tokens >= args.max_token_usage:
                            print(f"Reached maximum token usage after value call: {total_tokens}")
                            break
                    reward = 0
                    # Update last move stats if available
                    if hasattr(current_node.game_node, 'last_move') and current_node.game_node.last_move:
                        row, col, number = current_node.game_node.last_move
                        if solution_board[row][col] == number:
                            reward = 100
                            correct_moves += 1
                        else:
                            reward = 0
                        total_moves += 1
                    
                    # Backpropagate the value
                    current_node.backpropagate(full_resp["resp"])
                    estimated_value = full_resp["resp"]
                
                atts += 1
                backtrack_count += 1
                # Calculate and update metrics
                move_accuracy_data, board_accuracy_data = calculate_sudoku_accuracy_metrics(
                    board, 
                    current_node.game_node.board, 
                    solution_board, 
                    correct_moves, 
                    total_moves
                )
                print(f"Board Accuracy: {board_accuracy_data}")

                # Update metrics log with MCTS-specific data
                metrics_entry = update_mcts_metrics_log(
                    metrics_log, 
                    atts, 
                    explorer, 
                    token_usage, 
                    backtrack_count,
                    estimated_value=estimated_value if not current_node.is_terminal_node() else None
                )
                metrics_entry.update(move_accuracy_data)
                metrics_entry.update(board_accuracy_data)
                
                # Print progress with token information
                token_stats = calculate_token_stats(token_usage)
                total_tokens = token_stats["total_tokens"]
                print(f"Iteration: {atts}, Expanded Nodes: {explorer.expanded_nodes_count()}, Total Tokens: {total_tokens}, Backtracks: {backtrack_count}")
                
                if current_node.game_node.game_result() > 0:
                    print("We won!")
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
            board_accuracy_metrics = sudoku_utils.calculate_board_accuracy_metrics(
                board, final_board, solution_board
            )
            
            # Calculate token stats
            token_stats = calculate_token_stats(token_usage)
            total_tokens = token_stats["total_tokens"]
            print(f"Final State: Total Backtracks: {backtrack_count}, Total Tokens: {total_tokens}")
            api_call_count = token_stats["api_call_count"]
            
            # Extract metrics for easier access
            correct_cells = board_accuracy_metrics["correct_cells"]
            filled_cells = board_accuracy_metrics["filled_cells"]
            total_empty_cells = board_accuracy_metrics["total_empty_cells"]
            board_accuracy = board_accuracy_metrics["board_accuracy"]
            completion_percentage = board_accuracy_metrics["completion_percentage"]
            
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
                "total_tokens": total_tokens,
                "api_call_count": api_call_count,
                "backtrack_count": backtrack_count,  # Add backtrack count to output
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