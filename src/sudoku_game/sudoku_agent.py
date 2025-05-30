import regex
import tenacity
import openai
import numpy as np
from src.base_game.base_agent import BaseAgent, BaseInstructions
from src.utils.common_utils import check_json_list
from src.utils.sudoku_utils import board_to_string

class SudokuInstructions(BaseInstructions):
    """
    Class to store all the prompts and instructions for the Sudoku problem.
    
    This class centralizes all the templates used for interacting with the LLM,
    including system instructions, user requests, response formats, and error messages.
    """
    def __init__(self):
        # System instructions for MCTS
        super().__init__()
        self.system_instruction_prior = """
You are helping solve Sudoku puzzles using a tree-based search approach. Sudoku is a puzzle where you fill a grid with numbers 1 through {grid_size} so that each row, column, and box has no repeated numbers.

For this {grid_size}x{grid_size} Sudoku grid, the boxes are {box_width}x{box_height} in size. Each row, column, and box must contain all numbers from 1 to {grid_size} without repetition. This means:
1. Each row must contain each number from 1 to {grid_size} exactly once
2. Each column must contain each number from 1 to {grid_size} exactly once
3. Each {box_width}x{box_height} box must contain each number from 1 to {grid_size} exactly once

These constraints create a logical puzzle where placing a number in a cell immediately restricts what numbers can be placed in other cells in the same row, column, and box.

Board Structure:
- The Sudoku board is a {grid_size}x{grid_size} grid divided into {box_width}x{box_height} boxes
- Rows are numbered 0 to {grid_size_minus_one} from top to bottom
- Columns are numbered 0 to {grid_size_minus_one} from left to right
- Each cell is identified by its (row, column) coordinates
- Empty cells appear as periods (.) in the board representation
- Board state is represented as a nested list where board[row][column] gives the value at that position

When solving a Sudoku puzzle, we explore different possible number placements. Each step involves selecting an empty cell and placing a valid number in it. As we make selections, the set of valid moves for remaining cells may change.

Important considerations when evaluating possible actions:
1. How actions might create naked singles or hidden singles in other cells
2. Actions targeting cells with few remaining alternatives
3. How actions may constrain multiple other cells simultaneously 
4. How actions contribute to a balanced distribution of numbers across the board
5. Whether actions might lead to contradictions or cells with no legal moves

Your task is to evaluate the possible actions in the current state, scoring them based on how likely they are to help solve the Sudoku puzzle. The scores should form a probability distribution over the actions (sum to 1.0) and be returned as a dictionary mapping action indices to scores.

For example:

**Example {grid_size}x{grid_size} Sudoku Board**
{example_board}

**Example Possible Actions**
{example_prior_actions}

**Example Final Answer**

\\boxed{{
{{
"operation_scores": {example_operation_scores}
}}
}}
"""

        self.system_instruction_value = """
You are helping solve Sudoku puzzles using a tree-based search approach. Sudoku is a puzzle where you fill a grid with numbers 1 through {grid_size} so that each row, column, and box has no repeated numbers.

For this {grid_size}x{grid_size} Sudoku grid, the boxes are {box_width}x{box_height} in size. Each row, column, and box must contain all numbers from 1 to {grid_size} without repetition. This means:
1. Each row must contain each number from 1 to {grid_size} exactly once
2. Each column must contain each number from 1 to {grid_size} exactly once
3. Each {box_width}x{box_height} box must contain each number from 1 to {grid_size} exactly once

These constraints create a logical puzzle where placing a number in a cell immediately restricts what numbers can be placed in other cells in the same row, column, and box.

Board Structure:
- The Sudoku board is a {grid_size}x{grid_size} grid divided into {box_width}x{box_height} boxes
- Rows are numbered 0 to {grid_size_minus_one} from top to bottom
- Columns are numbered 0 to {grid_size_minus_one} from left to right
- Each cell is identified by its (row, column) coordinates
- Empty cells appear as periods (.) in the board representation
- Board state is represented as a nested list where board[row][column] gives the value at that position

When solving a Sudoku puzzle, we explore different possible number placements. Each step involves selecting an empty cell and placing a valid number in it. As we make selections, the set of valid moves for remaining cells may change.

Important considerations when estimating the value of a board state:

1. Factors that may indicate higher likelihood of success:
   - The number of cells with few possible remaining values
   - Whether all cells have at least one possible legal value
   - How close rows, columns, and boxes are to completion
   - The presence of obvious next moves such as naked or hidden singles
   
2. Factors that may indicate lower likelihood of success:
   - The presence of cells with zero possible legal values (contradictions)
   - Many cells having numerous possible values (high uncertainty)
   - Limited constraints between remaining empty cells
   - Patterns that typically lead to unsolvable states

Your task is to estimate the value of the current board state by determining the likelihood of solving the puzzle from this position. The score should range from 0 to 1.

For example:

**Example {grid_size}x{grid_size} Sudoku Board**
{example_board}

**Example Possible Actions**
{example_value_actions}

**Example Final Answer**

\\boxed{{
{{
"state_value_estimation": 0.75
}}
}}
"""

        # System instruction for exploration decision
        self.system_instruction_explore = """
You are helping solve Sudoku puzzles using a tree-based search approach. Sudoku is a puzzle where you fill a grid with numbers 1 through {grid_size} so that each row, column, and box has no repeated numbers.

For this {grid_size}x{grid_size} Sudoku grid, the boxes are {box_width}x{box_height} in size. Each row, column, and box must contain all numbers from 1 to {grid_size} without repetition. This means:
1. Each row must contain each number from 1 to {grid_size} exactly once
2. Each column must contain each number from 1 to {grid_size} exactly once
3. Each {box_width}x{box_height} box must contain each number from 1 to {grid_size} exactly once

These constraints create a logical puzzle where placing a number in a cell immediately restricts what numbers can be placed in other cells in the same row, column, and box.

Board Structure:
- The Sudoku board is a {grid_size}x{grid_size} grid divided into {box_width}x{box_height} boxes
- Rows are numbered 0 to {grid_size_minus_one} from top to bottom
- Columns are numbered 0 to {grid_size_minus_one} from left to right
- Each cell is identified by its (row, column) coordinates
- Empty cells appear as periods (.) in the board representation
- Board state is represented as a nested list where board[row][column] gives the value at that position

When solving a Sudoku puzzle, we explore different possible number placements. Each step involves selecting an empty cell and placing a valid number in it. As we make selections, the set of valid moves for remaining cells may change.

Important considerations when determining whether to continue with the current board state or explore a new state:
1. The presence of naked singles or hidden singles in the current board state
2. Whether the current board state contains contradictions or cells with no valid moves
3. The level of certainty in the remaining cells (many vs. few possible values)
4. Whether the board shows signs of making progress or appears to be in a deadlock

Your task is to decide whether to continue with the current board state or to visit an unexplored board state. Before deciding, carefully consider the current board and the available actions. Only choose to explore if you are certain that the current board state cannot lead to a solution and that switching to a new board state is the best use of time.

For example:

**Example {grid_size}x{grid_size} Sudoku Board**
{example_board}

**Example Possible Moves**
{example_explore_actions}

**Example Final Answer**

\\boxed{{
{{
"explore": false
}}
}}
"""

        # System instruction for evaluating child moves
        self.system_instruction_child_values = """
You are helping solve Sudoku puzzles using a tree-based search approach. Sudoku is a puzzle where you fill a grid with numbers 1 through {grid_size} so that each row, column, and box has no repeated numbers.

For this {grid_size}x{grid_size} Sudoku grid, the boxes are {box_width}x{box_height} in size. Each row, column, and box must contain all numbers from 1 to {grid_size} without repetition. This means:
1. Each row must contain each number from 1 to {grid_size} exactly once
2. Each column must contain each number from 1 to {grid_size} exactly once
3. Each {box_width}x{box_height} box must contain each number from 1 to {grid_size} exactly once

These constraints create a logical puzzle where placing a number in a cell immediately restricts what numbers can be placed in other cells in the same row, column, and box.

Board Structure:
- The Sudoku board is a {grid_size}x{grid_size} grid divided into {box_width}x{box_height} boxes
- Rows are numbered 0 to {grid_size_minus_one} from top to bottom
- Columns are numbered 0 to {grid_size_minus_one} from left to right
- Each cell is identified by its (row, column) coordinates
- Empty cells appear as periods (.) in the board representation
- Board state is represented as a nested list where board[row][column] gives the value at that position

Important considerations when evaluating possible moves:

1. Constraint Propagation: How each move affects future possibilities
   - Whether the move creates naked singles or hidden singles
   - How the move constrains other cells in the same row, column, and box

2. Strategic Value: The quality of the move in solving the puzzle
   - Whether the move targets cells with few remaining possibilities
   - Whether the move maintains flexibility in other cells
   - Whether the move creates a balanced distribution of numbers

3. Future Impact: How the move affects future solving paths
   - Whether the move opens up multiple solving techniques
   - Whether the move might lead to contradictions
   - Whether the move maintains good solving options

Your task is to evaluate each possible move and assign a value between 0 and 1 to each, where 1 means the move is extremely likely to lead to solving the puzzle and 0 means it's very unlikely to be helpful.

For example:

**Example {grid_size}x{grid_size} Sudoku Board**
{example_board}

**Example Possible Moves**
{example_moves}

**Example Final Answer**

\\boxed{{
{{
"move_values": {{"0": 0.8, "1": 0.5, "2": 0.3,...}}
}}
}}
"""

        # User request templates for MCTS
        self.user_request_prior = """
**Current {grid_size}x{grid_size} Sudoku Board**
{current_board}

**Possible Actions**
{action_list}

Evaluate each action based on how it creates constraints, identifies singles, minimizes branching, and maintains a balanced distribution of numbers as described in your instructions.

Assign a probability to each possible action based on how likely it is to lead to a solution of the Sudoku puzzle. The scores should sum to 1.0, representing a probability distribution over the actions.

Your response must include a valid JSON object, enclosed in a 'boxed', with an `operation_scores` field containing a dictionary mapping action indices to scores, formatted as follows:  

{prior_response_format}

Replace `<dictionary_of_scores>` with a dictionary mapping action indices to scores that MUST sum to 1.0.
"""

        self.prior_response_format = """
\\boxed{
{
"operation_scores": <dictionary_of_scores>
}
} 
"""

        self.user_request_value = """
**Current {grid_size}x{grid_size} Sudoku Board**
{current_board}

**Possible Actions**
{action_list}

Given the current board state and the possible actions, estimate the value of the current state. Consider factors like the number of cells with few possible values, whether there are contradictions, and whether there are obvious next moves as described in your instructions.

Provide a value ranging from 0-1, where 1 means it's certain to reach a solution and 0 means it's impossible.

Your response must include a valid JSON object, enclosed in a 'boxed', with a `state_value_estimation` field, formatted as follows:  

{value_response_format}

Replace `<value>` with your estimated probability (between 0 and 1) of solving the puzzle from this state.
"""

        self.value_response_format = """
\\boxed{
{
"state_value_estimation": <value>
}
} 
"""

        # User request for explore decision
        self.user_request_explore = """
**Current {grid_size}x{grid_size} Sudoku Board**
{current_board}

**Possible Moves**
{empty_cells}

Consider the current board state and the available actions. Evaluate whether the current state has promising moves like naked singles or hidden singles, or if it shows signs of contradictions or deadlocks as described in your instructions.

Reason through your options step by step and determine whether continuing with the current state or exploring a new state is the most optimal decision.

Respond with true if you should explore a new board state, or false if you should continue with the current one.

Your response must include a valid JSON object, enclosed in a 'boxed', with an `explore` field, where the value must be either true (to explore a new board state) or false (to continue with the current board state), formatted as follows:  

{explore_response_format}

Replace `<boolean>` with either true or false.
"""

        self.explore_response_format = """
\\boxed{
{
"explore": <boolean>
}
} 
"""

        # User request for child moves evaluation
        self.user_request_child_values = """
**Current {grid_size}x{grid_size} Sudoku Board**
{current_board}

**Possible Moves**
{moves_list}

Evaluate each possible move and assign a value between 0 and 1 to each, where 1 means the move is extremely likely to lead to solving the puzzle and 0 means it's very unlikely to be helpful.

Your response must include a valid JSON object, enclosed in a 'boxed', with a `move_values` field containing a dictionary mapping move indices to values between 0 and 1, formatted as follows:  

{child_moves_response_format}

Replace `<dictionary_of_values>` with a dictionary mapping move indices to values between 0 and 1."""

        self.child_moves_response_format = """
\\boxed{
{
"move_values": <dictionary_of_values>
}
}
"""

        # Common error handling messages
        self.try_again = """Your response must include a valid JSON object enclosed in a boxed format like this: \\boxed{{ {{...}} }}. Please ensure you follow this exact format and that your JSON is properly formatted according to the provided instructions."""
        self.correct_length = """Please ensure that the list of action scores is the same length as the list of possible actions. For the prior request, if you provide scores as a list, there must be exactly one score per action. Alternatively, provide scores as a dictionary mapping action indices to scores (e.g., {"0": 0.1, "1": 0.2, ...}). Respond in JSON format such that json.loads(json_resp) will not return any errors."""
        
        # Common regex parsing
        self.json_regex = r'\\boxed\{\s*(\{(?:[^{}]+|\{(?:[^{}]+|\{[^{}]*\})*\})*\})\s*\}'
        
        # Example boards for different box dimensions (raw arrays, not formatted strings)
        self.example_boards = {
            # 2x2 box (4x4 grid) - medium difficulty
            (2, 2): np.array([
                [4, 1, 3, 2],
                [0, 0, 0, 0],
                [1, 0, 2, 0],
                [2, 0, 1, 0]
            ]),
            
            # 2x3 box (6x6 grid) - medium difficulty
            (2, 3): np.array([
                [2, 0, 3, 0, 5, 6],
                [0, 4, 0, 2, 0, 0],
                [5, 0, 1, 6, 0, 2],
                [1, 0, 5, 0, 2, 4],
                [0, 2, 0, 4, 0, 0],
                [4, 5, 2, 0, 1, 0]
            ]),
            
            # 3x3 box (9x9 grid) - medium difficulty
            (3, 3): np.array([
                [2, 0, 0, 1, 4, 0, 5, 6, 9],
                [0, 1, 0, 6, 0, 8, 2, 4, 0],
                [4, 6, 9, 2, 5, 0, 3, 0, 0],
                [0, 0, 0, 8, 0, 9, 0, 7, 3],
                [0, 0, 0, 5, 7, 6, 1, 0, 0],
                [0, 0, 1, 3, 0, 0, 0, 8, 5],
                [0, 5, 7, 0, 0, 0, 8, 0, 6],
                [0, 8, 0, 0, 6, 5, 0, 3, 4],
                [0, 0, 0, 0, 0, 0, 7, 5, 0]
            ])
        }
        
        # Example actions for different board sizes
        self.example_actions_dict = {
            # 2x2 box (4x4 grid) examples
            (2, 2): {
                "actions": "{0: 'Consider cell (1, 0) where the following numbers [1, 2, 3, 4] have not been tried', 1: 'Consider cell (1, 1) where the following numbers [1, 2, 3, 4] have not been tried',...}",
                "prior_actions": "{0: 'Place 3 at cell (1, 0)', 1: 'Place 2 at cell (1, 0)', 2: 'Place 4 at cell (1, 1)', 3: 'Place 1 at cell (1, 1)',...}",
                "operation_scores": "{\"0\": 0.6, \"1\": 0.2, ...}",
                "value_actions": "{0: 'Place 3 at cell (1, 0)', 1: 'Place 2 at cell (1, 0)', 2: 'Place 4 at cell (1, 1)', 3: 'Place 1 at cell (1, 1)',...}",
                "explore_actions": "{0: 'Place 3 at cell (1, 0)', 1: 'Place 2 at cell (1, 0)', 2: 'Place 4 at cell (1, 1)', 3: 'Place 1 at cell (1, 1)',...}",
                "moves": "{0: 'Place 3 at cell (1, 0)', 1: 'Place 2 at cell (1, 0)', 2: 'Place 4 at cell (1, 1)', 3: 'Place 1 at cell (1, 1)',...}"
            },
            
            # 2x3 box (6x6 grid) examples
            (2, 3): {
                "actions": "{0: 'Consider cell (0, 1) where the following numbers [1, 2, 3, 4, 5, 6] have not been tried', 1: 'Consider cell (0, 3) where the following numbers [1, 2, 3, 4, 5, 6] have not been tried',...}",
                "prior_actions": "{0: 'Place 1 at cell (0, 1)', 1: 'Place 6 at cell (0, 1)', 2: 'Place 1 at cell (0, 3)', 3: 'Place 3 at cell (0, 3)', 4: 'Place 6 at cell (1, 2)',...}",
                "operation_scores": "{\"0\": 0.3, \"1\": 0.4, ...}",
                "value_actions": "{0: 'Place 1 at cell (0, 1)', 1: 'Place 6 at cell (0, 1)', 2: 'Place 1 at cell (0, 3)', 3: 'Place 3 at cell (0, 3)', 4: 'Place 6 at cell (1, 2)',...}",
                "explore_actions": "{0: 'Place 1 at cell (0, 1)', 1: 'Place 6 at cell (0, 1)', 2: 'Place 1 at cell (0, 3)', 3: 'Place 3 at cell (0, 3)', 4: 'Place 6 at cell (1, 2)',...}",
                "moves": "{0: 'Place 1 at cell (0, 1)', 1: 'Place 6 at cell (0, 1)', 2: 'Place 1 at cell (0, 3)', 3: 'Place 3 at cell (0, 3)', 4: 'Place 6 at cell (1, 2)',...}"
            },
            
            # 3x3 box (9x9 grid) examples
            (3, 3): {
                "actions": "{0: 'Consider cell (0, 1) where the following numbers [1, 2, 3, 4, 5, 6, 7, 8, 9] have not been tried', 1: 'Consider cell (0, 2) where the following numbers [1, 2, 3, 4, 5, 6, 7, 8, 9] have not been tried',...}",
                "prior_actions": "{0: 'Place 3 at cell (0, 1)', 1: 'Place 7 at cell (0, 1)', 2: 'Place 8 at cell (0, 2)', 3: 'Place 5 at cell (0, 2)', 4: 'Place 3 at cell (1, 0)', 5: 'Place 5 at cell (1, 0)',...}",
                "operation_scores": "{\"0\": 0.1, \"1\": 0.2, \"2\": 0.5, ...}",
                "value_actions": "{0: 'Place 3 at cell (0, 1)', 1: 'Place 7 at cell (0, 1)', 2: 'Place 8 at cell (0, 2)', 3: 'Place 5 at cell (0, 2)', 4: 'Place 3 at cell (1, 0)', 5: 'Place 5 at cell (1, 0)',...}",
                "explore_actions": "{0: 'Place 3 at cell (0, 1)', 1: 'Place 7 at cell (0, 1)', 2: 'Place 8 at cell (0, 2)', 3: 'Place 5 at cell (0, 2)', 4: 'Place 3 at cell (1, 0)', 5: 'Place 5 at cell (1, 0)',...}",
                "moves": "{0: 'Place 3 at cell (0, 1)', 1: 'Place 7 at cell (0, 1)', 2: 'Place 8 at cell (0, 2)', 3: 'Place 5 at cell (0, 2)', 4: 'Place 3 at cell (1, 0)', 5: 'Place 5 at cell (1, 0)',...}"
            }
        }

    def _format_board_example(self, board):
        """
        Format a nested list with None or zero values into a string representation suitable for examples.

        Args:
            board (list or numpy.ndarray): Representing the Sudoku board with 0 for empty cells

        Returns:
            str: Formatted board as a string with periods (".") for empty cells
        """
        # Convert to numpy array if it's not already
        if not isinstance(board, np.ndarray):
            board = np.array(board)

        # Use board_to_string from src.utils.game_specific.sudoku
        return board_to_string(board)

    def get_system_instruction(self, query_type, box_width=3, box_height=3):
        """
        Get the system instruction for the specified query type.
        
        Args:
            query_type (str): The type of query ('filter', 'action', 'best', 'prior', 'value', 'explore', 'child_moves')
            box_width (int): Width of each box in the Sudoku grid
            box_height (int): Height of each box in the Sudoku grid
            
        Returns:
            str: The system instruction for the specified query type
            
        Raises:
            ValueError: If there's no example for the specified box dimensions
        """
        # Calculate grid size based on box dimensions
        grid_size = box_width * box_height
        
        # Get the appropriate example board for these dimensions
        if (box_width, box_height) not in self.example_boards:
            raise ValueError(f"No example available for box dimensions {box_width}x{box_height}")
        
        # Format the example board using board_to_string
        example_board = board_to_string(self.example_boards[(box_width, box_height)])
        
        # Get the examples for the specific board size
        if (box_width, box_height) not in self.example_actions_dict:
            # Default to 3x3 examples if specific ones aren't available
            examples = self.example_actions_dict[(3, 3)]
        else:
            examples = self.example_actions_dict[(box_width, box_height)]
        
        # Select the appropriate template
        if query_type == "prior":
            template = self.system_instruction_prior
        elif query_type == "value":
            template = self.system_instruction_value
        elif query_type == "explore":
            template = self.system_instruction_explore
        elif query_type == "child_moves":
            template = self.system_instruction_child_values
        else:
            raise ValueError(f"Unknown query type: {query_type}")
            
        # Calculate grid_size_minus_one for the template
        grid_size_minus_one = grid_size - 1
            
        # Format the template with the grid size, box dimensions, and examples
        formatted_instruction = template.format(
            grid_size=grid_size,
            grid_size_minus_one=grid_size_minus_one,
            box_width=box_width,
            box_height=box_height,
            example_board=example_board,
            example_actions=examples.get("actions", ""),
            example_prior_actions=examples.get("prior_actions", ""),
            example_operation_scores=examples.get("operation_scores", ""),
            example_value_actions=examples.get("value_actions", ""),
            example_explore_actions=examples.get("explore_actions", ""),
            example_moves=examples.get("moves", ""),
            top_n=grid_size
        )
        
        return formatted_instruction
    
    def generate_request(self, query_type, **kwargs):
        """
        Generate a request for the specified query type with the given parameters.
        
        Args:
            query_type (str): The type of query ('filter', 'action', 'best', 'prior', 'value', 'explore', 'child_moves')
            **kwargs: Parameters for the request 
                - For 'prior' and 'value':
                    current_board (str or numpy.ndarray): Current Sudoku board state
                    action_list (list): List of possible actions to evaluate
                - For 'explore':
                    current_board (str or numpy.ndarray): Current Sudoku board state
                    empty_cells (list/dict): Available empty cells that can be filled
                - box_width (int): Width of each box in the Sudoku grid
                - box_height (int): Height of each box in the Sudoku grid
                
        Returns:
            str: The formatted request string
        """
        # Extract box dimensions and calculate grid size
        box_width = kwargs.get("box_width", 3)
        box_height = kwargs.get("box_height", 3)
        grid_size = box_width * box_height
        
        # Format board if it's a numpy array
        if query_type == "prior":
            current_board = kwargs.get("current_board")
            if isinstance(current_board, np.ndarray):
                current_board = board_to_string(current_board)
                
            # Convert action_list from a list to a dictionary format if it's a list
            action_list = kwargs.get("action_list")
            if isinstance(action_list, list):
                action_list = {i: action for i, action in enumerate(action_list)}
                
            return self.user_request_prior.format(
                grid_size=grid_size,
                current_board=current_board, 
                action_list=action_list,
                prior_response_format=self.prior_response_format
            )
        elif query_type == "value":
            current_board = kwargs.get("current_board")
            if isinstance(current_board, np.ndarray):
                current_board = board_to_string(current_board)
                
            # Convert action_list from a list to a dictionary format if it's a list
            action_list = kwargs.get("action_list")
            if isinstance(action_list, list):
                action_list = {i: action for i, action in enumerate(action_list)}
                
            return self.user_request_value.format(
                grid_size=grid_size,
                current_board=current_board, 
                action_list=action_list,
                value_response_format=self.value_response_format
            )
        elif query_type == "explore":
            current_board = kwargs.get("current_board")
            if isinstance(current_board, np.ndarray):
                current_board = board_to_string(current_board)
                
            return self.user_request_explore.format(
                grid_size=grid_size,
                current_board=current_board,
                empty_cells=kwargs.get("empty_cells"),
                explore_response_format=self.explore_response_format
            )
        elif query_type == "child_moves":
            current_board = kwargs.get("current_board")
            if isinstance(current_board, np.ndarray):
                current_board = board_to_string(current_board)
                
            return self.user_request_child_values.format(
                grid_size=box_width * box_height,
                current_board=current_board,
                moves_list=kwargs.get("moves_list"),
                child_moves_response_format=self.child_moves_response_format
            )
        else:
            raise ValueError(f"Unknown query type: {query_type}")


class SudokuAgent(BaseAgent):
    """
    Agent class for Sudoku problem solvers.
    
    This class handles communication with LLMs to assist in solving Sudoku puzzles.
    """
    def __init__(self, model, **kwargs):
        """
        Initialize the SudokuAgent.
        
        Args:
            model: OpenAI client instance
            **kwargs: Additional arguments
        """
        self.model = model  # OpenAI client instance
        self.model_name = kwargs.get("model_name", "gpt-4o")  # Model name to use with the client
        self.model_type = kwargs.get("model_type", "openai")  # Model name to use with the client
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.timeout = kwargs.get("timeout", 1000)
        self.message_history = []
        self.queries = []
        self.responses = []
        self.reasoning = kwargs.get("reasoning", 0)  # Whether to use reasoning mode (0=disabled, 1=enabled)
        
        # Box dimensions for the Sudoku grid
        self.box_width = kwargs.get("box_width", 3)
        self.box_height = kwargs.get("box_height", self.box_width)
        
        # Initialize the SudokuInstructions class to access all instructions
        self.instructions = SudokuInstructions()

    def reset(self, query_type):
        """Reset the message history based on the query type."""
        # Clear the message history
        self.message_history = []
        
        # Get the appropriate system instruction for this query type
        system_instruction = self.instructions.get_system_instruction(
            query_type, 
            box_width=self.box_width, 
            box_height=self.box_height
        )
        
        if self.model_type == "nvidia":
            # For NVIDIA models, use user messages instead of system messages
            message = {"role": "user", "content": system_instruction}
            self.update_message_history(message)
        else:
            # For other models, use system messages
            message = {"role": "system", "content": system_instruction}
            self.update_message_history(message)
        
    def update_message_history(self, new_message):
        """
        Update message history, combining consecutive messages of the same type.
        
        If the new message is of the same type (Human, System, or AI) as the last message
        in the history, their content will be joined into a single message.
        
        Args:
            new_message: The new message to add to the history (dictionary with 'role' and 'content')
        """
        # Also add to queries for tracking (don't add assistant messages to queries)
        if new_message["role"] != "assistant":
            self.queries.append(new_message)
        
        if not self.message_history:
            # If history is empty, just add the message
            self.message_history.append(new_message)
            return
        
        last_message = self.message_history[-1]
        
        # Check if both messages are of the same type
        if last_message["role"] == new_message["role"]:
            # Combine the content with a newline separator
            combined_content = last_message["content"] + "\n\n" + new_message["content"]
            
            # Replace the last message with a new combined message of the same type
            new_message_dict = {"role": last_message["role"], "content": combined_content}
            
            # If it's an assistant message and has response_metadata, preserve it
            if last_message["role"] == "assistant" and "response_metadata" in last_message:
                new_message_dict["response_metadata"] = last_message["response_metadata"]
            
            self.message_history[-1] = new_message_dict
        else:
            # If different types, just append
            self.message_history.append(new_message)

    def _get_api_params(self, messages, response_format=None):
        """
        Helper method to generate API parameters based on reasoning mode.
        
        Args:
            messages (list): List of message objects to send to the API
            response_format (dict, optional): Format specification for the response
            
        Returns:
            dict: Parameters to use for the API call
        """
        # Set up common parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "timeout": self.timeout
        }
        
        # Add response format if specified
        if response_format:
            params["response_format"] = response_format
        
        # Add parameters based on reasoning mode
        if self.reasoning == 1:
            # Parameters for reasoning mode
            params["reasoning_effort"] = "medium"
            params["store"] = False
        else:
            # Parameters for non-reasoning mode
            params["max_tokens"] = self.max_tokens
            params["temperature"] = self.temperature
        
        return params
    
    def _handle_retry(self, message_content, is_correction=False):
        """
        Helper method to handle retry attempts for boxed JSON responses.
        
        Args:
            message_content (str): The retry message content
            is_correction (bool): Whether this is a correction for an invalid number
            
        Returns:
            tuple: (json_list, full_response, token_usage)
        """
        # Add the retry message to history
        retry_message = {"role": "user", "content": message_content}
        self.update_message_history(retry_message)
        
        # Convert message history to OpenAI format (removing extra fields)
        openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.message_history]
        
        # Get API parameters for retry
        retry_params = self._get_api_params(
            openai_messages, 
            response_format={"type": "text"}
        )
        
        # Call OpenAI API for correction
        retry_response = self.model.chat.completions.create(**retry_params)
        
        # Extract completion content
        corrected_content = retry_response.choices[0].message.content
        
        # Create token usage info
        token_usage = {
            "prompt_tokens": retry_response.usage.prompt_tokens,
            "completion_tokens": retry_response.usage.completion_tokens,
            "total_tokens": retry_response.usage.total_tokens
        }
        
        # Create assistant message response
        corrected_act_message = {"role": "assistant", "content": corrected_content, "response_metadata": {"token_usage": token_usage}}
        
        # Add corrected AI response to history
        self.update_message_history(corrected_act_message)
        
        # Track response separately
        self.responses.append(corrected_act_message)
        
        # Parse JSON from response
        json_lst = regex.findall(self.instructions.json_regex, corrected_content)
        
        return json_lst, corrected_content, token_usage

    def _ask(self, query_type, attempt_num, **kwargs):
        """Internal implementation of asking the LLM."""
        if attempt_num < 2:
            # Generate the appropriate request for this query type
            user_message = {
                "role": "user", 
                "content": self.instructions.generate_request(query_type=query_type, **kwargs)
            }
        else:
            print(f"*** Attempt: {attempt_num} ***")
            user_message = {"role": "user", "content": self.instructions.try_again}
        
        # Use the new method to update message history
        self.update_message_history(user_message)
        
        # Convert message history to OpenAI format (removing any extra fields like response_metadata)
        openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.message_history]

        # Get API parameters
        params = self._get_api_params(openai_messages)
        
        # Call OpenAI API directly using self.model
        response = self.model.chat.completions.create(**params)
        
        # Extract completion content
        completion_content = response.choices[0].message.content
        
        # Create AI message response with metadata
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        act_message = {"role": "assistant", "content": completion_content, "response_metadata": {"token_usage": token_usage}}
        
        # Add AI response to message history
        self.update_message_history(act_message)
        
        # Still track all responses separately
        self.responses.append(act_message)
        
        # Handle response parsing and validation logic
        json_lst = regex.findall(self.instructions.json_regex, act_message["content"])
        full_response = act_message["content"]
        
        # Check if we have a properly formatted boxed JSON response
        if len(json_lst) == 0:
            # No properly boxed JSON found - try using the try_again prompt
            print("Response doesn't contain properly boxed JSON. Requesting correction with try_again prompt...")
            
            # Handle retry
            json_lst, _, _ = self._handle_retry(self.instructions.try_again)
        
        resp_dict = check_json_list(json_lst)
        if resp_dict is None:
            # If we couldn't parse any JSON, this will trigger a retry in the ask() method
            raise ValueError(f"Error loading agent message: {full_response}, {json_lst}")
            
        # Handle different response formats based on query type
        if query_type == "prior":
            resp = resp_dict["operation_scores"]
            original_action_list = kwargs.get("action_list", [])
            # Validate response format and ensure it's a dictionary
            if not isinstance(resp, dict) or len(resp) != len(original_action_list):
                print(f"Warning: Prior response is not a dictionary: {resp} or not the right length.")
                # Try to correct the format if it's a list
                if isinstance(resp, list) or len(resp) != len(original_action_list):                 
                    # Check if the response list has the same length as the action list
                    if len(resp) != len(original_action_list):
                        error_msg = f"Prior response list length ({len(resp)}) doesn't match action list length ({len(original_action_list)})"
                        print(f"Warning: {error_msg}")
                        
                        # Try to handle this with a specific correction request
                        retry_json_lst, full_response, _ = self._handle_retry(self.instructions.correct_length, is_correction=True)
                        
                        if retry_json_lst:
                            retry_resp_dict = check_json_list(retry_json_lst)
                            if retry_resp_dict and "operation_scores" in retry_resp_dict:
                                retry_resp = retry_resp_dict["operation_scores"]
                                
                                # Check if the corrected response is properly formatted
                                if isinstance(retry_resp, dict):
                                    # Use the corrected dictionary response
                                    resp = retry_resp
                                    print("Successfully obtained dictionary response after length correction.")
                                elif isinstance(retry_resp, list) and len(retry_resp) == len(original_action_list):
                                    # If still a list but now correct length, convert to dictionary
                                    resp = {str(i): score for i, score in enumerate(retry_resp)}
                                    print(f"Converted corrected response list to dictionary: {resp}")
                                else:
                                    # Still incorrect format or length
                                    raise ValueError(f"Prior response format or length still incorrect after retry")
                            else:
                                # No valid operation_scores in retry response
                                raise ValueError(f"No valid operation_scores in retry response: {retry_resp_dict}")
                        else:
                            # No valid JSON in retry response, use original error
                            raise ValueError(error_msg)
                    else:
                        # List has correct length, convert to dictionary
                        try:
                            resp = {str(i): score for i, score in enumerate(resp)}
                            print(f"Converted prior response from list to dictionary: {resp}")
                        except Exception as e:
                            raise ValueError(f"Could not convert prior response to dictionary: {e}")
                else:
                    raise ValueError(f"Prior response must be a dictionary, got: {resp}")
        elif query_type == "value":
            resp = resp_dict["state_value_estimation"]
        elif query_type == "explore":
            resp = resp_dict["explore"]
        elif query_type == "child_moves":
            try:
                # Try to get the move_values
                resp = resp_dict["move_values"]
                
                # Validate response format and ensure it's a dictionary
                original_moves_list = kwargs.get("moves_list", {})
                
                # Convert the moves list to a dictionary if it's not already
                if isinstance(original_moves_list, list):
                    original_moves_list = {i: move for i, move in enumerate(original_moves_list)}
                
                # If resp is a list, convert it to a dictionary
                if isinstance(resp, list):
                    if len(resp) != len(original_moves_list):
                        error_msg = f"Child moves response list length ({len(resp)}) doesn't match moves list length ({len(original_moves_list)})"
                        print(f"Warning: {error_msg}")
                        
                        # Try to correct the format with a specific message
                        retry_json_lst, full_response, _ = self._handle_retry(self.instructions.correct_length)
                        
                        if retry_json_lst:
                            retry_resp_dict = check_json_list(retry_json_lst)
                            if retry_resp_dict and "move_values" in retry_resp_dict:
                                retry_resp = retry_resp_dict["move_values"]
                                
                                # Check if the corrected response is properly formatted
                                if isinstance(retry_resp, dict):
                                    # Use the corrected dictionary response
                                    resp = retry_resp
                                    print("Successfully obtained dictionary response after correction.")
                                elif isinstance(retry_resp, list) and len(retry_resp) == len(original_moves_list):
                                    # If still a list but now correct length, convert to dictionary
                                    resp = {str(i): value for i, value in enumerate(retry_resp)}
                                    print(f"Converted corrected response list to dictionary: {resp}")
                                else:
                                    # Still incorrect format or length
                                    raise ValueError(f"Child moves response format or length still incorrect after retry")
                            else:
                                # No valid move_values in retry response
                                raise ValueError(f"No valid move_values in retry response: {retry_resp_dict}")
                        else:
                            # No valid JSON in retry response, use original error
                            raise ValueError(error_msg)
                    else:
                        # List has correct length, convert to dictionary
                        try:
                            resp = {str(i): value for i, value in enumerate(resp)}
                            print(f"Converted child moves response from list to dictionary: {resp}")
                        except Exception as e:
                            raise ValueError(f"Could not convert child moves response to dictionary: {e}")
                elif not isinstance(resp, dict):
                    # Not a list or dictionary, raise error
                    raise ValueError(f"Child moves response must be a dictionary or list, got: {type(resp)}")
                
                # Ensure all values in the dictionary are floats between 0 and 1
                validated_resp = {}
                for key, value in resp.items():
                    try:
                        value_float = float(value)
                        if 0 <= value_float <= 1:
                            validated_resp[key] = value_float
                        else:
                            raise ValueError(f"Move value {value_float} for key {key} is outside the range [0,1]")
                    except (ValueError, TypeError):
                        raise ValueError(f"Could not convert move value '{value}' for key {key} to float")
                
                # Replace the original response with the validated one
                resp = validated_resp
                
            except (KeyError, TypeError, ValueError) as e:
                # Raise error to trigger retry in ask()
                print(f"Error in child moves response: {e}")
                raise ValueError(f"Child moves response format error: {e}")
        else:
            raise ValueError(f"Unknown query type: {query_type}")
            
        return {"full_response": full_response,
                "resp": resp,
                "token_usage": token_usage}

    def ask(self, query_type, **kwargs):
        """Public method to ask the LLM, with retry logic."""
        self.reset(query_type)
        try:
            for attempt in tenacity.Retrying(
                    stop=tenacity.stop_after_attempt(5),
                    wait=tenacity.wait_exponential(2),  # wait time = 2*2^(attempts-1)
                    before_sleep=lambda retry_state: print(
                        f"Error occurred: {retry_state.outcome.exception()}, retrying..."
                    ),
                    retry=tenacity.retry_if_exception_type((openai.APIError, openai.APIConnectionError, ValueError)),
            ):
                with attempt:
                    try:
                        resp_dict = self._ask(query_type=query_type, attempt_num=attempt.retry_state.attempt_number, **kwargs)
                        return resp_dict
                    except (openai.BadRequestError, openai.RateLimitError) as e:
                        if any(phrase in str(e).lower() for phrase in ["maximum context length", "content too long", "token limit", "tokens in your"]):
                            print("Model exceeded maximum context length, stopping current game.")
                            return {
                                "full_response": str(e),
                                "resp": None,
                                "token_usage": None,
                                "context_length_exceeded": True,
                                "error_message": "Model exceeded maximum context length",
                            }
                        # Re-raise other errors to be caught by the retry logic
                        raise
            return None
        except tenacity.RetryError as e:
            return {
                "full_response": str(e),
                "resp": None,
                "token_usage": None,
            }