import openai
import regex
import tenacity

from src.base_game.base_agent import BaseAgent, BaseInstructions
from src.utils.common_utils import check_json_list


class CountdownInstructions(BaseInstructions):
    """
    Class to store all the prompts and instructions for the countdown problem.
    
    This class centralizes all the templates used for interacting with the LLM,
    including system instructions, user requests, response formats, and error messages.
    """

    def __init__(self):
        # System instructions for MCTS
        self.system_instruction_prior = """
You're playing the Countdown Numbers Game. Let me explain the rules and how to solve it:

Game Rules:
1. You are given a set of numbers and a target number to reach.
2. You can only use each number once.
3. You must combine numbers using only four operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Division is only allowed when it results in a whole number (no fractions or decimals).
5. You can only combine two numbers at a time to create a new number.
6. After each operation, the original numbers are removed, and the result is added to your available numbers.
7. You win when you have exactly one number left that matches the target.

For example, with target 50 and numbers [39, 66, 33, 13]:
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

**Action 2**
Operation: '52 - 2 = 50'
**State 3** (After performing 52 - 2 = 50)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2', '52 - 2 = 50']
Available Numbers: [50]
Game won!

Important considerations when assigning probabilities to operations:

1. Target Progress: How much closer the operation gets to the target
   - Operations resulting in numbers exactly at or very close to target should receive higher scores
   - Operations creating useful intermediate numbers should be favored

2. Number Creation: The utility of the resulting number
   - Creating small, flexible numbers (1-10) can be valuable
   - Creating numbers that are factors of the target
   - Creating numbers that offer efficient pathways to the target

3. Available Number Management: How the operation affects the number pool
   - Operations that use less useful numbers while preserving useful ones
   - Operations that create a more workable set of available numbers
   - Avoiding operations that result in unusable large numbers

4. Mathematical Strategy: Using operations optimally
   - Using division to create useful small numbers
   - Using multiplication for larger adjustments toward the target
   - Using addition/subtraction for precise movements toward the target

Your task is to evaluate the possible actions in the current state, scoring them based on how likely they are to help you achieve the target value. The scores should form a probability distribution over the actions.

For example:

**Example State Sequence**
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

Example Possible Operations: {0: '52 + 2 = 54', 1: '52 - 2 = 50', 2: '52 * 2 = 104', 3: '52 / 2 = 26'}

**Example Final Answer**

\\boxed{
{
"operation_scores": {"0": 0.15, "1": 0.35, "2": 0.35, "3": 0.15}
}
}
"""

        self.system_instruction_value = """
You're playing the Countdown Numbers Game. Let me explain the rules and how to solve it:

Game Rules:
1. You are given a set of numbers and a target number to reach.
2. You can only use each number once.
3. You must combine numbers using only four operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Division is only allowed when it results in a whole number (no fractions or decimals).
5. You can only combine two numbers at a time to create a new number.
6. After each operation, the original numbers are removed, and the result is added to your available numbers.
7. You win when you have exactly one number left that matches the target.

For example, with target 50 and numbers [39, 66, 33, 13]:
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

**Action 2**
Operation: '52 - 2 = 50'
**State 3** (After performing 52 - 2 = 50)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2', '52 - 2 = 50']
Available Numbers: [50]
Game won!

Important factors to consider when estimating state value:

1. Proximity to Target: How close the current numbers are to the target
   - States with numbers exactly equal to or close to the target are more valuable
   - States with numbers that can be easily combined to reach the target have higher value

2. Available Number Quality: How useful the remaining numbers are
   - Having small numbers (1-10) increases flexibility
   - Having numbers that are factors or multiples of target numbers is valuable
   - Having complementary numbers that work well together

3. State Progress: How much progress has been made
   - Number of operations performed so far
   - Reduction in the total number of available numbers
   - Quality of the operations performed so far

4. Potential for Success: Overall likelihood of reaching the target
   - Presence of clear pathways to the target
   - Absence of unusable or problematic numbers
   - Balance between large and small numbers

Your task is to estimate the value of the current state and possible operations by determining the likelihood of reaching the target number from it. The score should range from 0 to 1.

For example:

**Example State Sequence**
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

Example Possible Operations: ['52 + 2 = 54', '52 - 2 = 50', '52 * 2 = 104', '52 / 2 = 26']

**Example Final Answer**

\\boxed{
{
"state_value_estimation": 1.0
}
}
"""

        # System instruction for evaluating child actions
        self.system_instruction_child_values = """
You're playing the Countdown Numbers Game. Let me explain the rules and how to solve it:

Game Rules:
1. You are given a set of numbers and a target number to reach.
2. You can only use each number once.
3. You must combine numbers using only four operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Division is only allowed when it results in a whole number (no fractions or decimals).
5. You can only combine two numbers at a time to create a new number.
6. After each operation, the original numbers are removed, and the result is added to your available numbers.
7. You win when you have exactly one number left that matches the target.

For example, with target 50 and numbers [39, 66, 33, 13]:
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

**Action 2**
Operation: '52 - 2 = 50'
**State 3** (After performing 52 - 2 = 50)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2', '52 - 2 = 50']
Available Numbers: [50]
Game won!

Important considerations when evaluating possible operations:

1. Target Progress: How much each operation moves toward the target
   - Operations that result in numbers close to the target
   - Operations that create useful intermediate numbers for future steps

2. Number Creation: The strategic value of the resulting number
   - Creating small, useful numbers (1-10) for fine adjustments
   - Creating numbers that are easily combinable with others
   - Creating numbers that are factors or related to the target

3. Operation Strategy: How the operation affects solution paths
   - Using division to create useful small numbers
   - Using multiplication to make larger jumps toward the target
   - Using addition/subtraction for precise adjustments

4. Future Potential: How an operation affects future possibilities
   - Operations that open up multiple future paths
   - Operations that eliminate problematic numbers
   - Operations that maintain flexibility in the number set

Your task is to evaluate each possible operation and assign a value between 0 and 1 to each, where 1 means the operation is extremely likely to lead to solving the puzzle and 0 means it's very unlikely to be helpful.

For example:

**Example State Sequence**
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

Example Possible Operations: {0: '52 + 66 = 118', 1: '52 - 33 = 19', 2: '66 - 33 = 33', 3: '66 / 33 = 2'}

**Example Final Answer**

\\boxed{
{
"operation_values": {"0": 0.3, "1": 0.6, "2": 0.5, "3": 0.9}
}
}
"""

        # System instruction for exploration decision
        self.system_instruction_explore = """
You're playing the Countdown Numbers Game. Let me explain the rules and how to solve it:

Game Rules:
1. You are given a set of numbers and a target number to reach.
2. You can only use each number once.
3. You must combine numbers using only four operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Division is only allowed when it results in a whole number (no fractions or decimals).
5. You can only combine two numbers at a time to create a new number.
6. After each operation, the original numbers are removed, and the result is added to your available numbers.
7. You win when you have exactly one number left that matches the target.

For example, with target 50 and numbers [39, 66, 33, 13]:
**State 0**
Target: 50 
Operations: [] 
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50 
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52] 

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2'] 
Available Numbers: [52, 2] 

**Action 2**
Operation: '52 - 2 = 50'
**State 3** (After performing 52 - 2 = 50)
Target: 50 
Operations: ['39 + 13 = 52', '66 / 33 = 2', '52 - 2 = 50']
Available Numbers: [50]
Game won!

Important considerations when deciding whether to explore or continue:

1. Current Path Quality: How promising the current path appears
   - Presence of numbers close to the target
   - Quality and usefulness of available numbers
   - Clear pathways to reach the target from current numbers

2. Current Path Issues: Signs the current path may be problematic
   - Numbers far from the target with no clear way to combine them
   - Repeated patterns or circular operations
   - No beneficial operations remaining

3. Exploration Value: Potential benefit of trying other paths
   - Number of operations already performed on current path
   - Quality of alternative unexplored paths
   - Diminishing returns on current path

4. Decision Confidence: Certainty about current path viability
   - Clear evidence current path cannot reach target
   - Presence of obviously better unexplored paths
   - Risk assessment of continuing vs exploring

Your task is to decide whether to continue with the current state or to visit an unexplored state. Before deciding, carefully consider the current sequence of states and actions, as well as the available operations. Only choose to explore if you are certain that the current path cannot reach the target number and that switching to a new path is the best use of time.

For example:

**Example State and Action sequence**

**State 0**
Target: 50
Operations: []
Available Numbers: [39, 66, 33, 13]

**Action 0**
Operation: '39 + 13 = 52'
**State 1** (After performing 39 + 13 = 52)
Target: 50
Operations: ['39 + 13 = 52']
Available Numbers: [66, 33, 52]

**Action 1**
Operation: '66 / 33 = 2'
**State 2** (After performing 66 / 33 = 2)
Target: 50
Operations: ['39 + 13 = 52', '66 / 33 = 2']
Available Numbers: [52, 2]

Example Possible Operations: {0: '52 + 2 = 54', 1: '52 - 2 = 50', 2: '52 * 2 = 104', 3: '52 / 2 = 26'}

**Example Final Answer**

\\boxed{
{
"explore": false
}
}
"""

        # User request templates for MCTS
        self.user_request_prior = f"""
**Current State and Action sequence**
{{current_sequence}}
Possible Operations: {{action_list}}

What are the scores for each action/operation? Assign a probability to each possible operation based on how likely it is to lead to the target number.

Your response must include a valid JSON object, enclosed in a 'boxed', with an `operation_scores` field containing a dictionary mapping operation keys to scores, formatted as follows:  

{{prior_response_format}}

Replace `<dictionary_of_scores>` with a dictionary mapping operation keys to scores that must sum to 1.0."""

        self.prior_response_format = f"""
\\boxed{{
{{
"operation_scores": <dictionary_of_scores>
}}
}}  
"""

        self.user_request_value = f"""
**Current State and Action sequence**
{{current_sequence}}
Possible Operations: {{action_list}}

Given the current state and the possible operations, estimate the value of the current state, ranging from 0-1, where 1 means it's certain to reach the target number and 0 means it's impossible.

Your response must include a valid JSON object, enclosed in a 'boxed', with a `state_value_estimation` field, formatted as follows:  

{{value_response_format}}

Replace `<value>` with your estimated probability (between 0 and 1) of reaching the target from this state."""

        self.value_response_format = f"""
\\boxed{{
{{
"state_value_estimation": <value>
}}
}}  
"""

        # User request for explore decision
        self.user_request_explore = f"""
**Current State and Action sequence**
{{current_sequence}}

Possible Operations: {{action_list}}

Consider the current sequence of states and actions and the available operations. Reason through your options step by step and determine whether continuing with the current state or exploring a new state is the most optimal decision.

Your response must include a valid JSON object, enclosed in a 'boxed', with an `explore` field, where the value must be either true (to explore a new state) or false (to continue with the current state), formatted as follows:  

{{explore_response_format}}

Replace `<boolean>` with either true or false."""

        self.explore_response_format = f"""
\\boxed{{
{{
"explore": <boolean>
}}
}}  
"""

        # User request for child values
        self.user_request_child_values = f"""
**Current State and Action sequence**
{{current_sequence}}

Possible Operations: {{action_list}}

Evaluate each possible operation and assign a value between 0 and 1 to each, where 1 means the operation is extremely likely to lead to solving the puzzle and 0 means it's very unlikely to be helpful.

Your response must include a valid JSON object, enclosed in a 'boxed', with an `operation_values` field containing a dictionary mapping operation keys to values between 0 and 1, formatted as follows:  

{{child_values_response_format}}

Replace `<dictionary_of_values>` with a dictionary mapping operation keys to values between 0 and 1."""

        self.child_values_response_format = f"""
\\boxed{{
{{
"operation_values": <dictionary_of_values>
}}
}}  
"""

        # Common error handling messages
        self.try_again = f"""Your response must include a valid JSON object enclosed in a boxed format like this: \\boxed{{ {{...}} }}. Please ensure you follow this exact format and that your JSON is properly formatted according to the provided instructions."""
        self.correct_length = f"""Please ensure that the operation_scores dictionary contains scores for all possible operations. Each operation key should have a corresponding score value. Alternatively, if you're providing a list, ensure it has exactly one score per operation. All scores should sum to 1.0. Respond in JSON format such that json.loads(json_resp) will not return any errors."""

        # Common regex parsing
        self.json_regex = r'\\boxed\{\s*(\{(?:[^{}]+|\{(?:[^{}]+|\{[^{}]*\})*\})*\})\s*\}'

        super().__init__()

    def get_system_instruction(self, query_type):
        """
        Get the system instruction for the specified query type.
        
        Args:
            query_type: The type of query ('filter', 'action', 'best', 'prior', 'value')
            
        Returns:
            The system instruction for the specified query type
        """
        if query_type == "prior":
            return self.system_instruction_prior
        elif query_type == "value":
            return self.system_instruction_value
        elif query_type == "explore":
            return self.system_instruction_explore
        elif query_type == "child_values":
            return self.system_instruction_child_values
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def generate_request(self, query_type, **kwargs):
        """
        Generate a request for the specified query type with the given parameters.
        
        Args:
            query_type: The type of query ('filter', 'action', 'best', 'prior', 'value')
            **kwargs: Parameters for the request 
                - For 'filter' and 'best': target, unexplored_paths
                - For 'action', 'prior' and 'value': current_sequence, action_list
                
        Returns:
            The formatted request string
        """
        if query_type == "prior":
            return self.user_request_prior.format(
                current_sequence=kwargs.get("current_sequence"),
                action_list=kwargs.get("action_list"),
                prior_response_format=self.prior_response_format
            )
        elif query_type == "value":
            return self.user_request_value.format(
                current_sequence=kwargs.get("current_sequence"),
                action_list=kwargs.get("action_list"),
                value_response_format=self.value_response_format
            )
        elif query_type == "explore":
            return self.user_request_explore.format(
                current_sequence=kwargs.get("current_sequence"),
                action_list=kwargs.get("action_list"),
                explore_response_format=self.explore_response_format
            )
        elif query_type == "child_values":
            return self.user_request_child_values.format(
                current_sequence=kwargs.get("current_sequence"),
                action_list=kwargs.get("action_list"),
                child_values_response_format=self.child_values_response_format
            )
        else:
            raise ValueError(f"Unknown query type: {query_type}")


class CountdownAgent(BaseAgent):
    """
    Agent class for countdown problem solvers.
    
    This class handles communication with LLMs to assist in solving countdown puzzles.
    """

    def __init__(self, model, **kwargs):
        """
        Initialize the CountdownAgent.
        
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

        # Initialize the CountdownInstructions class to access all instructions
        self.instructions = CountdownInstructions()

    def reset(self, query_type):
        """Reset the message history based on the query type."""
        # Clear the message history
        self.message_history = []

        # Get the appropriate system instruction for this query type
        system_instruction = self.instructions.get_system_instruction(query_type)

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
            params["reasoning_effort"] = "low"
            params["store"] = False
        else:
            # Parameters for non-reasoning mode
            params["max_tokens"] = self.max_tokens
            params["temperature"] = self.temperature

        return params

    def _handle_retry(self, message_content):
        """
        Helper method to handle retry attempts for boxed JSON responses.
        
        Args:
            message_content (str): The retry message content
            
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
        corrected_act_message = {"role": "assistant", "content": corrected_content,
                                 "response_metadata": {"token_usage": token_usage}}

        # Add corrected AI response to history
        self.update_message_history(corrected_act_message)

        # Track response separately
        self.responses.append(corrected_act_message)

        # Parse JSON from response
        json_lst = regex.findall(self.instructions.json_regex, corrected_content)

        return json_lst, corrected_content, token_usage

    def _ask(self, query_type, attempt_num, **kwargs):
        """Internal implementation of asking the LLM."""
        # Generate the appropriate request for this query type
        if attempt_num > 1:
            print(f"*** Attempt: {attempt_num} ***")
        user_message = {
            "role": "user",
            "content": self.instructions.generate_request(query_type=query_type, **kwargs)
        }

        # Use the new method to update message history
        self.update_message_history(user_message)

        # Convert message history to OpenAI format (removing any extra fields like response_metadata)
        openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.message_history]

        # Get API parameters
        params = self._get_api_params(openai_messages)

        # Call OpenAI API directly using self.model
        response = self.model.chat.completions.create(**params)
        if isinstance(response, str):
            raise ValueError(f"Response is of type str: {response}")
        # Extract completion content
        completion_content = response.choices[0].message.content

        # Create AI message response with metadata
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        act_message = {"role": "assistant", "content": completion_content,
                       "response_metadata": {"token_usage": token_usage}}

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
            json_lst, full_response, token_usage = self._handle_retry(self.instructions.try_again)

        resp_dict = check_json_list(json_lst)
        if resp_dict is None:
            # If we couldn't parse any JSON, this will trigger a retry in the ask() method
            raise ValueError(f"Error loading agent message: {json_lst}")

        # Handle different response formats based on query type
        if query_type == "prior":
            resp = resp_dict["operation_scores"]
            # Validate response format and ensure it's a dictionary
            original_action_list = kwargs.get("action_list", {})

            # Convert the action list to a dictionary if it's not already
            if isinstance(original_action_list, list):
                original_action_list = {i: action for i, action in enumerate(original_action_list)}

            # Check if the response is in the correct dictionary format
            if not isinstance(resp, dict):
                print(f"Warning: Prior response is not a dictionary: {resp}")
                # Try to convert the response if it's a list
                if isinstance(resp, list):
                    # Check if the response list has the same length as the action list
                    if len(resp) != len(original_action_list):
                        error_msg = f"Prior response list length ({len(resp)}) doesn't match action list length ({len(original_action_list)})"
                        print(f"Warning: {error_msg}")

                        # Try to correct the format with a specific message
                        retry_json_lst, full_response, _ = self._handle_retry(self.instructions.correct_length)

                        if retry_json_lst:
                            retry_resp_dict = check_json_list(retry_json_lst)
                            if retry_resp_dict and "operation_scores" in retry_resp_dict:
                                retry_resp = retry_resp_dict["operation_scores"]

                                # Check if the corrected response is properly formatted
                                if isinstance(retry_resp, dict):
                                    # Use the corrected dictionary response
                                    resp = retry_resp
                                    print("Successfully obtained dictionary response after correction.")
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
        elif query_type == "value":
            resp = resp_dict["state_value_estimation"]
            # Validate the value is a number between 0 and 1
            if not isinstance(resp, (int, float)):
                try:
                    resp = float(resp)
                except (ValueError, TypeError):
                    raise ValueError(f"Could not convert value response to float: {resp}")

            # Ensure value is between 0 and 1
            if not (0 <= resp <= 1):
                raise ValueError(f"Value {resp} is outside the range [0,1]")
        elif query_type == "explore":
            resp = resp_dict["explore"]
        elif query_type == "child_values":
            resp = resp_dict["operation_values"]
            # Validate response format and ensure it's a dictionary
            original_action_list = kwargs.get("action_list", {})

            # Convert the action list to a dictionary if it's not already
            if isinstance(original_action_list, list):
                original_action_list = {i: action for i, action in enumerate(original_action_list)}

            # Check if the response is in the correct dictionary format
            if not isinstance(resp, dict):
                print(f"Warning: Child values response is not a dictionary: {resp}")
                # Try to convert the response if it's a list
                if isinstance(resp, list):
                    # Check if the response list has the same length as the action list
                    if len(resp) != len(original_action_list):
                        error_msg = f"Child values response list length ({len(resp)}) doesn't match action list length ({len(original_action_list)})"
                        print(f"Warning: {error_msg}")

                        # Try to correct the format with a specific message
                        retry_json_lst, full_response, _ = self._handle_retry(self.instructions.correct_length)

                        if retry_json_lst:
                            retry_resp_dict = check_json_list(retry_json_lst)
                            if retry_resp_dict and "operation_values" in retry_resp_dict:
                                retry_resp = retry_resp_dict["operation_values"]

                                # Check if the corrected response is properly formatted
                                if isinstance(retry_resp, dict):
                                    # Use the corrected dictionary response
                                    resp = retry_resp
                                    print("Successfully obtained dictionary response after correction.")
                                elif isinstance(retry_resp, list) and len(retry_resp) == len(original_action_list):
                                    # If still a list but now correct length, convert to dictionary
                                    resp = {str(i): value for i, value in enumerate(retry_resp)}
                                    print(f"Converted corrected response list to dictionary: {resp}")
                                else:
                                    # Still incorrect format or length
                                    raise ValueError(
                                        f"Child values response format or length still incorrect after retry")
                            else:
                                # No valid operation_values in retry response
                                raise ValueError(f"No valid operation_values in retry response: {retry_resp_dict}")
                        else:
                            # No valid JSON in retry response, use original error
                            raise ValueError(error_msg)
                    else:
                        # List has correct length, convert to dictionary
                        try:
                            resp = {str(i): value for i, value in enumerate(resp)}
                            print(f"Converted child values response from list to dictionary: {resp}")
                        except Exception as e:
                            raise ValueError(f"Could not convert child values response to dictionary: {e}")

            # Ensure all values in the dictionary are floats between 0 and 1
            validated_resp = {}
            for key, value in resp.items():
                try:
                    value_float = float(value)
                    if 0 <= value_float <= 1:
                        validated_resp[key] = value_float
                    else:
                        raise ValueError(f"Child value {value_float} for key {key} is outside the range [0,1]")
                except (ValueError, TypeError):
                    raise ValueError(f"Could not convert child value '{value}' for key {key} to float")

            # Replace the original response with the validated one
            resp = validated_resp
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
                        self.reset(query_type)
                        resp_dict = self._ask(query_type=query_type, attempt_num=attempt.retry_state.attempt_number,
                                              **kwargs)
                        return resp_dict
                    except (openai.BadRequestError, openai.RateLimitError) as e:
                        if any(phrase in str(e).lower() for phrase in
                               ["maximum context length", "content too long", "token limit", "tokens in your"]):
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
