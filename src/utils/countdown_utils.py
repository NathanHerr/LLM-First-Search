"""
Utility functions for countdown.py
"""

from src.utils.common_utils import query_agent


def combine_nums(a, b):
    # Implicitly makes assumptions about the order of operations and valid operations
    a = int(a)
    b = int(b)
    possible = [[a + b, f"{a}+{b}={a + b}"], [a * b, f"{a}*{b}={a * b}"]]
    if a <= b:
        possible.append([b - a, f"{b}-{a}={b - a}"])
        if a != 0 and b % a == 0:
            possible.append([b // a, f"{b}/{a}={round(b // a, 0)}"])
    else:
        possible.append([a - b, f"{a}-{b}={a - b}"])
        if b != 0 and a % b == 0:
            possible.append([a // b, f"{a}/{b}={round(a // b, 0)}"])
    return possible


def mult_heuristic(nums, target):
    # get closer to factors of target
    # return sum([1 if (nums[i] == 0 or target % nums[i] == 0 or nums[i] % target == 0) else 0 for i in range(len(nums))])
    # softer version, with distance to factors
    factors = [i for i in range(2, target + 1) if target % i == 0]
    return sum([min(abs(num - factor) for factor in factors) for num in nums])


def evaluate_child_node_values(agent, explorer, current_node, token_usage):
    """Evaluate all child nodes of the current node using the agent and store the values. CountDown Specific. """
    # Skip if no children
    if not current_node.children:
        return token_usage

    # Get the current sequence for the request
    current_sequence = explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(current_node))

    # Get the possible actions from the children
    action_list = current_node.get_possible_actions()

    # Skip if no valid actions
    if not action_list:
        return token_usage

    # Ask the agent to evaluate the possible operations
    res, usage = query_agent(
        agent,
        query_type="child_values",
        current_sequence=current_sequence,
        action_list=action_list
    )
    token_usage.append(usage)

    # Check if the response contains the operation values
    if res["resp"] is None:
        return token_usage

    # Get the operation values (which have already been validated by the agent)
    operation_values = res["resp"]

    # Set the value of each child node
    for action_key, value in operation_values.items():
        try:
            action_index = int(action_key)
            if action_index < len(current_node.children):
                child_node = current_node.children[action_index]
                child_node.value = value
                # Set adjusted_value equal to value initially
                child_node.adjusted_value = value
        except (ValueError, TypeError):
            # Skip invalid keys - validation should have handled most cases already
            continue

    return token_usage


def evaluate_countdown_node_value(agent, explorer, node, token_usage):
    """Evaluate a Countdown node's value using the agent and store it in the node."""
    node_sequence = explorer.sa_pairs_to_string(explorer.path_to_sa_pairs(node))
    res, usage = query_agent(
        agent,
        query_type="value",
        current_sequence=node_sequence,
        action_list=node.get_possible_actions()
    )

    token_usage.append(usage)
    node.value = res["resp"]
    return node.value, token_usage


def get_countdown_data_path(data_dir, split, countdown_difficulty):
    """
    Get the path to the input data file for Countdown.

    Args:
        data_dir (str): Directory for data
        split (str): Data split (train, test, etc.)
        countdown_difficulty (str): Difficulty level for Countdown puzzles

    Returns:
        str: Path to the input data file
    """
    return f"{data_dir}/{split}_{countdown_difficulty}.json"
