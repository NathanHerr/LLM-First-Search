"""
Countdown node implementation for the Countdown game.
"""
import itertools

from src.base_game.base_node import BaseNode
from src.utils.countdown_utils import combine_nums


class CountdownNode(BaseNode):
    def __init__(self, idx, parent, nums, operations, heuristic_func, target):
        super().__init__()
        self.nums = nums
        self.operations = operations
        self.heuristic_func = heuristic_func
        self.target = target
        self.parent = parent
        self.idx = idx
        # Compute the heuristic value for this node
        self.base_heuristic = self.compute_heuristic() if heuristic_func is not None else 0

    def compute_heuristic(self):
        """Compute the heuristic value for this node's state."""
        if self.heuristic_func is not None:
            return self.heuristic_func(self.nums, self.target)
        return 0

    def __lt__(self, other):
        return self.base_heuristic < other.base_heuristic

    def get_legal_next_nodes(self):
        # Generate successors for the current node
        next_possible_nodes = []
        node_index = 0
        for i, j in itertools.combinations(range(len(self.nums)), 2):
            for result, operation in combine_nums(self.nums[i], self.nums[j]):
                new_nums = [self.nums[k] for k in range(len(self.nums)) if
                            k != i and k != j] + [result]
                new_operations = self.operations + [operation]
                new_node = CountdownNode(node_index, self, new_nums, new_operations, self.heuristic_func, self.target)
                next_possible_nodes.append((None, new_node))
                node_index += 1
        return next_possible_nodes

    def select_next_node(self, next_node):
        """
        Return the new state after making the move.
        In this case the next move is just the next node - so dont need to do anything else.
        """
        return next_node

    def is_game_over(self):
        if len(self.nums) == 1:
            return True
        return False

    def game_result(self):
        # Maybe this should be the average value of all the heuristic socres for all the nodes in the path?
        if len(self.nums) == 1 and self.nums[0] == self.target:
            return 1
        return -1

    def get_action_description(self):
        """
        Returns a description of the action that led to this node.

        Returns:
            str: A string describing the action, or empty string for the root node.
        """
        if not self.operations:
            return ""
        return self.operations[-1]

    def get_state_description(self):
        """
        Returns a description of the state at this node.

        Returns:
            str: A string describing the current state.
        """
        return f"Target: {self.target}\nAvailable Numbers: {self.nums}\nOperations: {self.operations}"

