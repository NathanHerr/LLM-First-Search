"""
Tree-based exploration utilities for game problems.
This module provides reusable classes for tree representation and exploration
that work with both Countdown and Sudoku games.
"""

from collections import deque


class PathNode:
    """
    Represents a node in the exploration path tree.
    Generic implementation that works with game nodes of any type.
    
    Game nodes must implement the following methods:
    - get_legal_next_nodes(): Returns an iterable of (action, node) pairs
    - get_action_description(): Returns a string description of the action that led to this node
    - get_state_description(): Returns a string description of the state at this node
    """

    def __init__(self, game_node, parent=None):
        """
        Initialize a PathNode with a game node.
        
        Args:
            game_node: The game node (must implement the required interface)
            parent (PathNode, optional): Parent node in the tree
        """
        self.children = []
        self.parent = parent
        self.expanded = False
        self.game_node = game_node
        self.value = None  # Store the node's estimated value
        self.adjusted_value = None  # Store adjusted value that incorporates children's values

    def expand(self):
        """
        Expands the current node by generating all possible next nodes.
        Sets the expanded flag to True.
        """
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
                child_node = PathNode(next_node, parent=self)
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
                        child_node = PathNode(new_node, parent=self)
                        self.children.append(child_node)

    def get_possible_actions(self):
        """
        Returns a dictionary mapping indices to operations represented by the children.
        This delegates to the game node methods for getting action descriptions.
        
        By default, only returns actions that lead to unexpanded children.
        """
        possible_next_actions = {}

        for i, child in enumerate(self.children):
            # Only include unexpanded children
            if not child.expanded:
                if hasattr(child.game_node, 'get_action_description'):
                    possible_next_actions[i] = child.game_node.get_action_description()
                else:
                    possible_next_actions[i] = f"Action {i}: {child.game_node}"

        return possible_next_actions

    def __str__(self):
        """String representation of the path node."""
        if hasattr(self.game_node, 'get_action_description'):
            action_desc = self.game_node.get_action_description()
            if action_desc:
                return f"PathNode: {action_desc}"
            return f"PathNode: Root node"
        else:
            return f"PathNode: {self.game_node}"


class Explorer:
    """
    Provides functionality for exploring and visualizing a tree of PathNodes.
    """

    def __init__(self, explorer_root):
        """
        Initialize an Explorer with a root PathNode.
        
        Args:
            explorer_root (PathNode): The root node of the tree to explore
        """
        self.explorer_root = explorer_root

    @property
    def frontier_nodes(self):
        """
        Returns all unexpanded nodes in the tree.
        Uses breadth-first search to traverse the entire tree.
        """
        return self.get_unexpanded_nodes()

    @property
    def nodes_with_unexpanded_children(self):
        """
        Returns all nodes in the tree that have at least one unexpanded child.
        These are nodes that still have available actions to take.
        
        Returns:
            list: A list of PathNode objects that have unexpanded children.
        """
        if self.explorer_root is None:
            return []

        nodes_with_unexpanded_children = []
        queue = deque([self.explorer_root])

        while queue:
            node = queue.popleft()

            has_unexpanded_children = False
            for child in node.children:
                # Add all expanded children to the queue for further processing
                if child.expanded:
                    queue.append(child)
                else:
                    # If we find at least one unexpanded child, this node is what we're looking for
                    has_unexpanded_children = True

            if has_unexpanded_children:
                nodes_with_unexpanded_children.append(node)

        return nodes_with_unexpanded_children

    def get_unexpanded_nodes(self):
        """
        Returns all nodes in the tree that have not been expanded yet.
        Uses breadth-first search to traverse the entire tree.
        
        Returns:
            list: A list of all unexpanded PathNode objects in the tree.
        """
        if self.explorer_root is None:
            return []

        unexpanded_nodes = []
        queue = deque([self.explorer_root])

        while queue:
            node = queue.popleft()

            # If the node is not expanded, add it to our results
            if not node.expanded:
                unexpanded_nodes.append(node)

            # Add all children to the queue for processing
            for child in node.children:
                queue.append(child)

        return unexpanded_nodes

    def tree_size(self):
        """
        Returns the total number of nodes in the tree.
        """
        if self.explorer_root is None:
            return 0
        stack = [self.explorer_root]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)  # Add all children to the stack
        return count

    def expanded_nodes_count(self):
        """
        Returns the number of expanded nodes in the tree.
        """
        if self.explorer_root is None:
            return 0
        stack = [self.explorer_root]
        expanded = 0
        while stack:
            node = stack.pop()
            if node.expanded:
                expanded += 1
            stack.extend(node.children)  # Add all children to the stack
        return expanded

    @staticmethod
    def path_to_sa_pairs(path):
        """
        Converts a single path to a list of state-action pairs.
        
        Args:
            path: A PathNode object representing a path in the tree.
            
        Returns:
            List of (state, action) tuples.
        """
        state_action_pairs = []

        # Generic implementation for all game types
        current_path = path
        while current_path.parent:
            # Get state description
            if hasattr(current_path.game_node, 'get_state_description'):
                state = current_path.game_node.get_state_description()
            else:
                state = str(current_path.game_node)

            # Get action description
            if hasattr(current_path.game_node, 'get_action_description'):
                action = current_path.game_node.get_action_description()
            else:
                action = "Move to next state"

            state_action_pairs.append((state, action))
            current_path = current_path.parent

        # Add the initial state
        if hasattr(current_path.game_node, 'get_state_description'):
            state = current_path.game_node.get_state_description()
        else:
            state = str(current_path.game_node)

        state_action_pairs.append((state, ""))

        return state_action_pairs

    @staticmethod
    def sa_pairs_to_string(sa_pairs):
        """
        Converts a list of state-action pairs to a single string representation.
        
        Args:
            sa_pairs: List of (state, action) tuples.
            
        Returns:
            String representation of the state-action pairs.
        """
        temp = ""
        for j, (s, a) in enumerate(sa_pairs[::-1]):
            if a:
                temp += f"**Action {j - 1}**\n{a}\n"

            # Add annotation for states after the first state
            if j > 0 and a:
                temp += f"**State {j}** (After performing {a})\n{s}\n\n"
            else:
                temp += f"**State {j}**\n{s}\n\n"
        return temp
