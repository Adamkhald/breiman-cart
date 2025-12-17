"""
Cost-Complexity Pruning for CART trees with optimizations.
Implements the complete pruning algorithm from Breiman et al. 1984.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import copy
import logging

from .node import Node


logger = logging.getLogger(__name__)


class CostComplexityPruner:
    """
    Implements cost-complexity pruning with performance optimizations.
    
    The cost-complexity measure is:
        R_alpha(T) = R(T) + alpha * |T_leaves|
    
    where R(T) is the misclassification error and |T_leaves| is the number of leaves.
    
    Parameters
    ----------
    criterion : str, default="gini"
        Criterion used for building the tree: "gini" or "mse"
    
    Attributes
    ----------
    X_train : pd.DataFrame
        Training data (stored for recalculating leaf values)
    y_train : np.ndarray
        Training labels
    alpha_cache : dict
        Cache for computed alpha values
    """
    
    def __init__(self, criterion: str = "gini"):
        if criterion not in ["gini", "mse"]:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        self.criterion = criterion
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.alpha_cache: Dict[int, float] = {}
    
    def set_training_data(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Store training data for use during pruning.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training targets
        """
        self.X_train = X.copy()
        self.y_train = y if not isinstance(y, pd.Series) else y.values
        logger.debug(f"Set training data: {len(X)} samples")
    
    def calculate_node_error(self, node: Node) -> float:
        """
        Calculate the error at a node.
        
        Parameters
        ----------
        node : Node
            Node to calculate error for
        
        Returns
        -------
        float
            Weighted impurity (impurity * n_samples)
        """
        return node.impurity * node.n_samples
    
    def calculate_subtree_cost(self, node: Node, alpha: float = 0.0) -> float:
        """
        Calculate cost-complexity measure.
        
        Parameters
        ----------
        node : Node
            Root of subtree
        alpha : float, default=0.0
            Complexity parameter
        
        Returns
        -------
        float
            Cost-complexity measure: R_alpha(T) = R(T) + alpha * |T_leaves|
        """
        error = node.get_subtree_error()
        n_leaves = node.count_leaves()
        return error + alpha * n_leaves
    
    def calculate_alpha_for_node(self, node: Node) -> float:
        """
        Calculate effective alpha for pruning at this node.
        
        This is the alpha value at which pruning this node becomes beneficial.
        
        Parameters
        ----------
        node : Node
            Node to calculate alpha for
        
        Returns
        -------
        float
            Effective alpha: (R(t) - R(T_t)) / (|T_t_leaves| - 1)
        
        Notes
        -----
        Alpha represents the cost per leaf of keeping the subtree.
        When alpha exceeds this value, it's better to prune.
        """
        if node.is_leaf:
            return np.inf
        
        # Check cache
        node_id = id(node)
        if node_id in self.alpha_cache:
            return self.alpha_cache[node_id]
        
        node_error = self.calculate_node_error(node)
        subtree_error = node.get_subtree_error()
        n_leaves = node.count_leaves()
        
        if n_leaves <= 1:
            alpha = np.inf
        else:
            alpha = (node_error - subtree_error) / (n_leaves - 1)
            alpha = max(0, alpha)  # Alpha should be non-negative
        
        # Cache result
        self.alpha_cache[node_id] = alpha
        return alpha
    
    def find_weakest_link(self, node: Node) -> Tuple[Node, float]:
        """
        Find the node with smallest effective alpha (weakest link).
        
        This is the node that should be pruned next in the pruning sequence.
        
        Parameters
        ----------
        node : Node
            Root of subtree to search
        
        Returns
        -------
        weakest_node : Node
            Node with minimum alpha
        min_alpha : float
            The minimum alpha value found
        """
        if node.is_leaf:
            return node, np.inf
        
        min_alpha = self.calculate_alpha_for_node(node)
        weakest_node = node
        
        # Recursively check children
        left_node, left_alpha = self.find_weakest_link(node.left)
        if left_alpha < min_alpha:
            min_alpha = left_alpha
            weakest_node = left_node
        
        right_node, right_alpha = self.find_weakest_link(node.right)
        if right_alpha < min_alpha:
            min_alpha = right_alpha
            weakest_node = right_node
        
        return weakest_node, min_alpha
    
    def get_samples_at_node(
        self,
        node: Node,
        X: pd.DataFrame,
        y: np.ndarray,
        current_node: Node
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Traverse tree to find samples that reach a specific node.
        
        Parameters
        ----------
        node : Node
            Target node to find samples for
        X : pd.DataFrame
            Feature data
        y : np.ndarray
            Target data
        current_node : Node
            Current node in traversal (start with root)
        
        Returns
        -------
        X_node : pd.DataFrame
            Features of samples at target node
        y_node : np.ndarray
            Targets of samples at target node
        """
        # Use cached indices if available
        if node.sample_indices is not None:
            return X.iloc[node.sample_indices], y[node.sample_indices]
        
        if current_node is node:
            return X, y
        
        if current_node.is_leaf:
            return pd.DataFrame(), np.array([])
        
        # Split data according to current node's rule
        if current_node.is_categorical:
            feature_name = X.columns[current_node.feature_index]
            left_mask = X[feature_name].isin(current_node.categorical_subset)
        else:
            feature_name = X.columns[current_node.feature_index]
            left_mask = X[feature_name] <= current_node.threshold
        
        # Check left subtree
        X_left = X[left_mask]
        y_left = y[left_mask.values]
        result_X, result_y = self.get_samples_at_node(node, X_left, y_left, current_node.left)
        if len(result_X) > 0:
            return result_X, result_y
        
        # Check right subtree
        X_right = X[~left_mask]
        y_right = y[(~left_mask).values]
        return self.get_samples_at_node(node, X_right, y_right, current_node.right)
    
    def calculate_leaf_value(self, y: np.ndarray) -> float:
        """
        Calculate appropriate prediction value for a leaf.
        
        Parameters
        ----------
        y : np.ndarray
            Target values at the node
        
        Returns
        -------
        float or int
            Predicted value (majority class or mean)
        """
        if len(y) == 0:
            return 0.0
        
        if self.criterion == "gini":
            # Classification: return majority class
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]
        else:
            # Regression: return mean
            return np.mean(y)
    
    def convert_to_leaf(
        self,
        node: Node,
        root: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> None:
        """
        Convert an internal node to a leaf with proper prediction value.
        
        Parameters
        ----------
        node : Node
            Node to convert to leaf
        root : Node
            Root of the tree
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training targets
        """
        # Get samples that reach this node
        X_node, y_node = self.get_samples_at_node(node, X, y, root)
        
        # Calculate appropriate prediction
        if len(y_node) > 0:
            node.predicted_value = self.calculate_leaf_value(y_node)
            node.impurity = (
                self.gini_impurity(y_node) if self.criterion == "gini"
                else self.mse(y_node)
            )
        else:
            # Fallback if we can't find samples
            logger.warning("Could not find samples for node during pruning")
            node.predicted_value = 0
        
        # Convert to leaf
        node.is_leaf = True
        node.left = None
        node.right = None
    
    def gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0.0
        return np.var(y)
    
    def prune_tree_at_node(
        self,
        target_node: Node,
        root: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Node:
        """
        Create a pruned copy of the tree with target_node converted to leaf.
        
        Parameters
        ----------
        target_node : Node
            Node to prune
        root : Node
            Root of the tree
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training targets
        
        Returns
        -------
        Node
            Root of pruned tree
        """
        pruned_root = copy.deepcopy(root)
        
        # Build mapping from original to copied nodes
        node_map = {}
        
        def map_nodes(original, copied):
            node_map[id(original)] = copied
            if not original.is_leaf:
                map_nodes(original.left, copied.left)
                map_nodes(original.right, copied.right)
        
        map_nodes(root, pruned_root)
        
        # Get the copied version of target node
        copied_target = node_map.get(id(target_node))
        
        if copied_target is not None:
            self.convert_to_leaf(copied_target, pruned_root, X, y)
        
        # Clear alpha cache as tree structure changed
        self.alpha_cache.clear()
        
        return pruned_root
    
    def generate_pruning_sequence(
        self,
        root: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[Tuple[Node, float]]:
        """
        Generate the complete sequence of pruned trees.
        
        This implements the weakest link pruning algorithm from Breiman 1984.
        
        Parameters
        ----------
        root : Node
            Root of the tree to prune
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training targets
        
        Returns
        -------
        list of (Node, float)
            List of (tree_root, alpha) tuples representing the pruning sequence
        """
        sequence = [(copy.deepcopy(root), 0.0)]
        current_tree = copy.deepcopy(root)
        
        logger.debug(f"Starting pruning sequence with {current_tree.count_leaves()} leaves")
        
        iteration = 0
        while current_tree.count_leaves() > 1:
            iteration += 1
            
            # Find weakest link
            weakest_node, alpha = self.find_weakest_link(current_tree)
            
            if alpha == np.inf:
                logger.debug("No more nodes to prune")
                break
            
            logger.debug(
                f"Iteration {iteration}: Pruning node with alpha={alpha:.6f}, "
                f"leaves before={current_tree.count_leaves()}"
            )
            
            # Prune at weakest link
            current_tree = self.prune_tree_at_node(weakest_node, current_tree, X, y)
            sequence.append((copy.deepcopy(current_tree), alpha))
            
            logger.debug(f"Leaves after={current_tree.count_leaves()}")
        
        logger.debug(f"Pruning sequence complete with {len(sequence)} trees")
        return sequence
    
    def prune_tree(
        self,
        root: Node,
        alpha: float,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Node:
        """
        Prune tree for a given alpha value.
        
        Parameters
        ----------
        root : Node
            Root of tree to prune
        alpha : float
            Complexity parameter
        X : pd.DataFrame
            Training features
        y : np.ndarray
            Training targets
        
        Returns
        -------
        Node
            Root of pruned tree
        """
        pruned_tree = copy.deepcopy(root)
        
        # Iteratively prune at weakest links where alpha_node <= alpha
        improved = True
        iterations = 0
        max_iterations = 1000  # Safety limit
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            weakest_node, node_alpha = self.find_weakest_link(pruned_tree)
            
            if node_alpha <= alpha and node_alpha < np.inf:
                logger.debug(f"Pruning iteration {iterations}: node_alpha={node_alpha:.6f} <= alpha={alpha:.6f}")
                pruned_tree = self.prune_tree_at_node(weakest_node, pruned_tree, X, y)
                improved = True
        
        if iterations >= max_iterations:
            logger.warning(f"Pruning reached maximum iterations ({max_iterations})")
        
        return pruned_tree
    
    def evaluate_tree(
        self,
        node: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> float:
        """
        Evaluate tree error on validation set.
        
        Parameters
        ----------
        node : Node
            Root of tree to evaluate
        X : pd.DataFrame
            Validation features
        y : np.ndarray
            Validation targets
        
        Returns
        -------
        float
            Error measure (misclassification rate or MSE)
        """
        predictions = []
        
        for idx in range(len(X)):
            x = X.iloc[idx].values
            pred = node.predict_sample(x)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.criterion == "gini":
            # Classification: misclassification rate
            error = np.mean(predictions != y)
        else:
            # Regression: MSE
            error = np.mean((predictions - y) ** 2)
        
        return error
    
    def find_optimal_alpha(
        self,
        root: Node,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> float:
        """
        Find optimal alpha using validation set.
        
        Parameters
        ----------
        root : Node
            Root of tree to prune
        X_train : pd.DataFrame
            Training features
        y_train : np.ndarray
            Training targets
        X_val : pd.DataFrame
            Validation features
        y_val : np.ndarray
            Validation targets
        
        Returns
        -------
        float
            Optimal alpha value
        """
        # Generate pruning sequence
        logger.info("Generating pruning sequence...")
        sequence = self.generate_pruning_sequence(root, X_train, y_train)
        
        best_alpha = 0.0
        best_error = np.inf
        
        logger.info(f"Evaluating {len(sequence)} candidate trees...")
        
        for i, (tree_root, alpha) in enumerate(sequence):
            error = self.evaluate_tree(tree_root, X_val, y_val)
            
            logger.debug(
                f"Tree {i}: alpha={alpha:.6f}, leaves={tree_root.count_leaves()}, "
                f"val_error={error:.4f}"
            )
            
            if error < best_error:
                best_error = error
                best_alpha = alpha
        
        logger.info(f"Optimal alpha={best_alpha:.6f} with validation error={best_error:.4f}")
        
        return best_alpha