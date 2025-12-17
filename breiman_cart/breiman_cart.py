"""
Breiman CART Implementation (1984) - User-Friendly API
=======================================================

A pure NumPy/Pandas implementation of Classification and Regression Trees
following the original methodology from Breiman, Friedman, Olshen, and Stone (1984).

This module provides three main classes for easy use:
- BRCRegression: For regression tasks
- BRCClassification: For classification tasks  
- BRCInference: For making predictions with trained models

Author: Adam Khald
Version: 0.2.0
License: MIT
"""

from typing import Optional, List, Dict, Any, Set, Union, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import pickle
import json
import logging
from itertools import combinations
import warnings
import copy

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ============================================================================
# NODE CLASS - Tree Structure Building Block
# ============================================================================

class Node:
    """
    A node in the CART decision tree.
    
    Parameters
    ----------
    feature_index : int, optional
        Index of feature to split on (None for leaf nodes)
    threshold : float, optional
        Threshold value for numerical splits
    categorical_subset : set, optional
        Set of categories for left child (categorical splits)
    is_categorical : bool, default=False
        Whether this split is categorical
    left : Node, optional
        Left child node
    right : Node, optional
        Right child node
    predicted_value : int, float, or np.ndarray, optional
        Prediction value for leaf nodes
    impurity : float, default=0.0
        Node impurity (Gini or MSE)
    n_samples : int, default=0
        Number of samples at this node
    sample_indices : np.ndarray, optional
        Indices of samples at this node (for efficient pruning)
    """
    
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        categorical_subset: Optional[Set[Any]] = None,
        is_categorical: bool = False,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        predicted_value: Optional[Union[int, float, 'NDArray[np.float64]']] = None,
        impurity: float = 0.0,
        n_samples: int = 0,
        sample_indices: Optional['NDArray[np.int64]'] = None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.categorical_subset = categorical_subset
        self.is_categorical = is_categorical
        self.left = left
        self.right = right
        self.predicted_value = predicted_value
        self.impurity = impurity
        self.n_samples = n_samples
        self.sample_indices = sample_indices
        self.is_leaf = predicted_value is not None
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate node consistency."""
        if not self.is_leaf:
            if self.feature_index is None:
                raise ValueError("Non-leaf node must have feature_index")
            if self.feature_index < 0:
                raise ValueError(f"Invalid feature_index: {self.feature_index}")
            
            if self.is_categorical:
                if self.categorical_subset is None or len(self.categorical_subset) == 0:
                    raise ValueError("Categorical split must have non-empty subset")
            else:
                if self.threshold is None:
                    raise ValueError("Numerical split must have threshold")
    
    def predict_sample(self, x: 'NDArray[np.float64]') -> Union[int, float, 'NDArray[np.float64]']:
        """Predict a single sample by traversing the tree."""
        if len(x) == 0:
            raise ValueError("Empty feature vector provided")
        
        if self.is_leaf:
            return self.predicted_value
        
        if self.feature_index >= len(x):
            raise IndexError(
                f"Feature index {self.feature_index} out of bounds for vector of length {len(x)}"
            )
        
        if self.is_categorical:
            feature_value = x[self.feature_index]
            if feature_value in self.categorical_subset:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
        else:
            if x[self.feature_index] <= self.threshold:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
    
    def count_leaves(self) -> int:
        """Count the number of leaf nodes in this subtree."""
        if self.is_leaf:
            return 1
        return self.left.count_leaves() + self.right.count_leaves()
    
    def get_depth(self) -> int:
        """Calculate the depth of this subtree."""
        if self.is_leaf:
            return 0
        return 1 + max(self.left.get_depth(), self.right.get_depth())
    
    def get_n_nodes(self) -> int:
        """Count total nodes in this subtree."""
        if self.is_leaf:
            return 1
        return 1 + self.left.get_n_nodes() + self.right.get_n_nodes()
    
    def get_subtree_error(self) -> float:
        """Calculate total error for this subtree."""
        if self.is_leaf:
            return self.impurity * self.n_samples
        return self.left.get_subtree_error() + self.right.get_subtree_error()
    
    def get_leaves(self) -> List['Node']:
        """Get all leaf nodes in this subtree."""
        if self.is_leaf:
            return [self]
        leaves = []
        leaves.extend(self.left.get_leaves())
        leaves.extend(self.right.get_leaves())
        return leaves
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {
            'is_leaf': self.is_leaf,
            'n_samples': self.n_samples,
            'impurity': float(self.impurity)
        }
        
        if self.is_leaf:
            if isinstance(self.predicted_value, np.ndarray):
                result['predicted_value'] = self.predicted_value.tolist()
            else:
                result['predicted_value'] = self.predicted_value
        else:
            result['feature_index'] = self.feature_index
            result['is_categorical'] = self.is_categorical
            
            if self.is_categorical:
                result['categorical_subset'] = list(self.categorical_subset)
            else:
                result['threshold'] = float(self.threshold)
            
            result['left'] = self.left.to_dict()
            result['right'] = self.right.to_dict()
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the node."""
        if self.is_leaf:
            return f"Leaf(value={self.predicted_value}, n={self.n_samples}, impurity={self.impurity:.3f})"
        if self.is_categorical:
            return f"Node(feat={self.feature_index}, subset={self.categorical_subset}, n={self.n_samples})"
        return f"Node(feat={self.feature_index}, thr={self.threshold:.3f}, n={self.n_samples})"


# ============================================================================
# SPLITTER CLASS - Finding Optimal Splits
# ============================================================================

class Splitter:
    """
    Handles finding the best split for a node.
    Supports both numerical and categorical features with performance optimizations.
    """
    
    def __init__(
        self,
        criterion: str = "gini",
        categorical_features: Optional[List[str]] = None,
        max_categories_exhaustive: int = 8
    ):
        if criterion not in ["gini", "mse"]:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'gini' or 'mse'")
        
        self.criterion = criterion
        self.categorical_features = categorical_features or []
        self.max_categories_exhaustive = max_categories_exhaustive
    
    def gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0.0
        
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == "gini":
            return self.gini_impurity(y)
        elif self.criterion == "mse":
            return self.mse(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def information_gain(
        self,
        parent_y: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray
    ) -> float:
        """Calculate information gain from a split."""
        n = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_impurity = self.calculate_impurity(parent_y)
        left_impurity = self.calculate_impurity(left_y)
        right_impurity = self.calculate_impurity(right_y)
        
        weighted_child_impurity = (
            (n_left / n) * left_impurity +
            (n_right / n) * right_impurity
        )
        
        return parent_impurity - weighted_child_impurity
    
    def best_numerical_split(
        self,
        X_column: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Find best threshold for a numerical feature."""
        y = np.asarray(y)
        
        sorted_indices = np.argsort(X_column)
        X_sorted = X_column[sorted_indices]
        y_sorted = y[sorted_indices]
        
        best_gain = -np.inf
        best_threshold = None
        
        unique_values = np.unique(X_sorted)
        
        if len(unique_values) <= 1:
            return None, 0.0
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = X_sorted <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gain = self.information_gain(
                y_sorted,
                y_sorted[left_mask],
                y_sorted[right_mask]
            )
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def best_categorical_split(
        self,
        X_column: pd.Series,
        y: np.ndarray
    ) -> Tuple[Optional[Set[Any]], float]:
        """Find best subset split for a categorical feature."""
        y = np.asarray(y)
        
        unique_categories = X_column.unique()
        
        if len(unique_categories) <= 1:
            return None, 0.0
        
        # For binary classification, use optimal ordering trick
        if self.criterion == "gini" and len(np.unique(y)) == 2:
            return self._binary_class_categorical_split(X_column, y, unique_categories)
        
        # Decide between exhaustive and greedy
        if len(unique_categories) <= self.max_categories_exhaustive:
            return self._exhaustive_categorical_split(X_column, y, unique_categories)
        else:
            warnings.warn(
                f"Feature has {len(unique_categories)} categories. "
                f"Using greedy heuristic (threshold: {self.max_categories_exhaustive})",
                UserWarning
            )
            return self._greedy_categorical_split(X_column, y, unique_categories)
    
    def _binary_class_categorical_split(
        self,
        X_column: pd.Series,
        y: np.ndarray,
        categories: np.ndarray
    ) -> Tuple[Optional[Set[Any]], float]:
        """Optimal categorical split for binary classification."""
        category_proportions = []
        for cat in categories:
            mask = (X_column == cat).values
            if np.sum(mask) > 0:
                prop = np.mean(y[mask])
                category_proportions.append((cat, prop))
        
        category_proportions.sort(key=lambda x: x[1])
        sorted_categories = [cat for cat, _ in category_proportions]
        
        best_gain = -np.inf
        best_subset = None
        
        for i in range(1, len(sorted_categories)):
            subset = set(sorted_categories[:i])
            
            left_mask = X_column.isin(subset).values
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gain = self.information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_subset = subset
        
        return best_subset, best_gain
    
    def _exhaustive_categorical_split(
        self,
        X_column: pd.Series,
        y: np.ndarray,
        categories: np.ndarray
    ) -> Tuple[Optional[Set[Any]], float]:
        """Exhaustive search for categorical splits."""
        best_gain = -np.inf
        best_subset = None
        
        n_cats = len(categories)
        for r in range(1, n_cats):
            for subset_tuple in combinations(categories, r):
                subset = set(subset_tuple)
                
                left_mask = X_column.isin(subset).values
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_subset = subset
        
        return best_subset, best_gain
    
    def _greedy_categorical_split(
        self,
        X_column: pd.Series,
        y: np.ndarray,
        categories: np.ndarray
    ) -> Tuple[Optional[Set[Any]], float]:
        """Greedy heuristic for categorical splits with many categories."""
        category_means = []
        for cat in categories:
            mask = (X_column == cat).values
            if np.sum(mask) > 0:
                mean_y = np.mean(y[mask])
                category_means.append((cat, mean_y))
        
        category_means.sort(key=lambda x: x[1])
        sorted_categories = [cat for cat, _ in category_means]
        
        best_gain = -np.inf
        best_subset = None
        
        for i in range(1, len(sorted_categories)):
            subset = set(sorted_categories[:i])
            
            left_mask = X_column.isin(subset).values
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gain = self.information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_subset = subset
        
        return best_subset, best_gain
    
    def find_best_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Optional[Set[Any]], bool, float]:
        """Find the best split across all features."""
        y = np.asarray(y)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_subset = None
        best_is_categorical = False
        
        for feature_idx, feature_name in enumerate(X.columns):
            is_categorical = feature_name in self.categorical_features
            
            if is_categorical:
                subset, gain = self.best_categorical_split(X[feature_name], y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_subset = subset
                    best_threshold = None
                    best_is_categorical = True
            else:
                threshold, gain = self.best_numerical_split(
                    X[feature_name].values, y
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_subset = None
                    best_is_categorical = False
        
        return best_feature, best_threshold, best_subset, best_is_categorical, best_gain


# ============================================================================
# PRUNER CLASS - Cost-Complexity Pruning
# ============================================================================

class CostComplexityPruner:
    """
    Implements cost-complexity pruning with performance optimizations.
    """
    
    def __init__(self, criterion: str = "gini"):
        if criterion not in ["gini", "mse"]:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        self.criterion = criterion
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.alpha_cache: Dict[int, float] = {}
    
    def set_training_data(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Store training data for use during pruning."""
        self.X_train = X.copy()
        self.y_train = y if not isinstance(y, pd.Series) else y.values
        logger.debug(f"Set training data: {len(X)} samples")
    
    def calculate_node_error(self, node: Node) -> float:
        """Calculate the error at a node."""
        return node.impurity * node.n_samples
    
    def calculate_subtree_cost(self, node: Node, alpha: float = 0.0) -> float:
        """Calculate cost-complexity measure."""
        error = node.get_subtree_error()
        n_leaves = node.count_leaves()
        return error + alpha * n_leaves
    
    def calculate_alpha_for_node(self, node: Node) -> float:
        """Calculate effective alpha for pruning at this node."""
        if node.is_leaf:
            return np.inf
        
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
            alpha = max(0, alpha)
        
        self.alpha_cache[node_id] = alpha
        return alpha
    
    def find_weakest_link(self, node: Node) -> Tuple[Node, float]:
        """Find the node with smallest effective alpha (weakest link)."""
        if node.is_leaf:
            return node, np.inf
        
        min_alpha = self.calculate_alpha_for_node(node)
        weakest_node = node
        
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
        """Traverse tree to find samples that reach a specific node."""
        if node.sample_indices is not None:
            return X.iloc[node.sample_indices], y[node.sample_indices]
        
        if current_node is node:
            return X, y
        
        if current_node.is_leaf:
            return pd.DataFrame(), np.array([])
        
        if current_node.is_categorical:
            feature_name = X.columns[current_node.feature_index]
            left_mask = X[feature_name].isin(current_node.categorical_subset)
        else:
            feature_name = X.columns[current_node.feature_index]
            left_mask = X[feature_name] <= current_node.threshold
        
        X_left = X[left_mask]
        y_left = y[left_mask.values]
        result_X, result_y = self.get_samples_at_node(node, X_left, y_left, current_node.left)
        if len(result_X) > 0:
            return result_X, result_y
        
        X_right = X[~left_mask]
        y_right = y[(~left_mask).values]
        return self.get_samples_at_node(node, X_right, y_right, current_node.right)
    
    def calculate_leaf_value(self, y: np.ndarray) -> float:
        """Calculate appropriate prediction value for a leaf."""
        if len(y) == 0:
            return 0.0
        
        if self.criterion == "gini":
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]
        else:
            return np.mean(y)
    
    def convert_to_leaf(
        self,
        node: Node,
        root: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> None:
        """Convert an internal node to a leaf."""
        X_node, y_node = self.get_samples_at_node(node, X, y, root)
        
        if len(y_node) > 0:
            node.predicted_value = self.calculate_leaf_value(y_node)
            node.impurity = (
                self.gini_impurity(y_node) if self.criterion == "gini"
                else self.mse(y_node)
            )
        else:
            logger.warning("Could not find samples for node during pruning")
            node.predicted_value = 0
        
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
        """Create a pruned copy of the tree."""
        pruned_root = copy.deepcopy(root)
        
        node_map = {}
        
        def map_nodes(original, copied):
            node_map[id(original)] = copied
            if not original.is_leaf:
                map_nodes(original.left, copied.left)
                map_nodes(original.right, copied.right)
        
        map_nodes(root, pruned_root)
        
        copied_target = node_map.get(id(target_node))
        
        if copied_target is not None:
            self.convert_to_leaf(copied_target, pruned_root, X, y)
        
        self.alpha_cache.clear()
        
        return pruned_root
    
    def generate_pruning_sequence(
        self,
        root: Node,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[Tuple[Node, float]]:
        """Generate the complete sequence of pruned trees."""
        sequence = [(copy.deepcopy(root), 0.0)]
        current_tree = copy.deepcopy(root)
        
        logger.debug(f"Starting pruning sequence with {current_tree.count_leaves()} leaves")
        
        iteration = 0
        while current_tree.count_leaves() > 1:
            iteration += 1
            
            weakest_node, alpha = self.find_weakest_link(current_tree)
            
            if alpha == np.inf:
                logger.debug("No more nodes to prune")
                break
            
            logger.debug(
                f"Iteration {iteration}: Pruning node with alpha={alpha:.6f}, "
                f"leaves before={current_tree.count_leaves()}"
            )
            
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
        """Prune tree for a given alpha value."""
        pruned_tree = copy.deepcopy(root)
        
        improved = True
        iterations = 0
        max_iterations = 1000
        
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
        """Evaluate tree error on validation set."""
        predictions = []
        
        for idx in range(len(X)):
            x = X.iloc[idx].values
            pred = node.predict_sample(x)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.criterion == "gini":
            error = np.mean(predictions != y)
        else:
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
        """Find optimal alpha using validation set."""
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


# ============================================================================
# BASE CLASS - Common Functionality
# ============================================================================

class CARTBase:
    """
    Base class for CART trees with common functionality.
    """
    
    def __init__(
        self,
        criterion: str,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        categorical_features: Optional[List[str]] = None,
        verbose: int = 0
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features or []
        self.verbose = verbose
        
        if verbose > 0:
            logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        
        self.root: Optional[Node] = None
        self.splitter: Optional[Splitter] = None
        self.feature_names: Optional[List[str]] = None
        self.pruner: Optional[CostComplexityPruner] = None
        
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        
        self.training_history_: Dict[str, Any] = {}
    
    def _validate_input(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        if len(X) != len(y):
            raise ValueError(f"X and y lengths don't match: {len(X)} != {len(y)}")
        
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"X contains NaN values in columns: {nan_cols}")
        
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")
        
        for cat_feat in self.categorical_features:
            if cat_feat not in X.columns:
                raise ValueError(f"Categorical feature '{cat_feat}' not found in X")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'CARTBase':
        """Fit the CART tree to training data."""
        self._validate_input(X, y)
        
        self.X_train = X.copy()
        self.y_train = y if not isinstance(y, pd.Series) else y.values
        
        self.feature_names = list(X.columns)
        self.splitter = Splitter(
            criterion=self.criterion,
            categorical_features=self.categorical_features
        )
        
        logger.info(f"Building CART tree with {len(X)} samples and {len(X.columns)} features")
        logger.info(f"Criterion: {self.criterion}, Max depth: {self.max_depth}")
        
        self.root = self._grow_tree(X, self.y_train, depth=0)
        
        self.training_history_ = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'max_depth_reached': self.root.get_depth(),
            'n_leaves': self.root.count_leaves(),
            'n_nodes': self.root.get_n_nodes()
        }
        
        if self.verbose > 0:
            self._print_tree_info()
        
        return self
    
    def _print_tree_info(self) -> None:
        """Print information about the fitted tree."""
        print("\n" + "="*50)
        print("Tree Training Complete")
        print("="*50)
        print(f"Samples: {self.training_history_['n_samples']}")
        print(f"Features: {self.training_history_['n_features']}")
        print(f"Max depth: {self.training_history_['max_depth_reached']}")
        print(f"Number of leaves: {self.training_history_['n_leaves']}")
        print(f"Total nodes: {self.training_history_['n_nodes']}")
        print("="*50 + "\n")
    
    def _grow_tree(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        depth: int,
        sample_indices: Optional[np.ndarray] = None
    ) -> Node:
        """Recursively grow the tree."""
        if isinstance(y, pd.Series):
            y = y.values
        
        n_samples = len(y)
        impurity = self.splitter.calculate_impurity(y)
        
        if sample_indices is None:
            sample_indices = np.arange(n_samples)
        
        if self.verbose > 1:
            logger.debug(f"Growing node at depth {depth}, samples={n_samples}, impurity={impurity:.4f}")
        
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            impurity == 0.0
        )
        
        if should_stop:
            if self.verbose > 1:
                reason = (
                    "max depth" if self.max_depth is not None and depth >= self.max_depth
                    else "min samples split" if n_samples < self.min_samples_split
                    else "min samples leaf" if n_samples < 2 * self.min_samples_leaf
                    else "pure node"
                )
                logger.debug(f"Creating leaf (reason: {reason})")
            
            return self._create_leaf(y, impurity, n_samples, sample_indices)
        
        feature_idx, threshold, subset, is_categorical, gain = \
            self.splitter.find_best_split(X, y)
        
        if feature_idx is None or gain <= 0:
            if self.verbose > 1:
                logger.debug("No beneficial split found, creating leaf")
            return self._create_leaf(y, impurity, n_samples, sample_indices)
        
        if is_categorical:
            feature_name = X.columns[feature_idx]
            left_mask = X[feature_name].isin(subset)
            right_mask = ~left_mask
        else:
            feature_name = X.columns[feature_idx]
            left_mask = X[feature_name] <= threshold
            right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            if self.verbose > 1:
                logger.debug(f"Split violates min_samples_leaf, creating leaf")
            return self._create_leaf(y, impurity, n_samples, sample_indices)
        
        if self.verbose > 1:
            logger.debug(
                f"Splitting on feature {feature_idx} ({'categorical' if is_categorical else 'numerical'}), "
                f"gain={gain:.4f}, left={np.sum(left_mask)}, right={np.sum(right_mask)}"
            )
        
        left_indices = sample_indices[left_mask.values] if sample_indices is not None else None
        right_indices = sample_indices[right_mask.values] if sample_indices is not None else None
        
        left_child = self._grow_tree(
            X[left_mask], y[left_mask.values], depth + 1, left_indices
        )
        right_child = self._grow_tree(
            X[right_mask], y[right_mask.values], depth + 1, right_indices
        )
        
        return Node(
            feature_index=feature_idx,
            threshold=threshold,
            categorical_subset=subset,
            is_categorical=is_categorical,
            left=left_child,
            right=right_child,
            impurity=impurity,
            n_samples=n_samples,
            sample_indices=sample_indices
        )
    
    def _create_leaf(
        self,
        y: np.ndarray,
        impurity: float,
        n_samples: int,
        sample_indices: Optional[np.ndarray] = None
    ) -> Node:
        """Create a leaf node with appropriate prediction."""
        raise NotImplementedError("Subclasses must implement _create_leaf")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values for samples in X."""
        if self.root is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        if len(X.columns) != len(self.feature_names):
            raise ValueError(
                f"X has {len(X.columns)} features but tree was fitted with "
                f"{len(self.feature_names)} features"
            )
        
        predictions = []
        for idx in range(len(X)):
            x = X.iloc[idx].values
            pred = self.root.predict_sample(x)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Calculate accuracy (classification) or R² (regression)."""
        predictions = self.predict(X)
        
        if self.criterion == "gini":
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def get_feature_importance(self) -> np.ndarray:
        """Calculate feature importance based on total impurity reduction."""
        if self.root is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        importances = np.zeros(len(self.feature_names))
        
        def traverse(node):
            if node.is_leaf:
                return
            
            total_samples = node.n_samples
            left_samples = node.left.n_samples
            right_samples = node.right.n_samples
            
            impurity_reduction = (
                node.impurity -
                (left_samples * node.left.impurity + 
                 right_samples * node.right.impurity) / total_samples
            )
            
            importances[node.feature_index] += impurity_reduction * total_samples
            
            traverse(node.left)
            traverse(node.right)
        
        traverse(self.root)
        
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances
    
    def prune(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        alpha: Optional[float] = None
    ) -> 'CARTBase':
        """Prune the tree using cost-complexity pruning."""
        if self.root is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not stored. Cannot prune.")
        
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        logger.info("Starting pruning process...")
        
        self.pruner = CostComplexityPruner(self.criterion)
        
        if alpha is None:
            logger.info("Finding optimal alpha using validation set...")
            alpha = self.pruner.find_optimal_alpha(
                self.root,
                self.X_train,
                self.y_train,
                X_val,
                y_val
            )
            logger.info(f"Optimal alpha: {alpha:.6f}")
        
        logger.info(f"Pruning tree with alpha={alpha:.6f}")
        leaves_before = self.root.count_leaves()
        
        self.root = self.pruner.prune_tree(
            self.root,
            alpha,
            self.X_train,
            self.y_train
        )
        
        leaves_after = self.root.count_leaves()
        logger.info(f"Pruning complete: {leaves_before} -> {leaves_after} leaves")
        
        self.training_history_['pruned'] = True
        self.training_history_['alpha'] = alpha
        self.training_history_['n_leaves_pruned'] = leaves_after
        
        return self
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path where the model should be saved
        
        Examples
        --------
        >>> inference.save_model('my_model_v2.pkl')
        """
        self.model.save(filepath)
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export tree structure to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path where the JSON should be saved
        
        Examples
        --------
        >>> inference.export_to_json('tree_structure.json')
        """
        self.model.to_json(filepath)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns
        -------
        info : dict
            Dictionary containing model metadata and statistics
        
        Examples
        --------
        >>> info = inference.get_model_info()
        >>> print(info)
        {
            'model_type': 'regression',
            'n_features': 5,
            'feature_names': ['x1', 'x2', 'x3', 'x4', 'x5'],
            'tree_depth': 4,
            'n_leaves': 8,
            'n_nodes': 15,
            'training_samples': 1000
        }
        """
        info = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'tree_depth': self.get_tree_depth(),
            'n_leaves': self.get_n_leaves(),
            'n_nodes': self.get_n_nodes(),
        }
        
        # Add training history if available
        if hasattr(self.model, 'training_history_'):
            info['training_history'] = self.model.training_history_
        
        # Add classes for classification
        if self.model_type == 'classification' and self.classes_ is not None:
            info['classes'] = self.classes_.tolist()
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the inference engine."""
        return (
            f"BRCInference(model_type='{self.model_type}', "
            f"n_features={len(self.feature_names)}, "
            f"tree_depth={self.get_tree_depth()}, "
            f"n_leaves={self.get_n_leaves()})"
        )
    
    def __str__(self) -> str:
        """Pretty string representation."""
        info = self.get_model_info()
        output = "="*50 + "\n"
        output += "Breiman CART Inference Engine\n"
        output += "="*50 + "\n"
        output += f"Model Type: {info['model_type'].capitalize()}\n"
        output += f"Features: {info['n_features']}\n"
        output += f"Tree Depth: {info['tree_depth']}\n"
        output += f"Leaves: {info['n_leaves']}\n"
        output += f"Total Nodes: {info['n_nodes']}\n"
        
        if 'classes' in info:
            output += f"Classes: {info['classes']}\n"
        
        output += "="*50 + "\n"
        return output


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__version__ = "0.2.0"
__author__ = "Adam Khald"
__all__ = [
    'BRCRegression',
    'BRCClassification',
    'BRCInference'
]


# ============================================================================
# QUICK START GUIDE (in docstring)
# ============================================================================

"""
Quick Start Guide
=================

REGRESSION EXAMPLE
------------------
>>> import pandas as pd
>>> import numpy as np
>>> import breiman_cart as brc
>>> 
>>> # Prepare data
>>> X_train = pd.DataFrame({
...     'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
...     'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
... })
>>> y_train = np.array([3, 6, 9, 12, 15, 18, 21, 24])
>>> 
>>> # Train model
>>> model = brc.BRCRegression(max_depth=5, min_samples_split=2)
>>> model.fit(X_train, y_train)
>>> 
>>> # Make predictions
>>> X_test = pd.DataFrame({'feature1': [2.5, 5.5], 'feature2': [5, 11]})
>>> predictions = model.predict(X_test)
>>> print(predictions)
>>> 
>>> # Or use inference engine
>>> inference = brc.BRCInference(model)
>>> predictions = inference.predict(X_test)

CLASSIFICATION EXAMPLE
----------------------
>>> # Prepare data
>>> X_train = pd.DataFrame({
...     'age': [25, 30, 35, 40, 45, 50, 55, 60],
...     'income': [30000, 35000, 50000, 60000, 70000, 80000, 90000, 100000]
... })
>>> y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])
>>> 
>>> # Train model
>>> model = brc.BRCClassification(max_depth=3, min_samples_leaf=2)
>>> model.fit(X_train, y_train)
>>> 
>>> # Make predictions
>>> X_test = pd.DataFrame({'age': [28, 48], 'income': [32000, 75000]})
>>> predictions = model.predict(X_test)
>>> 
>>> # Evaluate
>>> accuracy = model.score(X_test, [0, 1])
>>> print(f"Accuracy: {accuracy:.2%}")

INFERENCE ONLY
--------------
>>> # Load saved model
>>> inference = brc.BRCInference('trained_model.pkl')
>>> 
>>> # Get model info
>>> print(inference.get_model_info())
>>> 
>>> # Make predictions
>>> predictions = inference.predict(X_new)
>>> 
>>> # Get feature importance
>>> importance = inference.get_feature_importance()
>>> for feature, score in importance.items():
...     print(f"{feature}: {score:.3f}")

PRUNING EXAMPLE
---------------
>>> # Train a full tree
>>> model = brc.BRCRegression(max_depth=10)
>>> model.fit(X_train, y_train)
>>> 
>>> # Prune using validation set
>>> model.prune(X_val, y_val)  # Automatically finds optimal alpha
>>> 
>>> # Or specify alpha manually
>>> model.prune(X_val, y_val, alpha=0.01)

CATEGORICAL FEATURES
--------------------
>>> X_train = pd.DataFrame({
...     'color': ['red', 'blue', 'red', 'green', 'blue'],
...     'size': [10, 20, 15, 25, 30]
... })
>>> y_train = np.array([1, 0, 1, 0, 0])
>>> 
>>> # Specify categorical features
>>> model = brc.BRCClassification(
...     max_depth=5,
...     categorical_features=['color']
... )
>>> model.fit(X_train, y_train)

SAVING AND LOADING
------------------
>>> # Save model
>>> model.save('my_model.pkl')
>>> 
>>> # Load for inference
>>> inference = brc.BRCInference('my_model.pkl')
>>> predictions = inference.predict(X_new)
>>> 
>>> # Or load the full model
>>> from breiman_cart import BRCRegression
>>> loaded_model = BRCRegression.load('my_model.pkl')

FEATURE IMPORTANCE
------------------
>>> model = brc.BRCClassification()
>>> model.fit(X_train, y_train)
>>> 
>>> # Get importance array
>>> importance = model.get_feature_importance()
>>> 
>>> # Or use inference for named dictionary
>>> inference = brc.BRCInference(model)
>>> importance_dict = inference.get_feature_importance()
>>> print(importance_dict)
{'age': 0.65, 'income': 0.35}

VERBOSE MODE
------------
>>> # See detailed training progress
>>> model = brc.BRCRegression(max_depth=5, verbose=1)
>>> model.fit(X_train, y_train)
# Building CART tree with 100 samples and 5 features...
# Tree Training Complete
# Samples: 100
# Max depth: 5
# Number of leaves: 12

SKLEARN COMPATIBILITY
---------------------
>>> # Get/set parameters like sklearn
>>> params = model.get_params()
>>> print(params)
>>> 
>>> model.set_params(max_depth=10, min_samples_split=5)
>>> model.fit(X_train, y_train)
"""
def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        self.model.save(filepath)
    
def export_to_json(self, filepath: str) -> None:
        """Export tree structure to JSON file."""
        self.model.to_json(filepath)
    
@staticmethod
def load(filepath: str) -> 'CARTBase':
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
def to_dict(self) -> Dict:
        """Export tree structure as dictionary."""
        if self.root is None:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'criterion': self.criterion,
            'feature_names': self.feature_names,
            'training_history': self.training_history_,
            'tree': self.root.to_dict()
        }
    
def to_json(self, filepath: str) -> None:
        """Export tree to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Tree exported to {filepath}")
    
def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'categorical_features': self.categorical_features,
            'verbose': self.verbose
        }
    
def set_params(self, **params) -> 'CARTBase':
        """Set the parameters of this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ============================================================================
# USER-FACING CLASSES - Clean API
# ============================================================================

class BRCRegression(CARTBase):
    """
    Breiman CART for Regression Tasks
    ==================================
    
    A decision tree regressor following Breiman et al. (1984) methodology.
    Uses mean squared error (MSE) as the splitting criterion.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. If None, nodes expand until all leaves
        are pure or contain fewer than min_samples_split samples.
        
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
        
    categorical_features : list of str, optional
        Names of categorical features in your dataset.
        
    verbose : int, default=0
        Controls verbosity of tree building process.
        0 = silent, 1 = info, 2 = debug
    
    Attributes
    ----------
    root : Node
        Root node of the fitted tree
        
    feature_names : list of str
        Names of features used during training
        
    training_history_ : dict
        Training statistics (depth, number of nodes, etc.)
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import breiman_cart as brc
    >>> 
    >>> # Prepare data
    >>> X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
    >>> y = np.array([2.5, 4.5, 6.5, 8.5, 10.5])
    >>> 
    >>> # Create and train model
    >>> model = brc.BRCRegression(max_depth=5, min_samples_split=2)
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)
    >>> 
    >>> # Evaluate
    >>> r2_score = model.score(X, y)
    >>> print(f"R² Score: {r2_score:.3f}")
    >>> 
    >>> # Get feature importance
    >>> importance = model.get_feature_importance()
    >>> 
    >>> # Prune if needed
    >>> X_val = pd.DataFrame({'x1': [1.5, 2.5], 'x2': [3, 5]})
    >>> y_val = np.array([3.5, 5.5])
    >>> model.prune(X_val, y_val)
    >>> 
    >>> # Save model
    >>> model.save('my_regression_model.pkl')
    
    Methods
    -------
    fit(X, y)
        Build the regression tree from training data
        
    predict(X)
        Predict continuous values for samples in X
        
    score(X, y)
        Return R² score on test data
        
    get_feature_importance()
        Get feature importances based on impurity reduction
        
    prune(X_val, y_val, alpha=None)
        Prune tree using validation data
        
    save(filepath)
        Save model to disk
        
    load(filepath)
        Load model from disk (static method)
    
    References
    ----------
    Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
    Classification and Regression Trees. Wadsworth.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        categorical_features: Optional[List[str]] = None,
        verbose: int = 0
    ):
        super().__init__(
            criterion="mse",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            verbose=verbose
        )
    
    def _create_leaf(
        self,
        y: np.ndarray,
        impurity: float,
        n_samples: int,
        sample_indices: Optional[np.ndarray] = None
    ) -> Node:
        """Create a leaf node with mean value."""
        predicted_value = np.mean(y)
        
        return Node(
            predicted_value=predicted_value,
            impurity=impurity,
            n_samples=n_samples,
            sample_indices=sample_indices
        )


class BRCClassification(CARTBase):
    """
    Breiman CART for Classification Tasks
    ======================================
    
    A decision tree classifier following Breiman et al. (1984) methodology.
    Uses Gini impurity as the splitting criterion.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. If None, nodes expand until all leaves
        are pure or contain fewer than min_samples_split samples.
        
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
        
    categorical_features : list of str, optional
        Names of categorical features in your dataset.
        
    verbose : int, default=0
        Controls verbosity of tree building process.
        0 = silent, 1 = info, 2 = debug
    
    Attributes
    ----------
    root : Node
        Root node of the fitted tree
        
    feature_names : list of str
        Names of features used during training
        
    classes_ : np.ndarray
        Unique class labels found during training
        
    training_history_ : dict
        Training statistics (depth, number of nodes, etc.)
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import breiman_cart as brc
    >>> 
    >>> # Prepare data
    >>> X = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45],
    ...     'income': [30000, 45000, 60000, 75000, 90000]
    ... })
    >>> y = np.array([0, 0, 1, 1, 1])  # Binary classification
    >>> 
    >>> # Create and train model
    >>> model = brc.BRCClassification(max_depth=3, min_samples_split=2)
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)
    >>> 
    >>> # Evaluate
    >>> accuracy = model.score(X, y)
    >>> print(f"Accuracy: {accuracy:.3f}")
    >>> 
    >>> # Get feature importance
    >>> importance = model.get_feature_importance()
    >>> 
    >>> # Prune if needed
    >>> X_val = pd.DataFrame({'age': [28, 38], 'income': [35000, 70000]})
    >>> y_val = np.array([0, 1])
    >>> model.prune(X_val, y_val)
    >>> 
    >>> # Save model
    >>> model.save('my_classification_model.pkl')
    
    Methods
    -------
    fit(X, y)
        Build the classification tree from training data
        
    predict(X)
        Predict class labels for samples in X
        
    score(X, y)
        Return accuracy score on test data
        
    get_feature_importance()
        Get feature importances based on impurity reduction
        
    prune(X_val, y_val, alpha=None)
        Prune tree using validation data
        
    save(filepath)
        Save model to disk
        
    load(filepath)
        Load model from disk (static method)
    
    References
    ----------
    Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
    Classification and Regression Trees. Wadsworth.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        categorical_features: Optional[List[str]] = None,
        verbose: int = 0
    ):
        super().__init__(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            verbose=verbose
        )
        self.classes_ = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BRCClassification':
        """Fit the classifier to training data."""
        self.classes_ = np.unique(y)
        return super().fit(X, y)
    
    def _create_leaf(
        self,
        y: np.ndarray,
        impurity: float,
        n_samples: int,
        sample_indices: Optional[np.ndarray] = None
    ) -> Node:
        """Create a leaf node with majority class."""
        classes, counts = np.unique(y, return_counts=True)
        predicted_class = classes[np.argmax(counts)]
        
        return Node(
            predicted_value=predicted_class,
            impurity=impurity,
            n_samples=n_samples,
            sample_indices=sample_indices
        )


class BRCInference:
    """
    Breiman CART Inference Engine
    ==============================
    
    Make predictions using a trained CART model.
    This class provides a clean interface for loading models and making predictions
    without needing to retrain or access training data.
    
    Parameters
    ----------
    model : BRCRegression or BRCClassification or str
        Either a trained model object or path to a saved model file (.pkl)
    
    Attributes
    ----------
    model : BRCRegression or BRCClassification
        The loaded model used for inference
        
    model_type : str
        Type of model ('regression' or 'classification')
        
    feature_names : list of str
        Names of features the model expects
    
    Examples
    --------
    >>> import pandas as pd
    >>> import breiman_cart as brc
    >>> 
    >>> # Option 1: Use with a trained model directly
    >>> model = brc.BRCRegression(max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> inference = brc.BRCInference(model)
    >>> predictions = inference.predict(X_test)
    >>> 
    >>> # Option 2: Load from saved model file
    >>> inference = brc.BRCInference('my_model.pkl')
    >>> predictions = inference.predict(X_test)
    >>> 
    >>> # Get model information
    >>> print(f"Model type: {inference.model_type}")
    >>> print(f"Features: {inference.feature_names}")
    >>> print(f"Tree depth: {inference.get_tree_depth()}")
    >>> print(f"Number of leaves: {inference.get_n_leaves()}")
    
    Methods
    -------
    predict(X)
        Make predictions for new data
        
    score(X, y)
        Evaluate model performance on test data
        
    get_feature_importance()
        Get feature importance from the model
        
    get_tree_depth()
        Get the depth of the decision tree
        
    get_n_leaves()
        Get the number of leaf nodes
        
    get_n_nodes()
        Get the total number of nodes in the tree
        
    export_tree_dict()
        Export tree structure as a dictionary
        
    save_model(filepath)
        Save the model to a new location
    """
    
    def __init__(self, model: Union[BRCRegression, BRCClassification, str]):
        """
        Initialize the inference engine.
        
        Parameters
        ----------
        model : BRCRegression, BRCClassification, or str
            Trained model or path to saved model file
        
        Raises
        ------
        ValueError
            If model is not fitted or invalid type
        FileNotFoundError
            If model file path doesn't exist
        TypeError
            If model is not a valid CART model
        """
        if isinstance(model, str):
            # Load from file
            try:
                self.model = CARTBase.load(model)
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found: {model}")
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        elif isinstance(model, (BRCRegression, BRCClassification, CARTBase)):
            # Use provided model
            self.model = model
        else:
            raise TypeError(
                f"Model must be BRCRegression, BRCClassification, or file path. "
                f"Got {type(model)}"
            )
        
        # Validate model is fitted
        if self.model.root is None:
            raise ValueError("Model is not fitted. Train the model before using for inference.")
        
        # Store model metadata
        self.feature_names = self.model.feature_names
        self.model_type = 'regression' if self.model.criterion == 'mse' else 'classification'
        
        if self.model_type == 'classification':
            self.classes_ = getattr(self.model, 'classes_', None)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features)
            Must have the same features as the training data
        
        Returns
        -------
        predictions : np.ndarray
            Predicted values (continuous for regression, class labels for classification)
        
        Raises
        ------
        ValueError
            If X has wrong number of features or missing features
        
        Examples
        --------
        >>> X_new = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        >>> predictions = inference.predict(X_new)
        >>> print(predictions)
        [1.5, 2.5]
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Validate features match
        if list(X.columns) != self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            
            error_msg = "Feature mismatch:\n"
            if missing:
                error_msg += f"  Missing features: {missing}\n"
            if extra:
                error_msg += f"  Extra features: {extra}\n"
            error_msg += f"  Expected features: {self.feature_names}"
            
            raise ValueError(error_msg)
        
        return self.model.predict(X)
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Test features
        y : np.ndarray
            True target values
        
        Returns
        -------
        score : float
            Accuracy (classification) or R² score (regression)
        
        Examples
        --------
        >>> score = inference.score(X_test, y_test)
        >>> print(f"Model score: {score:.3f}")
        0.95
        """
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns
        -------
        importance_dict : dict
            Dictionary mapping feature names to importance scores
        
        Examples
        --------
        >>> importance = inference.get_feature_importance()
        >>> for feature, score in importance.items():
        ...     print(f"{feature}: {score:.3f}")
        age: 0.65
        income: 0.35
        """
        importances = self.model.get_feature_importance()
        return dict(zip(self.feature_names, importances))
    
    def get_tree_depth(self) -> int:
        """
        Get the maximum depth of the decision tree.
        
        Returns
        -------
        depth : int
            Maximum depth from root to any leaf
        
        Examples
        --------
        >>> depth = inference.get_tree_depth()
        >>> print(f"Tree depth: {depth}")
        Tree depth: 5
        """
        return self.model.root.get_depth()
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes in the tree.
        
        Returns
        -------
        n_leaves : int
            Number of leaf nodes
        
        Examples
        --------
        >>> n_leaves = inference.get_n_leaves()
        >>> print(f"Number of leaves: {n_leaves}")
        Number of leaves: 12
        """
        return self.model.root.count_leaves()
    
    def get_n_nodes(self) -> int:
        """
        Get the total number of nodes in the tree.
        
        Returns
        -------
        n_nodes : int
            Total number of nodes (internal + leaves)
        
        Examples
        --------
        >>> n_nodes = inference.get_n_nodes()
        >>> print(f"Total nodes: {n_nodes}")
        Total nodes: 23
        """
        return self.model.root.get_n_nodes()
    
    def export_tree_dict(self) -> Dict:
        """
        Export tree structure as a dictionary.
        
        Returns
        -------
        tree_dict : dict
            Complete tree structure with metadata
        
        Examples
        --------
        >>> tree_dict = inference.export_tree_dict()
        >>> print(tree_dict.keys())
        dict_keys(['fitted', 'criterion', 'feature_names', 'training_history', 'tree'])
        """
        return self.model.to_dict()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path where the model should be saved
        
        Examples
        --------
        >>> inference.save_model("my_model_v2.pkl")
        """
        self.model.save(filepath)