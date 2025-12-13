"""
Node class for CART tree structure.
Supports both numerical and categorical splits.
Improved version with better validation, type hints, and utility methods.
"""

from typing import Optional, Any, Set, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    
    Attributes
    ----------
    is_leaf : bool
        Whether this is a leaf node
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
        
        # Validate node consistency
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
        """
        Predict a single sample by traversing the tree.
        
        Parameters
        ----------
        x : np.ndarray
            Feature vector
        
        Returns
        -------
        int, float, or np.ndarray
            Predicted value
        
        Raises
        ------
        ValueError
            If feature vector is empty
        IndexError
            If feature_index is out of bounds
        """
        if len(x) == 0:
            raise ValueError("Empty feature vector provided")
        
        if self.is_leaf:
            return self.predicted_value
        
        # Validate feature index
        if self.feature_index >= len(x):
            raise IndexError(
                f"Feature index {self.feature_index} out of bounds for vector of length {len(x)}"
            )
        
        if self.is_categorical:
            # Categorical split: check if value is in subset
            feature_value = x[self.feature_index]
            if feature_value in self.categorical_subset:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
        else:
            # Numerical split: check threshold
            if x[self.feature_index] <= self.threshold:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
    
    def count_leaves(self) -> int:
        """
        Count the number of leaf nodes in this subtree.
        
        Returns
        -------
        int
            Number of leaf nodes
        """
        if self.is_leaf:
            return 1
        return self.left.count_leaves() + self.right.count_leaves()
    
    def get_depth(self) -> int:
        """
        Calculate the depth of this subtree.
        
        Returns
        -------
        int
            Maximum depth from this node to any leaf
        """
        if self.is_leaf:
            return 0
        return 1 + max(self.left.get_depth(), self.right.get_depth())
    
    def get_n_nodes(self) -> int:
        """
        Count total nodes in this subtree.
        
        Returns
        -------
        int
            Total number of nodes (internal + leaves)
        """
        if self.is_leaf:
            return 1
        return 1 + self.left.get_n_nodes() + self.right.get_n_nodes()
    
    def get_subtree_error(self) -> float:
        """
        Calculate total error for this subtree.
        
        Returns
        -------
        float
            Total weighted impurity across all leaves
        """
        if self.is_leaf:
            return self.impurity * self.n_samples
        return self.left.get_subtree_error() + self.right.get_subtree_error()
    
    def get_leaves(self) -> list['Node']:
        """
        Get all leaf nodes in this subtree.
        
        Returns
        -------
        list of Node
            All leaf nodes
        """
        if self.is_leaf:
            return [self]
        leaves = []
        leaves.extend(self.left.get_leaves())
        leaves.extend(self.right.get_leaves())
        return leaves
    
    def to_dict(self) -> dict:
        """
        Convert node to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation of the node
        """
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
    
    def __str__(self) -> str:
        """Pretty string representation."""
        return self._str_helper(0)
    
    def _str_helper(self, depth: int) -> str:
        """Helper for pretty printing tree structure."""
        indent = "  " * depth
        if self.is_leaf:
            return f"{indent}→ Leaf: {self.predicted_value} (n={self.n_samples})\n"
        
        result = ""
        if self.is_categorical:
            result += f"{indent}Feature {self.feature_index} in {self.categorical_subset}?\n"
        else:
            result += f"{indent}Feature {self.feature_index} <= {self.threshold:.3f}?\n"
        
        result += f"{indent}├─ Yes:\n"
        result += self.left._str_helper(depth + 1)
        result += f"{indent}└─ No:\n"
        result += self.right._str_helper(depth + 1)
        
        return result