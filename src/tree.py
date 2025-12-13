"""
Complete CART tree implementation with improvements.
Implements Classification and Regression Trees (Breiman et al., 1984).
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import pickle
import json
import logging

from .node import Node
from .splitter import Splitter
from .pruning import CostComplexityPruner


logger = logging.getLogger(__name__)


class CARTBase:
    """
    Base class for CART trees with common functionality.
    
    Parameters
    ----------
    criterion : str
        Split criterion: "gini" for classification, "mse" for regression
    max_depth : int, optional
        Maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    categorical_features : list of str, optional
        List of categorical feature names
    verbose : int, default=0
        Verbosity level (0=silent, 1=info, 2=debug)
    
    Attributes
    ----------
    root : Node
        Root node of the fitted tree
    feature_names : list of str
        Names of features used for training
    training_history_ : dict
        Training statistics (depth, n_nodes, etc.)
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
        
        # Set logging level
        if verbose > 0:
            logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)
        
        self.root: Optional[Node] = None
        self.splitter: Optional[Splitter] = None
        self.feature_names: Optional[List[str]] = None
        self.pruner: Optional[CostComplexityPruner] = None
        
        # Store training data for pruning
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        
        # Training statistics
        self.training_history_: Dict[str, Any] = {}
    
    def _validate_input(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Validate input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target values
        
        Raises
        ------
        TypeError
            If X is not a DataFrame
        ValueError
            If data is invalid (empty, mismatched lengths, contains NaN)
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        if len(X) != len(y):
            raise ValueError(f"X and y lengths don't match: {len(X)} != {len(y)}")
        
        # Check for NaN values
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"X contains NaN values in columns: {nan_cols}")
        
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")
        
        # Validate categorical features exist
        for cat_feat in self.categorical_features:
            if cat_feat not in X.columns:
                raise ValueError(f"Categorical feature '{cat_feat}' not found in X")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'CARTBase':
        """
        Fit the CART tree to training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features)
        y : np.ndarray
            Target values with shape (n_samples,)
        
        Returns
        -------
        self : CARTBase
            Fitted estimator
        
        Raises
        ------
        ValueError
            If input data is invalid
        """
        # Validate input
        self._validate_input(X, y)
        
        # Store training data
        self.X_train = X.copy()
        self.y_train = y if not isinstance(y, pd.Series) else y.values
        
        self.feature_names = list(X.columns)
        self.splitter = Splitter(
            criterion=self.criterion,
            categorical_features=self.categorical_features
        )
        
        logger.info(f"Building CART tree with {len(X)} samples and {len(X.columns)} features")
        logger.info(f"Criterion: {self.criterion}, Max depth: {self.max_depth}")
        
        # Build tree
        self.root = self._grow_tree(X, self.y_train, depth=0)
        
        # Collect training statistics
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
        """
        Recursively grow the tree.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for this node
        y : np.ndarray
            Target values for this node
        depth : int
            Current depth in the tree
        sample_indices : np.ndarray, optional
            Original indices of samples at this node
        
        Returns
        -------
        Node
            The created node
        """
        # Convert y to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        n_samples = len(y)
        impurity = self.splitter.calculate_impurity(y)
        
        if sample_indices is None:
            sample_indices = np.arange(n_samples)
        
        if self.verbose > 1:
            logger.debug(f"Growing node at depth {depth}, samples={n_samples}, impurity={impurity:.4f}")
        
        # Check stopping criteria
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
        
        # Find best split
        feature_idx, threshold, subset, is_categorical, gain = \
            self.splitter.find_best_split(X, y)
        
        # If no good split found, create leaf
        if feature_idx is None or gain <= 0:
            if self.verbose > 1:
                logger.debug("No beneficial split found, creating leaf")
            return self._create_leaf(y, impurity, n_samples, sample_indices)
        
        # Split the data
        if is_categorical:
            feature_name = X.columns[feature_idx]
            left_mask = X[feature_name].isin(subset)
            right_mask = ~left_mask
        else:
            feature_name = X.columns[feature_idx]
            left_mask = X[feature_name] <= threshold
            right_mask = ~left_mask
        
        # Check minimum samples in leaves
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
        
        # Recursively grow children
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
        """
        Predict target values for samples in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features)
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted values with shape (n_samples,)
        
        Raises
        ------
        ValueError
            If tree is not fitted or X has wrong number of features
        """
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Should be implemented by CARTClassifier.
        """
        raise NotImplementedError("predict_proba only available for classification")
    
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Calculate accuracy (classification) or R² (regression).
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            True target values
        
        Returns
        -------
        float
            Score (accuracy or R²)
        """
        predictions = self.predict(X)
        
        if self.criterion == "gini":
            # Classification: accuracy
            return np.mean(predictions == y)
        else:
            # Regression: R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance based on total impurity reduction.
        
        Returns
        -------
        importances : np.ndarray
            Feature importances (normalized to sum to 1)
        """
        if self.root is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        importances = np.zeros(len(self.feature_names))
        
        def traverse(node):
            if node.is_leaf:
                return
            
            # Calculate weighted impurity reduction
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
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances
    
    def prune(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        alpha: Optional[float] = None
    ) -> 'CARTBase':
        """
        Prune the tree using cost-complexity pruning.
        
        Parameters
        ----------
        X_val : pd.DataFrame
            Validation features
        y_val : np.ndarray
            Validation targets
        alpha : float, optional
            Regularization parameter. If None, finds optimal value
        
        Returns
        -------
        self : CARTBase
            Pruned estimator
        
        Raises
        ------
        ValueError
            If tree not fitted or training data not available
        """
        if self.root is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not stored. Cannot prune.")
        
        # Convert y_val to numpy if needed
        if isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        logger.info("Starting pruning process...")
        
        # Initialize pruner
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
        
        # Prune the tree
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
        
        # Update training history
        self.training_history_['pruned'] = True
        self.training_history_['alpha'] = alpha
        self.training_history_['n_leaves_pruned'] = leaves_after
        
        return self
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'CARTBase':
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        
        Returns
        -------
        CARTBase
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def to_dict(self) -> Dict:
        """
        Export tree structure as dictionary.
        
        Returns
        -------
        dict
            Tree structure and metadata
        """
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
        """
        Export tree to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save JSON
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Tree exported to {filepath}")
    
    def __str__(self) -> str:
        """String representation of the tree."""
        if self.root is None:
            return "CART Tree (not fitted)"
        
        info = f"CART Tree ({self.criterion})\n"
        info += f"Depth: {self.root.get_depth()}, "
        info += f"Leaves: {self.root.count_leaves()}, "
        info += f"Nodes: {self.root.get_n_nodes()}\n"
        info += "\nTree structure:\n"
        info += str(self.root)
        
        return info


class CARTClassifier(CARTBase):
    """
    CART classifier for classification tasks.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf
    categorical_features : list of str, optional
        Names of categorical features
    verbose : int, default=0
        Verbosity level
    
    Examples
    --------
    >>> from src import CARTClassifier
    >>> import pandas as pd
    >>> X = pd.DataFrame({'x1': [0, 1, 2, 3], 'x2': [0, 1, 1, 0]})
    >>> y = np.array([0, 1, 1, 0])
    >>> clf = CARTClassifier(max_depth=3)
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 1, 1, 0])
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
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'CARTClassifier':
        """Fit the classifier."""
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


class CARTRegressor(CARTBase):
    """
    CART regressor for regression tasks.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf
    categorical_features : list of str, optional
        Names of categorical features
    verbose : int, default=0
        Verbosity level
    
    Examples
    --------
    >>> from src import CARTRegressor
    >>> import pandas as pd
    >>> X = pd.DataFrame({'x': [1, 2, 3, 4]})
    >>> y = np.array([2, 4, 6, 8])
    >>> reg = CARTRegressor(max_depth=3)
    >>> reg.fit(X, y)
    >>> reg.predict(X)
    array([2., 4., 6., 8.])
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