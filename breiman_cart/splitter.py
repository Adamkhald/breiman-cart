"""
Splitter for CART trees with optimized performance.
Handles finding the best split for a node with support for both 
numerical and categorical features.
"""

from typing import Tuple, Optional, Set, List, Any
import numpy as np
import pandas as pd
from itertools import combinations
import warnings


class Splitter:
    """
    Handles finding the best split for a node.
    Supports both numerical and categorical features with performance optimizations.
    
    Parameters
    ----------
    criterion : str, default="gini"
        Split quality criterion: "gini" for classification, "mse" for regression
    categorical_features : list of str, optional
        List of categorical feature names
    max_categories_exhaustive : int, default=8
        Maximum number of categories to use exhaustive search
        (uses greedy heuristic above this threshold)
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
        """
        Calculate Gini impurity.
        
        The Gini impurity measures the probability of incorrectly classifying
        a randomly chosen element.
        
        Parameters
        ----------
        y : np.ndarray
            Target values
        
        Returns
        -------
        float
            Gini impurity in [0, 1-1/n_classes]
        
        Notes
        -----
        Gini impurity is calculated as: i(t) = 1 - sum(p_j^2)
        where p_j is the proportion of samples of class j.
        """
        if len(y) == 0:
            return 0.0
        
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def mse(self, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Parameters
        ----------
        y : np.ndarray
            Target values
        
        Returns
        -------
        float
            Mean squared error
        """
        if len(y) == 0:
            return 0.0
        
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity based on criterion.
        
        Parameters
        ----------
        y : np.ndarray
            Target values
        
        Returns
        -------
        float
            Impurity measure
        """
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
        """
        Calculate information gain from a split.
        
        Parameters
        ----------
        parent_y : np.ndarray
            Parent node targets
        left_y : np.ndarray
            Left child targets
        right_y : np.ndarray
            Right child targets
        
        Returns
        -------
        float
            Information gain (reduction in impurity)
        """
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
        """
        Find best threshold for a numerical feature using optimized algorithm.
        
        Parameters
        ----------
        X_column : np.ndarray
            Feature values
        y : np.ndarray
            Target values
        
        Returns
        -------
        best_threshold : float or None
            Best threshold value
        best_gain : float
            Information gain from best split
        """
        # Ensure numpy array
        y = np.asarray(y)
        
        # Sort by feature value
        sorted_indices = np.argsort(X_column)
        X_sorted = X_column[sorted_indices]
        y_sorted = y[sorted_indices]
        
        best_gain = -np.inf
        best_threshold = None
        
        # Try splits between unique values only
        unique_values = np.unique(X_sorted)
        
        if len(unique_values) <= 1:
            return None, 0.0
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = X_sorted <= threshold
            right_mask = ~left_mask
            
            # Skip degenerate splits
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
        """
        Find best subset split for a categorical feature.
        Uses 1984 CART methodology with performance optimizations.
        
        For binary classification: sorts categories by class proportion
        For multi-class/regression: uses exhaustive search (small n) or greedy heuristic
        
        Parameters
        ----------
        X_column : pd.Series
            Categorical feature values
        y : np.ndarray
            Target values
        
        Returns
        -------
        best_subset : set or None
            Best subset of categories for left child
        best_gain : float
            Information gain from best split
        """
        # Ensure numpy array
        y = np.asarray(y)
        
        unique_categories = X_column.unique()
        
        if len(unique_categories) <= 1:
            return None, 0.0
        
        # For binary classification, use optimal ordering trick (Breiman 1984)
        if self.criterion == "gini" and len(np.unique(y)) == 2:
            return self._binary_class_categorical_split(X_column, y, unique_categories)
        
        # For other cases, decide between exhaustive and greedy
        if len(unique_categories) <= self.max_categories_exhaustive:
            return self._exhaustive_categorical_split(X_column, y, unique_categories)
        else:
            # Too many categories for exhaustive search
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
        """
        Optimal categorical split for binary classification.
        
        Uses the property that optimal splits can be found by ordering
        categories by class proportion and trying splits in sequence.
        """
        # Calculate proportion of positive class for each category
        category_proportions = []
        for cat in categories:
            mask = (X_column == cat).values
            if np.sum(mask) > 0:
                prop = np.mean(y[mask])
                category_proportions.append((cat, prop))
        
        # Sort by proportion
        category_proportions.sort(key=lambda x: x[1])
        sorted_categories = [cat for cat, _ in category_proportions]
        
        # Try splits in order (only n-1 splits needed)
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
        """
        Exhaustive search for categorical splits.
        Try all possible binary partitions.
        """
        best_gain = -np.inf
        best_subset = None
        
        # Try all possible non-empty proper subsets
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
        """
        Greedy heuristic for categorical splits with many categories.
        Sort by mean target value and try splits in sequence.
        """
        # Calculate mean target for each category
        category_means = []
        for cat in categories:
            mask = (X_column == cat).values
            if np.sum(mask) > 0:
                mean_y = np.mean(y[mask])
                category_means.append((cat, mean_y))
        
        # Sort by mean
        category_means.sort(key=lambda x: x[1])
        sorted_categories = [cat for cat, _ in category_means]
        
        # Try splits in order
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
        """
        Find the best split across all features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target values
        
        Returns
        -------
        feature_index : int or None
            Index of best feature to split on
        threshold : float or None
            Threshold for numerical split
        categorical_subset : set or None
            Subset for categorical split
        is_categorical : bool
            Whether the split is categorical
        gain : float
            Information gain from the split
        """
        # Ensure y is numpy array
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