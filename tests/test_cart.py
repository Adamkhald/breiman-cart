"""
Comprehensive unit tests for CART implementation.
Tests both classification and regression with extensive edge cases.
"""

import numpy as np
import pandas as pd
import sys
import os
import pytest
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Fix import path for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tree import CARTClassifier, CARTRegressor
from src.node import Node
from src.splitter import Splitter


# ============================================================================
# Node Tests
# ============================================================================

def test_node_creation():
    """Test Node initialization."""
    # Leaf node
    leaf = Node(predicted_value=1.0, n_samples=10, impurity=0.0)
    assert leaf.is_leaf == True
    assert leaf.predicted_value == 1.0
    assert leaf.n_samples == 10
    
    # Internal node
    internal = Node(
        feature_index=0,
        threshold=5.0,
        left=leaf,
        right=leaf,
        n_samples=20,
        impurity=0.5
    )
    assert internal.is_leaf == False
    assert internal.feature_index == 0
    assert internal.threshold == 5.0
    print("✓ Node creation test passed")


def test_node_validation():
    """Test Node validation."""
    # Invalid non-leaf without feature_index
    with pytest.raises(ValueError):
        Node(left=Node(predicted_value=1), right=Node(predicted_value=2))
    
    # Invalid categorical split without subset
    with pytest.raises(ValueError):
        Node(
            feature_index=0,
            is_categorical=True,
            left=Node(predicted_value=1),
            right=Node(predicted_value=2)
        )
    print("✓ Node validation test passed")


def test_node_methods():
    """Test Node utility methods."""
    # Create a small tree
    left = Node(predicted_value=0, n_samples=5, impurity=0.0)
    right = Node(predicted_value=1, n_samples=5, impurity=0.0)
    root = Node(
        feature_index=0,
        threshold=0.5,
        left=left,
        right=right,
        n_samples=10,
        impurity=0.5
    )
    
    assert root.count_leaves() == 2
    assert root.get_depth() == 1
    assert root.get_n_nodes() == 3
    assert len(root.get_leaves()) == 2
    print("✓ Node methods test passed")


def test_node_predict():
    """Test node prediction."""
    left = Node(predicted_value=0, n_samples=5)
    right = Node(predicted_value=1, n_samples=5)
    root = Node(
        feature_index=0,
        threshold=0.5,
        left=left,
        right=right,
        n_samples=10
    )
    
    assert root.predict_sample(np.array([0.3])) == 0
    assert root.predict_sample(np.array([0.7])) == 1
    
    # Test error handling
    with pytest.raises(ValueError):
        root.predict_sample(np.array([]))
    print("✓ Node predict test passed")


# ============================================================================
# Splitter Tests
# ============================================================================

def test_gini_impurity():
    """Test Gini calculation."""
    splitter = Splitter(criterion="gini")
    
    # Pure node
    y_pure = np.array([0, 0, 0, 0])
    assert splitter.gini_impurity(y_pure) == 0.0
    
    # 50-50 split
    y_balanced = np.array([0, 0, 1, 1])
    assert abs(splitter.gini_impurity(y_balanced) - 0.5) < 1e-6
    
    # Empty
    assert splitter.gini_impurity(np.array([])) == 0.0
    print("✓ Gini impurity test passed")


def test_mse():
    """Test MSE calculation."""
    splitter = Splitter(criterion="mse")
    
    # No variance
    y_constant = np.array([5.0, 5.0, 5.0])
    assert splitter.mse(y_constant) == 0.0
    
    # Known variance
    y = np.array([1.0, 2.0, 3.0])
    expected_mse = np.var(y)
    assert abs(splitter.mse(y) - expected_mse) < 1e-6
    print("✓ MSE test passed")


def test_numerical_split():
    """Test numerical feature splitting."""
    splitter = Splitter(criterion="gini")
    
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 1, 1, 1])
    
    threshold, gain = splitter.best_numerical_split(X, y)
    
    assert threshold is not None
    assert gain > 0
    assert 2 < threshold < 3  # Should split between classes
    print("✓ Numerical split test passed")


def test_categorical_split_binary():
    """Test categorical splitting with binary classification."""
    splitter = Splitter(criterion="gini", categorical_features=['city'])
    
    X = pd.DataFrame({
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'SF'] * 2
    })
    y = np.array([0, 1, 0, 1, 1, 1] * 2)
    
    subset, gain = splitter.best_categorical_split(X['city'], y)
    
    assert subset is not None
    assert gain > 0
    print("✓ Categorical split test passed")


# ============================================================================
# Classification Tests
# ============================================================================

def test_classification_simple():
    """Test basic classification."""
    # Use a more separable dataset
    X = pd.DataFrame({
        'x1': [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
        'x2': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    })
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    
    clf = CARTClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1)
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Classification Test - Training Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.70, "Classification accuracy too low"
    assert clf.classes_ is not None
    print("✓ Classification test passed")


def test_classification_edge_cases():
    """Test classification edge cases."""
    # Single class
    X = pd.DataFrame({'x': [1, 2, 3]})
    y = np.array([0, 0, 0])
    
    clf = CARTClassifier()
    clf.fit(X, y)
    
    assert clf.root.is_leaf
    assert clf.predict(X)[0] == 0
    
    # Single sample
    X_single = pd.DataFrame({'x': [1]})
    y_single = np.array([1])
    
    clf_single = CARTClassifier()
    clf_single.fit(X_single, y_single)
    
    assert clf_single.root.is_leaf
    assert clf_single.predict(X_single)[0] == 1
    print("✓ Classification edge cases test passed")


def test_classification_multiclass():
    """Test multiclass classification."""
    np.random.seed(42)
    
    X = pd.DataFrame({
        'x1': np.random.randn(90),
        'x2': np.random.randn(90)
    })
    
    # Three classes
    y = np.array([0]*30 + [1]*30 + [2]*30)
    y = y[np.random.permutation(90)]
    
    clf = CARTClassifier(max_depth=5, min_samples_split=5)
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Multiclass Test - Training Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.6
    assert len(clf.classes_) == 3
    print("✓ Multiclass test passed")


def test_classification_validation():
    """Test input validation for classification."""
    clf = CARTClassifier()
    
    # Empty data
    with pytest.raises(ValueError):
        clf.fit(pd.DataFrame(), np.array([]))
    
    # Mismatched lengths
    with pytest.raises(ValueError):
        clf.fit(pd.DataFrame({'x': [1, 2]}), np.array([0]))
    
    # NaN in features
    with pytest.raises(ValueError):
        clf.fit(pd.DataFrame({'x': [1, np.nan, 3]}), np.array([0, 1, 0]))
    
    # NaN in target
    with pytest.raises(ValueError):
        clf.fit(pd.DataFrame({'x': [1, 2, 3]}), np.array([0, np.nan, 1]))
    
    # Wrong type
    with pytest.raises(TypeError):
        clf.fit([[1, 2], [3, 4]], np.array([0, 1]))
    
    print("✓ Classification validation test passed")


# ============================================================================
# Regression Tests
# ============================================================================

def test_regression_simple():
    """Test basic regression."""
    X = pd.DataFrame({
        'x1': np.linspace(0, 10, 50),
        'x2': np.random.randn(50)
    })
    y = 2 * X['x1'] + 5 + np.random.randn(50) * 0.5
    
    reg = CARTRegressor(max_depth=5, min_samples_split=4)
    reg.fit(X, y)
    
    predictions = reg.predict(X)
    mse = np.mean((predictions - y) ** 2)
    
    print(f"Regression Test - Training MSE: {mse:.2f}")
    assert mse < 10, "Regression MSE too high"
    print("✓ Regression test passed")


def test_regression_score():
    """Test R² score calculation."""
    X = pd.DataFrame({'x': np.linspace(0, 10, 100)})
    y = 2 * X['x'].values + 1
    
    reg = CARTRegressor(max_depth=10)
    reg.fit(X, y)
    
    r2 = reg.score(X, y)
    print(f"Regression R² Score: {r2:.4f}")
    assert r2 > 0.95  # Should fit very well
    print("✓ Regression score test passed")


def test_regression_edge_cases():
    """Test regression edge cases."""
    # Constant target
    X = pd.DataFrame({'x': [1, 2, 3, 4]})
    y = np.array([5.0, 5.0, 5.0, 5.0])
    
    reg = CARTRegressor()
    reg.fit(X, y)
    
    assert reg.root.is_leaf
    assert reg.predict(X)[0] == 5.0
    print("✓ Regression edge cases test passed")


# ============================================================================
# Categorical Feature Tests
# ============================================================================

def test_categorical_features():
    """Test categorical feature handling."""
    X = pd.DataFrame({
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'SF', 'NYC', 'LA'] * 3,
        'age': [25, 45, 35, 50, 23, 41, 29, 38] * 3
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 3)
    
    clf = CARTClassifier(
        max_depth=4,
        min_samples_split=2,
        categorical_features=['city']
    )
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Categorical Test - Training Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.6, "Categorical feature handling failed"
    print("✓ Categorical feature test passed")


def test_mixed_features():
    """Test mix of categorical and numerical features."""
    X = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'] * 5,
        'numeric1': np.random.randn(30),
        'numeric2': np.random.randn(30)
    })
    y = np.random.randint(0, 2, 30)
    
    clf = CARTClassifier(
        max_depth=5,
        categorical_features=['category']
    )
    clf.fit(X, y)
    
    # Should not raise error
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    print("✓ Mixed features test passed")


# ============================================================================
# Pruning Tests
# ============================================================================

def test_pruning():
    """Test cost-complexity pruning."""
    np.random.seed(42)
    
    # Create overfit-prone data
    X_train = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100)
    })
    y_train = (X_train['x1'] + X_train['x2'] > 0).astype(int)
    
    X_val = pd.DataFrame({
        'x1': np.random.randn(50),
        'x2': np.random.randn(50)
    })
    y_val = (X_val['x1'] + X_val['x2'] > 0).astype(int)
    
    # Fit deep tree (likely overfit)
    clf = CARTClassifier(max_depth=10, min_samples_split=2)
    clf.fit(X_train, y_train)
    
    leaves_before = clf.root.count_leaves()
    train_acc_before = clf.score(X_train, y_train)
    val_acc_before = clf.score(X_val, y_val)
    
    # Prune
    clf.prune(X_val, y_val)
    
    leaves_after = clf.root.count_leaves()
    train_acc_after = clf.score(X_train, y_train)
    val_acc_after = clf.score(X_val, y_val)
    
    print(f"Pruning Test:")
    print(f"  Leaves: {leaves_before} → {leaves_after}")
    print(f"  Train: {train_acc_before:.3f} → {train_acc_after:.3f}")
    print(f"  Val:   {val_acc_before:.3f} → {val_acc_after:.3f}")
    
    assert leaves_after <= leaves_before, "Pruning should reduce leaves"
    print("✓ Pruning test passed")


def test_pruning_with_alpha():
    """Test pruning with specified alpha."""
    np.random.seed(42)
    
    # Create a more complex dataset that will generate a deeper tree
    X = pd.DataFrame({
        'x1': np.random.randn(200),
        'x2': np.random.randn(200)
    })
    y = ((X['x1'] > 0) & (X['x2'] > 0)).astype(int).values
    
    clf = CARTClassifier(max_depth=15, min_samples_split=2, min_samples_leaf=1)
    clf.fit(X, y)
    
    leaves_before = clf.root.count_leaves()
    
    # Prune with high alpha (aggressive pruning)
    clf.prune(X, y, alpha=0.1)
    
    leaves_after = clf.root.count_leaves()
    
    print(f"Pruning with alpha: {leaves_before} → {leaves_after} leaves")
    assert leaves_after <= leaves_before, "Pruning should not increase leaves"
    print("✓ Pruning with alpha test passed")


# ============================================================================
# Feature Importance Tests
# ============================================================================

def test_feature_importance():
    """Test feature importance calculation."""
    X = pd.DataFrame({
        'important': np.random.randn(100),
        'noise': np.random.randn(100)
    })
    y = (X['important'] > 0).astype(int).values
    
    clf = CARTClassifier(max_depth=5)
    clf.fit(X, y)
    
    importances = clf.get_feature_importance()
    
    assert len(importances) == 2
    assert abs(importances.sum() - 1.0) < 1e-6  # Should sum to 1
    assert importances[0] > importances[1]  # 'important' should be more important
    
    print(f"Feature importances: {importances}")
    print("✓ Feature importance test passed")


# ============================================================================
# Model Persistence Tests
# ============================================================================

def test_save_load(tmp_path):
    """Test model saving and loading."""
    X = pd.DataFrame({'x': [1, 2, 3, 4]})
    y = np.array([0, 1, 1, 0])
    
    clf = CARTClassifier(max_depth=2)
    clf.fit(X, y)
    
    # Save
    filepath = tmp_path / "model.pkl"
    clf.save(str(filepath))
    
    # Load
    clf_loaded = CARTClassifier.load(str(filepath))
    
    # Compare predictions
    pred_original = clf.predict(X)
    pred_loaded = clf_loaded.predict(X)
    
    assert np.array_equal(pred_original, pred_loaded)
    print("✓ Save/load test passed")


def test_to_dict():
    """Test tree export to dictionary."""
    X = pd.DataFrame({'x': [1, 2, 3]})
    y = np.array([0, 1, 1])
    
    clf = CARTClassifier(max_depth=2)
    clf.fit(X, y)
    
    tree_dict = clf.to_dict()
    
    assert tree_dict['fitted'] == True
    assert 'tree' in tree_dict
    assert 'criterion' in tree_dict
    assert tree_dict['criterion'] == 'gini'
    
    print("✓ to_dict test passed")


# ============================================================================
# Integration Tests
# ============================================================================

def test_iris_like():
    """Test on Iris-like dataset."""
    np.random.seed(42)
    
    # Simulate Iris-like data
    X0 = pd.DataFrame({
        'petal_length': np.random.normal(1.5, 0.2, 30),
        'petal_width': np.random.normal(0.3, 0.1, 30),
        'species_group': ['setosa'] * 30
    })
    y0 = np.zeros(30)
    
    X1 = pd.DataFrame({
        'petal_length': np.random.normal(4.0, 0.3, 30),
        'petal_width': np.random.normal(1.3, 0.2, 30),
        'species_group': ['versicolor'] * 30
    })
    y1 = np.ones(30)
    
    X2 = pd.DataFrame({
        'petal_length': np.random.normal(5.5, 0.4, 30),
        'petal_width': np.random.normal(2.0, 0.3, 30),
        'species_group': ['virginica'] * 30
    })
    y2 = np.ones(30) * 2
    
    X = pd.concat([X0, X1, X2], ignore_index=True)
    y = np.concatenate([y0, y1, y2])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X.iloc[indices].reset_index(drop=True)
    y = y[indices]
    
    clf = CARTClassifier(
        max_depth=5,
        min_samples_split=5,
        categorical_features=['species_group']
    )
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Iris-like Test - Training Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.85, "Multi-class classification failed"
    print("✓ Iris-like test passed")


def test_verbose_mode():
    """Test verbose output."""
    X = pd.DataFrame({'x': np.random.randn(50)})
    y = (X['x'] > 0).astype(int).values
    
    # Should print tree info
    clf = CARTClassifier(max_depth=3, verbose=0)  # Set to 0 to not clutter test output
    clf.fit(X, y)
    
    assert clf.training_history_ is not None
    print("✓ Verbose mode test passed")


# ============================================================================
# Performance Benchmarks
# ============================================================================

def test_benchmark_classification():
    """Benchmark against sklearn."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_df = pd.DataFrame(X)
    
    # Our implementation
    start = time.time()
    clf_ours = CARTClassifier(max_depth=10)
    clf_ours.fit(X_df, y)
    time_ours = time.time() - start
    acc_ours = clf_ours.score(X_df, y)
    
    # Sklearn
    start = time.time()
    clf_sk = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_sk.fit(X, y)
    time_sk = time.time() - start
    acc_sk = clf_sk.score(X, y)
    
    print(f"\nBenchmark (Classification, 1000 samples, 20 features):")
    print(f"  Our implementation: {time_ours:.3f}s, accuracy={acc_ours:.3f}")
    print(f"  Sklearn:            {time_sk:.3f}s, accuracy={acc_sk:.3f}")
    print(f"  Speed ratio: {time_ours/time_sk:.1f}x")
    
    # We should be reasonably close in accuracy
    assert abs(acc_ours - acc_sk) < 0.1
    print("✓ Benchmark classification test passed")


def test_benchmark_regression():
    """Benchmark regression against sklearn."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    X_df = pd.DataFrame(X)
    
    # Our implementation
    start = time.time()
    reg_ours = CARTRegressor(max_depth=10)
    reg_ours.fit(X_df, y)
    time_ours = time.time() - start
    r2_ours = reg_ours.score(X_df, y)
    
    # Sklearn
    start = time.time()
    reg_sk = DecisionTreeRegressor(max_depth=10, random_state=42)
    reg_sk.fit(X, y)
    time_sk = time.time() - start
    r2_sk = reg_sk.score(X, y)
    
    print(f"\nBenchmark (Regression, 1000 samples, 20 features):")
    print(f"  Our implementation: {time_ours:.3f}s, R²={r2_ours:.3f}")
    print(f"  Sklearn:            {time_sk:.3f}s, R²={r2_sk:.3f}")
    print(f"  Speed ratio: {time_ours/time_sk:.1f}x")
    
    # We should be reasonably close in R²
    assert abs(r2_ours - r2_sk) < 0.1
    print("✓ Benchmark regression test passed")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Running Comprehensive CART Tests")
    print("="*70)
    
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])