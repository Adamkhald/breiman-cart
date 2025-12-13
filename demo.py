"""
Demo scripts for running CART on Breast Cancer (classification) 
and Diabetes (regression) datasets.

Make sure you have the breiman_cart package structure set up:
breiman_cart/
├── src/
│   ├── __init__.py
│   ├── node.py
│   ├── splitter.py
│   ├── tree.py
│   └── pruning.py
└── demo.py (this file)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Import your CART implementation
from src.tree import CARTClassifier, CARTRegressor


def demo_classification():
    """Demo: Breast Cancer Classification"""
    print("=" * 70)
    print("CART CLASSIFICATION DEMO: Breast Cancer Dataset")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Convert to DataFrame (CART expects DataFrame)
    feature_names = data.feature_names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Classes: {np.unique(y)}")
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training into train/validation for pruning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Validation samples: {len(X_val)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Train CART Classifier
    print("\n3. Training CART Classifier...")
    clf = CARTClassifier(
        max_depth=10,           # Allow deep tree initially
        min_samples_split=5,    # Minimum samples to split a node
        min_samples_leaf=2,     # Minimum samples in a leaf
        categorical_features=[] # No categorical features in this dataset
    )
    
    clf.fit(X_train, y_train)
    print("   ✓ Training complete!")
    
    # Evaluate before pruning
    print("\n4. Evaluation BEFORE pruning:")
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"   - Training Accuracy:   {train_acc:.4f}")
    print(f"   - Validation Accuracy: {val_acc:.4f}")
    print(f"   - Test Accuracy:       {test_acc:.4f}")
    print(f"   - Tree leaves:         {clf.root.count_leaves()}")
    
    # Prune the tree
    print("\n5. Pruning tree using validation set...")
    clf.prune(X_val, y_val)
    print("   ✓ Pruning complete!")
    
    # Evaluate after pruning
    print("\n6. Evaluation AFTER pruning:")
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    train_acc_pruned = accuracy_score(y_train, y_train_pred)
    val_acc_pruned = accuracy_score(y_val, y_val_pred)
    test_acc_pruned = accuracy_score(y_test, y_test_pred)
    
    print(f"   - Training Accuracy:   {train_acc_pruned:.4f}")
    print(f"   - Validation Accuracy: {val_acc_pruned:.4f}")
    print(f"   - Test Accuracy:       {test_acc_pruned:.4f}")
    print(f"   - Tree leaves:         {clf.root.count_leaves()}")
    
    # Detailed classification report
    print("\n7. Detailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=data.target_names))
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION DEMO COMPLETE!")
    print("=" * 70 + "\n")


def demo_regression():
    """Demo: Diabetes Regression"""
    print("=" * 70)
    print("CART REGRESSION DEMO: Diabetes Dataset")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading Diabetes dataset...")
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Convert to DataFrame
    feature_names = data.feature_names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Further split for pruning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Validation samples: {len(X_val)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Train CART Regressor
    print("\n3. Training CART Regressor...")
    reg = CARTRegressor(
        max_depth=8,            # Allow reasonably deep tree
        min_samples_split=10,   # Prevent overfitting
        min_samples_leaf=5,     # Minimum samples in leaves
        categorical_features=[] # No categorical features
    )
    
    reg.fit(X_train, y_train)
    print("   ✓ Training complete!")
    
    # Evaluate before pruning
    print("\n4. Evaluation BEFORE pruning:")
    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)
    y_test_pred = reg.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"   - Training MSE: {train_mse:.2f} | R²: {train_r2:.4f}")
    print(f"   - Validation MSE: {val_mse:.2f} | R²: {val_r2:.4f}")
    print(f"   - Test MSE: {test_mse:.2f} | R²: {test_r2:.4f}")
    print(f"   - Tree leaves: {reg.root.count_leaves()}")
    
    # Prune the tree
    print("\n5. Pruning tree using validation set...")
    reg.prune(X_val, y_val)
    print("   ✓ Pruning complete!")
    
    # Evaluate after pruning
    print("\n6. Evaluation AFTER pruning:")
    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)
    y_test_pred = reg.predict(X_test)
    
    train_mse_pruned = mean_squared_error(y_train, y_train_pred)
    train_r2_pruned = r2_score(y_train, y_train_pred)
    val_mse_pruned = mean_squared_error(y_val, y_val_pred)
    val_r2_pruned = r2_score(y_val, y_val_pred)
    test_mse_pruned = mean_squared_error(y_test, y_test_pred)
    test_r2_pruned = r2_score(y_test, y_test_pred)
    
    print(f"   - Training MSE: {train_mse_pruned:.2f} | R²: {train_r2_pruned:.4f}")
    print(f"   - Validation MSE: {val_mse_pruned:.2f} | R²: {val_r2_pruned:.4f}")
    print(f"   - Test MSE: {test_mse_pruned:.2f} | R²: {test_r2_pruned:.4f}")
    print(f"   - Tree leaves: {reg.root.count_leaves()}")
    
    # Show some predictions
    print("\n7. Sample Predictions (first 10 test samples):")
    print("   Actual  | Predicted | Error")
    print("   " + "-" * 35)
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        pred = y_test_pred[i]
        error = abs(actual - pred)
        print(f"   {actual:6.1f}  | {pred:9.1f} | {error:6.1f}")
    
    print("\n" + "=" * 70)
    print("REGRESSION DEMO COMPLETE!")
    print("=" * 70 + "\n")


def demo_categorical_example():
    """Demo: Categorical features example"""
    print("=" * 70)
    print("BONUS: CATEGORICAL FEATURES DEMO")
    print("=" * 70)
    
    print("\n1. Creating synthetic dataset with categorical features...")
    
    # Create a dataset with mixed numerical and categorical features
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n_samples)
    })
    
    # Create target based on complex rules
    y = (
        (X['age'] > 35).astype(int) +
        (X['income'] > 55000).astype(int) +
        (X['city'].isin(['NYC', 'LA'])).astype(int) +
        (X['education'].isin(['Master', 'PhD'])).astype(int)
    )
    y = (y >= 2).astype(int)  # Binary classification
    
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {X.columns.tolist()}")
    print(f"   - Categorical features: ['city', 'education']")
    print(f"   - Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train with categorical features
    print("\n2. Training CART with categorical features...")
    clf = CARTClassifier(
        max_depth=6,
        min_samples_split=10,
        categorical_features=['city', 'education']  # Specify categorical columns
    )
    
    clf.fit(X_train, y_train)
    print("   ✓ Training complete!")
    
    # Evaluate
    print("\n3. Evaluation:")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   - Test Accuracy: {acc:.4f}")
    print(f"   - Tree leaves: {clf.root.count_leaves()}")
    
    print("\n   Note: The tree used SUBSET SPLITTING for categorical features!")
    print("   Example: city in {NYC, LA} vs {Chicago, Houston}")
    print("   This is more efficient than one-hot encoding!")
    
    print("\n" + "=" * 70)
    print("CATEGORICAL DEMO COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """
    Run all demos.
    
    Usage:
        python demo.py
    """
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BREIMAN CART (1984) DEMONSTRATION" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    # Run classification demo
    demo_classification()
    input("Press Enter to continue to regression demo...")
    
    # Run regression demo
    demo_regression()
    input("Press Enter to continue to categorical demo...")
    
    # Run categorical demo
    demo_categorical_example()
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "ALL DEMOS COMPLETED!" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\nThank you for testing Breiman CART implementation!")
    print("For more info, see README.md and docs/theory.tex\n")