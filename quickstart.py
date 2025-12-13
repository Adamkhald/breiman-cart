"""
Quick Start: Minimal examples to get you running immediately
Save this as: quickstart.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.tree import CARTClassifier, CARTRegressor


# ============================================================================
# EXAMPLE 1: Classification (Breast Cancer)
# ============================================================================

print("🌳 EXAMPLE 1: Classification\n")

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf = CARTClassifier(max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of leaves: {clf.root.count_leaves()}\n")


# ============================================================================
# EXAMPLE 2: Regression (Diabetes)
# ============================================================================

print("🌳 EXAMPLE 2: Regression\n")

# Load data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
reg = CARTRegressor(max_depth=5, min_samples_split=10)
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")
print(f"Number of leaves: {reg.root.count_leaves()}\n")


# ============================================================================
# EXAMPLE 3: With Pruning
# ============================================================================

print("🌳 EXAMPLE 3: With Pruning\n")

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train/val/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train deep tree
clf = CARTClassifier(max_depth=15, min_samples_split=2)
clf.fit(X_train, y_train)

# Accuracy before pruning
acc_before = accuracy_score(y_test, clf.predict(X_test))
leaves_before = clf.root.count_leaves()

# Prune using validation set
clf.prune(X_val, y_val)

# Accuracy after pruning
acc_after = accuracy_score(y_test, clf.predict(X_test))
leaves_after = clf.root.count_leaves()

print(f"BEFORE pruning: Accuracy={acc_before:.4f}, Leaves={leaves_before}")
print(f"AFTER pruning:  Accuracy={acc_after:.4f}, Leaves={leaves_after}")
print(f"→ Pruning removed {leaves_before - leaves_after} leaves!\n")


# ============================================================================
# EXAMPLE 4: Categorical Features
# ============================================================================

print("🌳 EXAMPLE 4: Categorical Features\n")

# Create synthetic data with categorical features
np.random.seed(42)
X = pd.DataFrame({
    'age': np.random.randint(18, 65, 300),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 300),
    'job': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], 300)
})

# Target: complex rule involving categorical features
y = ((X['city'] == 'NYC') | (X['city'] == 'LA')) & (X['age'] > 35)
y = y.astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with categorical features specified
clf = CARTClassifier(
    max_depth=4,
    min_samples_split=10,
    categorical_features=['city', 'job']  # ← Specify categorical columns
)
clf.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy: {acc:.4f}")
print(f"Number of leaves: {clf.root.count_leaves()}")
print("✓ Used SUBSET SPLITTING (not one-hot encoding)!\n")


print("=" * 60)
print("✅ All examples completed successfully!")
print("=" * 60)