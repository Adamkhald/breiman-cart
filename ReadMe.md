# Breiman CART Implementation (1984)

A production-grade, from-scratch implementation of Classification and Regression Trees (CART) following the original 1984 methodology by Breiman, Friedman, Olshen, and Stone.

## Key Features

- **True Categorical Handling**: Native subset splitting for categorical features (not one-hot encoding)
- **Cost-Complexity Pruning**: Minimal cost-complexity pruning with alpha-based regularization
- **No ML Libraries**: Pure NumPy/Pandas implementation for educational and research purposes
- **Type-Safe**: Full type hints throughout the codebase

## Installation

```bash
pip install numpy pandas
cd breiman_cart
```

## Quick Start

### Classification Example

```python
import numpy as np
import pandas as pd
from src.tree import CARTClassifier

# Create sample data
X = pd.DataFrame({
    'age': [25, 45, 35, 50, 23],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA']
})
y = np.array([0, 1, 0, 1, 0])

# Specify categorical columns
categorical_features = ['city']

# Fit the tree
tree = CARTClassifier(
    max_depth=5,
    min_samples_split=2,
    categorical_features=categorical_features
)
tree.fit(X, y)

# Predict
predictions = tree.predict(X)

# Prune the tree
tree.prune(X_val, y_val)
```

### Regression Example

```python
from src.tree import CARTRegressor

X = pd.DataFrame({
    'sqft': [1200, 1800, 1500, 2200],
    'neighborhood': ['A', 'B', 'A', 'C']
})
y = np.array([300000, 450000, 350000, 550000])

tree = CARTRegressor(
    max_depth=4,
    min_samples_split=2,
    categorical_features=['neighborhood']
)
tree.fit(X, y)
predictions = tree.predict(X)
```

## Architecture

- `node.py`: Node data structure with categorical support
- `splitter.py`: Splitting logic (Gini, MSE, subset splitting)
- `tree.py`: Main CART classes (Classifier/Regressor)
- `pruning.py`: Cost-complexity pruning implementation

## Mathematical Background

See `docs/theory.tex` for detailed mathematical formulation.

## Differences from Scikit-Learn

1. **Categorical Features**: True subset splitting vs one-hot encoding
2. **Pruning**: Implements full cost-complexity pruning sequence
3. **Transparency**: Educational codebase following the original paper

## Testing

```bash
python -m pytest tests/
```

## References

Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
