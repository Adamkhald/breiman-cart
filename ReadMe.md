# 🌳 Breiman CART

[![PyPI version](https://badge.fury.io/py/breiman-cart.svg)](https://badge.fury.io/py/breiman-cart)
[![Python](https://img.shields.io/pypi/pyversions/breiman-cart.svg)](https://pypi.org/project/breiman-cart/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/breiman-cart)](https://pepy.tech/project/breiman-cart)

A pure Python implementation of **Classification and Regression Trees (CART)** following the original methodology from Breiman, Friedman, Olshen, and Stone (1984).

**Why use this?** Unlike sklearn's implementation, this provides a faithful reproduction of the original CART algorithm with full access to the tree-building process, cost-complexity pruning, and educational transparency.

---

## ✨ Features

- 🎯 **Original CART Algorithm** - Faithful implementation of Breiman et al. (1984)
- 🔢 **Regression & Classification** - Support for both continuous and categorical targets
- 📊 **Categorical Features** - Native handling of categorical variables without encoding
- ✂️ **Cost-Complexity Pruning** - Automatic tree pruning to prevent overfitting
- 🎨 **Feature Importance** - Built-in feature importance calculation
- 💾 **Model Persistence** - Easy save/load functionality
- 🔍 **Tree Inspection** - Full access to tree structure and node information
- 🤖 **Sklearn Compatible** - Familiar API for ML practitioners

---

## 🚀 Quick Start

### Installation

```bash
pip install breiman-cart
```

### Basic Usage

```python
import breiman_cart as brc
import pandas as pd
import numpy as np

# Prepare your data
X = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [30000, 35000, 50000, 60000, 70000, 80000, 90000, 100000]
})
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Train a classifier
model = brc.BRCClassification(max_depth=5, min_samples_split=2)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 📚 Complete Examples

### 🔵 Regression Example

```python
import breiman_cart as brc
import pandas as pd
import numpy as np

# Create regression data
X_train = pd.DataFrame({
    'square_feet': [1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
    'bedrooms': [2, 3, 3, 4, 4, 4, 5, 5],
    'age': [10, 8, 5, 3, 2, 1, 0, 0]
})
y_train = np.array([250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000])

# Train model
model = brc.BRCRegression(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    verbose=1  # Show training progress
)
model.fit(X_train, y_train)

# Make predictions
X_test = pd.DataFrame({
    'square_feet': [1600, 2400],
    'bedrooms': [3, 4],
    'age': [7, 2]
})
predictions = model.predict(X_test)
print(f"Predicted prices: ${predictions}")

# Get R² score
r2 = model.score(X_train, y_train)
print(f"R² Score: {r2:.3f}")

# Feature importance
importance = model.get_feature_importance()
for feature, score in zip(X_train.columns, importance):
    print(f"{feature}: {score:.3f}")
```

### 🟢 Classification Example

```python
import breiman_cart as brc
import pandas as pd
import numpy as np

# Create classification data
X_train = pd.DataFrame({
    'temperature': [30, 25, 20, 15, 10, 5, 0, -5],
    'humidity': [80, 75, 70, 65, 60, 55, 50, 45],
    'wind_speed': [5, 10, 15, 20, 25, 30, 35, 40]
})
y_train = np.array([1, 1, 1, 0, 0, 0, 0, 0])  # 1=Rain, 0=No Rain

# Train classifier
clf = brc.BRCClassification(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1
)
clf.fit(X_train, y_train)

# Predictions
X_test = pd.DataFrame({
    'temperature': [22, 8],
    'humidity': [72, 58],
    'wind_speed': [12, 28]
})
predictions = clf.predict(X_test)
print(f"Predictions: {predictions}")  # [1, 0]

# Accuracy
accuracy = clf.score(X_train, y_train)
print(f"Accuracy: {accuracy:.2%}")
```

### 🔮 Inference Engine

Use the inference engine for production deployments:

```python
import breiman_cart as brc

# Option 1: From trained model
model = brc.BRCRegression(max_depth=5)
model.fit(X_train, y_train)

inference = brc.BRCInference(model)

# Option 2: Load from saved file
inference = brc.BRCInference('my_model.pkl')

# Make predictions
predictions = inference.predict(X_new)

# Get model information
info = inference.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Tree depth: {info['tree_depth']}")
print(f"Number of leaves: {info['n_leaves']}")

# Feature importance (as dictionary)
importance = inference.get_feature_importance()
print(importance)  # {'feature1': 0.65, 'feature2': 0.35}

# Pretty print
print(inference)
```

---

## 🎓 Advanced Features

### Cost-Complexity Pruning

Prevent overfitting with automatic pruning:

```python
# Train a full tree
model = brc.BRCRegression(max_depth=10)
model.fit(X_train, y_train)

# Prune using validation set
# (automatically finds optimal alpha)
model.prune(X_val, y_val)

# Or specify alpha manually
model.prune(X_val, y_val, alpha=0.01)
```

### Categorical Features

Native support for categorical variables:

```python
X = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green', 'blue', 'green'],
    'size': ['S', 'M', 'L', 'M', 'S', 'L'],
    'price': [10, 20, 15, 25, 12, 30]
})
y = np.array([0, 1, 0, 1, 0, 1])

# Specify categorical features
model = brc.BRCClassification(
    max_depth=5,
    categorical_features=['color', 'size']  # These won't be treated as numerical
)
model.fit(X, y)
```

### Model Persistence

Save and load models:

```python
# Save model
model.save('my_cart_model.pkl')

# Load model later
loaded_model = brc.BRCRegression.load('my_cart_model.pkl')

# Or use with inference
inference = brc.BRCInference('my_cart_model.pkl')
```

### Tree Inspection

Explore the tree structure:

```python
# Get tree statistics
print(f"Tree depth: {model.root.get_depth()}")
print(f"Number of leaves: {model.root.count_leaves()}")
print(f"Total nodes: {model.root.get_n_nodes()}")

# Export tree structure
tree_dict = model.to_dict()

# Save as JSON
model.to_json('tree_structure.json')
```

---

## 🔧 API Reference

### BRCRegression

**Decision tree regressor using MSE criterion.**

```python
brc.BRCRegression(
    max_depth=None,           # Maximum tree depth
    min_samples_split=2,      # Min samples to split node
    min_samples_leaf=1,       # Min samples at leaf
    categorical_features=[],  # List of categorical feature names
    verbose=0                 # Verbosity level (0, 1, or 2)
)
```

**Methods:**
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `score(X, y)` - Calculate R² score
- `get_feature_importance()` - Get feature importances
- `prune(X_val, y_val, alpha=None)` - Prune the tree
- `save(filepath)` - Save model to disk
- `load(filepath)` - Load model from disk (static method)

### BRCClassification

**Decision tree classifier using Gini impurity criterion.**

```python
brc.BRCClassification(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    categorical_features=[],
    verbose=0
)
```

**Methods:** Same as BRCRegression, but `score()` returns accuracy.

### BRCInference

**Inference engine for making predictions with trained models.**

```python
brc.BRCInference(model)  # Pass trained model or file path
```

**Methods:**
- `predict(X)` - Make predictions
- `score(X, y)` - Evaluate performance
- `get_feature_importance()` - Get feature importance dict
- `get_tree_depth()` - Get tree depth
- `get_n_leaves()` - Get number of leaves
- `get_n_nodes()` - Get total nodes
- `get_model_info()` - Get comprehensive model info
- `save_model(filepath)` - Save model
- `export_to_json(filepath)` - Export tree as JSON

---

## 📊 Comparison with Sklearn

| Feature | breiman-cart | sklearn DecisionTree |
|---------|--------------|---------------------|
| CART Algorithm | ✅ Original (1984) | Modified |
| Cost-Complexity Pruning | ✅ Automatic | Manual (`ccp_alpha`) |
| Categorical Features | ✅ Native | Requires encoding |
| Tree Inspection | ✅ Full access | Limited |
| Educational Value | ✅ High | Medium |
| Performance | Good | Excellent (C++) |
| Interpretability | ✅ Excellent | Good |

**Use breiman-cart when:**
- You want the original CART algorithm
- You need educational transparency
- You're working with categorical features
- You want easy tree inspection
- You need cost-complexity pruning

**Use sklearn when:**
- Performance is critical
- You need integration with sklearn pipelines
- You're working with very large datasets

---

## 🧪 Testing

Run the test suite:

```python
# test_example.py
import breiman_cart as brc
import pandas as pd
import numpy as np

# Test regression
X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
y = np.array([3, 6, 9, 12, 15])

reg = brc.BRCRegression(max_depth=5)
reg.fit(X, y)
assert reg.score(X, y) > 0.9, "Regression test failed"

# Test classification
y_class = np.array([0, 0, 1, 1, 1])
clf = brc.BRCClassification(max_depth=3)
clf.fit(X, y_class)
assert clf.score(X, y_class) >= 0.8, "Classification test failed"

# Test inference
inference = brc.BRCInference(reg)
preds = inference.predict(X)
assert len(preds) == len(X), "Inference test failed"

print("✅ All tests passed!")
```

---

## 📖 Background

This implementation follows the seminal work:

> **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).**  
> *Classification and Regression Trees.*  
> Wadsworth International Group.

CART is a decision tree algorithm that:
1. **Recursively partitions** the feature space into rectangular regions
2. **Uses Gini impurity** (classification) or **MSE** (regression) for splitting
3. **Employs binary splits** only (unlike ID3/C4.5)
4. **Prunes using cost-complexity** to find the optimal subtree

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Adam Khald** - adamkhald@outlook.com

Project Link: [https://github.com/Adamkhald/breiman-cart](https://github.com/Adamkhald/breiman-cart)

---

## 🙏 Acknowledgments

- Leo Breiman and colleagues for the original CART algorithm
- The scikit-learn team for API inspiration
- The Python data science community

---

## 📈 Changelog

### Version 0.2.0 (Current)
- ✨ **NEW:** Clean sklearn-compatible API with `BRCRegression`, `BRCClassification`, `BRCInference`
- ✨ **NEW:** `BRCInference` class for production deployments
- 🎨 Improved user experience with intuitive class names
- 📚 Comprehensive documentation and examples
- 🐛 Better error messages and validation
- 🔧 Feature importance as named dictionary in `BRCInference`

### Version 0.1.1
- Added comprehensive input validation
- Improved performance for categorical features
- Added model save/load functionality
- Enhanced pruning algorithm efficiency

### Version 0.1.0
- Initial release
- Basic CART implementation
- Cost-complexity pruning
- Categorical feature support

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=Adamkhald/breiman-cart&type=Date)](https://star-history.com/#Adamkhald/breiman-cart&Date)

---

**Made with ❤️ by Adam Khald**