"""
Breiman CART Implementation (1984)
====================================

A pure NumPy/Pandas implementation of Classification and Regression Trees
following the original methodology from Breiman, Friedman, Olshen, and Stone (1984).

Features
--------
- Binary decision trees for classification and regression
- Support for both numerical and categorical features
- Cost-complexity pruning for model selection
- Feature importance calculation
- Model persistence (save/load)
- Comprehensive validation and error handling
- Clean sklearn-compatible API

Quick Start
-----------
>>> import breiman_cart as brc
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Create sample data
>>> X = pd.DataFrame({'x1': [0, 1, 2, 3], 'x2': [0, 1, 1, 0]})
>>> y = np.array([0, 1, 1, 0])
>>>
>>> # Classification
>>> clf = brc.BRCClassification(max_depth=3)
>>> clf.fit(X, y)
>>> predictions = clf.predict(X)
>>>
>>> # Regression
>>> reg = brc.BRCRegression(max_depth=5)
>>> reg.fit(X, y)
>>>
>>> # Inference
>>> inference = brc.BRCInference(clf)
>>> predictions = inference.predict(X)

Classes
-------
BRCClassification : Classification tree (main user class)
BRCRegression : Regression tree (main user class)
BRCInference : Inference engine for predictions

Legacy Classes (still available)
---------------------------------
CARTClassifier : Original classification tree
CARTRegressor : Original regression tree
Node : Tree node structure
Splitter : Split finding algorithm
CostComplexityPruner : Pruning algorithm

References
----------
Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
Classification and Regression Trees. Wadsworth.
"""

# Import new user-friendly classes
from .breiman_cart import BRCRegression, BRCClassification, BRCInference

# Import legacy classes for backward compatibility
from .breiman_cart import Node, Splitter, CostComplexityPruner

__version__ = "0.2.0"
__author__ = "Adam Khald"
__doc_url__ = "https://github.com/Adamkhald/breiman-cart"
__license__ = "MIT"

# Main exports (what users should use)
__all__ = [
    "BRCRegression",
    "BRCClassification",
    "BRCInference",
    # Legacy classes still available
    "Node",
    "Splitter",
    "CostComplexityPruner"
]

# Version history
__changelog__ = """
Version 0.2.0 (Current)
-----------------------
- NEW: Clean sklearn-compatible API with BRCRegression, BRCClassification, BRCInference
- NEW: BRCInference class for easy model loading and prediction
- Improved user experience with intuitive class names
- All original functionality preserved
- Comprehensive documentation and examples
- Better error messages and validation
- Feature importance as named dictionary in BRCInference

Version 0.1.1
-------------
- Added comprehensive input validation
- Improved performance for categorical features
- Added feature importance calculation
- Added model save/load functionality
- Added extensive logging support
- Improved pruning algorithm efficiency
- Added comprehensive test suite
- Better error messages and handling
- Added tree visualization helpers
- Improved documentation

Version 0.1.0
-------------
- Initial implementation
- Basic classification and regression
- Cost-complexity pruning
- Categorical feature support
"""
